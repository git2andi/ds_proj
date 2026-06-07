"""
orchestrator.py
---------------
Coordinates one dialogue run.

Refactored runtime model:
  - The LLM only writes the next utterance.
  - `policy.select_next_speakers()` only routes speakers.
  - Live decisions use explicit votes + explicit accept/reject statements.
  - DialogueMemory keeps only compact prompt/logging memory.
"""

from __future__ import annotations

import datetime
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from logger import DialogueLogger
from policy import extract_discourse, repetition_pressure as compute_repetition_pressure, sample_hard_blocker, select_next_speakers
from state import DialogueMemory
from utils import OptionResolver


# ---------------------------------------------------------------------------
# Live dialogue state
# ---------------------------------------------------------------------------

@dataclass
class DialogueState:
    phase: str = "opening"
    turn_index: int = 0

    has_asked_narrowing: bool = False
    agreement_reached: bool = False

    preferred_option: Optional[str] = None
    current_leading_option: Optional[str] = None
    candidate_option: Optional[str] = None

    last_addressed: Optional[str] = None
    pending_question_target: Optional[str] = None

    repetition_pressure: float = 0.0
    confirmation_rejection_count: int = 0

    # Live-control state. These are the authoritative decision signals.
    explicit_votes: dict[str, str] = field(default_factory=dict)
    explicit_accepts: dict[str, set[str]] = field(default_factory=dict)
    explicit_rejects: dict[str, dict[str, str]] = field(default_factory=dict)

    # Context for short confirmation replies. When the moderator asks
    # "Léa, could you live with Option C?", a bare "that's fine" must be
    # recorded as Léa accepting C.
    pending_confirmation_target: Optional[str] = None
    pending_confirmation_candidate: Optional[str] = None

    vote_changes: dict = field(default_factory=dict)
    last_known_vote: dict = field(default_factory=dict)
    rejected_options_by_speaker: dict = field(default_factory=dict)
    outcome_reason: str = ""

    # Moderator style/control: avoid repeating the same full holdout prompt.
    candidate_prompt_counts: dict[str, int] = field(default_factory=dict)

    # Only a targeted confirmation "no" hard-excludes a candidate. Discussion-
    # phase "not sold" lines remain useful context but should not prevent the
    # moderator from testing a single-option compromise.
    confirmation_rejected_options: set[str] = field(default_factory=set)


_ACCEPT_RE = re.compile(
    r"\b(?:can\s+live\s+with|could\s+live\s+with|works?\s+for\s+me|"
    r"works?\s+for\s+us|works?\s+(?:well\s+enough|fine|okay|ok)|"
    r"works?\s+(?:as\s+)?(?:a\s+)?compromise|is\s+acceptable|acceptable\s+as\s+compromise|"
    r"(?:is|seems|sounds)\s+(?:okay|ok|fine|workable|acceptable)|"
    r"good\s+with|fine\s+with|okay\s+with|ok\s+with|"
    r"i'?m\s+good\s+with|i'?m\s+fine\s+with|i\s+can\s+accept|"
    r"i'?d\s+accept|i\s+could\s+accept|i\s+can\s+do|i\s+could\s+do|"
    r"i'?m\s+in\s+for|i\s+can\s+go\s+with|could\s+go\s+with)\b",
    re.I,
)
_RULE_OUT_RE = re.compile(r"\b(?:i(?:\'d| would)?|we should|let(?:\'s| us)|given .{0,40}i(?:\'d| would)?)\s+rule\s+out\s+(?:Option\s+)?([A-D])\b", re.I)
_CHANGED_MIND_RE = re.compile(r"\b(?:changed my mind|change my mind|reconsidered|after thinking|on second thought|actually|despite what I said|I know I ruled it out)\b", re.I)

_REJECT_RE = re.compile(
    r"\b(?:not\s+sold\s+on|can'?t\s+live\s+with|couldn'?t\s+live\s+with|"
    r"not\s+okay\s+with|not\s+good\s+with|not\s+fine\s+with|"
    r"don'?t\s+want|wouldn'?t\s+do|can'?t\s+do|no\s+to|rather\s+not|"
    r"still\s+not\s+convinced|doesn'?t\s+work\s+for\s+me|won'?t\s+work)\b",
    re.I,
)
_YES_RE = re.compile(
    r"(?:^\s*(?:yes|yeah|yep|sure|ok|okay|fine|agreed|works|works\s+for\s+me|"
    r"that\s+works|that'?s\s+fine|that\s+is\s+fine|sounds\s+good|good\s+with\s+me|"
    r"i\s+can\s+live\s+with\s+that)\b|\bbut\s+yeah\b|\byeah\s*$)",
    re.I,
)
_NO_RE = re.compile(r"^\s*(?:no|nope|nah|not\s+really|not\s+sold|still\s+not)\b", re.I)


class Orchestrator:

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.sims: list[Any] = []

        self._llm = get_llm_client()
        self.state = DialogueState()

        self.dialogue_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.outcome: str = "pending"

        self.options, self.opening_question = self._generate_options()
        self.resolver = OptionResolver(self.options)
        self.history: list[str] = self._build_opening_history()

        self._logger = DialogueLogger(self.dialogue_id, topic)
        self._memory: Optional[DialogueMemory] = None

    def add_sim(self, sim: Any) -> None:
        self.sims.append(sim)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _generate_options(self) -> tuple[list[str], str]:
        attempts = int(getattr(cfg.option_generation, "max_attempts", 1))
        last_error = ""
        for _ in range(max(1, attempts)):
            data = self._llm.generate_json(prompts.option_generation(self.topic))
            options_raw = data.get("options", [])
            question = str(data.get("opening_question", "")).strip()
            try:
                cleaned = self._clean_generated_options(options_raw)
                self._validate_generated_options(cleaned)
            except ValueError as exc:
                last_error = str(exc)
                continue
            if not question:
                last_error = "Option generation returned no opening_question."
                continue
            decision_kind = str(data.get("decision_kind", "")).strip().lower()
            if decision_kind:
                print(f"  [options] decision_kind={decision_kind}")
            return cleaned, question
        raise ValueError(f"Option generation failed validation: {last_error}")

    def _clean_generated_options(self, options_raw: Any) -> list[str]:
        expected = int(cfg.option_generation.option_count)
        if not isinstance(options_raw, list) or len(options_raw) != expected:
            raise ValueError(f"Option generation expected {expected} options, got: {options_raw!r}")
        cleaned: list[str] = []
        for i, raw in enumerate(options_raw):
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Option generation returned an empty option at index {i}.")
            label = chr(ord("A") + i)
            text = raw.strip()
            if not text.lower().startswith(f"option {label.lower()}"):
                text = f"Option {label} - {text}"
            cleaned.append(text)
        return cleaned

    def _validate_generated_options(self, options: list[str]) -> None:
        required = [str(x).lower() for x in getattr(cfg.option_generation, "required_sections", [])]
        forbidden = [str(x).lower() for x in getattr(cfg.option_generation, "forbidden_terms", [])]
        for option in options:
            lower = option.lower()
            missing = [section for section in required if section not in lower]
            if missing:
                raise ValueError(f"Option card missing section(s) {missing}: {option}")
            blocked = [term for term in forbidden if term and term in lower]
            if blocked:
                raise ValueError(f"Option card contains live-checking term(s) {blocked}: {option}")

    def _display_option(self, option: str) -> str:
        """Readable transcript form for option cards.

        Internally the simulator keeps the structured `attrs: key=value` form for
        prompts and parsing. The public transcript should not look like a raw
        database row, so opening lines are lightly humanised here.
        """
        if ": attrs:" not in option:
            return option
        head, rest = option.split(": attrs:", 1)
        attrs_part, sep, tail = rest.partition("; upside:")

        labels = {
            "price_per_person_eur": lambda v: f"about €{v}/person",
            "price_per_night_eur": lambda v: f"about €{v}/night",
            "price_eur": lambda v: f"about €{v}",
            "travel_time_min": lambda v: f"{v} min away",
            "expected_wait_min": lambda v: f"{v} min expected wait",
            "city_center_min": lambda v: f"{v} min from center",
            "transit_walk_min": lambda v: f"{v} min walk to transit",
            "room_size_m2": lambda v: f"{v} m² room",
            "noise_level_1_5": lambda v: f"noise {v}/5",
            "menu_variety_1_5": lambda v: f"menu variety {v}/5",
            "vegetarian_options_1_5": lambda v: f"vegetarian options {v}/5",
            "allergen_safety_1_5": lambda v: f"allergen safety {v}/5",
            "local_business_1_5": lambda v: f"local-business fit {v}/5",
            "reservation_possible": lambda v: "reservations possible" if str(v).lower() == "true" else "no reservations",
            "cancellation_flexibility_1_5": lambda v: f"cancellation flexibility {v}/5",
            "breakfast_included": lambda v: "breakfast included" if str(v).lower() == "true" else "breakfast not included",
            "departure_time": lambda v: f"departs {v}",
            "duration_min": lambda v: f"{v} min flight",
            "stops": lambda v: f"{v} stops",
            "baggage_included": lambda v: "baggage included" if str(v).lower() == "true" else "baggage extra",
            "change_fee_eur": lambda v: f"€{v} change fee",
            "comfort_1_5": lambda v: f"comfort {v}/5",
            "schedule_buffer_1_5": lambda v: f"schedule buffer {v}/5",
            "cost_1_5": lambda v: f"cost {v}/5",
            "effort_1_5": lambda v: f"effort {v}/5",
            "novelty_1_5": lambda v: f"novelty {v}/5",
            "group_fit_1_5": lambda v: f"group fit {v}/5",
            "risk_1_5": lambda v: f"risk {v}/5",
            "impact_1_5": lambda v: f"impact {v}/5",
            "feasibility_1_5": lambda v: f"feasibility {v}/5",
            "safety_1_5": lambda v: f"safety {v}/5",
        }
        details: list[str] = []
        for item in attrs_part.split(","):
            if "=" not in item:
                continue
            key, val = [x.strip() for x in item.split("=", 1)]
            formatter = labels.get(key)
            if formatter:
                details.append(formatter(val))
            else:
                details.append(f"{key.replace('_', ' ')} {val}")
        display_limit = int(cfg.option_generation.display_attribute_limit)
        detail_text = ", ".join(details[:display_limit])
        if sep:
            tail_text = "upside:" + tail
            # Keep only the human-readable qualitative part after attributes.
            return f"{head}: {detail_text}; {tail_text}" if detail_text else f"{head}: {tail_text}"
        return f"{head}: {detail_text}" if detail_text else option

    def _build_opening_history(self) -> list[str]:
        lines = [
            "Moderator: Hey everyone, let's get started.",
            f"Moderator: Today we are deciding: {self.topic}",
            "Moderator: Here are the options on the table:",
        ]
        lines.extend(f"Moderator: {self._display_option(opt)}" for opt in self.options)
        lines.append(f"Moderator: {self.opening_question}")
        return lines

    # ------------------------------------------------------------------
    # Logging / state update
    # ------------------------------------------------------------------

    def _store_line(
        self,
        line: str,
        selected_reason: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        verification_result: Optional[dict] = None,
    ) -> None:
        self.history.append(line)
        print(f"-> {line}")
        self._logger.append_chat_line(line)
        self._logger.buffer(
            line, selected_reason, self.state,
            tokens_in=tokens_in, tokens_out=tokens_out,
            verification_result=verification_result,
        )

        if self._memory is not None:
            self._memory.update(
                line, self.state.phase,
                selected_reason=selected_reason,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        # Keep the lightweight public discourse fields in sync for prompts and
        # speaker selection. This now includes implicit-address detection.
        disc = extract_discourse(self.history, {s.name for s in self.sims})
        self.state.last_addressed = disc.get("last_addressed")
        self.state.pending_question_target = disc.get("pending_question_target")
        self._update_control_from_line(line)

    def _store_moderator(self, text: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self._store_line(f"Moderator: {text}", selected_reason="moderator", tokens_in=tokens_in, tokens_out=tokens_out)

    def _option_refs(self, text: str) -> list[str]:
        """Option references for live control.

        OptionResolver intentionally avoids many bare-letter matches for general
        text safety. Live control needs common chat shorthand too: "I can live
        with B", "still prefer C". We therefore add conservative uppercase
        single-letter references on top of resolver aliases.
        """
        refs = list(self.resolver.options_in(text))
        for letter in self.resolver.letters:
            if letter not in refs and re.search(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", text):
                refs.append(letter)
        return refs

    def _vote_ref(self, text: str) -> Optional[str]:
        vote = self.resolver.vote_in(text)
        if vote:
            return vote
        # Common shorthand: "I'd go with C", "I'm leaning toward D", "my pick is B".
        m = re.search(
            r"\b(?:i\s*(?:'d|would)?\s*(?:go\s+with|pick|choose|prefer|vote\s+for)|"
            r"i\s*(?:'m|am)\s*(?:leaning\s+(?:toward|towards)|going\s+with)|"
            r"my\s+(?:pick|choice|vote)\s+(?:is|would\s+be))\s+(?:Option\s+)?([A-D])\b",
            text,
            re.I,
        )
        return m.group(1).upper() if m else None

    def _mentions_option(self, text: str, option: str) -> bool:
        return option in self._option_refs(text)

    def _context_accepts_candidate(self, text: str, candidate: str) -> bool:
        """Confirmation-specific acceptance.

        A sentence such as "I still prefer A, but Option B works well enough"
        must count as accepting B. Generic vote parsing cannot infer this
        reliably because the sentence mentions two options.
        """
        if _YES_RE.search(text) or _ACCEPT_RE.search(text):
            if not self._option_refs(text) or self._mentions_option(text, candidate):
                return True
        if self._mentions_option(text, candidate) and re.search(
            r"\b(?:works?|acceptable|fine|okay|ok|good)\b.{0,30}\b(?:enough|compromise|for\s+me|for\s+us)?\b",
            text,
            re.I,
        ):
            return True
        return False

    def _update_control_from_line(self, line: str) -> None:
        if ":" not in line:
            return
        speaker, text = line.split(":", 1)
        speaker, text = speaker.strip(), text.strip()
        if speaker in cfg.EXCLUDED_SPEAKERS:
            return

        # Context-bound confirmation handling. A reply like "That's fine" or
        # "I'd still prefer B, but yeah" is meaningless without the moderator's
        # previous candidate question; with that context it is an explicit
        # acceptance of the pending candidate. This must run before generic
        # option-reference parsing.
        pending_target = self.state.pending_confirmation_target
        pending_candidate = self.state.pending_confirmation_candidate
        if (
            self.state.phase == "confirmation"
            and pending_target == speaker
            and pending_candidate
        ):
            if _NO_RE.search(text):
                chars = int(cfg.structured_control.state_reject_excerpt_chars)
                self.state.explicit_rejects.setdefault(speaker, {})[pending_candidate] = text[:chars]
                self.state.confirmation_rejected_options.add(pending_candidate)
                self.state.rejected_options_by_speaker[speaker] = pending_candidate
                self.state.explicit_accepts.setdefault(speaker, set()).discard(pending_candidate)
            elif self._context_accepts_candidate(text, pending_candidate):
                self.state.explicit_accepts.setdefault(speaker, set()).add(pending_candidate)
                self.state.explicit_rejects.setdefault(speaker, {}).pop(pending_candidate, None)
                if self.state.rejected_options_by_speaker.get(speaker) == pending_candidate:
                    self.state.rejected_options_by_speaker.pop(speaker, None)
            # Clear only after the addressed speaker replies.
            self.state.pending_confirmation_target = None
            self.state.pending_confirmation_candidate = None

        vote = self._vote_ref(text)
        if vote and self.state.phase == "narrowing":
            prev = self.state.explicit_votes.get(speaker)
            if prev and prev != vote:
                self.state.vote_changes[speaker] = self.state.vote_changes.get(speaker, 0) + 1
            self.state.explicit_votes[speaker] = vote
            self.state.last_known_vote[speaker] = vote
            self.state.explicit_accepts.setdefault(speaker, set()).add(vote)
            # If the speaker explicitly changed their mind, clear a prior local
            # rejection. Otherwise the verifier should already have repaired
            # this before it reached the state update.
            if _CHANGED_MIND_RE.search(text):
                self.state.explicit_rejects.setdefault(speaker, {}).pop(vote, None)

        rule_out_match = None if "?" in text else _RULE_OUT_RE.search(text)
        explicit_rule_out = rule_out_match.group(1).upper() if rule_out_match else None

        for opt in self._option_refs(text):
            # Outside a targeted confirmation, a leading "No update..." or
            # "No idea..." must not become a rejection of every option named
            # in the sentence. Only explicit rejection phrases count here.
            is_reject = bool(_REJECT_RE.search(text)) or (explicit_rule_out == opt)
            if is_reject:
                chars = int(cfg.structured_control.state_reject_excerpt_chars)
                self.state.explicit_rejects.setdefault(speaker, {})[opt] = text[:chars]
                self.state.rejected_options_by_speaker[speaker] = opt
                self.state.explicit_accepts.setdefault(speaker, set()).discard(opt)
            elif _ACCEPT_RE.search(text) or (_YES_RE.search(text) and self.state.candidate_option == opt):
                self.state.explicit_accepts.setdefault(speaker, set()).add(opt)
                self.state.explicit_rejects.setdefault(speaker, {}).pop(opt, None)
                if self.state.rejected_options_by_speaker.get(speaker) == opt:
                    self.state.rejected_options_by_speaker.pop(speaker, None)

    def _apply_structured_vote(self, speaker: str, option: Optional[str]) -> None:
        if not option:
            return
        prev = self.state.explicit_votes.get(speaker)
        if prev and prev != option:
            self.state.vote_changes[speaker] = self.state.vote_changes.get(speaker, 0) + 1
        self.state.explicit_votes[speaker] = option
        self.state.last_known_vote[speaker] = option
        self.state.explicit_accepts.setdefault(speaker, set()).add(option)
        self.state.explicit_rejects.setdefault(speaker, {}).pop(option, None)

    def _apply_structured_confirmation(
        self,
        speaker: str,
        action: str,
        option: Optional[str],
        message: str,
    ) -> None:
        if not option:
            return
        sc = cfg.structured_control
        if action == sc.accept_action:
            self.state.explicit_accepts.setdefault(speaker, set()).add(option)
            self.state.explicit_rejects.setdefault(speaker, {}).pop(option, None)
            if self.state.rejected_options_by_speaker.get(speaker) == option:
                self.state.rejected_options_by_speaker.pop(speaker, None)
        elif action == sc.reject_action:
            chars = int(cfg.structured_control.state_reject_excerpt_chars)
            self.state.explicit_rejects.setdefault(speaker, {})[option] = message[:chars]
            self.state.confirmation_rejected_options.add(option)
            self.state.rejected_options_by_speaker[speaker] = option
            self.state.confirmation_rejection_count += 1
            self.state.explicit_accepts.setdefault(speaker, set()).discard(option)
        if self.state.pending_confirmation_target == speaker:
            self.state.pending_confirmation_target = None
            self.state.pending_confirmation_candidate = None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _primary_sim(self) -> Optional[Any]:
        return next((s for s in self.sims if s.persona.is_primary), None)

    def _ordered_sims(self) -> list[Any]:
        primary = self._primary_sim()
        return ([primary] if primary else []) + [s for s in self.sims if s is not primary]

    def _structured(self):
        return self._memory

    def _all_names(self) -> list[str]:
        return [s.name for s in self.sims]

    def _participant_for_name(self, name: str) -> Optional[Any]:
        return next((s for s in self.sims if s.name == name), None)

    def _update_repetition_and_candidate(self) -> None:
        self.state.repetition_pressure = compute_repetition_pressure(self.history)
        if self.state.explicit_votes:
            self.state.candidate_option = Counter(self.state.explicit_votes.values()).most_common(1)[0][0]
            self.state.current_leading_option = self.state.candidate_option

    def _private_accepts(self, sim: Any, option: str) -> bool:
        beliefs = getattr(sim.persona, "beliefs", None)
        if not beliefs:
            return False
        return option in (beliefs.acceptable or []) and option not in (beliefs.rejected or [])

    def _explicit_accepts_all(self, option: str) -> bool:
        for sim in self.sims:
            if self.state.explicit_votes.get(sim.name) == option:
                continue
            if option in self.state.explicit_accepts.get(sim.name, set()):
                continue
            return False
        return True

    def _rejected_by_anyone(self, option: str) -> bool:
        # Only targeted confirmation no's hard-exclude a candidate. Earlier
        # discussion-phase objections are context and score penalties, not final
        # blockers.
        return option in self.state.confirmation_rejected_options

    def _candidate_from_votes(self) -> Optional[str]:
        if not self.state.explicit_votes:
            return None
        counts = Counter(self.state.explicit_votes.values())
        return counts.most_common(1)[0][0]

    def _candidate_order(self, exclude_rejected_live: bool = True) -> list[str]:
        """Rank candidates for compromise testing.

        The first attempted candidate is usually the plurality vote. If it is
        explicitly rejected, it must not be selected again during finalization.
        Ranking combines fresh votes, explicit accepts, private acceptability,
        and primary-speaker acceptability. It intentionally does not use the old
        stance table.
        """
        letters = list(self.resolver.letters or ["A", "B", "C", "D"])
        primary = self._primary_sim()
        vote_counts = Counter(self.state.explicit_votes.values())

        def score(opt: str) -> tuple[int, int, int, int, int, int, int]:
            if exclude_rejected_live and self._rejected_by_anyone(opt):
                return (-999, -999, -999, -999, -999, -999, -999)
            explicit_accepts = sum(1 for opts in self.state.explicit_accepts.values() if opt in opts)
            votes = vote_counts.get(opt, 0)
            private_accepts = sum(1 for s in self.sims if self._private_accepts(s, opt))
            primary_ok = int(primary is not None and self._private_accepts(primary, opt))
            discussion_rejects = sum(1 for rejects in self.state.explicit_rejects.values() if opt in rejects)
            private_rejects = sum(
                1 for s in self.sims
                if opt in (getattr(s.persona.beliefs, "rejected", []) if s.persona.beliefs else [])
            )
            # Explicit votes/accepts are public evidence; private acceptability
            # decides which fallback to test next. Discussion objections penalize
            # but do not block a candidate.
            return (votes, explicit_accepts, private_accepts, primary_ok, -discussion_rejects, -private_rejects, -letters.index(opt))

        ranked = sorted(letters, key=score, reverse=True)
        return [o for o in ranked if score(o)[0] > -999]

    def _holdouts_for(self, option: str) -> list[Any]:
        holdouts: list[Any] = []
        for sim in self.sims:
            name = sim.name
            if self.state.explicit_votes.get(name) == option:
                continue
            if option in self.state.explicit_accepts.get(name, set()):
                continue
            holdouts.append(sim)
        return holdouts

    def _compromise_rationale(self, option: str) -> str:
        """Explain the final candidate using only actual support for it.

        Do not mention unrelated rejected options here. Earlier versions said
        things like "Option D did not work" while finalizing Option A, which
        made the moderator sound confused.
        """
        voted_for = [name for name, v in self.state.explicit_votes.items() if v == option]
        accepted = [
            name for name, opts in self.state.explicit_accepts.items()
            if option in opts and self.state.explicit_votes.get(name) != option
        ]
        if voted_for and accepted:
            return f"Option {option} is the shared fallback: {', '.join(voted_for)} picked it, and {', '.join(accepted)} can live with it."
        if accepted:
            return f"Option {option} is the shared fallback: {', '.join(accepted)} can live with it."
        if voted_for:
            return f"Option {option} has the clearest support from {', '.join(voted_for)}."
        return f"Option {option} is the option everyone can live with."

    def _participant_turn_records(self, phase: Optional[str] = None):
        structured = self._structured()
        if not structured:
            return []
        return [
            t for t in structured.turns
            if not t.is_moderator
            and t.speaker not in cfg.EXCLUDED_SPEAKERS
            and (phase is None or t.phase == phase)
        ]

    def _discussion_option_coverage(self) -> set[str]:
        opts: set[str] = set()
        min_words = int(cfg.turns.readiness_min_reason_words)
        for t in self._participant_turn_records("negotiation"):
            text = t.text
            if len(text.split()) < min_words:
                continue
            if not t.mentioned_options:
                continue
            # Count an option as substantively discussed only if the turn
            # contains a reason/trade-off style marker, not just "rule out X".
            if re.search(
                r"\b(because|since|due to|so |offers?|helps?|fits?|matters?|"
                r"worth|useful|better|worse|risk|tradeoff|drawback|concern|"
                r"price|cost|wait|time|travel|noise|menu|variety|safety|"
                r"allergen|local|quiet|loud|comfort|amenit|flexib|reliable)\b",
                text,
                re.I,
            ):
                opts.update(t.mentioned_options)
        return opts

    def _recent_thread_active(self) -> bool:
        """Whether the current local thread should get one more turn.

        This is deliberately not a coverage checklist. If the last turn is a
        directed question or the last few turns are still developing the same
        concrete issue, let it breathe instead of immediately moving to a vote.
        """
        window = int(cfg.turns.readiness_thread_window)
        recs = self._participant_turn_records("negotiation")[-window:]
        if not recs:
            return False
        last = recs[-1]
        if last.is_question and (last.addressees or self.state.pending_question_target):
            return True
        question_window = int(cfg.turns.readiness_question_window)
        if any(t.is_question and t.addressees for t in recs[-question_window:]):
            return True
        opt_counts = Counter(o for t in recs for o in t.mentioned_options)
        min_mentions = int(cfg.turns.readiness_same_option_thread_mentions)
        if opt_counts and opt_counts.most_common(1)[0][1] >= min_mentions:
            # A short same-option thread is active if it has not yet produced a
            # compromise/decision-move. Do not continue forever.
            recent_text = " ".join(t.text.lower() for t in recs)
            if not re.search(r"\b(can live with|works for me|between|landing|go with|pick|vote)\b", recent_text):
                return True
        return False

    def _discussion_ready_for_narrowing(self) -> bool:
        """Minimum quality gate before asking for votes.

        This is not a fixed script. It asks whether enough of the group has made
        option-linked reasons, enough alternatives have been discussed, no local
        question is hanging, and novelty is dropping.
        """
        n = max(1, len(self.sims))
        recs = self._participant_turn_records("negotiation")
        if not recs:
            return False
        reason_ratio = float(cfg.turns.readiness_reason_ratio)
        needed_reasons = max(1, math.ceil(n * reason_ratio))
        reasons = sum(1 for s in self.sims if self._has_substantive_reason(s.name))
        if reasons < needed_reasons:
            return False
        min_options = int(cfg.turns.readiness_min_options_discussed)
        if len(self._discussion_option_coverage()) < min_options:
            return False
        if self._recent_thread_active():
            return False
        return self.state.repetition_pressure >= float(cfg.turns.readiness_repetition_threshold)

    def _has_substantive_reason(self, name: str) -> bool:
        """Whether a participant has made at least one option-linked reason.

        This is intentionally simple. It prevents voting after only shallow
        fragments like "Can we rule out C?" or "Still think A is worth it".
        """
        reason_markers = re.compile(
            r"\b(because|since|due to|as |so |offers?|helps?|fits?|matters?|"
            r"worth|useful|better|worse|risk|tradeoff|drawback|concern|"
            r"comfortable|convenient|authentic|quality|price|cost|location|"
            r"atmosphere|service|quiet|loud|flexib|reliable|wait|time|travel|menu|variety|safety|allergen|local|amenit)\b",
            re.I,
        )
        for line in self.history:
            if not line.startswith(f"{name}:"):
                continue
            text = line.split(":", 1)[1].strip()
            if len(text.split()) < int(cfg.turns.readiness_min_reason_words):
                continue
            if self._option_refs(text) and reason_markers.search(text):
                return True
        return False

    # ------------------------------------------------------------------
    # Participant turn helpers
    # ------------------------------------------------------------------

    def _speaker_turn(self, sim: Any, reason: str) -> None:
        self.state.turn_index += 1
        text, tok_in, tok_out = sim.generate_turn(
            self.history,
            self.state,
            all_names=self._all_names(),
            structured=self._structured(),
        )
        if text and "[SILENCE]" not in text.upper():
            v_result = getattr(sim, "_last_verification", None)
            self._store_line(
                f"{sim.name}: {text}", selected_reason=reason,
                tokens_in=tok_in, tokens_out=tok_out,
                verification_result=v_result,
            )

    def _control_turn(self, sim: Any, reason: str, kind: str) -> None:
        self.state.turn_index += 1
        text, action, option, tok_in, tok_out, v_result = sim.generate_control_turn(
            self.history,
            self.state,
            structured=self._structured(),
            kind=kind,
        )
        if text and "[SILENCE]" not in text.upper():
            self._store_line(
                f"{sim.name}: {text}",
                selected_reason=reason,
                tokens_in=tok_in,
                tokens_out=tok_out,
                verification_result=v_result,
            )
            if kind == "vote":
                self._apply_structured_vote(sim.name, option)
            elif kind == "confirmation":
                self._apply_structured_confirmation(sim.name, action, option, text)

    def _run_opening(self) -> None:
        self.state.phase = "opening"
        for sim in self._ordered_sims():
            self._speaker_turn(sim, "opening")

    def _run_discussion(self) -> None:
        self.state.phase = "negotiation"
        n = max(1, len(self.sims))
        min_turns = max(
            int(cfg.turns.discussion_min_total_turns),
            n * int(cfg.turns.discussion_min_turns_per_participant),
        )
        soft_max = max(min_turns, n * int(cfg.turns.discussion_max_turns_per_participant))
        max_turns = max(min_turns, min(int(cfg.turns.discussion_max_total_turns), soft_max))

        for _ in range(max_turns):
            self._update_repetition_and_candidate()
            selected = select_next_speakers(self.sims, self.history, self.state)
            if not selected:
                break
            self._speaker_turn(selected[0], "discussion")

            discussion_turns = len(self._participant_turn_records("negotiation"))
            if discussion_turns < min_turns:
                continue

            self._update_repetition_and_candidate()
            # If the local thread is still active, allow it to finish unless the
            # hard maximum has been reached. This avoids the "one point, move on"
            # failure while still keeping generation bounded.
            if self._recent_thread_active():
                continue

            # Stop early only when the discussion is substantively ready.
            if self._discussion_ready_for_narrowing():
                break

    def _run_vote_round(self) -> None:
        self.state.phase = "narrowing"
        self.state.has_asked_narrowing = True
        # Fresh vote round: discussion-phase support is useful context, but it
        # must not replace a current explicit vote after the moderator asks.
        self.state.explicit_votes.clear()
        self.state.vote_changes.clear()
        self.state.last_known_vote.clear()
        self.state.pending_confirmation_target = None
        self.state.pending_confirmation_candidate = None
        self._store_moderator("Okay, we have enough on the table now. Where is everyone landing -- which option is your current pick?")

        for sim in self._ordered_sims():
            self._control_turn(sim, "vote", "vote")
        self._update_repetition_and_candidate()

    def _candidate_test_rationale(self, candidate: str) -> str:
        """One short moderator rationale before testing a compromise candidate."""
        voters = [name for name, opt in self.state.explicit_votes.items() if opt == candidate]
        accepts = [
            name for name, opts in self.state.explicit_accepts.items()
            if candidate in opts and self.state.explicit_votes.get(name) != candidate
        ]
        if voters and accepts:
            return f"Option {candidate} is worth checking: {', '.join(voters)} picked it and {', '.join(accepts)} can live with it."
        if voters:
            return f"Option {candidate} is worth checking because it has current vote support."
        return f"Option {candidate} is worth checking as a possible shared fallback."

    def _ask_holdout(self, candidate: str, holdout: Any) -> None:
        current = self.state.explicit_votes.get(holdout.name, "")
        self.state.phase = "confirmation"
        self.state.candidate_option = candidate
        self.state.current_leading_option = candidate
        self.state.pending_confirmation_target = holdout.name
        self.state.pending_confirmation_candidate = candidate

        count = self.state.candidate_prompt_counts.get(candidate, 0)
        self.state.candidate_prompt_counts[candidate] = count + 1
        if count == 0:
            rationale = self._candidate_test_rationale(candidate)
            if current and current != candidate:
                text = (f"{rationale} {holdout.name}, you picked Option {current}. "
                        f"Could Option {candidate} work for you, or is that still a no?")
            else:
                text = f"{rationale} {holdout.name}, could Option {candidate} work for you?"
        else:
            # Same candidate, next holdout: avoid repeating the full rationale.
            if current and current != candidate:
                text = (f"{holdout.name}, same candidate: you picked Option {current}. "
                        f"Could Option {candidate} work too, or no?")
            else:
                text = f"{holdout.name}, same candidate: could Option {candidate} work too?"
        self._store_moderator(text)
        self._control_turn(holdout, "targeted_holdout", "confirmation")

    def _run_compromise(self) -> Optional[str]:
        """Test compromise candidates until one is explicitly accepted by all.

        Earlier versions tested a vote winner, then a single fallback, and then
        finalization recomputed the original winner after rejection. That caused
        logs where B was accepted by all holdouts but the moderator still failed
        on A. This loop keeps a tested/excluded set and never returns to an
        explicitly rejected candidate.
        """
        tested: set[str] = set()
        max_candidates = int(cfg.structured_control.max_candidates_to_test)
        while True:
            order = [c for c in self._candidate_order(exclude_rejected_live=True) if c not in tested]
            if not order or len(tested) >= max_candidates:
                return None
            candidate = order[0]
            tested.add(candidate)
            self.state.candidate_option = candidate
            self.state.current_leading_option = candidate

            # If votes/accepts already imply agreement, finish immediately.
            if self._explicit_accepts_all(candidate):
                return candidate

            for holdout in list(self._holdouts_for(candidate)):
                self._ask_holdout(candidate, holdout)
                if self._rejected_by_anyone(candidate):
                    break
                if self._explicit_accepts_all(candidate):
                    return candidate

            if self._explicit_accepts_all(candidate):
                return candidate
            # Candidate was rejected or remained unaccepted; try the next best
            # public/private fallback. Do not force or loop endlessly here.
            continue

    # ------------------------------------------------------------------
    # Ending
    # ------------------------------------------------------------------

    def _finalize_success(self, option: str, outcome: str) -> None:
        self.outcome = outcome
        self.state.outcome_reason = "all participants explicitly voted for or accepted the final option"
        self.state.preferred_option = option
        self.state.candidate_option = option
        self.state.current_leading_option = option
        self.state.agreement_reached = True
        if outcome == "success":
            self._store_moderator(f"Agreed -- Option {option}. Done.")
        else:
            rationale = self._compromise_rationale(option)
            self._store_moderator(f"{rationale} Compromise works -- Option {option}.")
        self._run_closure()

    def _run_closure(self) -> None:
        self.state.phase = "closure"
        cap = int(getattr(getattr(cfg, "closure", object()), "participants_to_close", 1))
        if cap <= 0:
            return
        ordered = self._ordered_sims()
        for sim in ordered[:min(cap, len(ordered))]:
            self._speaker_turn(sim, "closure")

    def _finalize_force_or_failure(self) -> None:
        order = self._candidate_order(exclude_rejected_live=True)
        candidate = (order[0] if order else (self.state.candidate_option or self._candidate_from_votes() or "A"))
        self.state.candidate_option = candidate
        self.state.preferred_option = candidate
        if self._rejected_by_anyone(candidate):
            self.outcome = "failed_no_viable_compromise"
            self.state.outcome_reason = "a tested candidate was explicitly rejected and no shared fallback reached all participants"
            self._store_moderator(f"No full agreement. Option {candidate} was the closest tested fallback, but this is not real consensus.")
        else:
            self.outcome = "force_close"
            self.state.outcome_reason = "maximum turn budget reached without explicit acceptance from every participant"
            self._store_moderator(f"No full agreement. The closest workable option is Option {candidate}, so I'm stopping there.")
        # Do not ask participants to close after force-close/failure. That
        # produced contradictory lines like "sounds good" after a failed
        # compromise. The moderator's terminal line is the honest ending.

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(self, setup_tokens_in: int = 0, setup_tokens_out: int = 0) -> None:
        self._memory = DialogueMemory(
            participant_names=[s.name for s in self.sims],
            options=self.options,
            resolver=self.resolver,
        )
        self._memory.attach_personas(self.sims)
        for line in self.history:
            self._memory.update(line, "opening", selected_reason="moderator")

        sample_hard_blocker(list(self._memory.participants.values()))

        self._logger.write_header(
            participant_names=[s.name for s in self.sims],
            personas=[s.persona for s in self.sims],
            opening_lines=self.history,
        )
        for line in self.history:
            self._logger.buffer(line, "moderator", self.state)

        print("\n--- Dialogue started ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        try:
            self._run_opening()
            self._run_discussion()
            self._run_vote_round()

            votes = self.state.explicit_votes
            if len(votes) == len(self.sims) and len(set(votes.values())) == 1:
                self._finalize_success(next(iter(votes.values())), "success")
            else:
                candidate = self._run_compromise()
                if candidate:
                    self._finalize_success(candidate, "compromise_success")
                else:
                    self._finalize_force_or_failure()

        finally:
            dialogue_in = self._llm.session_tokens_in
            dialogue_out = self._llm.session_tokens_out
            self._logger.flush(
                outcome=self.outcome,
                sims=self.sims,
                state=self.state,
                memory=self._memory,
                setup_tokens_in=setup_tokens_in,
                setup_tokens_out=setup_tokens_out,
                dialogue_tokens_in=dialogue_in,
                dialogue_tokens_out=dialogue_out,
            )
            chat_path, eval_path = self._logger.paths
            total_in = setup_tokens_in + dialogue_in
            total_out = setup_tokens_out + dialogue_out
            print(f"\n[Tokens] setup={setup_tokens_in}/{setup_tokens_out} "
                  f"dialogue={dialogue_in}/{dialogue_out} total={total_in}/{total_out}")
            print(f"[Outcome] {self.outcome}")
            print(f"[Saved]   {chat_path} | {eval_path}")
