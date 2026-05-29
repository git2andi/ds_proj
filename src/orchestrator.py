"""
orchestrator.py
---------------
Coordinates one dialogue run.

Refactored runtime model:
  - The LLM only writes the next utterance.
  - `policy.select_next_speakers()` only routes speakers.
  - Live decisions use explicit votes + explicit accept/reject statements.
  - The older rich `StateTracker` still exists for logging/evaluation, but it no
    longer decides when the dialogue is done.

This removes the previous failure mode where stance-table inference, open
question invitations, challenge gates, and force-close logic kept a dialogue
running after a workable compromise already existed.
"""

from __future__ import annotations

import datetime
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from logger import DialogueLogger
from moderation import ModerationEngine
from policy import extract_discourse, repetition_pressure as compute_repetition_pressure, sample_hard_blocker, select_next_speakers
from state import StateTracker
from utils import OptionResolver, participant_turn_count


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
    priority_next_speaker: Optional[str] = None

    repetition_pressure: float = 0.0
    stall_rounds: int = 0
    post_narrowing_rounds: int = 0
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

    # One-shot prompt nudge used before voting if a participant has not yet
    # contributed a concrete option-specific reason.
    required_reason_target: Optional[str] = None

    # Kept for logger compatibility.
    vote_changes: dict = field(default_factory=dict)
    last_known_vote: dict = field(default_factory=dict)
    rejected_options_by_speaker: dict = field(default_factory=dict)
    last_rejected_option: Optional[str] = None
    last_rejecting_speaker: Optional[str] = None
    consensus_cooldown: int = 0
    nudged_participants: set[str] = field(default_factory=set)
    has_entered_emergence: bool = False
    info_gap_cooldown: int = 0
    facilitate_cooldown: int = 0
    outcome_reason: str = ""

    # Moderator style/control: avoid repeating the same full holdout prompt.
    candidate_prompt_counts: dict[str, int] = field(default_factory=dict)

    # Conditional compromise support, e.g. Option C + "split it over two nights".
    # These are execution terms attached to an existing option, not new options.
    compromise_terms: dict[str, list[str]] = field(default_factory=dict)

    # Only a targeted confirmation "no" hard-excludes a candidate. Discussion-
    # phase "not sold" lines remain useful context but should not prevent the
    # moderator from testing a condition-backed compromise.
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
        self._mod: Optional[ModerationEngine] = None
        self._tracker: Optional[StateTracker] = None

    def add_sim(self, sim: Any) -> None:
        self.sims.append(sim)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _generate_options(self) -> tuple[list[str], str]:
        data = self._llm.generate_json(prompts.option_generation(self.topic))
        options_raw = data.get("options", [])
        question = str(data.get("opening_question", "")).strip()
        if not question:
            raise ValueError("Option generation returned no opening_question.")
        if not isinstance(options_raw, list) or len(options_raw) != 4:
            raise ValueError(f"Option generation expected 4 options, got: {options_raw!r}")

        cleaned: list[str] = []
        for i, raw in enumerate(options_raw):
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Option generation returned an empty option at index {i}.")
            label = chr(ord("A") + i)
            text = raw.strip()
            if not text.lower().startswith(f"option {label.lower()}"):
                text = f"Option {label} - {text}"
            cleaned.append(text)

        decision_kind = str(data.get("decision_kind", "")).strip().lower()
        if decision_kind:
            print(f"  [options] decision_kind={decision_kind}")
        return cleaned, question

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
        detail_text = ", ".join(details[:8])
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

        if self._tracker is not None:
            self._tracker.update(
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

    def _is_hard_rejected(self, option: str) -> bool:
        return option in self.state.confirmation_rejected_options

    def _terms_for(self, option: str) -> list[str]:
        return list(dict.fromkeys(self.state.compromise_terms.get(option, [])))

    def _terms_text(self, option: str) -> str:
        terms = self._terms_for(option)
        if not terms:
            return ""
        if len(terms) == 1:
            return f" if we {terms[0]}"
        return " if we " + " and ".join(terms[:2])

    def _add_compromise_term(self, option: str, term: str) -> None:
        term = re.sub(r"\s+", " ", term.strip(" .,!?:;\"'")).lower()
        if not term:
            return
        # Keep terms short and implementation-like. They must not create a new option.
        term = re.sub(r"^(we|you|they|someone)\s+", "", term)
        if len(term.split()) > 9:
            term = " ".join(term.split()[:9])
        terms = self.state.compromise_terms.setdefault(option, [])
        if term not in terms:
            terms.append(term)

    def _extract_compromise_terms(self, text: str) -> list[str]:
        """Extract feasible execution conditions from natural compromise talk.

        These terms keep the final decision as Option A-D while allowing human
        compromises such as \"Option C, but split it over two nights\".
        """
        lower = text.lower()
        terms: list[str] = []
        if re.search(r"\b(split|break\s+up|two\s+nights?|two-part|two\s+sessions?)\b", lower):
            terms.append("split it over two nights")
        if re.search(r"\b(brief|short)\s+intro\b|\bset\s+context\b|\bintro\s+first\b", lower):
            terms.append("add a brief intro first")
        if re.search(r"\b(go|arrive|leave|get\s+there)\s+early\b|\bearlier\b", lower):
            terms.append("go early")
        if re.search(r"\bdiscussion\s+guide\b", lower):
            terms.append("use a discussion guide")
        if re.search(r"\bkeep\s+(?:the\s+)?(?:next\s+)?morning\s+light\b", lower):
            terms.append("keep the next morning light")
        if re.search(r"\bavoid\s+(?:the\s+)?(?:peak|crowd|crowds|busy\s+time)\b", lower):
            terms.append("avoid peak crowds")

        for pat in (
            r"\bas long as\s+([^.;!?]+)",
            r"\bonly if\s+([^.;!?]+)",
            r"\bif we\s+([^.;!?]+)",
            r"\bif it\s+([^.;!?]+)",
            r"\bprovided\s+that\s+([^.;!?]+)",
        ):
            for m in re.finditer(pat, lower):
                phrase = m.group(1).strip()
                # Avoid treating clear rejection conditions as accepted terms.
                if re.match(r"^(had|has|gets|becomes|is)\s+better\b", phrase):
                    continue
                phrase = re.sub(r"^(we\s+)?", "", phrase)
                if phrase and len(phrase.split()) <= 9:
                    terms.append(phrase)
        return list(dict.fromkeys(terms))

    def _maybe_store_compromise_terms(self, text: str, candidate_hint: Optional[str] = None) -> None:
        terms = self._extract_compromise_terms(text)
        if not terms:
            return
        refs = self._option_refs(text)
        if candidate_hint and candidate_hint not in refs:
            refs.append(candidate_hint)
        # If the speaker talks about a named option title such as \"the sci-fi epic\",
        # OptionResolver should already resolve it. Otherwise keep terms only when
        # there is a live candidate, to avoid attaching vague conditions randomly.
        for opt in refs:
            for term in terms:
                self._add_compromise_term(opt, term)

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
                self.state.explicit_rejects.setdefault(speaker, {})[pending_candidate] = text[:120]
                self.state.confirmation_rejected_options.add(pending_candidate)
                self.state.rejected_options_by_speaker[speaker] = pending_candidate
                self.state.last_rejected_option = pending_candidate
                self.state.last_rejecting_speaker = speaker
                self.state.explicit_accepts.setdefault(speaker, set()).discard(pending_candidate)
                self._maybe_store_compromise_terms(text, pending_candidate)
            elif self._context_accepts_candidate(text, pending_candidate):
                self.state.explicit_accepts.setdefault(speaker, set()).add(pending_candidate)
                self.state.explicit_rejects.setdefault(speaker, {}).pop(pending_candidate, None)
                if self.state.rejected_options_by_speaker.get(speaker) == pending_candidate:
                    self.state.rejected_options_by_speaker.pop(speaker, None)
                self._maybe_store_compromise_terms(text, pending_candidate)
            # Clear only after the addressed speaker replies.
            self.state.pending_confirmation_target = None
            self.state.pending_confirmation_candidate = None

        self._maybe_store_compromise_terms(text, self.state.candidate_option)

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
                self.state.explicit_rejects.setdefault(speaker, {})[opt] = text[:120]
                self.state.rejected_options_by_speaker[speaker] = opt
                self.state.last_rejected_option = opt
                self.state.last_rejecting_speaker = speaker
                self.state.explicit_accepts.setdefault(speaker, set()).discard(opt)
            elif _ACCEPT_RE.search(text) or (_YES_RE.search(text) and self.state.candidate_option == opt):
                self.state.explicit_accepts.setdefault(speaker, set()).add(opt)
                self.state.explicit_rejects.setdefault(speaker, {}).pop(opt, None)
                if self.state.rejected_options_by_speaker.get(speaker) == opt:
                    self.state.rejected_options_by_speaker.pop(speaker, None)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _primary_sim(self) -> Optional[Any]:
        return next((s for s in self.sims if s.persona.is_primary), None)

    def _ordered_sims(self) -> list[Any]:
        primary = self._primary_sim()
        return ([primary] if primary else []) + [s for s in self.sims if s is not primary]

    def _structured(self):
        return self._tracker.state if self._tracker else None

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

    def _has_accepted(self, name: str, option: str) -> bool:
        if self.state.explicit_votes.get(name) == option:
            return True
        if option in self.state.explicit_accepts.get(name, set()):
            return True
        sim = self._participant_for_name(name)
        return bool(sim and self._private_accepts(sim, option) and self.state.phase in {"confirmation", "closure"})

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
        # blockers, because they may be resolved by a condition.
        return option in self.state.confirmation_rejected_options

    def _candidate_from_votes(self) -> Optional[str]:
        if not self.state.explicit_votes:
            return None
        counts = Counter(self.state.explicit_votes.values())
        return counts.most_common(1)[0][0]

    def _candidate_from_private_acceptability(self) -> Optional[str]:
        order = self._candidate_order()
        return order[0] if order else None

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

        def score(opt: str) -> tuple[int, int, int, int, int, int, int, int]:
            if exclude_rejected_live and self._rejected_by_anyone(opt):
                return (-999, -999, -999, -999, -999, -999, -999, -999)
            explicit_accepts = sum(1 for opts in self.state.explicit_accepts.values() if opt in opts)
            votes = vote_counts.get(opt, 0)
            private_accepts = sum(1 for s in self.sims if self._private_accepts(s, opt))
            primary_ok = int(primary is not None and self._private_accepts(primary, opt))
            compromise_terms = len(self._terms_for(opt))
            discussion_rejects = sum(1 for rejects in self.state.explicit_rejects.values() if opt in rejects)
            private_rejects = sum(
                1 for s in self.sims
                if opt in (getattr(s.persona.beliefs, "rejected", []) if s.persona.beliefs else [])
            )
            # Explicit votes/accepts are public evidence; private acceptability
            # and compromise terms decide which fallback to test next. Discussion
            # objections penalize but do not block a candidate.
            return (votes, explicit_accepts, private_accepts, compromise_terms, primary_ok, -discussion_rejects, -private_rejects, -letters.index(opt))

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
        terms = self._terms_for(option)
        term_text = ""
        if terms:
            term_text = " with the condition that we " + " and ".join(terms[:2])
        if voted_for and accepted:
            return f"Option {option} is the shared fallback{term_text}: {', '.join(voted_for)} picked it, and {', '.join(accepted)} can live with it."
        if accepted:
            return f"Option {option} is the shared fallback{term_text}: {', '.join(accepted)} can live with it."
        if voted_for:
            return f"Option {option} has the clearest support from {', '.join(voted_for)}{term_text}."
        return f"Option {option} is the option everyone can live with{term_text}."

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
        for t in self._participant_turn_records("negotiation"):
            text = t.text
            if len(text.split()) < 7:
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
        recs = self._participant_turn_records("negotiation")[-3:]
        if not recs:
            return False
        last = recs[-1]
        if last.is_question and (last.addressees or self.state.pending_question_target):
            return True
        opt_counts = Counter(o for t in recs for o in t.mentioned_options)
        if opt_counts and opt_counts.most_common(1)[0][1] >= 2:
            # A short same-option thread is active if it has not yet produced a
            # compromise/decision-move. Do not continue forever.
            recent_text = " ".join(t.text.lower() for t in recs)
            if not re.search(r"\b(can live with|works for me|between|landing|go with|pick|vote)\b", recent_text):
                return True
        return False

    def _discussion_ready_for_narrowing(self) -> bool:
        """Minimum quality gate before asking for votes.

        This is not "touch every option once". One issue may deserve several
        turns. The gate only requires enough substance: each participant has had
        room to speak, each has contributed at least one concrete option-linked
        reason, and at least two options have been substantively compared.
        """
        n = max(1, len(self.sims))
        recs = self._participant_turn_records("negotiation")
        turns_by_speaker = Counter(t.speaker for t in recs)
        min_per = 3
        if any(turns_by_speaker.get(s.name, 0) < min_per for s in self.sims):
            return False
        if any(not self._has_substantive_reason(s.name) for s in self.sims):
            return False
        if len(self._discussion_option_coverage()) < 2:
            return False
        if self._recent_thread_active():
            return False
        return len(recs) >= n * int(getattr(cfg.turns, "min_before_narrowing_per_participant", 4))

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
            if len(text.split()) < 8:
                continue
            if self._option_refs(text) and reason_markers.search(text):
                return True
        return False

    def _run_reason_floor(self) -> None:
        """Before voting, give missing participants one chance to add substance.

        This is a safety net, not a checklist-driven mini-round. Normal
        discussion should do most of the work; the floor only prevents a vote
        after someone contributed only rule-outs or vague preference fragments.
        """
        self.state.phase = "negotiation"
        for sim in self._ordered_sims():
            if self._has_substantive_reason(sim.name):
                continue
            self.state.required_reason_target = sim.name
            before = self._has_substantive_reason(sim.name)
            self._speaker_turn(sim, "reason_floor")
            # If still not substantive, do not loop. The prompt/verifier should
            # make this rare, and endless repair-like turns hurt naturalness.
            after = self._has_substantive_reason(sim.name)
            self.state.required_reason_target = None

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

    def _run_opening(self) -> None:
        self.state.phase = "opening"
        for sim in self._ordered_sims():
            self._speaker_turn(sim, "opening")

    def _run_discussion(self) -> None:
        self.state.phase = "negotiation"
        n = max(1, len(self.sims))
        min_turns = max(n * int(getattr(cfg.turns, "min_before_narrowing_per_participant", 4)), n * 3)
        max_turns = max(n * int(getattr(cfg.turns, "narrow_after_per_participant", 5)), min_turns)

        for _ in range(max_turns):
            self._update_repetition_and_candidate()
            discourse = self._structured().discourse if self._structured() else None
            selected = select_next_speakers(self.sims, self.history, self.state, discourse)
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

            # Stop early only when the discussion is substantively ready and is
            # beginning to repeat. Otherwise use the configured max for a little
            # more natural elaboration.
            if self._discussion_ready_for_narrowing() and self.state.repetition_pressure >= 0.50:
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
            self._speaker_turn(sim, "vote")
        self._update_repetition_and_candidate()

    def _candidate_test_rationale(self, candidate: str) -> str:
        """One short moderator rationale before testing a compromise candidate."""
        voters = [name for name, opt in self.state.explicit_votes.items() if opt == candidate]
        accepts = [
            name for name, opts in self.state.explicit_accepts.items()
            if candidate in opts and self.state.explicit_votes.get(name) != candidate
        ]
        terms = self._terms_text(candidate)
        if voters and accepts:
            return f"Option {candidate} is worth checking{terms}: {', '.join(voters)} picked it and {', '.join(accepts)} can live with it."
        if voters:
            return f"Option {candidate} is worth checking{terms} because it has current vote support."
        return f"Option {candidate} is worth checking{terms} as a possible shared fallback."

    def _ask_holdout(self, candidate: str, holdout: Any) -> None:
        current = self.state.explicit_votes.get(holdout.name, "")
        self.state.phase = "confirmation"
        self.state.candidate_option = candidate
        self.state.current_leading_option = candidate
        self.state.pending_confirmation_target = holdout.name
        self.state.pending_confirmation_candidate = candidate

        count = self.state.candidate_prompt_counts.get(candidate, 0)
        self.state.candidate_prompt_counts[candidate] = count + 1
        term_text = self._terms_text(candidate)
        if count == 0:
            rationale = self._candidate_test_rationale(candidate)
            if current and current != candidate:
                text = (f"{rationale} {holdout.name}, you picked Option {current}. "
                        f"Could Option {candidate}{term_text} work for you, or is that still a no?")
            else:
                text = f"{rationale} {holdout.name}, could Option {candidate}{term_text} work for you?"
        else:
            # Same candidate, next holdout: avoid repeating the full rationale.
            if current and current != candidate:
                text = (f"{holdout.name}, same candidate: you picked Option {current}. "
                        f"Could Option {candidate}{term_text} work too, or no?")
            else:
                text = f"{holdout.name}, same candidate: could Option {candidate}{term_text} work too?"
        self._store_moderator(text)
        self._speaker_turn(holdout, "targeted_holdout")

    def _run_peer_compromise_probe(self) -> None:
        """Let participants surface a compromise before moderator holdout checks.

        This keeps compromise from feeling entirely moderator-imposed. It is
        intentionally short: one or two Sims may propose/accept a fallback, then
        explicit confirmation still verifies it.
        """
        if not bool(getattr(getattr(cfg, "compromise", object()), "peer_probe_enabled", True)):
            return
        candidate = self._candidate_order(exclude_rejected_live=True)[0] if self._candidate_order(exclude_rejected_live=True) else None
        if not candidate:
            return
        self.state.phase = "compromise"
        self.state.candidate_option = candidate
        self.state.current_leading_option = candidate
        self._store_moderator(f"Votes are split. Before I check one by one, can we make Option {candidate} work with a simple condition, or is there a better fallback?")
        # Ask holdouts first so compromise is not only moderator-imposed. A
        # supporter may then suggest a condition that addresses the objection.
        speakers: list[Any] = []
        for h in self._holdouts_for(candidate):
            speakers.append(h)
            if len(speakers) >= 2:
                break
        if len(speakers) < 2:
            for s in self._ordered_sims():
                if s not in speakers and self.state.explicit_votes.get(s.name) == candidate:
                    speakers.append(s)
                    break
        for sim in speakers[:2]:
            self._speaker_turn(sim, "peer_compromise")
            if self._explicit_accepts_all(candidate):
                break
        self.state.phase = "confirmation"

    def _run_compromise(self) -> Optional[str]:
        """Test compromise candidates until one is explicitly accepted by all.

        Earlier versions tested a vote winner, then a single fallback, and then
        finalization recomputed the original winner after rejection. That caused
        logs where B was accepted by all holdouts but the moderator still failed
        on A. This loop keeps a tested/excluded set and never returns to an
        explicitly rejected candidate.
        """
        tested: set[str] = set()
        while True:
            order = [c for c in self._candidate_order(exclude_rejected_live=True) if c not in tested]
            if not order:
                return None
            candidate = order[0]
            tested.add(candidate)
            self.state.candidate_option = candidate
            self.state.current_leading_option = candidate

            # If votes/accepts already imply agreement, finish immediately.
            if self._explicit_accepts_all(candidate):
                return candidate

            rejected_before = set(self.state.explicit_rejects.get(n, {}).get(candidate) for n in self._all_names())
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
            terms = self._terms_text(candidate)
            self._store_moderator(f"No full agreement. Option {candidate}{terms} was the closest tested fallback, but this is not real consensus.")
        else:
            self.outcome = "force_close"
            self.state.outcome_reason = "maximum turn budget reached without explicit acceptance from every participant"
            terms = self._terms_text(candidate)
            self._store_moderator(f"No full agreement. The closest workable option is Option {candidate}{terms}, so I'm stopping there.")
        # Do not ask participants to close after force-close/failure. That
        # produced contradictory lines like "sounds good" after a failed
        # compromise. The moderator's terminal line is the honest ending.

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(self, setup_tokens_in: int = 0, setup_tokens_out: int = 0) -> None:
        self._mod = ModerationEngine(self.topic, self.options, self.sims, self.resolver)
        self._tracker = StateTracker(
            participant_names=[s.name for s in self.sims],
            options=self.options,
            resolver=self.resolver,
        )
        self._tracker.attach_personas(self.sims)
        for line in self.history:
            self._tracker.update(line, "opening", selected_reason="moderator")

        sample_hard_blocker(list(self._tracker.state.participants.values()))

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
            self._run_reason_floor()
            self._run_vote_round()

            votes = self.state.explicit_votes
            if len(votes) == len(self.sims) and len(set(votes.values())) == 1:
                self._finalize_success(next(iter(votes.values())), "success")
            else:
                self._run_peer_compromise_probe()
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
                structured=(self._tracker.state if self._tracker else None),
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
