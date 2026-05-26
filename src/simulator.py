"""
simulator.py
------------
Simulator - wraps a Persona and generates one dialogue turn via LLM.

Responsibilities:
- Format the turn prompt (delegates all text to prompts.py)
- Strip common model artefacts (name prefix, silence)
- Build the beliefs block injected into every turn
- Provide position discipline grounded in the stable belief state
- Track forbidden openers from recent history
- Detect repeated phrases and add them to the forbidden list
"""

from __future__ import annotations

import random
import re
from typing import Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from persona import Persona
from realize.grounding import fact_check as _fact_check, repair_directive as _repair_directive

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from policy.act_planner import TurnPlan


class Simulator:

    _GOODBYES = ["Later!", "See ya!", "Bye!", "Bye everyone!", "See you!", "Cheers!", "Take care!"]

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options
        self._llm = get_llm_client()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_turn(
        self,
        history: list[str],
        state: "DialogueState",
        all_names: list[str] | None = None,
        forced_adaptation: bool = False,
        turn_plan: Optional["TurnPlan"] = None,
    ) -> tuple[str, int, int]:
        """Returns (text, tokens_in, tokens_out). Tokens are 0 when skipped."""
        all_names = all_names or []

        if state.phase == "closure":
            return self._closure_line(state), 0, 0

        raw = self._generate_decision(history, state, forced_adaptation, turn_plan=turn_plan)
        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out

        if not raw:
            return "[SILENCE]", tok_in, tok_out

        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        raw = self._enforce_word_budget(raw, state.phase)

        return raw or "[SILENCE]", tok_in, tok_out

    # ------------------------------------------------------------------
    # Turn generation
    # ------------------------------------------------------------------

    def _has_voted(self, history: list[str]) -> bool:
        """True if this sim has explicitly stated an option letter in any of their turns."""
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() != self.name:
                continue
            if re.search(r"\boption\s+([a-d])(?![\w'\-])", msg.lower()):
                return True
        return False

    def _generate_decision(
        self, history: list[str], state: "DialogueState", forced_adaptation: bool,
        turn_plan: Optional["TurnPlan"] = None,
    ) -> str:
        add_brevity = (
            (state.repetition_pressure >= cfg.repetition.add_brevity_threshold
             or state.post_narrowing_rounds >= 2)
            and state.phase not in {"greeting", "opening", "closure", "confirmation"}
        )
        phase_instr = prompts.phase_instruction_text(
            phase=state.phase,
            add_brevity=add_brevity,
            has_voted=self._has_voted(history),
        )

        use_compact = cfg.prompt_budget.use_compact_prompt
        if use_compact:
            return self._generate_compact(history, state, forced_adaptation, phase_instr,
                                           turn_plan=turn_plan)
        return self._generate_legacy(history, state, forced_adaptation, phase_instr)

    def _generate_legacy(
        self, history: list[str], state: "DialogueState",
        forced_adaptation: bool, phase_instr: str,
    ) -> str:
        """Original monolithic prompt — kept for A/B comparison."""
        is_closure = state.phase == "closure"
        prompt = prompts.sim_turn(
            name=self.name,
            topic=self.topic,
            options_text=self._format_options(),
            goal=self.persona.goal,
            backstory=self.persona.backstory,
            personality_summary=self.persona.personality_summary(),
            style_rule=self.persona.style_rule(),
            max_words=self.persona.max_words(state.phase),
            phase=state.phase,
            phase_instruction=phase_instr,
            interaction_instruction=self._interaction_instruction(history, state),
            own_recent_points=self._own_recent_points_block(history),
            recent_history=self._recent_history(history, max_lines=4 if is_closure else 8),
            forbidden_openers=self._recent_openers(history),
            beliefs_block="" if is_closure else self._beliefs_block(),
            last_speaker_line="" if is_closure else self._last_participant_line(history),
            position_discipline="" if is_closure else self._position_discipline(state),
            forced_adaptation=forced_adaptation,
        )
        try:
            result = self._llm.generate(prompt).strip()
            return self._ground_check(result, prompt)
        except Exception as exc:
            print(f"!! Turn generation error for {self.name}: {exc}")
            return "[SILENCE]"

    def _generate_compact(
        self, history: list[str], state: "DialogueState",
        forced_adaptation: bool, phase_instr: str,
        turn_plan: Optional["TurnPlan"] = None,
    ) -> str:
        """Speaker-card compact prompt (Stage 4). ~50-60% fewer input tokens."""
        from realize.prompt_context import (
            build_speaker_card,
            build_relevant_options,
            build_group_state,
            build_rolling_summary,
            build_local_context,
            build_move_instruction,
            build_output_contract,
        )

        is_closure = state.phase == "closure"
        n_recent: int = cfg.prompt_budget.recent_turns_short

        speaker_card = build_speaker_card(self.persona)

        candidate = state.candidate_option or state.current_leading_option
        relevant_opts = build_relevant_options(self.options, self.persona, candidate)

        group_state = "" if is_closure else build_group_state(state)

        rolling = build_rolling_summary(history, older_than_n=n_recent)
        local_ctx = build_local_context(history, n_recent=n_recent, rolling_summary=rolling)

        interaction_instr = "" if is_closure else self._interaction_instruction(history, state)
        position_disc = "" if is_closure else self._position_discipline(state)
        forbidden_openers = self._recent_openers(history)

        move_instr = build_move_instruction(
            phase_instruction=phase_instr,
            interaction_instruction=interaction_instr,
            position_discipline=position_disc,
            forced_adaptation=forced_adaptation,
            forbidden_openers=forbidden_openers,
            turn_plan=turn_plan,
        )

        output_contract = build_output_contract(
            max_words=self.persona.max_words(state.phase),
            name=self.name,
        )

        prompt = prompts.sim_turn_compact(
            speaker_card=speaker_card,
            relevant_options=relevant_opts,
            group_state=group_state,
            local_context=local_ctx,
            move_instruction=move_instr,
            output_contract=output_contract,
        )

        try:
            result = self._llm.generate(prompt).strip()
            return self._ground_check(result, prompt)
        except Exception as exc:
            print(f"!! Turn generation error for {self.name}: {exc}")
            return "[SILENCE]"

    # ------------------------------------------------------------------
    # Grounding check (Stage 10)
    # ------------------------------------------------------------------

    def _ground_check(self, turn_text: str, original_prompt: str) -> str:
        """
        Deterministic fact-check.  If suspicious invented facts are found and
        cfg.grounding.repair_attempts >= 1, regenerate once with an appended
        directive.  Returns the final turn text (repaired or original).
        """
        grounding_cfg = cfg.grounding
        if not grounding_cfg.enable_fact_check:
            return turn_text

        flags = _fact_check(turn_text, self.options, self.topic)
        if not flags:
            return turn_text

        repair_limit = grounding_cfg.repair_attempts
        if repair_limit < 1:
            print(f"  [grounding] {self.name}: invented facts flagged: {flags[:3]}")
            return turn_text

        print(f"  [grounding] {self.name}: repairing — flagged {flags[:3]}")
        repair_prompt = original_prompt + "\n\n" + _repair_directive()
        try:
            repaired = self._llm.generate(repair_prompt).strip()
            remaining = _fact_check(repaired, self.options, self.topic)
            if remaining:
                print(f"  [grounding] {self.name}: repair did not fully resolve flags {remaining[:3]}")
            return repaired
        except Exception as exc:
            print(f"  [grounding] repair error for {self.name}: {exc}")
            return turn_text

    # ------------------------------------------------------------------
    # Beliefs block
    # ------------------------------------------------------------------

    def _beliefs_block(self) -> str:
        """Compact belief-state anchor injected into every non-closure turn."""
        b = self.persona.beliefs
        if not b:
            return ""

        parts = ["Private anchor - stay coherent, do not recite:"]
        parts.append(f"  Prefer {b.preferred}; concern: {b.key_concern}")

        other_acceptable = [x for x in b.acceptable if x != b.preferred]
        if other_acceptable:
            opts_str = ", ".join(f"Option {x}" for x in other_acceptable)
            parts.append(f"  Could accept: {opts_str}; if: {b.concession}")

        if b.rejected:
            opts_str = ", ".join(f"Option {x}" for x in b.rejected)
            parts.append(f"  Opposed to: {opts_str}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Position discipline
    # ------------------------------------------------------------------

    def _position_discipline(self, state: "DialogueState") -> str:
        """Compute belief-state flags and delegate prose to prompts.position_discipline_block()."""
        if state.phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
            return ""

        beliefs = self.persona.beliefs
        if not beliefs:
            return ""

        preferred = beliefs.preferred
        flips = state.vote_changes.get(self.name, 0)
        current_text_vote = state.last_known_vote.get(self.name)

        if current_text_vote and current_text_vote != preferred and flips >= 1:
            anchor = current_text_vote
            prefix = f"You switched to Option {anchor}."
        else:
            anchor = preferred
            prefix = f"You lean toward Option {anchor} ({beliefs.key_concern})."

        candidate = (state.candidate_option or state.current_leading_option) if state.phase == "emergence" else None
        rejected_list = beliefs.rejected or []
        can_soften = (
            (self.persona.agreeableness >= 3 or self.persona.conscientiousness <= 2)
            and state.emergence_rounds >= 3
        )

        return prompts.position_discipline_block(
            phase=state.phase,
            prefix=prefix,
            anchor=anchor,
            flips=flips,
            candidate=candidate,
            candidate_in_acceptable=bool(candidate and candidate in beliefs.acceptable and candidate != anchor),
            candidate_in_rejected=bool(candidate and candidate in rejected_list),
            candidate_is_anchor=bool(candidate and candidate == anchor),
            can_soften=can_soften,
            concession_text=beliefs.concession or "",
            high_agreeableness=self.persona.agreeableness >= 4,
            low_agreeableness_or_high_neuroticism=(self.persona.agreeableness <= 2 or self.persona.neuroticism >= 4),
        )

    # ------------------------------------------------------------------
    # Context formatters
    # ------------------------------------------------------------------

    def _format_options(self) -> str:
        return "\n".join(f"  {opt}" for opt in self.options)

    def _recent_history(self, history: list[str], max_lines: int = 12) -> str:
        return "\n".join(history[-max_lines:])

    def _interaction_instruction(self, history: list[str], state: "DialogueState") -> str:
        """Compute dialogue-state flags and delegate prose to prompts.interaction_instruction_block()."""
        if state.phase in {"greeting", "closure", "confirmation"}:
            return ""

        last = self._last_participant_line(history)
        recent = self._recent_participant_messages(history, limit=5)
        question_count = sum(1 for msg in recent if "?" in msg)

        last_has_question = bool(last and "?" in last and last.split(":", 1)[0].strip() != self.name)

        compromise = getattr(state, "compromise_option", None) if state.phase == "emergence" else None
        b = self.persona.beliefs
        compromise_in_acceptable = bool(compromise and b and compromise in b.acceptable)

        rejected = getattr(state, "last_rejected_option", None)
        rejecting_speaker = getattr(state, "last_rejecting_speaker", None)
        rejecting_self = bool(rejected and rejecting_speaker == self.name and state.phase in {"narrowing", "emergence"})
        turns_since_rejection = len(self._own_recent_turns(history, limit=10)) if rejecting_self else 0

        own_last_was_question = bool(self._last_own_turn(history) and "?" in self._last_own_turn(history))

        own_turns = self._own_recent_turns(history, limit=4)
        speculative_count = sum(1 for t in own_turns if re.search(r"\bwhat if\b", t, re.IGNORECASE))

        repeated_kws = self._repeated_concern_keywords(history)
        self_repeated = self._own_recent_repetition(history) if len(repeated_kws) < 2 else False

        return prompts.interaction_instruction_block(
            last_has_question=last_has_question,
            question_count=question_count,
            compromise_option=compromise,
            compromise_in_acceptable=compromise_in_acceptable,
            rejected_option=rejected if rejecting_self else None,
            rejecting_self=rejecting_self,
            turns_since_rejection=turns_since_rejection,
            escalation_threshold=cfg.repetition.turns_since_rejection_escalation,
            own_last_was_question=own_last_was_question,
            speculative_count=speculative_count,
            repeated_kws=repeated_kws,
            self_repeated=self_repeated,
        )

    def _own_recent_points_block(self, history: list[str], limit: int = 3) -> str:
        turns = self._own_recent_turns(history, limit=limit)
        if not turns:
            return ""
        compact = " | ".join(turns)
        return (
            "\nYour recent points: "
            f"{compact}\nDo not repeat these points. Add a new angle or respond to someone else's point."
        )

    def _own_recent_repetition(self, history: list[str]) -> bool:
        turns = self._own_recent_turns(history, limit=2)
        if len(turns) < 2:
            return False
        a = set(re.sub(r"[^\w\s]", "", turns[0].lower()).split())
        b = set(re.sub(r"[^\w\s]", "", turns[1].lower()).split())
        if not a or not b:
            return False
        return len(a & b) / max(1, min(len(a), len(b))) >= cfg.repetition.jaccard_threshold_self

    def _repeated_concern_keywords(self, history: list[str]) -> list[str]:
        """Keywords this sim mentioned in 2+ of their last 5 turns — signals semantic looping."""
        turns = self._own_recent_turns(history, limit=5)
        if len(turns) < 2:
            return []
        stopwords = {
            "that", "this", "with", "from", "have", "would", "could", "should",
            "there", "their", "about", "option", "think", "just", "like", "work",
            "good", "need", "want", "make", "sure", "know", "okay", "also", "even",
            "very", "more", "some", "than", "when", "what", "which", "been", "will",
            "still", "into", "back", "each", "much", "both", "your", "they", "them",
            "then", "well", "here", "feel", "does", "might", "only", "most", "over",
            "really", "think", "right", "actually", "though", "going", "mean",
        }
        from collections import Counter as _Counter
        turn_word_sets: list[set[str]] = []
        for turn in turns:
            words = set(re.sub(r"[^\w\s]", "", turn.lower()).split())
            meaningful = {w for w in words if len(w) >= 4 and w not in stopwords}
            turn_word_sets.append(meaningful)
        word_counts: _Counter = _Counter()
        for ws in turn_word_sets:
            word_counts.update(ws)
        return [w for w, count in word_counts.items() if count >= 2]

    def _own_recent_turns(self, history: list[str], limit: int) -> list[str]:
        turns: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == self.name:
                turns.append(msg.strip())
                if len(turns) >= limit:
                    break
        return turns

    def _recent_participant_messages(self, history: list[str], limit: int) -> list[str]:
        messages: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            messages.append(msg.strip())
            if len(messages) >= limit:
                break
        return messages

    def _last_own_turn(self, history: list[str]) -> str:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == self.name:
                return msg.strip()
        return ""

    def _enforce_word_budget(self, text: str, phase: str) -> str:
        """Hard trim overlong LLM output so the transcript remains chat-like."""
        max_words = self.persona.max_words(phase)
        clean = re.sub(r"\s+", " ", text).strip()
        words = clean.split()
        if len(words) <= max_words:
            return clean

        sentences = re.split(r"(?<=[.!?])\s+", clean)
        kept: list[str] = []
        count = 0
        for sentence in sentences:
            s_words = sentence.split()
            if not s_words:
                continue
            if count + len(s_words) <= max_words:
                kept.append(sentence.strip())
                count += len(s_words)
            elif kept:
                break
            else:
                kept = s_words[:max_words]
                break

        trimmed = " ".join(kept)

        trimmed = trimmed.strip(" ,;:")
        if not trimmed.endswith((".", "!", "?")):
            trimmed += "."
        return trimmed

    def _recent_openers(self, history: list[str], n: int = 6) -> str:
        """First 1-2 words of recent participant turns — prevents repetitive openings.

        Tracks both single-word openers (e.g. "honestly") and two-word phrases
        (e.g. "hmm yeah", "but what") so the prompt can forbid them by exact string.
        """
        openers: list[str] = []
        seen: set[str] = set()
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            words = msg.strip().split()
            if not words:
                continue
            w1 = words[0].rstrip(",.!?").lower()
            w2 = (words[1].rstrip(",.!?").lower() if len(words) > 1 else "")
            two_word = f"{w1} {w2}".strip() if w2 else w1
            key = two_word
            if key and key not in seen:
                openers.append(two_word)
                seen.add(key)
                seen.add(w1)  # also block the bare first word
            if len(openers) >= n:
                break
        return ", ".join(f'"{o}"' for o in openers) if openers else ""

    # ------------------------------------------------------------------
    # Nudges
    # ------------------------------------------------------------------

    def _last_participant_line(self, history: list[str]) -> str:
        """Most recent line from someone other than self, for soft context."""
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() not in cfg.EXCLUDED_SPEAKERS and speaker.strip() != self.name:
                return f"{speaker.strip()}: {msg.strip()}"
        return ""

    def _closure_line(self, state: "DialogueState") -> str:
        option = getattr(state, "preferred_option", None)
        if option and random.random() < cfg.moderation.closure.template_probability:
            return random.choice(prompts.closure_templates(option))
        return random.choice(self._GOODBYES)
