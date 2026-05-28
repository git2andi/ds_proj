"""
simulator.py
------------
Simulator -- wraps a Persona and generates one dialogue turn via the LLM,
using the compact speaker-card prompt. Stage 1c + Stage 6: the prompt now
carries a per-sim memory block built from StructuredState.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from persona import Persona
from prompt_context import (
    build_group_state,
    build_local_context,
    build_memory_block,
    build_move_instruction,
    build_output_contract,
    build_relevant_options,
    build_speaker_card,
)
from reasoning import fact_check, repair_directive
from utils import OptionResolver

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from policy import TurnPlan
    from state import StructuredState


class Simulator:

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options
        self._resolver = OptionResolver(options)
        self._llm = get_llm_client()

    # ------------------------------------------------------------------

    def generate_turn(
        self,
        history: list[str],
        state: "DialogueState",
        all_names: Optional[list[str]] = None,
        turn_plan: Optional["TurnPlan"] = None,
        structured: Optional["StructuredState"] = None,
    ) -> tuple[str, int, int]:
        del all_names  # reserved for future use
        raw = self._generate(history, state, turn_plan, structured)
        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out

        if not raw:
            return "[SILENCE]", tok_in, tok_out

        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        raw = self._enforce_word_budget(raw, state.phase)
        return raw or "[SILENCE]", tok_in, tok_out

    # ------------------------------------------------------------------

    def _has_voted(self, history: list[str]) -> bool:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() != self.name:
                continue
            if self._resolver.vote_in(msg):
                return True
        return False

    def _generate(
        self,
        history: list[str],
        state: "DialogueState",
        turn_plan: Optional["TurnPlan"],
        structured: Optional["StructuredState"],
    ) -> str:
        is_closure = state.phase == "closure"
        phase_instr = prompts.phase_instruction_text(
            phase=state.phase, has_voted=self._has_voted(history),
            final_option=getattr(state, "preferred_option", None) if is_closure else None,
        )

        n_recent = cfg.prompt_budget.recent_turns_short

        speaker_card = build_speaker_card(self.persona)
        candidate = state.candidate_option or state.current_leading_option
        relevant_opts = build_relevant_options(self.options, self.persona, candidate)
        group_state = "" if is_closure else build_group_state(state)
        local_ctx = build_local_context(history, n_recent=n_recent)
        memory_block = "" if is_closure else build_memory_block(self.name, structured)

        open_challenger = self._open_challenger(structured)
        interaction_instr = "" if is_closure else self._interaction_instruction(
            history, state, open_challenger,
        )
        position_disc = "" if is_closure else self._position_discipline(state)

        plan_to_show = turn_plan if (turn_plan is not None and turn_plan.directive) else None

        move_instr = build_move_instruction(
            phase_instruction=phase_instr,
            interaction_instruction=interaction_instr,
            position_discipline=position_disc,
            turn_plan=plan_to_show,
        )
        output_contract = build_output_contract(
            max_words=self.persona.max_words(state.phase), name=self.name,
        )

        prompt = prompts.sim_turn_compact(
            speaker_card=speaker_card,
            relevant_options=relevant_opts,
            group_state=group_state,
            local_context=local_ctx,
            memory_block=memory_block,
            move_instruction=move_instr,
            output_contract=output_contract,
        )

        result = self._llm.generate(prompt).strip()
        return self._ground_check(result, prompt)

    # ------------------------------------------------------------------

    def _ground_check(self, turn_text: str, original_prompt: str) -> str:
        if not cfg.grounding.enable_fact_check or cfg.grounding.repair_attempts < 1:
            return turn_text

        flags = fact_check(turn_text, self.options, self.topic)
        if not flags:
            return turn_text

        if len(turn_text.split()) < cfg.grounding.min_words_to_check:
            if not any(f[:1].isdigit() for f in flags):
                return turn_text

        return self._llm.generate(original_prompt + "\n\n" + repair_directive()).strip()

    # ------------------------------------------------------------------

    def _open_challenger(
        self, structured: Optional["StructuredState"]
    ) -> Optional[str]:
        if structured is None:
            return None
        for ch in reversed(structured.discourse.challenges):
            if ch.target == self.name and ch.answered_turn_id is None:
                return ch.challenger
        return None

    def _interaction_instruction(
        self,
        history: list[str],
        state: "DialogueState",
        open_challenger: Optional[str],
    ) -> str:
        if state.phase in {"opening", "closure", "confirmation"}:
            return ""

        last = self._last_participant_line(history)
        last_has_question = bool(
            last and "?" in last and last.split(":", 1)[0].strip() != self.name
        )

        last_claim_speaker: Optional[str] = None
        if not last_has_question and last and ":" in last:
            spk = last.split(":", 1)[0].strip()
            if spk != self.name and spk not in cfg.EXCLUDED_SPEAKERS:
                last_claim_speaker = spk

        repetition_high = state.repetition_pressure >= cfg.repetition.stall_increment_threshold

        return prompts.interaction_instruction_block(
            last_has_question=last_has_question,
            last_claim_speaker=last_claim_speaker,
            repetition_high=repetition_high,
            open_challenge_from=open_challenger,
        )

    def _position_discipline(self, state: "DialogueState") -> str:
        if state.phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
            return ""
        beliefs = self.persona.beliefs
        if not beliefs:
            return ""

        anchor = beliefs.preferred
        candidate = state.candidate_option or state.current_leading_option

        return prompts.position_discipline_block(
            phase=state.phase,
            anchor=anchor,
            candidate=candidate,
            candidate_in_acceptable=bool(
                candidate and candidate in beliefs.acceptable and candidate != anchor
            ),
            candidate_in_rejected=bool(candidate and candidate in (beliefs.rejected or [])),
            candidate_is_anchor=bool(candidate and candidate == anchor),
            reconsider_text=beliefs.would_reconsider_if or "",
        )

    # ------------------------------------------------------------------

    def _last_participant_line(self, history: list[str]) -> str:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() not in cfg.EXCLUDED_SPEAKERS and speaker.strip() != self.name:
                return f"{speaker.strip()}: {msg.strip()}"
        return ""

    def _enforce_word_budget(self, text: str, phase: str) -> str:
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

        trimmed = " ".join(kept).strip(" ,;:")
        if trimmed and trimmed[-1] not in (".", "!", "?"):
            trimmed += "."
        return trimmed
