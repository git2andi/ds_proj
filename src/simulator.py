"""
simulator.py
------------
Simulator -- wraps a Persona and generates one dialogue turn via the LLM,
using the compact speaker-card prompt.

Generation flow (per turn):
  1. _generate_raw()    : build prompt + call LLM -> (raw_text, prompt)
  2. strip name prefix  : remove "Name: " if model echoed it
  3. _verify_and_repair : verify; repair once if needed; store result in
                          self._last_verification for the logger to read
  4. _enforce_word_budget : hard cap on word count
"""

from __future__ import annotations

import re
from typing import Any, Optional, TYPE_CHECKING

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
    pick_surface_move_kind,
)
from utils import OptionResolver
from verifier import VerificationResult, verify_participant_turn

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from state import StructuredState


class Simulator:

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options
        self._resolver = OptionResolver(options)
        self._llm = get_llm_client()
        self._last_verification: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------

    def generate_turn(
        self,
        history: list[str],
        state: "DialogueState",
        all_names: Optional[list[str]] = None,
        structured: Optional["StructuredState"] = None,
    ) -> tuple[str, int, int]:
        del all_names  # reserved for future use
        raw, gen_prompt = self._generate_raw(history, state, structured)
        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out

        if not raw:
            self._last_verification = None
            return "[SILENCE]", tok_in, tok_out

        # Strip "Name: " prefix if the model echoed it
        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        # Verify and repair (stores result in self._last_verification)
        raw = self._verify_and_repair(raw, gen_prompt, history, state, structured)

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

    def _generate_raw(
        self,
        history: list[str],
        state: "DialogueState",
        structured: Optional["StructuredState"],
    ) -> tuple[str, str]:
        """Build the full prompt, call the LLM, and return (raw_text, prompt).
        No verification or grounding check here -- that happens in generate_turn.
        """
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

        # Update.md §4.2 -- tiny stochastic surface-move nudge. Suppressed when
        # the simulator already has a hard obligation (open challenge or
        # pending question) so we don't fight the existing instruction.
        last_line = self._last_participant_line(history)
        last_has_question = bool(
            last_line and "?" in last_line and last_line.split(":", 1)[0].strip() != self.name
        )
        surface_kind = None if is_closure else pick_surface_move_kind(
            phase=state.phase,
            repetition_high=state.repetition_pressure >= cfg.repetition.stall_increment_threshold,
            has_open_challenge=open_challenger is not None,
            has_open_question=last_has_question,
        )

        move_instr = build_move_instruction(
            phase_instruction=phase_instr,
            interaction_instruction=interaction_instr,
            position_discipline=position_disc,
            surface_move_kind=surface_kind,
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
        return result, prompt

    # ------------------------------------------------------------------
    # Verification + repair
    # ------------------------------------------------------------------

    def _verify_and_repair(
        self,
        text: str,
        gen_prompt: str,
        history: list[str],
        state: "DialogueState",
        structured: Optional["StructuredState"],
    ) -> str:
        """Verify the generated text; repair once if needed.

        Stores a summary dict in self._last_verification for the logger.
        Returns the (possibly repaired) text.
        """
        if not cfg.verification.enabled:
            self._last_verification = None
            return text

        ps = structured.participants.get(self.name) if structured else None
        candidate = state.candidate_option or state.current_leading_option

        result: VerificationResult = verify_participant_turn(
            text=text,
            speaker_name=self.name,
            phase=state.phase,
            options=self.options,
            history=history,
            persona_state=ps,
            resolver=self._resolver,
            candidate=candidate,
        )

        if not result.needs_repair:
            self._last_verification = result.as_dict() if result.issues else None
            return text

        # --- Attempt repair -----------------------------------------------
        repair_prompt = self._build_repair_prompt(text, gen_prompt, result, state)
        repaired = self._llm.generate(repair_prompt).strip()

        # Strip name prefix again
        if repaired.lower().startswith(f"{self.name.lower()}:"):
            repaired = repaired.split(":", 1)[1].strip()

        # Verify the repaired text
        result2: VerificationResult = verify_participant_turn(
            text=repaired,
            speaker_name=self.name,
            phase=state.phase,
            options=self.options,
            history=history,
            persona_state=ps,
            resolver=self._resolver,
            candidate=candidate,
        )

        result.repair_attempted = True
        result.repair_succeeded = not result2.needs_repair
        self._last_verification = result.as_dict()

        if not result2.needs_repair and repaired:
            return repaired

        # --- Deterministic fallback for phase-critical failures -----------
        fallback = self._deterministic_fallback(state, result)
        if fallback:
            return fallback

        # Accept original if no fallback is applicable
        return text

    def _build_repair_prompt(
        self,
        original_text: str,
        gen_prompt: str,
        result: "VerificationResult",
        state: "DialogueState",
    ) -> str:
        """Choose the right repair prompt based on the primary issue code.

        Priority order (high-impact / phase-critical first):
          phase obligations (vote, confirmation) -> validity (option refs/facts)
          -> naturalness (ack-loop, semantic repeat, self-repetition).
        """
        repair_codes = {i.code for i in result.issues if i.severity == "repair"}

        if "MISSING_EXPLICIT_VOTE" in repair_codes:
            return prompts.repair_vote(self.options)

        if "UNCLEAR_CONFIRMATION" in repair_codes:
            candidate = state.candidate_option or state.current_leading_option or "?"
            return prompts.repair_confirmation(candidate)

        if "VALID_OPTION_DENIED" in repair_codes or "INVALID_OPTION_REFERENCE" in repair_codes:
            return prompts.repair_invalid_option(original_text, self.options)

        if "INVENTED_OPTION_FACT" in repair_codes:
            return prompts.repair_invented_fact(gen_prompt)

        # Update.md §4.3 -- group-level acknowledgement loop.
        if "ACK_LOOP" in repair_codes:
            return prompts.repair_ack_loop(original_text)

        # Update.md §4.4 -- semantic repeat (use the matched prior point in the
        # repair so the model knows which point to avoid).
        if "SEMANTIC_POINT_REPEAT" in repair_codes:
            prior = next(
                (i.message.split("'")[1] for i in result.issues
                 if i.code == "SEMANTIC_POINT_REPEAT" and "'" in i.message),
                "",
            )
            return prompts.repair_semantic_repeat(original_text, prior)

        if "SELF_REPETITION" in repair_codes:
            return prompts.repair_repetition(original_text)

        # Default: treat like repetition
        return prompts.repair_repetition(original_text)

    def _deterministic_fallback(
        self,
        state: "DialogueState",
        result: "VerificationResult",
    ) -> Optional[str]:
        """Deterministic safe line for phase-critical failures that survived repair."""
        codes = {i.code for i in result.issues}
        beliefs = self.persona.beliefs

        if "MISSING_EXPLICIT_VOTE" in codes and state.phase == "narrowing":
            preferred = beliefs.preferred if beliefs else (self._resolver.letters[0] if self._resolver.letters else "A")
            return f"I'd go with Option {preferred}."

        if "UNCLEAR_CONFIRMATION" in codes and state.phase == "confirmation":
            candidate = state.candidate_option or state.current_leading_option
            if candidate and beliefs and candidate in (beliefs.rejected or []):
                return "No, I'm still not convinced."
            return "Yeah, that works for me."

        return None

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
