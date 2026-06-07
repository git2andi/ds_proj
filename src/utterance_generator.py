"""One-turn LLM utterance generation."""

from __future__ import annotations

from typing import Optional

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from schemas import DialogueState, MoveIntent, Persona, Phase, ValidationResult
from utils import normalise_ws, strip_speaker_prefix


class UtteranceGenerator:
    def __init__(self) -> None:
        self._llm = get_llm_client()

    def generate(self, state: DialogueState, intent: MoveIntent) -> tuple[str, str, int, int]:
        persona = state.persona_by_id(intent.speaker_id)
        max_words = self._max_words(intent, state.phase, persona)
        prompt = prompts.sim_utterance(
            persona=persona,
            scenario=state.scenario,
            recent_lines=self._recent_lines(state),
            state_summary=self._state_summary(state),
            intent=intent,
            addressee_name=state.name_for(intent.addressee_id) if intent.addressee_id else None,
            max_words=max_words,
        )
        text = self._llm.generate(prompt, profile="dialogue")
        cleaned = self._clean(text, persona.name, max_words)
        return cleaned, prompt, self._llm.last_tokens_in, self._llm.last_tokens_out

    def repair(self, original: str, validation: ValidationResult, state: DialogueState, intent: MoveIntent) -> tuple[str, str, int, int]:
        persona = state.persona_by_id(intent.speaker_id)
        max_words = self._max_words(intent, state.phase, persona)
        prompt = prompts.repair_utterance(
            original_text=original,
            issue_codes=validation.codes(),
            persona=persona,
            scenario=state.scenario,
            recent_lines=self._recent_lines(state),
            state_summary=self._state_summary(state),
            intent=intent,
            addressee_name=state.name_for(intent.addressee_id) if intent.addressee_id else None,
            max_words=max_words,
        )
        text = self._llm.generate(prompt, profile="repair")
        cleaned = self._clean(text, persona.name, max_words)
        return cleaned, prompt, self._llm.last_tokens_in, self._llm.last_tokens_out

    def fallback(self, state: DialogueState, intent: MoveIntent) -> str:
        persona = state.persona_by_id(intent.speaker_id)
        candidate = intent.option_focus[0] if intent.option_focus else state.candidate_option
        text = prompts.deterministic_fallback(intent, persona, candidate)
        return self._clean(text, persona.name, self._max_words(intent, state.phase, persona))

    def _clean(self, text: str, speaker_name: str, max_words: int) -> str:
        text = strip_speaker_prefix(normalise_ws(text), speaker_name)
        text = text.strip().strip('"')
        # Keep only first line if the model created multiple paragraphs.
        if "\n" in text:
            text = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
        words = text.split()
        hard_cap = max_words + int(cfg.utterances.hard_cap_extra_words)
        if len(words) > hard_cap:
            text = " ".join(words[:hard_cap]).rstrip(" ,;:") + "."
        return text

    def _max_words(self, intent: MoveIntent, phase: Phase, persona: Persona) -> int:
        budgets = cfg.utterances.word_budgets
        if phase == Phase.OPENING:
            base = int(budgets.opening)
        elif intent.act.value == "vote":
            base = int(budgets.vote)
        elif intent.act.value in {"accept", "reject"}:
            base = int(budgets.confirm)
        elif phase == Phase.CLOSURE:
            base = int(budgets.closure)
        else:
            base = int(getattr(budgets, intent.length_hint))
        trait_factor = 0.75 + (persona.traits.response_length - 1) * 0.125
        return max(6, int(base * trait_factor))

    def _recent_lines(self, state: DialogueState) -> list[str]:
        n = int(cfg.utterances.max_recent_turns_in_prompt)
        return [f"{t.speaker_name}: {t.text}" for t in state.turns[-n:]]

    def _state_summary(self, state: DialogueState) -> dict:
        return {
            "phase": state.phase.value,
            "candidate_option": state.candidate_option,
            "readiness_score": state.readiness_score,
            "open_questions": [
                {"target": state.name_for(q.target_id), "asked_by": state.name_for(q.asked_by), "text": q.text}
                for q in state.open_questions
                if q.asked_by != "moderator" and q.target_id != "moderator"
            ],
            "participant_state": {
                state.name_for(pid): {
                    "vote": rt.explicit_vote,
                    "accepted": sorted(rt.accepted_options),
                    "rejected": sorted(rt.rejected_options),
                    "turn_count": rt.turn_count,
                }
                for pid, rt in state.runtimes.items()
            },
            "coverage": {
                oid: {
                    "mentions": cov.mentions,
                    "reasons": cov.reasons,
                    "objections": cov.objections,
                    "acceptances": cov.acceptances,
                }
                for oid, cov in state.coverage.items()
            },
        }
