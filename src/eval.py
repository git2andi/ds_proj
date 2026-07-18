"""Small deterministic metric adapter used by batch evaluation scripts."""

from __future__ import annotations

from typing import Any

from logger import metrics_for
from models import DialogueState, RunOutcome


def flat_metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    metrics = metrics_for(state, outcome)
    return {
        "outcome": outcome.status,
        "final_option": outcome.final_option or "",
        "participant_turns": metrics["participant_turns"],
        "voluntary_turns": metrics["voluntary_turns"],
        "moderator_turns": metrics["moderator_turns"],
        "moderator_ratio": metrics["moderator_ratio"],
        "avg_words_per_participant_turn": metrics["avg_words_per_participant_turn"],
        "visible_preference_changes": metrics["visible_preference_changes"],
        "repairs": metrics["repair_turns"],
        "dropped_turns": metrics["dropped_turns"],
        "fallback_turns": metrics["fallback_turns"],
        "response_failures": metrics["response_failures"],
        "protocol_error_count": metrics["protocol_errors"],
        "vote_outcome_consistent": metrics["vote_outcome_consistent"],
        "llm_calls": metrics["llm_calls"],
        "tokens_in": metrics["input_tokens"],
        "tokens_out": metrics["output_tokens"],
    }


__all__ = ["metrics_for", "flat_metrics_for"]
