"""Deterministic structural evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from logger import metrics_for  # noqa: E402
from models import DialogueState, RunOutcome  # noqa: E402


def flat_metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    metrics = metrics_for(state, outcome)
    return {
        "outcome": outcome.status,
        "final_option": outcome.final_option or "",
        "participant_turns": metrics["turns"]["participant"],
        "voluntary_turns": metrics["turns"]["voluntary"],
        "self_selected_turns": metrics["turns"]["self_selected"],
        "mandatory_turns": metrics["turns"]["mandatory"],
        "moderator_turns": metrics["turns"]["moderator"],
        "repairs": metrics["generation"]["repairs"],
        "dropped_turns": metrics["generation"]["dropped"],
        "liveness_forced_turns": metrics["generation"]["liveness_forced"],
        "questions_opened": metrics["questions"]["opened"],
        "questions_answered": metrics["questions"]["answered"],
        "issues_opened": metrics["issues"]["opened"],
        "issues_resolved": metrics["issues"]["resolved"],
        "issues_stale": metrics["issues"]["stale"],
        "concerns_opened": metrics["issues"]["concerns_opened"],
        "concerns_resolved": metrics["issues"]["concerns_resolved"],
        "concerns_stale": metrics["issues"]["concerns_stale"],
        "visible_switches": metrics["stances"]["switches"],
        "public_acceptances": metrics["stances"]["acceptances"],
        "narrowing_movements": metrics["stances"]["narrowing_movements"],
        "grounded_movements": metrics["stances"]["grounded_movements"],
        "unexplained_movements": metrics["stances"]["unexplained_movements"],
        "compromise_proposals": metrics["compromise"]["proposals"],
        "compromise_acceptances": metrics["compromise"]["acceptances"],
        "revote_skipped_no_movement": metrics["votes"]["revote_skipped"],
        "semantic_reason_reuse": metrics["generation"]["semantic_reason_reuse"],
        "vote_fallbacks": metrics["generation"]["vote_fallbacks"],
        "mandatory_movement_failures": metrics["generation"]["mandatory_movement_failures"],
        "movement_fallbacks": metrics["generation"]["movement_fallbacks"],
        "selected_movement_actions": metrics["generation"]["selected_movement_actions"],
        "committed_movement_actions": metrics["generation"]["committed_movement_actions"],
        "movement_realization_failures": metrics["generation"]["movement_realization_failures"],
        "repair_causes": metrics["generation"]["repair_causes"],
        "valid_final_votes": metrics["votes"]["valid"],
        "unclear_final_votes": metrics["votes"]["unclear"],
        "vote_protocol_degraded": metrics["votes"]["protocol_degraded"],
        "llm_calls": metrics["tokens"]["llm_calls"],
        "tokens_in": metrics["tokens"]["input"],
        "tokens_out": metrics["tokens"]["output"],
    }


__all__ = ["metrics_for", "flat_metrics_for"]
