"""Deterministic structural evaluation for one autonomous-simulator run.

The runtime already records compact defensible metrics. This module exposes
those metrics to evaluation scripts and supplies a flat row for CSV summaries.
No evaluation or validator LLM is used.
"""

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
    turns = metrics["turns"]
    generation = metrics["generation"]
    issues = metrics["issues"]
    stances = metrics["stances"]
    tokens = metrics["tokens"]
    return {
        "outcome": outcome.status,
        "final_option": outcome.final_option or "",
        "participant_turns": turns["participant_turns"],
        "voluntary_turns": turns["voluntary_turns"],
        "mandatory_answers": turns["mandatory_answers"],
        "openings": turns["openings"],
        "votes": turns["votes"],
        "moderator_turns": turns["moderator_turns"],
        "repairs": generation["repairs"],
        "dropped_turns": generation["dropped_turns"],
        "liveness_forced_turns": generation["liveness_forced_turns"],
        "suppressed_repetitions": generation["suppressed_repetitions"],
        "issues_opened": issues["opened"],
        "issues_resolved": issues["resolved"],
        "issues_stale": issues["stale"],
        "issue_follow_ups": issues["follow_ups"],
        "questions_answered": issues["questions_answered"],
        "questions_resolved": issues["questions_resolved"],
        "concerns_resolved": issues["concerns_resolved"],
        "concerns_maintained": issues["concerns_maintained"],
        "concerns_partially_addressed": issues["concerns_partially_addressed"],
        "visible_switches": stances["visible_switches"],
        "public_acceptances": stances["public_acceptance_count"],
        "vote_failures": metrics["votes"]["non_valid_final_statuses"],
        "narrowing_focus_adherence": metrics["narrowing"]["focus_adherence"],
        "llm_calls": tokens["llm_calls"],
        "repair_calls": tokens["repair_calls"],
        "tokens_in": tokens["input_tokens"],
        "tokens_out": tokens["output_tokens"],
    }


__all__ = ["metrics_for", "flat_metrics_for"]
