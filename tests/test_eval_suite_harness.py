from __future__ import annotations

import tests  # noqa: F401

from eval.run_eval_suite import SUITE_VERSION, format_summary_row


def test_suite_version_invalidates_previous_closeout_runs():
    assert SUITE_VERSION.endswith("v7")


def test_summary_uses_grouped_metric_columns():
    row = {
        "case_id": "c02",
        "returncode": 0,
        "outcome_status": "majority",
        "final_option": "A",
        "participant_turn_count": 20,
        "runtime_validator_calls": 0,
        "repair_rate": 0.05,
        "dropped_turns": 0,
        "critical_grounding_interventions": 0,
        "total_input_tokens": 24000,
        "flags": "",
    }
    line = format_summary_row(row)
    assert "outcome=majority" in line
    assert "turns=20" in line
    assert "validator_calls=0" in line
    assert "drops=0" in line
    assert "critical_grounding=0" in line
    assert "tokens_in=24000" in line
    assert "v/turn=" not in line
    assert "outcome=None" not in line
