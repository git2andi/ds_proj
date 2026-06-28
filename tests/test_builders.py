"""Deterministic setup-world validation."""

from __future__ import annotations

import pytest

from builders import SetupBuilder
from prompts import setup_scenario


def _scenario_raw(shared_context: list[str]) -> dict:
    return {
        "decision_kind": "test",
        "opening_question": "What matters most?",
        "shared_context": shared_context,
        "options": [
            {
                "id": option_id,
                "name": f"Choice {option_id}",
                "short_name": f"Choice {option_id}",
                "attrs": {"cost": "$10", "time": "1 hour", "effort": "low"},
                "upside": "Useful benefit",
                "tradeoff": "Real tradeoff",
                "concern": "Known concern",
                "best_for": "people prioritizing value",
            }
            for option_id in ["A", "B", "C", "D"]
        ],
    }


def test_scenario_prompt_receives_exact_participant_count():
    prompt = setup_scenario("Choose something", 3)
    assert "exactly 3 participants" in prompt


def test_mismatched_group_size_reference_rejects_scenario():
    builder = object.__new__(SetupBuilder)
    builder.topic = "Choose something"

    with pytest.raises(ValueError, match="participant count"):
        builder._parse_scenario(_scenario_raw(["A group of 5 friends is deciding."]), 3)


def test_contradictory_generated_scores_are_rejected():
    with pytest.raises(ValueError, match="score/list contradiction"):
        SetupBuilder._build_scores(
            {"A": 5, "B": 1, "C": 1, "D": 1},
            ["A", "B", "C", "D"],
            preferred="A",
            acceptable=["A", "B"],
            soft=["C"],
            hard=[],
            pid="p1",
        )
