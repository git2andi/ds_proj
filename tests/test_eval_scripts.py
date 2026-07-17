"""Deterministic checks for the eval2/ batch scripts (no LLM calls)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval2"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from experiment_common import (  # noqa: E402
    cfg,
    config_overrides,
    read_scenarios,
    run_dialogue,
    set_config_value,
    setup_fingerprint,
)
from run_config_sweep import build_experiments  # noqa: E402
from judge_transcripts import DIMENSIONS, rotated_judges, validate_response  # noqa: E402
from validate_judge import (  # noqa: E402
    corrupt_grounding,
    corrupt_outcome,
    corrupt_personas,
    corrupt_turn_order,
)
from builders import SetupBuilder  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from tests.fixtures import make_personas, make_scenario  # noqa: E402

import pytest  # noqa: E402


def test_scenarios_file_parses_with_balanced_supported_counts():
    cases = read_scenarios()
    counts = Counter(case.participants for case in cases)
    minimum = int(cfg.simulation.min_participants)
    maximum = int(cfg.simulation.max_participants)
    assert set(counts) == set(range(minimum, maximum + 1))
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len({case.topic for case in cases}) == len(cases)


def test_scenario_topics_do_not_contradict_their_participant_count():
    for case in read_scenarios():
        # Raises when a topic names a group size that differs from the count.
        SetupBuilder._validate_topic_participant_count(case.topic, case.participants)


def test_config_overrides_restore_previous_values():
    original = cfg._raw["conversation"]["recent_turns_in_prompt"]
    with config_overrides({("conversation", "recent_turns_in_prompt"): original + 5}):
        assert cfg._raw["conversation"]["recent_turns_in_prompt"] == original + 5
        assert cfg.conversation.recent_turns_in_prompt == original + 5
    assert cfg._raw["conversation"]["recent_turns_in_prompt"] == original
    assert cfg.conversation.recent_turns_in_prompt == original


def test_set_config_value_updates_raw_and_attribute_views():
    previous = set_config_value("simulator", "unknown_information_question_probability", 0.5)
    try:
        assert cfg._raw["simulator"]["unknown_information_question_probability"] == 0.5
        assert cfg.simulator.unknown_information_question_probability == 0.5
    finally:
        set_config_value("simulator", "unknown_information_question_probability", previous)


def test_sweep_is_limited_to_the_four_observed_problem_areas():
    experiments = build_experiments(3)
    assert [experiment.name for experiment in experiments] == [
        "duplicate_detection",
        "issue_follow_up",
        "consecutive_turns",
        "small_group_closure",
    ]
    sections = {
        section
        for experiment in experiments
        for variant in experiment.variants
        for section, _ in variant.overrides
    }
    assert sections == {"conversation", "language"}

def test_every_sweep_variant_passes_full_config_validation():
    for participants in (3, 6):
        for experiment in build_experiments(participants):
            for variant in experiment.variants:
                assert variant.overrides
                with config_overrides(variant.overrides):
                    cfg._validate()  # raises on any constraint violation


def test_llm_client_accepts_judge_provider_override():
    # 'uni' builds without network access or API keys; the judge defaults to it.
    client = LLMClient(provider="uni")
    assert client.provider == "uni"
    assert client.model_id == str(cfg.llm.models.get("uni"))
    override = LLMClient(provider="uni", model="llama3.1:8b")
    assert override.model_id == "llama3.1:8b"
    with pytest.raises(ValueError):
        LLMClient(provider="nonsense")


def test_small_group_closure_experiment_is_only_used_for_small_groups():
    assert "small_group_closure" in {experiment.name for experiment in build_experiments(3)}
    assert "small_group_closure" not in {experiment.name for experiment in build_experiments(6)}


def test_setup_fingerprint_is_stable_and_sensitive_to_personas():
    scenario = make_scenario()
    personas = make_personas(("A", "B", "C"))
    first = setup_fingerprint(scenario, personas)
    second = setup_fingerprint(scenario, personas)
    changed = setup_fingerprint(scenario, make_personas(("A", "A", "C")))
    assert first == second
    assert first != changed


def test_run_dialogue_rejects_partial_paired_setup():
    row = run_dialogue(
        "Choose a study location",
        participants=3,
        seed=1,
        scenario=make_scenario(),
        personas=None,
    )
    assert row["outcome"] == "error"
    assert "supplied together" in row["error"]


def test_judge_response_validation_is_strict_and_complete():
    scores, verdict = validate_response(
        {"scores": {dimension: 4 for dimension in DIMENSIONS}, "verdict": "Minor repetition only."}
    )
    assert scores == {dimension: 4 for dimension in DIMENSIONS}
    assert verdict
    with pytest.raises(ValueError, match="missing score"):
        validate_response({"scores": {"naturalness": 4}, "verdict": "Incomplete."})
    with pytest.raises(ValueError, match="integer"):
        validate_response(
            {"scores": {dimension: 4.5 for dimension in DIMENSIONS}, "verdict": "Invalid."}
        )


def test_judge_order_is_deterministic_and_contains_distinct_roles():
    first = rotated_judges("run-17", 3)
    second = rotated_judges("run-17", 3)
    assert first == second
    assert len({name for name, _ in first}) == 3


def test_judge_corruptions_modify_only_copies():
    payload = {
        "run_id": "sample",
        "scenario": {"options": [{"id": "A"}, {"id": "B"}]},
        "personas": [
            {"id": "p1", "name": "Nora", "background": "one"},
            {"id": "p2", "name": "Ben", "background": "two"},
        ],
        "turns": [
            {"speaker_id": "p1", "moderator": False, "phase": "DISCUSSION", "text": "First"},
            {"speaker_id": "p2", "moderator": False, "phase": "DISCUSSION", "text": "Second"},
            {"speaker_id": "p1", "moderator": False, "phase": "NARROWING", "text": "Third"},
        ],
        "votes": {"p1": "A", "p2": "A"},
        "outcome": {"status": "successful", "final_option": "A"},
    }
    shuffled = corrupt_turn_order(payload, 3)
    grounded = corrupt_grounding(payload, 3)
    personas = corrupt_personas(payload, 3)
    outcome = corrupt_outcome(payload, 3)
    assert payload["turns"][0]["text"] == "First"
    assert [turn["text"] for turn in shuffled["turns"]] != ["First", "Second", "Third"]
    assert "42 percent" in grounded["turns"][0]["text"]
    assert personas["personas"][0]["id"] == "p1"
    assert personas["personas"][0]["background"] == "two"
    assert outcome["outcome"]["final_option"] != "A"
