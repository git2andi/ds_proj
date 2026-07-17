"""Deterministic checks for the eval/ batch scripts (no LLM calls)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from experiment_common import cfg, config_overrides, read_scenarios, set_config_value  # noqa: E402
from run_config_sweep import build_parameters  # noqa: E402
from builders import SetupBuilder  # noqa: E402
from llm_client import LLMClient  # noqa: E402

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


def test_sweep_covers_all_three_config_sections():
    parameters = build_parameters(3)
    sections = {parameter.section for parameter in parameters}
    assert sections == {"conversation", "simulator", "language"}
    swept = [parameter for parameter in parameters if parameter.variants]
    assert len(swept) >= 20


def test_every_sweep_variant_passes_full_config_validation():
    for participants in (3, 6):
        for parameter in build_parameters(participants):
            for variant in parameter.variants:
                assert variant.value != parameter.current, parameter.name
                with config_overrides({(parameter.section, parameter.key): variant.value}):
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


def test_large_group_knobs_are_skipped_at_small_sizes_and_swept_at_large_sizes():
    def variants_for(participants: int, key: str) -> int:
        return sum(
            len(parameter.variants)
            for parameter in build_parameters(participants)
            if parameter.key == key
        )

    assert variants_for(3, "large_group_narrowing_final_position_cap") == 0
    assert variants_for(6, "large_group_narrowing_final_position_cap") > 0
    assert variants_for(6, "small_group_extra_no_bid_rounds") == 0
    assert variants_for(3, "small_group_extra_no_bid_rounds") > 0
