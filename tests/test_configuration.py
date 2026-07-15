from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from config_loader import Config, cfg


def test_configured_trait_mappings_are_complete_and_monotonic():
    bids = [cfg.level_value("simulator", "bid_probability_by_engagement", level) for level in range(1, 6)]
    movements = [cfg.level_value("simulator", "movement_probability_by_stubbornness", level) for level in range(1, 6)]
    words = [cfg.level_value("language", "max_words_by_verbosity", level, cast=int) for level in range(1, 6)]
    assert bids == sorted(bids)
    assert movements == sorted(movements, reverse=True)
    assert movements[-1] == 0.0
    assert words == sorted(words)


def test_scaled_conversation_budgets():
    assert cfg.conversation_turn_budgets(2) == (4, 10, 14)
    assert cfg.conversation_turn_budgets(3) == (6, 15, 21)
    assert cfg.conversation_turn_budgets(4) == (8, 20, 28)
    assert cfg.conversation_turn_budgets(6) == (12, 22, 30)
    assert cfg.conversation_turn_budgets(7) == (14, 22, 30)
    assert int(cfg.conversation.compromise_window_max_turns) == 1
    assert int(cfg.conversation.narrowing_reaction_turn_cap) == 2
    assert int(cfg.conversation.large_group_narrowing_final_position_cap) == 3
    assert int(cfg.conversation.recent_turns_in_prompt) == 7
    assert [cfg.level_value("language", "max_words_by_verbosity", level, cast=int) for level in range(1, 6)] == [8, 12, 16, 22, 27]


def _write_config(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load((Path(cfg.root) / "config.yaml").read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_config_rejects_non_monotonic_engagement(tmp_path):
    path = _write_config(tmp_path, lambda data: data["simulator"]["bid_probability_by_engagement"].update({3: 0.1}))
    with pytest.raises(ValueError, match="non-decreasing"):
        Config(path)


def test_config_rejects_nonzero_hardblocker_movement_probability(tmp_path):
    path = _write_config(tmp_path, lambda data: data["simulator"]["movement_probability_by_stubbornness"].update({5: 0.1}))
    with pytest.raises(ValueError, match="zero movement probability"):
        Config(path)


def test_config_rejects_missing_directness_level(tmp_path):
    def mutate(data):
        data["language"]["directness_instructions"].pop(4)
    path = _write_config(tmp_path, mutate)
    with pytest.raises(ValueError, match="missing level 4"):
        Config(path)


def test_config_rejects_invalid_per_participant_pacing(tmp_path):
    def mutate(data):
        data["conversation"]["soft_target_voluntary_turns_per_participant"] = 8
        data["conversation"]["hard_max_voluntary_turns_per_participant"] = 5
    path = _write_config(tmp_path, mutate)
    with pytest.raises(ValueError, match="per-participant budgets"):
        Config(path)


def test_config_rejects_invalid_unknown_information_probability(tmp_path):
    path = _write_config(
        tmp_path,
        lambda data: data["simulator"].update(
            {"unknown_information_question_probability": 1.5}
        ),
    )
    with pytest.raises(ValueError, match="unknown_information_question_probability"):
        Config(path)


def test_config_rejects_negative_issue_response_caps(tmp_path):
    def mutate(data):
        data["conversation"]["direct_question_optional_follow_up_cap"] = -1
    path = _write_config(tmp_path, mutate)
    with pytest.raises(ValueError, match="direct_question_optional_follow_up_cap"):
        Config(path)
