"""Corpus preset resolution and dominance weighting (issue #12). No LLM calls."""

from __future__ import annotations

import pytest

from config_loader import apply_corpus_preset
from utils import preset_dominance_weight


def _data(preset: dict | None, active: str | None = "delidata") -> dict:
    return {
        "simulation": {"num_participants": 3, "min_participants": 2, "max_participants": 7},
        "conversation": {
            "min_discussion_turns_per_participant": 3.0,
            "target_discussion_turns_per_participant": 4.0,
            "max_discussion_turns_per_participant": 6.0,
        },
        "corpus": {"preset": active, "presets": {"delidata": preset} if preset else {}},
    }


def test_no_preset_leaves_config_untouched():
    data = _data(None, active=None)
    assert apply_corpus_preset(data) is None
    assert data["simulation"]["num_participants"] == 3
    assert data["conversation"]["target_discussion_turns_per_participant"] == 4.0


def test_preset_overrides_group_size_and_pacing():
    data = _data({
        "turns_per_participant": 4.5,
        "preferred_group_size": 5,
        "top_speaker_share": 0.40,
        "dominance_range": [0.30, 0.50],
        "imbalance_tolerance": 0.15,
    })
    resolved = apply_corpus_preset(data)
    assert resolved["name"] == "delidata"
    assert data["simulation"]["num_participants"] == 5
    conv = data["conversation"]
    assert conv["target_discussion_turns_per_participant"] == 4.5
    assert conv["min_discussion_turns_per_participant"] == pytest.approx(3.38, abs=0.01)
    assert conv["max_discussion_turns_per_participant"] == pytest.approx(6.75, abs=0.01)
    assert resolved["top_speaker_share"] == 0.40
    assert resolved["dominance_range"] == (0.30, 0.50)


def test_preset_group_size_clamped_to_configured_range():
    data = _data({"preferred_group_size": 12})
    apply_corpus_preset(data)
    assert data["simulation"]["num_participants"] == 7


def test_unknown_preset_name_rejected():
    data = _data({"turns_per_participant": 4.0}, active="missing")
    with pytest.raises(ValueError):
        apply_corpus_preset(data)


def test_invalid_dominance_fields_rejected():
    with pytest.raises(ValueError):
        apply_corpus_preset(_data({"top_speaker_share": 1.5}))
    with pytest.raises(ValueError):
        apply_corpus_preset(_data({"dominance_range": [0.6, 0.4]}))


PRESET = {"top_speaker_share": 0.40, "dominance_range": (0.30, 0.50), "imbalance_tolerance": 0.15}


def test_dominant_speaker_boosted_below_target_share():
    # 2 of 10 turns = 0.2 share, target 0.4 -> boosted well above base.
    w = preset_dominance_weight(1.0, True, 2, 10, 5, PRESET, quiet_boost=1.25)
    assert w > 1.0


def test_dominant_speaker_suppressed_above_band():
    # 6 of 10 turns = 0.6 share > dominance high bound 0.5 -> suppressed.
    w = preset_dominance_weight(1.0, True, 6, 10, 5, PRESET, quiet_boost=1.25)
    assert w < 1.0


def test_others_rebalanced_only_outside_tolerance():
    # fair share 0.2, tolerance 0.15: 0.1 share stays inside the band -> unchanged.
    inside = preset_dominance_weight(1.0, False, 1, 10, 5, PRESET, quiet_boost=1.25)
    assert inside == 1.0
    # 0 of 20 turns = 0.0 share, below 0.05 -> quiet boost applies.
    quiet = preset_dominance_weight(1.0, False, 0, 20, 5, PRESET, quiet_boost=1.25)
    assert quiet == pytest.approx(2.25)
    # 8 of 20 turns = 0.4 share, above 0.35 -> damped.
    loud = preset_dominance_weight(1.0, False, 8, 20, 5, PRESET, quiet_boost=1.25)
    assert loud == pytest.approx(0.5)
