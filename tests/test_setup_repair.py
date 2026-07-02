"""Deterministic setup repair of persona preferences. No LLM calls."""

from __future__ import annotations

import pytest

from builders import repair_preferred_options


def test_missing_required_preference_inserted_first():
    assert repair_preferred_options(["B"], None, "A", single_only=False) == ["A", "B"]


def test_reordered_required_preference_moved_to_front():
    assert repair_preferred_options(["B", "A"], None, "A", single_only=False) == ["A", "B"]


def test_empty_preferences_rescued_by_required():
    assert repair_preferred_options([], None, "C", single_only=False) == ["C"]


def test_result_capped_at_two_options():
    assert repair_preferred_options(["B", "C"], None, "A", single_only=False) == ["A", "B"]


def test_hard_blocker_keeps_exactly_one():
    assert repair_preferred_options(["A", "B"], "D", "A", single_only=True) == ["A"]


def test_rejection_of_required_option_raises():
    with pytest.raises(ValueError):
        repair_preferred_options(["B"], "A", "A", single_only=False)


def test_no_required_leaves_list_unchanged():
    assert repair_preferred_options(["B", "C"], None, None, single_only=False) == ["B", "C"]
