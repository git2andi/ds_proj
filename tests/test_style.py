"""Deterministic tests for the local surface-style tracker."""

from __future__ import annotations

from style import (
    leading_name,
    leading_option,
    name_prefix_fraction,
    opening_signature,
    option_opening_fraction,
    repeated_opening_token,
    repeated_pattern,
    strip_leading_name,
    surface_pattern,
)

ALIASES = ["rooftop lounge", "hotel banquet", "Go with Gin"]

NAMES = ["Anton", "Kenji", "Lila"]


def test_leading_name_detected():
    assert leading_name("Kenji, that's fair but...", NAMES) == "Kenji"
    assert leading_name("The midday flight is better", NAMES) is None


def test_name_prefix_fraction():
    texts = ["Kenji, sure", "The flight is fine", "Lila, true but", "Anton, I get it"]
    assert name_prefix_fraction(texts, NAMES) == 0.75


def test_strip_leading_name():
    assert strip_leading_name("Kenji, the cost is too high", NAMES) == "The cost is too high"
    assert strip_leading_name("The cost is too high", NAMES) == "The cost is too high"


def test_surface_pattern_buckets():
    assert surface_pattern("I vote for Option B.") == "vote"
    assert surface_pattern("I get that, but the cost is high.") == "concede_but"
    assert surface_pattern("It's cheap but I worry about comfort.") == "worry_but"
    assert surface_pattern("How long is the layover?") == "question"
    assert surface_pattern("The direct flight is fastest.") == "statement"


def test_repeated_pattern_flags_templated_streak():
    texts = [
        "It's cheap but I worry about comfort.",
        "I get that, but the timing is bad.",
        "Fast, but the price is steep.",
    ]
    assert repeated_pattern(texts, 3) in {"concede_but", "worry_but", "tradeoff_but"}


def test_repeated_pattern_ignores_varied_turns():
    texts = [
        "I vote for Option B.",
        "How long is the layover?",
        "The direct flight is fastest.",
    ]
    assert repeated_pattern(texts, 3) is None


def test_leading_option_detects_option_openings():
    assert leading_option("The rooftop lounge is lively.", ALIASES) is True
    assert leading_option("Go with Gin offers top performance.", ALIASES) is True
    assert leading_option("Option C looks best.", ALIASES) is True
    assert leading_option("I think we should move fast.", ALIASES) is False


def test_option_opening_fraction():
    texts = ["The hotel banquet is elegant", "I worry about cost", "Go with Gin scales well"]
    assert option_opening_fraction(texts, ALIASES) == round(2 / 3, 10) or abs(option_opening_fraction(texts, ALIASES) - 2 / 3) < 1e-9


def test_opening_signature_and_repeat():
    assert opening_signature("The rooftop lounge is nice") == "rooftop"
    assert opening_signature("Kenji, we should decide") == "we"
    texts = ["Maybe we wait", "Maybe try B", "Maybe go cheaper"]
    assert repeated_opening_token(texts, 3) == "maybe"
    assert repeated_opening_token(["Yes", "No", "Maybe"], 3) is None
