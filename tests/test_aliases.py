"""Shared option-alias contract across setup, prompts, and parsing."""

from __future__ import annotations

from aliases import short_alias_map, validated_short_alias
from models import OptionCard
from parsing import OptionResolver


def test_short_alias_must_be_recognizable_and_long_enough():
    assert validated_short_alias("Codenames", "Spy") == ""
    assert validated_short_alias("Ticket to Ride", "Rails") == ""
    assert validated_short_alias("Structured Daily Quiz Sessions", "Quiz") == "Quiz"


def test_colliding_aliases_are_not_exposed_to_prompts_or_parser():
    options = [
        OptionCard(id="A", name="Alpha Project Plan", short_name="Project"),
        OptionCard(id="B", name="Beta Project Plan", short_name="Project"),
    ]

    aliases = short_alias_map(options)
    resolver = OptionResolver(options)

    assert aliases["A"] != "Project"
    assert aliases["B"] != "Project"
    assert aliases["A"] != aliases["B"]
    assert resolver.ids_in_text(aliases["A"]) == ["A"]
    assert resolver.ids_in_text(aliases["B"]) == ["B"]
