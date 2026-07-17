from builders import SetupBuilder
from models import OptionCard


class InvalidAliasLLM:
    def generate_json(self, *_args, **_kwargs):
        return {
            "short_names": {
                "B": "BA via LHR",
                "D": "Delta 2-Stop",
            }
        }


def _card(option_id: str, name: str, short_name: str = "") -> OptionCard:
    return OptionCard(
        id=option_id,
        name=name,
        short_name=short_name,
        attrs={"carrier": "listed", "stops": "listed", "route": "listed"},
        upside="public advantage",
        concern="public drawback",
    )


def test_alias_repair_has_deterministic_fallback_for_long_flight_names():
    builder = SetupBuilder("Book a flight from Miami to Stockholm", llm=InvalidAliasLLM())
    options = [
        _card("A", "Direct Flight with Scandinavian Airlines", "Direct Flight"),
        _card("B", "One-Stop Flight via London Heathrow with British Airways"),
        _card("C", "Overnight Flight with Icelandair", "Overnight Flight"),
        _card("D", "Two-Stop Flight via Atlanta and Copenhagen with Delta Airlines"),
    ]
    proposed = {
        "A": "Direct Flight",
        "B": "BA via LHR",
        "C": "Overnight Flight",
        "D": "Delta 2-Stop",
    }

    notes = builder._ensure_valid_aliases(options, proposed)

    aliases = {option.id: option.short_name for option in options}
    assert aliases["B"] == "British Airways"
    assert aliases["D"] == "Delta Airlines"
    assert len({alias.casefold() for alias in aliases.values()}) == 4
    assert any("alias_repaired_deterministically" in note for note in notes)


def test_alias_candidates_handle_route_without_carrier_phrase():
    candidates = SetupBuilder._alias_candidates(
        "Two-Stop Flight via New York and Copenhagen"
    )
    assert "New York" in candidates or "Copenhagen" in candidates


def test_full_scenario_parse_survives_invalid_flight_abbreviations():
    builder = SetupBuilder("Book a flight from Miami to Stockholm", llm=InvalidAliasLLM())
    raw = {
        "shared_context": ["The group is comparing publicly listed flight options."],
        "options": [
            {
                "id": "A",
                "name": "Direct Flight with Scandinavian Airlines",
                "short_name": "Direct Flight",
                "attrs": {"carrier": "Scandinavian Airlines", "stops": "none", "route": "Miami to Stockholm"},
                "upside": "no connection",
                "concern": "highest listed fare",
            },
            {
                "id": "B",
                "name": "One-Stop Flight via London Heathrow with British Airways",
                "short_name": "BA via LHR",
                "attrs": {"carrier": "British Airways", "stops": "one", "route": "via London Heathrow"},
                "upside": "one connection",
                "concern": "airport transfer time",
            },
            {
                "id": "C",
                "name": "Overnight Flight with Icelandair",
                "short_name": "Overnight Flight",
                "attrs": {"carrier": "Icelandair", "stops": "one", "route": "overnight itinerary"},
                "upside": "overnight schedule",
                "concern": "overnight travel",
            },
            {
                "id": "D",
                "name": "Two-Stop Flight via Atlanta and Copenhagen with Delta Airlines",
                "short_name": "Delta 2-Stop",
                "attrs": {"carrier": "Delta Airlines", "stops": "two", "route": "via Atlanta and Copenhagen"},
                "upside": "multiple routing options",
                "concern": "two connections",
            },
        ],
    }

    scenario = builder._parse_scenario(raw, 3)

    assert scenario.option("B").short_name == "British Airways"
    assert scenario.option("D").short_name == "Delta Airlines"
    assert any("alias_repaired_deterministically" in note for note in scenario.setup_notes)


def test_generated_scenario_context_is_a_single_paragraph_string():
    from prompts import setup_scenario

    prompt = setup_scenario("Ship a fragile prototype", 3)
    assert '"shared_context": "One or two complete sentences' in prompt
    assert "never output it as a list or bullets" in prompt
    assert "Every context statement must be able to coexist with every option" in prompt


def test_shared_context_normalization_accepts_one_or_two_sentences_only():
    from builders import normalize_shared_context
    import pytest

    assert normalize_shared_context(
        "The shipment is fragile. The destination needs the complete prototype."
    ) == ["The shipment is fragile. The destination needs the complete prototype."]
    with pytest.raises(ValueError, match="1..2 complete sentences"):
        normalize_shared_context("First condition. Second condition. Third condition.")


def test_scenario_parser_accepts_string_context_and_stores_one_paragraph():
    builder = SetupBuilder("Book a flight from Miami to Stockholm", llm=InvalidAliasLLM())
    raw = {
        "shared_context": (
            "The travelers leave from Miami and must arrive together in Stockholm. "
            "They are comparing economy itineraries listed for the same travel date."
        ),
        "options": [
            {
                "id": option_id,
                "name": name,
                "short_name": short_name,
                "attrs": {"carrier": "listed", "stops": "listed", "route": "listed"},
                "upside": "public advantage",
                "concern": "public drawback",
            }
            for option_id, name, short_name in (
                ("A", "Direct Flight with Scandinavian Airlines", "Direct Flight"),
                ("B", "One-Stop Flight via London with British Airways", "British Airways"),
                ("C", "Overnight Flight with Icelandair", "Overnight Flight"),
                ("D", "Two-Stop Flight with Delta Airlines", "Delta Airlines"),
            )
        ],
    }

    scenario = builder._parse_scenario(raw, 3)
    assert len(scenario.shared_context) == 1
    assert scenario.context_text.startswith("The travelers leave from Miami")


def test_setup_sampling_is_reproducible_from_the_run_rng():
    import random

    first = SetupBuilder("Choose a project workspace", llm=InvalidAliasLLM(), rng=random.Random(777))
    second = SetupBuilder("Choose a project workspace", llm=InvalidAliasLLM(), rng=random.Random(777))

    first_traits = first._trait_rows(4)
    first_shape = first._preference_shape(4, 4)
    first_preferences = first._preference_assignments(4, ["A", "B", "C", "D"], first_shape)

    second_traits = second._trait_rows(4)
    second_shape = second._preference_shape(4, 4)
    second_preferences = second._preference_assignments(4, ["A", "B", "C", "D"], second_shape)

    assert first_traits == second_traits
    assert first_shape == second_shape
    assert first_preferences == second_preferences


def test_shared_context_stop_cap_rejects_incompatible_option():
    builder = SetupBuilder("Book a flight", llm=InvalidAliasLLM())
    raw = {
        "shared_context": "All listed flights include at most one layover.",
        "options": [
            {
                "id": option_id,
                "name": name,
                "short_name": short_name,
                "attrs": {
                    "stops": stops,
                    "duration": "listed",
                    "price": "listed",
                },
                "upside": "public advantage",
                "concern": "public drawback",
            }
            for option_id, name, short_name, stops in (
                ("A", "Direct Flight", "Direct", "0"),
                ("B", "Flight via London", "London", "1"),
                ("C", "Flight via Reykjavik", "Reykjavik", "1"),
                ("D", "Flight via New York and Copenhagen", "New York", "2"),
            )
        ],
    }
    scenario = builder._parse_scenario(raw, 3)
    from builders import shared_option_constraint_violations

    violations = shared_option_constraint_violations(scenario)
    assert len(violations) == 1
    assert "option D" in violations[0]
    assert "maximum of 1" in violations[0]
