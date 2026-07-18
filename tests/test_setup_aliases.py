import random

import pytest

from aliases import (
    normalize_option_text,
    resolve_option_mentions,
    unique_generated_aliases,
    validate_unique_aliases,
)
from builders import SetupBuilder, normalize_shared_context
from tests.fixtures import make_scenario


class ScenarioLLM:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0

    def generate_json(self, prompt, *, profile="setup"):
        self.calls += 1
        return self.rows.pop(0)


def valid_raw(*, duplicate_names=False):
    board = make_scenario()
    options = []
    for option in board.options:
        options.append({
            "id": option.id,
            "name": board.option("A").name if duplicate_names and option.id == "B" else option.name,
            "attrs": option.attrs,
            "upside": option.upside,
            "concern": option.concern,
        })
    return {"scenario": {"shared_context": board.context_text, "options": options}}


def alias_raw():
    return {
        "aliases": [
            {"id": "A", "aliases": ["Library"]},
            {"id": "B", "aliases": ["Cafe", "Riverside Cafe"]},
            {"id": "C", "aliases": ["Lab"]},
            {"id": "D", "aliases": ["Online"]},
        ]
    }


def test_aliases_include_generated_short_references():
    board = make_scenario()
    board.option("A").aliases = ("Central",)
    assert resolve_option_mentions("Library is my pick", board) == {"A"}
    assert resolve_option_mentions("Central is my pick", board) == {"A"}
    assert resolve_option_mentions("Central Library is my pick", board) == {"A"}
    assert resolve_option_mentions("Option A is my pick", board) == {"A"}
    assert resolve_option_mentions("the quiet place is my pick", board) == set()


def test_generated_aliases_must_be_derived_and_unique():
    names = {"A": "Chicago City Stay", "B": "Chicago Airport Hotel"}
    accepted = unique_generated_aliases(
        names,
        {"A": ["Chicago", "Chicago City"], "B": ["Chicago", "Airport Hotel", "Downtown"]},
    )
    assert accepted["A"] == ("Chicago City",)
    assert accepted["B"] == ("Airport Hotel",)


def test_alias_normalization_handles_articles_and_accents():
    assert normalize_option_text("The Café") == normalize_option_text("Cafe")


def test_invalid_generated_scenario_is_regenerated_before_alias_call():
    llm = ScenarioLLM([valid_raw(duplicate_names=True), valid_raw(), alias_raw()])
    builder = SetupBuilder("Choose a study location", llm=llm, rng=random.Random(1))
    scenario = builder._generate_scenario(3)
    assert llm.calls == 3
    assert scenario.option("B").short_name == "Cafe"
    assert "scenario_regenerated_after_validation_error" in scenario.setup_notes
    assert "generated_option_aliases" in scenario.setup_notes


def test_invalid_alias_payload_does_not_regenerate_valid_scenario():
    llm = ScenarioLLM([valid_raw(), {"aliases": "broken"}])
    builder = SetupBuilder("Choose a study location", llm=llm, rng=random.Random(1))
    scenario = builder._generate_scenario(3)
    assert llm.calls == 2
    assert scenario.option("B").short_name == "Riverside Cafe"
    assert "alias_generation_fell_back_to_full_names" in scenario.setup_notes
    validate_unique_aliases(scenario)


def test_invalid_scenario_fails_after_two_attempts():
    llm = ScenarioLLM([valid_raw(duplicate_names=True), valid_raw(duplicate_names=True)])
    builder = SetupBuilder("Choose a study location", llm=llm, rng=random.Random(1))
    with pytest.raises(RuntimeError, match="after 2 attempt"):
        builder._generate_scenario(3)


def test_shared_context_accepts_one_or_two_sentences():
    assert normalize_shared_context("One sentence.") == ["One sentence."]
    assert normalize_shared_context("First sentence. Second sentence.") == ["First sentence. Second sentence."]
    with pytest.raises(ValueError):
        normalize_shared_context("One. Two. Three.")


def test_generated_location_alias_is_accepted_in_opening_text():
    from models import OptionCard, Scenario
    from validation import validate_realization
    from dialogue import initialise_state
    from tests.fixtures import make_personas
    from models import ActionType, BidPriority, UserAction

    board = Scenario(
        topic="Choose a trip",
        shared_context=["The group needs one destination."],
        options=[
            OptionCard("A", "Chicago City Stay", {}, "museums", "busy", "Chicago", ("Chicago",)),
            OptionCard("B", "Vermont Countryside", {}, "quiet", "long trip", "Vermont", ("Vermont",)),
            OptionCard("C", "Miami Beach Hotel", {}, "beach", "humid", "Miami", ("Miami",)),
            OptionCard("D", "Aspen Mountain Resort", {}, "hiking", "costly", "Aspen", ("Aspen",)),
        ],
    )
    state = initialise_state(board, make_personas())
    action = UserAction("p1", True, BidPriority.REQUIRED, ActionType.OPENING, ("A",), reason="museums")
    assert validate_realization(state, state.persona("p1"), action, "Hi, Chicago is my current choice because of the museums.") == []


def test_invalid_generated_person_name_uses_local_unique_fallback():
    board = make_scenario()
    builder = SetupBuilder("Choose a study location", llm=ScenarioLLM([]), rng=random.Random(2))
    traits = [
        {"id": "p1", "traits": {"engagement": 3, "verbosity": 3, "directness": 3, "stubbornness": 2}, "hard_blocker": False},
        {"id": "p2", "traits": {"engagement": 3, "verbosity": 3, "directness": 3, "stubbornness": 2}, "hard_blocker": False},
    ]
    preferences = {"p1": "A", "p2": "B"}
    rows = []
    for pid, name in (("p1", ""), ("p2", "Alex")):
        rows.append({
            "id": pid,
            "name": name,
            "background": "Works on the project.",
            "private_goal": "Needs a practical choice.",
            "age": 30,
            "option_stances": {
                option.id: {
                    "rank": 5 if option.id == preferences[pid] else 3,
                    "reason_for": option.upside,
                    "reason_against": option.concern,
                }
                for option in board.options
            },
        })
    personas = builder._parse_personas(rows, traits, board, preferences)
    assert len({persona.name for persona in personas}) == 2
    assert all(persona.name for persona in personas)
    assert any(note.startswith("fallback_name_assigned:p1") for note in board.setup_notes)
