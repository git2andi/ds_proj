"""I11/I16: the LLM grounding judge only runs when a regex tripwire finds a
suspicious concrete claim or a cross-option fact transfer. No LLM calls."""

from __future__ import annotations

from dialogue import DialogueRunner, initialise_state
from models import OptionCard, Persona, Scenario, SimulatorParameters, TraitProfile
from parsing import OptionResolver


def _world():
    options = [
        OptionCard(id="A", name="Sunny Side Cafe", attrs={"cost": "$20", "distance": "2 miles"},
                   upside="cozy", tradeoff="limited vegetarian options", concern="crowded", best_for="speed"),
        OptionCard(id="B", name="Green Garden Bistro", attrs={"cost": "$25", "backup": "reliable backup power"},
                   upside="healthy", tradeoff="pricier", concern="far", best_for="diets"),
    ]
    scenario = Scenario(
        topic="t", decision_kind="generic_decision", opening_question="q",
        options=options, shared_context=["Budget is $25 per person"],
    )
    persona = Persona(
        id="p1", name="P1", traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=SimulatorParameters(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        background="b", private_goal="g", preferred_options=["A"],
    )
    state = initialise_state(scenario, [persona])
    runner = DialogueRunner.__new__(DialogueRunner)
    runner._resolver = OptionResolver(options)
    return runner, state


def test_numbers_from_cards_do_not_trip():
    runner, state = _world()
    assert runner._grounding_tripwire("Sunny Side is $20 and just 2 miles away.", state) is False


def test_invented_number_trips():
    runner, state = _world()
    assert runner._grounding_tripwire("They seat 120 people on weekends.", state) is True


def test_policy_claim_trips():
    runner, state = _world()
    assert runner._grounding_tripwire("Green Garden includes a free refill policy.", state) is True


def test_allergy_claim_trips():
    runner, state = _world()
    assert runner._grounding_tripwire("Some people are allergic to their peanut sauce.", state) is True


def test_plain_opinion_does_not_trip():
    runner, state = _world()
    assert runner._grounding_tripwire("Honestly the cozy vibe matters more to me than price.", state) is False


def test_world_cache_refreshes_per_state():
    runner, state = _world()
    assert runner._grounding_tripwire("It costs $20.", state) is False
    _, state2 = _world()
    state2.scenario.options[0].attrs["cost"] = "$99"
    runner2 = runner  # same runner, different state
    assert runner2._grounding_tripwire("It costs $99.", state2) is False


# --- I16: cross-option fact transfer ---


def test_other_options_fact_on_named_option_trips():
    """The standing-desk failure: option A credited with B's distinctive feature."""
    runner, state = _world()
    assert runner._grounding_tripwire("Sunny Side has reliable backup power, that settles it.", state) is True


def test_own_card_fact_does_not_trip():
    runner, state = _world()
    assert runner._grounding_tripwire("Sunny Side stays cozy but can get crowded.", state) is False


def test_mixed_card_facts_in_one_line_trip():
    """Facts distinctive to two different cards in one claim get judged."""
    runner, state = _world()
    assert runner._grounding_tripwire("Cozy beats healthy any day of the week.", state) is True


def test_unmentioned_option_fact_alone_does_not_trip():
    """One card's distinctive fact with no other option named stays cheap."""
    runner, state = _world()
    assert runner._grounding_tripwire("A cozy spot would suit the morning crowd.", state) is False
