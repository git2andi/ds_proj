"""I11: the LLM grounding judge only runs when a regex tripwire finds a
suspicious concrete claim. No LLM calls."""

from __future__ import annotations

from dialogue import DialogueRunner, initialise_state
from models import OptionCard, Persona, Scenario, SimulatorParameters, TraitProfile


def _world():
    options = [
        OptionCard(id="A", name="Sunny Side Cafe", attrs={"cost": "$20", "distance": "2 miles"},
                   upside="cozy", tradeoff="limited vegetarian options", concern="crowded", best_for="speed"),
        OptionCard(id="B", name="Green Garden Bistro", attrs={"cost": "$25"},
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
