"""I10: moderator interventions are targeted and evidence-based; vote calls
stay option-neutral. No LLM calls."""

from __future__ import annotations

from dialogue import DialogueRunner, initialise_state
from models import OptionCard, Persona, Scenario, SimulatorParameters, TraitProfile
from parsing import OptionResolver


def _persona(pid: str, pref: str) -> Persona:
    return Persona(
        id=pid,
        name=pid.upper(),
        traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=SimulatorParameters(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        background="b",
        private_goal="g",
        preferred_options=[pref],
    )


def _world(prefs=("A", "B", "C")):
    options = [OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    personas = [_persona(f"p{i+1}", p) for i, p in enumerate(prefs)]
    state = initialise_state(scenario, personas)
    return state, personas


def _runner() -> DialogueRunner:
    runner = DialogueRunner.__new__(DialogueRunner)
    runner._resolver = OptionResolver([OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]])
    return runner


def test_group_vote_call_is_option_neutral():
    state, _ = _world()
    target, action, focus = _runner()._moderator_intervention_details(state, "A", voting=True)
    assert target is None
    assert focus == []
    assert "do not name" in action


def test_single_unresolved_voter_is_addressed_without_candidate():
    state, _ = _world()
    state.runtimes["p1"].explicit_vote = "A"
    state.runtimes["p2"].explicit_vote = "A"
    target, action, focus = _runner()._moderator_intervention_details(state, "A", voting=True)
    assert target == "p3"
    assert focus == []
    assert "do not name" in action


def test_blocker_on_candidate_is_probed_once():
    state, _ = _world()
    state.runtimes["p2"].hard_rejections["A"] = "dealbreaker"
    runner = _runner()
    target, action, focus = runner._moderator_intervention_details(state, "A")
    assert target == "p2" and focus == ["A"]
    assert "work for them" in action
    # second stall nudge must not repeat the probe
    target2, action2, _ = runner._moderator_intervention_details(state, "A")
    assert not (target2 == "p2" and "work for them" in action2)


def test_visible_split_requests_head_to_head():
    state, _ = _world()
    state.runtimes["p1"].accepted_options.add("A")
    state.runtimes["p2"].accepted_options.add("C")
    target, action, focus = _runner()._moderator_intervention_details(state, None)
    assert target is None
    assert set(focus) == {"A", "C"}
    assert "weigh" in action
