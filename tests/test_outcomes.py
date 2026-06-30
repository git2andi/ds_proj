"""Deterministic tests for vote-overwrite protection and outcome logic (no LLM)."""

from __future__ import annotations

from dialogue import ConsensusManager, DialogueRunner, initialise_state
from models import (
    OptionCard,
    ParticipantRuntime,
    Persona,
    Scenario,
    SimulatorParameters,
    TraitProfile,
)


def _params() -> SimulatorParameters:
    return SimulatorParameters(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)


def _persona(pid: str, pref: str) -> Persona:
    return Persona(
        id=pid,
        name=pid.upper(),
        traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=_params(),
        background="b",
        private_goal="g",
        preferred_options=[pref],
    )


def _state(votes: dict[str, str | None]):
    options = [OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    personas = [_persona(pid, "A") for pid in votes]
    state = initialise_state(scenario, personas)
    for pid, vote in votes.items():
        state.runtimes[pid].explicit_vote = vote
    return state


def test_set_vote_protects_existing_clear_vote():
    rt = ParticipantRuntime(persona_id="p1", explicit_vote="B")
    DialogueRunner._set_vote(rt, "C", "I vote for C now, it is cheaper.")
    assert rt.explicit_vote == "B"  # silent flip is ignored


def test_set_vote_allows_explicit_change():
    rt = ParticipantRuntime(persona_id="p1", explicit_vote="B")
    DialogueRunner._set_vote(rt, "C", "Actually I vote for C after all.")
    assert rt.explicit_vote == "C"


def test_set_vote_sets_when_empty():
    rt = ParticipantRuntime(persona_id="p1")
    DialogueRunner._set_vote(rt, "A", "I vote for A.")
    assert rt.explicit_vote == "A"


def test_outcome_successful_unanimous():
    out = ConsensusManager.finalize(_state({"p1": "A", "p2": "A", "p3": "A"}))
    assert out.status == "successful" and out.final_option == "A"


def test_outcome_majority():
    out = ConsensusManager.finalize(_state({"p1": "A", "p2": "A", "p3": "B"}))
    assert out.status == "majority" and out.final_option == "A"


def test_outcome_unresolved_split():
    out = ConsensusManager.finalize(_state({"p1": "A", "p2": "B", "p3": "C"}))
    assert out.status == "unresolved" and out.final_option is None


def test_outcome_unresolved_tie():
    out = ConsensusManager.finalize(_state({"p1": "A", "p2": "A", "p3": "B", "p4": "B"}))
    assert out.status == "unresolved"


def test_outcome_unresolved_no_votes():
    out = ConsensusManager.finalize(_state({"p1": None, "p2": None}))
    assert out.status == "unresolved"
