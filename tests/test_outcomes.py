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


def test_transcript_vote_round_yields_majority():
    """End-to-end regression for logs/20260702_092804_559743: the exact vote-round
    lines that once closed as unresolved must parse and produce a 2/3 majority."""
    from parsing import OptionResolver, visible_commitment

    options = [
        OptionCard(id="A", name="Amazon Redshift Data Warehouse", short_name="Redshift"),
        OptionCard(id="B", name="Google BigQuery Serverless Analytics", short_name="BigQuery"),
        OptionCard(id="C", name="PostgreSQL on Single VM", short_name="PostgreSQL VM"),
        OptionCard(id="D", name="ClickHouse Managed Cloud Service", short_name="ClickHouse Cloud"),
    ]
    resolver = OptionResolver(options)
    lines = {
        "p1": "I'm going with ClickHouse for its fast queries, low maintenance, and reasonable cost.",
        "p2": "I'd go with Redshift for its scalability and seamless AWS integration despite the higher cost.",
        "p3": "My vote is ClickHouse for fast, low-maintenance analytics that fit our $120 monthly budget and provide millisecond responses.",
    }
    votes = {}
    for pid, text in lines.items():
        commitment = visible_commitment(text, resolver)
        assert commitment is not None, f"{pid} vote must parse: {text}"
        stance, option_id = commitment
        assert stance == "vote"
        votes[pid] = option_id

    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    personas = [_persona(pid, "D") for pid in votes]
    state = initialise_state(scenario, personas)
    for pid, vote in votes.items():
        state.runtimes[pid].explicit_vote = vote
    out = ConsensusManager.finalize(state)
    assert out.status == "majority" and out.final_option == "D"


def test_direct_vote_overrides_accept_derived_vote():
    """#23: 'Daily News seems like a solid pick' (accept) must not lock out the
    formal 'Comedy Stories gets my vote' round vote."""
    rt = ParticipantRuntime(persona_id="p1", explicit_vote="A", vote_stance="accept")
    DialogueRunner._set_vote(rt, "B", "Comedy Stories from Real Life gets my vote for keeping things light.")
    assert rt.explicit_vote == "B" and rt.vote_stance == "vote"


def test_direct_vote_still_protected_from_direct_overwrite():
    rt = ParticipantRuntime(persona_id="p1", explicit_vote="A", vote_stance="vote")
    DialogueRunner._set_vote(rt, "B", "B gets my vote now.")
    assert rt.explicit_vote == "A"  # no explicit change signal


def test_accept_does_not_override_direct_vote():
    rt = ParticipantRuntime(persona_id="p1", explicit_vote="A", vote_stance="vote")
    DialogueRunner._set_vote(rt, "B", "B works for me too.", stance="accept")
    assert rt.explicit_vote == "A"


def test_switch_turn_gets_budget_headroom():
    """#24: a compromise switch (allow_vote_change) must have room for the bridge clause."""
    from models import ActType, MoveIntent
    persona = _persona("p1", "A")
    plain = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r")
    switch = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", allow_vote_change=True)
    _, plain_max = DialogueRunner._word_bounds(plain, persona)
    _, switch_max = DialogueRunner._word_bounds(switch, persona)
    assert switch_max > plain_max
