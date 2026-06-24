"""Essential tests for consensus detection and support fraction."""

from __future__ import annotations

import pytest
from conftest import _make_persona, _make_options
from models import DialogueState, OptionCoverage, ParticipantRuntime, Phase, Scenario
from dialogue import ConsensusManager


@pytest.fixture()
def consensus_state() -> DialogueState:
    options = _make_options()
    scenario = Scenario(topic="Test", decision_kind="test", opening_question="What?", options=options)
    personas = [_make_persona("p1", "Alice", preferred="A"), _make_persona("p2", "Bob", preferred="B"), _make_persona("p3", "Carol", preferred="A")]
    st = DialogueState(scenario=scenario, personas=personas, phase=Phase.CONFIRMATION)
    for p in personas:
        st.runtimes[p.id] = ParticipantRuntime(persona_id=p.id, current_preference=p.preferred_option)
    for o in options:
        st.coverage[o.id] = OptionCoverage()
    return st


class TestConsensus:
    def test_unanimous_vote_is_consensus(self, consensus_state):
        cm = ConsensusManager()
        for pid in ["p1", "p2", "p3"]:
            consensus_state.runtimes[pid].explicit_vote = "A"
        outcome = cm.detect(consensus_state)
        assert outcome is not None and outcome.status == "consensus"

    def test_partial_vote_no_consensus(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "A"
        assert cm.detect(consensus_state) is None

    def test_accept_counts_as_support(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].accepted_options = {"A"}
        consensus_state.runtimes["p3"].explicit_vote = "A"
        assert cm.support_fraction(consensus_state, "A") == 1.0

    def test_vote_overrides_preference(self, consensus_state):
        cm = ConsensusManager()
        for pid in ["p1", "p2", "p3"]:
            consensus_state.runtimes[pid].explicit_vote = "B"
        assert cm.support_fraction(consensus_state, "B") == 1.0
        assert cm.support_fraction(consensus_state, "A") == 0.0

    def test_hard_blocker_prevents_fallback(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "A"
        consensus_state.runtimes["p3"].explicit_vote = "B"
        consensus_state.runtimes["p3"].hard_rejections = {"A": "blocked"}
        outcome = cm.finalize(consensus_state)
        assert outcome.final_option != "A" or outcome.status != "fallback"

    def test_majority_fallback(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "B"
        consensus_state.runtimes["p3"].explicit_vote = "A"
        consensus_state.candidate_option = "A"
        outcome = cm.finalize(consensus_state)
        assert outcome.status == "fallback" and outcome.final_option == "A"
