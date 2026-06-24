"""Unit tests for consensus/outcome state consistency (O7)."""

from __future__ import annotations

import pytest
from conftest import _make_persona
from models import (
    DialogueState,
    OptionCoverage,
    ParticipantRuntime,
    Phase,
    Scenario,
)
from conftest import _make_options
from dialogue import ConsensusManager


@pytest.fixture()
def consensus_state() -> DialogueState:
    options = _make_options()
    scenario = Scenario(
        topic="Test topic",
        decision_kind="test",
        opening_question="What?",
        options=options,
    )
    personas = [
        _make_persona("p1", "Alice", preferred="A"),
        _make_persona("p2", "Bob", preferred="B"),
        _make_persona("p3", "Carol", preferred="A"),
    ]
    st = DialogueState(scenario=scenario, personas=personas, phase=Phase.CONFIRMATION)
    for p in personas:
        st.runtimes[p.id] = ParticipantRuntime(persona_id=p.id, current_preference=p.preferred_option)
    for o in options:
        st.coverage[o.id] = OptionCoverage()
    return st


class TestSupportFraction:
    def test_unanimous_vote(self, consensus_state):
        cm = ConsensusManager()
        for pid in ["p1", "p2", "p3"]:
            consensus_state.runtimes[pid].explicit_vote = "A"
        assert cm.support_fraction(consensus_state, "A") == 1.0

    def test_split_vote(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "B"
        consensus_state.runtimes["p3"].explicit_vote = "A"
        fraction = cm.support_fraction(consensus_state, "A")
        assert abs(fraction - 2/3) < 0.01

    def test_vote_overrides_preference(self, consensus_state):
        """A persona who preferred A but voted B should count as B supporter."""
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "B"
        consensus_state.runtimes["p2"].explicit_vote = "B"
        consensus_state.runtimes["p3"].explicit_vote = "B"
        assert cm.support_fraction(consensus_state, "B") == 1.0
        assert cm.support_fraction(consensus_state, "A") == 0.0

    def test_accept_counts_as_support(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].accepted_options = {"A"}
        consensus_state.runtimes["p3"].explicit_vote = "A"
        assert cm.support_fraction(consensus_state, "A") == 1.0

    def test_preference_only_when_no_vote(self, consensus_state):
        """Original preference counts only when no vote or accept has been made."""
        cm = ConsensusManager()
        # p1 preferred A and hasn't voted -> should count
        # p2 preferred B and accepted A -> should count for A
        # p3 preferred A and voted B -> should count for B, NOT A
        consensus_state.runtimes["p2"].accepted_options = {"A"}
        consensus_state.runtimes["p3"].explicit_vote = "B"
        fraction_a = cm.support_fraction(consensus_state, "A")
        # p1 (preference A, no vote) + p2 (accepted A) = 2/3
        assert abs(fraction_a - 2/3) < 0.01


class TestConsensusDetection:
    def test_consensus_requires_all(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "A"
        # p3 hasn't voted or accepted
        outcome = cm.detect(consensus_state)
        assert outcome is None

    def test_consensus_when_all_voted(self, consensus_state):
        cm = ConsensusManager()
        for pid in ["p1", "p2", "p3"]:
            consensus_state.runtimes[pid].explicit_vote = "A"
        outcome = cm.detect(consensus_state)
        assert outcome is not None
        assert outcome.status == "consensus"
        assert outcome.final_option == "A"

    def test_fallback_with_majority(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "B"
        consensus_state.runtimes["p3"].explicit_vote = "A"
        consensus_state.candidate_option = "A"
        outcome = cm.finalize(consensus_state)
        assert outcome.status == "fallback"
        assert outcome.final_option == "A"

    def test_hard_blocker_prevents_fallback(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "A"
        consensus_state.runtimes["p3"].explicit_vote = "B"
        consensus_state.runtimes["p3"].hard_rejections = {"A": "blocked"}
        outcome = cm.finalize(consensus_state)
        # A has 2/3 support but p3 hard-rejects it
        assert outcome.final_option != "A" or outcome.status != "fallback"
