"""Essential tests for consensus detection and support fraction."""

from __future__ import annotations

import pytest
from conftest import _make_persona, _make_options
from models import DialogueState, OptionCoverage, ParticipantRuntime, Phase, Scenario
from dialogue import ConsensusManager
from models import OpenQuestion
from router import TurnRouter


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
        assert outcome is not None and outcome.status == "successful"

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

    def test_rejection_state_does_not_veto_visible_majority(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "A"
        consensus_state.runtimes["p3"].explicit_vote = "B"
        consensus_state.runtimes["p3"].hard_rejections = {"A": "blocked"}
        outcome = cm.finalize(consensus_state)
        assert outcome.status == "majority" and outcome.final_option == "A"

    def test_majority_fallback(self, consensus_state):
        cm = ConsensusManager()
        consensus_state.runtimes["p1"].explicit_vote = "A"
        consensus_state.runtimes["p2"].explicit_vote = "B"
        consensus_state.runtimes["p3"].explicit_vote = "A"
        consensus_state.candidate_option = "A"
        outcome = cm.finalize(consensus_state)
        assert outcome.status == "majority" and outcome.final_option == "A"


class TestBestAnswerer:
    """R24: _best_answerer must never route the asker back to answer their own question,
    or route any persona who themselves has an open question (they don't know either)."""

    def test_asker_not_chosen_for_own_question(self, consensus_state):
        router = TurnRouter()
        q = OpenQuestion(turn_id=1, asked_by="p1", target_id="p2", text="What about cost?")
        consensus_state.open_questions = [q]
        result = router._best_answerer(consensus_state, q)
        assert result != "p1"

    def test_prefers_option_champion(self, consensus_state):
        router = TurnRouter()
        # p2 prefers "B" (sole champion); p1 prefers "A"; p3 is the asker → must pick p2
        q = OpenQuestion(turn_id=1, asked_by="p3", target_id="p1", text="How good is B?", option_focus=["B"])
        consensus_state.open_questions = [q]
        result = router._best_answerer(consensus_state, q)
        assert result == "p2"

    def test_other_open_askers_excluded(self, consensus_state):
        """If p1 asked Q1 and p2 asked Q2, routing for Q2 should not pick p1."""
        router = TurnRouter()
        q1 = OpenQuestion(turn_id=1, asked_by="p1", target_id="p3", text="What about cost?")
        q2 = OpenQuestion(turn_id=2, asked_by="p2", target_id="p1", text="How good is A?")
        consensus_state.open_questions = [q1, q2]
        # When answering q2, neither p1 (asked q1) nor p2 (asked q2) should be chosen
        result = router._best_answerer(consensus_state, q2)
        assert result not in {"p1", "p2"}


class TestUpdateQuestions:
    """R26: ANSWER turns that still carry QUESTION_ECHO must not propagate as new OpenQuestions."""

    def test_question_echo_in_answer_not_propagated(self, consensus_state):
        from conftest import make_turn
        from dialogue import StateTracker
        from models import ActType, OpenQuestion

        tracker = StateTracker(consensus_state)
        # Simulate: Q1 was asked, router routes p2 to ANSWER, but p2 echoes Q1
        q1 = OpenQuestion(turn_id=1, asked_by="p1", target_id="p2", text="Do they have room?")
        consensus_state.open_questions = [q1]
        # A turn record that represents p2's failed ANSWER (QUESTION_ECHO remaining)
        turn = make_turn(index=2, speaker_id="p2", speaker_name="Bob", text="Do they have room?", act_type=ActType.ANSWER)
        from models import MoveIntent
        turn.intent = MoveIntent(speaker_id="p2", act=ActType.ANSWER, reason="answer", respond_to_turn=1)
        turn.validation_issues = ["QUESTION_ECHO"]
        # Manually set question_target_id on the act (as parse_dialogue_act would)
        turn.act.question_target_id = "p1"

        tracker._update_questions(consensus_state, turn)
        # Q1 should be cleared (respond_to_turn=1 matches), but no new Q added
        assert len(consensus_state.open_questions) == 0

    def test_normal_question_still_registered(self, consensus_state):
        from conftest import make_turn
        from dialogue import StateTracker
        from models import ActType, MoveIntent

        tracker = StateTracker(consensus_state)
        turn = make_turn(index=1, speaker_id="p1", speaker_name="Alice", text="Do they have parking?", act_type=ActType.ASK)
        turn.intent = MoveIntent(speaker_id="p1", act=ActType.ASK, reason="ask")
        turn.act.question_target_id = "p2"

        tracker._update_questions(consensus_state, turn)
        assert len(consensus_state.open_questions) == 1


def test_response_target_uses_exact_latest_turn_from_addressee(consensus_state):
    from conftest import make_turn

    consensus_state.turns.extend(
        [
            make_turn(1, "p2", "Bob", "The beach is closer."),
            make_turn(2, "p3", "Carol", "The city is cheaper."),
            make_turn(3, "p2", "Bob", "The beach still has the easier schedule."),
        ]
    )

    assert TurnRouter()._response_turn_for(consensus_state, "p1", "p2") == 3


def test_local_response_target_prefers_recent_turn_about_focus(consensus_state):
    from conftest import make_turn

    beach = make_turn(1, "p2", "Bob", "The beach drive is too long.")
    beach.act.option_refs = ["B"]
    city = make_turn(2, "p3", "Carol", "The city costs less.")
    city.act.option_refs = ["C"]
    consensus_state.turns.extend([beach, city])

    assert TurnRouter()._local_response_turn_for(consensus_state, "p1", ["B"]) == 1
