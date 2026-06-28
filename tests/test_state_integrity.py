"""State mutation guards for turns that remain invalid after repair."""

from __future__ import annotations

from conftest import make_intent
from dialogue import StateTracker
from models import ActType, Phase
from parsing import TurnMove


def test_unclear_accept_is_logged_but_does_not_mutate_binding_state(state):
    state.phase = Phase.CONFIRMATION
    tracker = StateTracker(state)
    intent = make_intent("p1", ActType.ACCEPT, ["B"])
    move = TurnMove(act=ActType.ACCEPT, option="B", stance="accept", present=True)

    record = tracker.apply_participant(
        state,
        intent,
        "I still prefer Mountain Retreat.",
        move,
        tokens_in=100,
        tokens_out=10,
        validation_issues=["UNCLEAR_ACCEPT"],
        repaired=True,
    )

    assert record.act.accepts == ["B"]  # retained for diagnostics
    assert record.state_mutation_blocked is True
    assert state.runtimes["p1"].turn_count == 1
    assert state.runtimes["p1"].accepted_options == set()
    assert state.runtimes["p1"].current_preference is None
    assert state.coverage["B"].acceptances == 0


def test_grounding_failure_does_not_mutate_rejection_or_coverage(state):
    state.phase = Phase.CONFIRMATION
    tracker = StateTracker(state)
    intent = make_intent("p1", ActType.REJECT, ["B"])
    move = TurnMove(act=ActType.REJECT, option="B", stance="reject", present=True)

    record = tracker.apply_participant(
        state,
        intent,
        "Beach Resort costs $9999, so no.",
        move,
        tokens_in=100,
        tokens_out=10,
        validation_issues=["INVENTED_OPTION_ATTRIBUTE"],
        repaired=True,
    )

    assert state.runtimes["p1"].hard_rejections == {}
    assert record.state_mutation_blocked is True
    assert state.coverage["B"].mentions == 0
    assert state.coverage["B"].objections == 0


def test_style_warning_does_not_block_valid_vote(state):
    state.phase = Phase.NARROWING
    tracker = StateTracker(state)
    intent = make_intent("p1", ActType.VOTE, ["B"])
    move = TurnMove(act=ActType.VOTE, option="B", stance="vote", present=True)

    record = tracker.apply_participant(
        state,
        intent,
        "Beach Resort gets my vote.",
        move,
        tokens_in=100,
        tokens_out=10,
        validation_issues=["REPETITIVE_OPENER"],
        repaired=False,
    )

    assert state.runtimes["p1"].explicit_vote == "B"
    assert record.state_mutation_blocked is False
    assert state.coverage["B"].mentions == 1
