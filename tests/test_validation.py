from models import ActionType, BidPriority, DiscussionThread, StanceUpdate, StanceUpdateKind, ThreadKind, UserAction
from tests.fixtures import make_persona, make_state
from validation import validate_action, validate_realization


def test_workable_is_visible_acceptance():
    state = make_state()
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.ACCEPT, ("B",),
        reason="relaxed atmosphere",
        stance_update=StanceUpdate(StanceUpdateKind.MAKE_ACCEPTABLE, "B", "A", "relaxed atmosphere"),
    )
    assert validate_realization(state, state.persona("p1"), action, "Cafe is workable for me now.") == []


def test_unrelated_answer_is_rejected():
    state = make_state()
    state.active_thread = DiscussionThread(
        "t1", ThreadKind.QUESTION, "p1", ("B",), ("B", "noise"),
        "Does background noise make Cafe unsuitable?", required_answer_pending=True,
    )
    action = UserAction("p2", True, BidPriority.REQUIRED, ActionType.ANSWER, ("B",), reason="background noise")
    errors = validate_realization(state, state.persona("p2"), action, "Cafe has a relaxed atmosphere.")
    assert "answer does not address the active question" in errors


def test_relevant_answer_is_accepted():
    state = make_state()
    state.active_thread = DiscussionThread(
        "t1", ThreadKind.QUESTION, "p1", ("B",), ("B", "noise"),
        "Does background noise make Cafe unsuitable?", required_answer_pending=True,
    )
    action = UserAction("p2", True, BidPriority.REQUIRED, ActionType.ANSWER, ("B",), reason="background noise")
    assert validate_realization(state, state.persona("p2"), action, "No, Cafe still works because the background noise is manageable.") == []


def test_unsupported_number_is_rejected():
    state = make_state()
    action = UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="quiet")
    errors = validate_realization(state, state.persona("p1"), action, "Library costs 50 euros and is quiet.")
    assert any("unsupported numeric claim" in error for error in errors)


def test_vote_must_match_structured_vote():
    state = make_state()
    action = UserAction("p1", True, BidPriority.REQUIRED, ActionType.VOTE, ("A",), vote_option="A")
    errors = validate_realization(state, state.persona("p1"), action, "My final vote is Cafe.")
    assert "visible vote does not match the structured vote" in errors


def test_hard_blocker_action_validation_rejects_switch():
    state = make_state()
    blocker = make_persona("p1", "Nora", "A", hard_blocker=True)
    state.personas[0] = blocker
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.ACCEPT, ("B",),
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "B", "A"),
    )
    assert "hard blocker cannot change stance" in validate_action(state, blocker, action)


def test_contextual_reaction_may_omit_repeated_option_name():
    state = make_state()
    prior = UserAction("p2", True, BidPriority.NORMAL, ActionType.SUPPORT, ("B",), reason="relaxed atmosphere")
    from models import TurnRecord, Phase
    state.turns.append(TurnRecord(0, Phase.DISCUSSION, "p2", "Ben", "Cafe feels relaxed.", action=prior))
    action = UserAction("p1", True, BidPriority.NORMAL, ActionType.REACT, ("B",), reason="background noise")
    assert validate_realization(state, state.persona("p1"), action, "That noise would still bother me.") == []


def test_movement_still_requires_explicit_option_reference():
    state = make_state()
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.ACCEPT, ("B",),
        reason="relaxed atmosphere",
        stance_update=StanceUpdate(StanceUpdateKind.MAKE_ACCEPTABLE, "B", "A", "relaxed atmosphere"),
    )
    errors = validate_realization(state, state.persona("p1"), action, "That is workable for me now.")
    assert "focused option is not visible" in errors
