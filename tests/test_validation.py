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


def test_answer_is_not_semantically_reinterpreted_by_validator():
    state = make_state()
    state.active_thread = DiscussionThread(
        "t1", ThreadKind.QUESTION, "p1", ("B",), ("B", "noise"),
        "Does background noise make Cafe unsuitable?", required_answer_pending=True,
    )
    action = UserAction("p2", True, BidPriority.REQUIRED, ActionType.ANSWER, ("B",), reason="background noise")
    errors = validate_realization(state, state.persona("p2"), action, "Cafe has a relaxed atmosphere.")
    assert errors == []


def test_relevant_answer_is_accepted():
    state = make_state()
    state.active_thread = DiscussionThread(
        "t1", ThreadKind.QUESTION, "p1", ("B",), ("B", "noise"),
        "Does background noise make Cafe unsuitable?", required_answer_pending=True,
    )
    action = UserAction("p2", True, BidPriority.REQUIRED, ActionType.ANSWER, ("B",), reason="background noise")
    assert validate_realization(state, state.persona("p2"), action, "No, Cafe still works because the background noise is manageable.") == []




def test_number_inside_focused_option_name_is_not_treated_as_a_claim():
    state = make_state()
    option = state.scenario.option("A")
    option.name = "Quiet Hours Starting at 11 PM"
    option.short_name = "11 PM Quiet Hours"
    option.aliases = ("11 PM Quiet Hours",)
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.OPENING, ("A",), reason="quieter nights"
    )
    errors = validate_realization(
        state, state.persona("p1"), action, "I prefer 11 PM Quiet Hours because nights stay calmer."
    )
    assert not any("unsupported numeric claim" in error for error in errors)


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


def test_movement_accepts_natural_positive_shift_phrasing():
    state = make_state()
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.ACCEPT, ("B",),
        reason="relaxed atmosphere",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED,
            "B",
            "A",
            "relaxed atmosphere",
        ),
    )
    assert validate_realization(
        state,
        state.persona("p1"),
        action,
        "Cafe works better for me now because the atmosphere is more relaxed.",
    ) == []


def test_relevant_evaluative_answer_does_not_require_yes_or_no():
    state = make_state()
    state.active_thread = DiscussionThread(
        "t1",
        ThreadKind.QUESTION,
        "p1",
        ("B",),
        ("B", "noise"),
        "How does the background noise affect whether Cafe works?",
        required_answer_pending=True,
    )
    action = UserAction(
        "p2",
        True,
        BidPriority.REQUIRED,
        ActionType.ANSWER,
        ("B",),
        reason="background noise",
    )
    assert validate_realization(
        state,
        state.persona("p2"),
        action,
        "The background noise makes focused work harder for me.",
    ) == []


def test_ensures_is_not_a_hard_grounding_failure():
    state = make_state()
    action = UserAction(
        "p1",
        True,
        BidPriority.NORMAL,
        ActionType.SUPPORT,
        ("A",),
        reason="quiet and predictable",
    )
    errors = validate_realization(
        state,
        state.persona("p1"),
        action,
        "Library ensures a calmer session for me.",
    )
    assert not any("strengthened claim" in error for error in errors)


def test_comparison_requires_two_matching_grounded_sources():
    from models import ReasonSource

    state = make_state()
    action = UserAction(
        "p1",
        True,
        BidPriority.NORMAL,
        ActionType.COMPARE,
        ("A", "B"),
        comparison_sources=(ReasonSource("A", "cost", "free"),),
    )
    assert "comparison requires two grounded sources" in validate_action(
        state, state.persona("p1"), action
    )


def test_comparison_missing_exact_alias_is_not_a_hard_failure():
    from models import ReasonSource

    state = make_state()
    action = UserAction(
        "p1",
        True,
        BidPriority.NORMAL,
        ActionType.COMPARE,
        ("A", "B"),
        comparison_sources=(
            ReasonSource("A", "cost", "free"),
            ReasonSource("B", "cost", "8 euros"),
        ),
    )
    errors = validate_realization(
        state,
        state.persona("p1"),
        action,
        "Library is free, which matters more to me.",
    )
    assert "focused option is not visible" not in errors
    assert validate_realization(
        state,
        state.persona("p1"),
        action,
        "Library is free, whereas Cafe costs 8 euros; the difference matters to me.",
    ) == []


def test_guarantees_is_not_a_hard_grounding_failure():
    state = make_state()
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",),
        reason="quiet and predictable",
    )
    errors = validate_realization(
        state, state.persona("p1"), action,
        "That setup guarantees a calmer session for me.",
    )
    assert not any("strengthened claim" in error for error in errors)
