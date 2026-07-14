from __future__ import annotations

import random

from models import (
    ActionType,
    BidPriority,
    Phase,
    ReasonSource,
    StanceUpdateKind,
    TurnRecord,
    UserAction,
)
from simulator import FloorManager, UserSimulator, bid_probability, reason_key, movement_probability
from tests.fixtures import make_persona, make_scenario, make_state


def test_engagement_mapping_is_exact():
    assert [bid_probability(level) for level in range(1, 6)] == [0.20, 0.35, 0.50, 0.70, 0.90]


def test_stubbornness_mapping_is_exact():
    assert [movement_probability(level) for level in range(1, 6)] == [0.80, 0.60, 0.40, 0.20, 0.0]
    assert movement_probability(2, hard_blocker=True) == 0.0


def test_higher_engagement_submits_more_bids_over_repeated_trials():
    scenario = make_scenario()
    low_persona = make_persona("p1", "Low", "A", engagement=1)
    high_persona = make_persona("p1", "High", "A", engagement=5)
    from dialogue import initialise_state
    low_state = initialise_state(scenario, [low_persona])
    high_state = initialise_state(scenario, [high_persona])
    low = UserSimulator(low_persona, random.Random(12))
    high = UserSimulator(high_persona, random.Random(12))
    low_count = sum(low.propose(low_state).wants_to_speak for _ in range(500))
    high_count = sum(high.propose(high_state).wants_to_speak for _ in range(500))
    assert high_count > low_count * 2


def test_floor_uses_highest_category_without_rewriting():
    state = make_state(("A", "B", "C"))
    bids = [
        UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="one"),
        UserAction("p2", True, BidPriority.ISSUE_RESPONSE, ActionType.CONCERN, ("B",), reason="two"),
        UserAction("p3", True, BidPriority.NORMAL, ActionType.ASK, ("C",), reason="three"),
    ]
    original = bids[1].copy()
    selected = FloorManager(random.Random(1)).select(state, bids)
    assert selected is not None
    assert selected.action is bids[1]
    assert selected.action == original


def test_floor_prefers_different_speaker_within_same_category():
    state = make_state(("A", "B", "C"))
    state.turns.append(TurnRecord(0, Phase.DISCUSSION, "p1", "Nora", "previous"))
    bids = [
        UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="one"),
        UserAction("p2", True, BidPriority.NORMAL, ActionType.SUPPORT, ("B",), reason="two"),
    ]
    selected = FloorManager(random.Random(0)).select(state, bids)
    assert selected and selected.action.speaker_id == "p2"


def test_raw_attributes_do_not_become_ordinary_reason_candidates():
    state = make_state(("A", "B", "C"))
    simulator = UserSimulator(state.persona("p1"), random.Random(2))
    actions = simulator._candidate_actions(state, state.runtimes["p1"])
    reasons = " ".join(action.reason for action in actions)
    assert "20:00" not in reasons
    assert "standard desks" not in reasons


def test_question_is_direct_and_not_reopened_after_use():
    state = make_state(("A", "B", "C"))
    for pid, option in zip(("p1", "p2", "p3"), ("A", "B", "C")):
        state.runtimes[pid].public_preference = option
    simulator = UserSimulator(state.persona("p1"), random.Random(1))
    action = simulator._question_action(state, state.runtimes["p1"])
    assert action is not None and action.addressee_id in {"p2", "p3"}
    state.runtimes["p1"].asked_question_keys.add(reason_key(action))
    next_action = simulator._question_action(state, state.runtimes["p1"])
    assert next_action is None or reason_key(next_action) != reason_key(action)


def test_hard_blocker_vote_never_moves():
    scenario = make_scenario()
    persona = make_persona("p1", "Mira", "C", hard_blocker=True)
    from dialogue import initialise_state
    state = initialise_state(scenario, [persona])
    state.narrowing_options = ("A", "C")
    action = UserSimulator(persona, random.Random(1)).decide_vote(state, revote=True)
    assert action.vote_option == "C"
    assert action.stance_update is None


def test_reason_identity_is_semantic_not_act_specific():
    source = ReasonSource("A", "upside", "quiet and predictable")
    support = UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="quiet", reason_source=source)
    comment = UserAction("p1", True, BidPriority.ISSUE_RESPONSE, ActionType.COMMENT, ("A",), reason="quiet", reason_source=source)
    assert reason_key(support) == reason_key(comment)


def test_reason_identity_ignores_addressee():
    source = ReasonSource("B", "concern", "background noise")
    to_ben = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.ASK, ("B",),
        addressee_id="p2", reason="background noise", reason_source=source,
    )
    to_mira = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.ASK, ("B",),
        addressee_id="p3", reason="background noise", reason_source=source,
    )
    assert reason_key(to_ben) == reason_key(to_mira)


def test_question_targets_a_concrete_concern_not_repeated_rationale():
    state = make_state(("A", "B", "C"))
    for pid, option in zip(("p1", "p2", "p3"), ("A", "B", "C")):
        state.runtimes[pid].public_preference = option
    simulator = UserSimulator(state.persona("p1"), random.Random(1))
    action = simulator._question_action(state, state.runtimes["p1"])
    assert action is not None
    assert "ask why" not in action.reason.casefold()
    assert action.reason.strip()


def test_rank_two_concern_can_be_resolved_for_non_blocker(monkeypatch):
    from models import ActiveIssue, IssueKind, IssueStatus, STANCE_DISLIKED

    state = make_state(("A", "B", "C"))
    runtime = state.runtimes["p1"]
    runtime.ranks["B"] = STANCE_DISLIKED
    issue = ActiveIssue(
        id="i1", kind=IssueKind.CONCERN, option_focus=("B",),
        opened_by="p1", addressed_to=None, summary="background noise",
        status=IssueStatus.OPEN, opened_at_turn=0, last_relevant_turn=1,
        response_count=1,
    )
    response = UserAction(
        "p2", True, BidPriority.ISSUE_RESPONSE, ActionType.COMMENT,
        ("B",), reason="the benefit outweighs the drawback", issue_id="i1",
        issue_effect=__import__("models").IssueEffect.RESPOND,
    )
    state.turns.append(TurnRecord(0, Phase.DISCUSSION, "p2", "Ben", "It can still work.", action=response))
    simulator = UserSimulator(state.persona("p1"), random.Random(1))
    actions = simulator._owner_reaction(state, runtime, issue)
    assert actions[0].stance_update is not None
    assert actions[0].stance_update.kind is StanceUpdateKind.MAKE_ACCEPTABLE


def test_stagnation_compromise_is_simulator_owned_and_optional(monkeypatch):
    import simulator as simulator_module

    state = make_state(("A", "B", "C"))
    state.compromise_opportunity = True
    sim = UserSimulator(state.persona("p1"), random.Random(3))

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 1.0)
    action = sim._compromise_action(state, state.runtimes["p1"])
    assert action is not None
    assert action.act is ActionType.COMPROMISE
    assert action.stance_update is not None
    assert action.stance_update.kind in {
        StanceUpdateKind.MAKE_ACCEPTABLE,
        StanceUpdateKind.SWITCH_PREFERRED,
    }

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 0.0)
    assert sim._compromise_action(state, state.runtimes["p1"]) is None


def test_hard_blocker_never_proposes_compromise(monkeypatch):
    import simulator as simulator_module
    from dialogue import initialise_state

    persona = make_persona("p1", "Mira", "C", hard_blocker=True)
    state = initialise_state(make_scenario(), [persona])
    state.compromise_opportunity = True
    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 1.0)
    sim = UserSimulator(persona, random.Random(2))
    assert sim._compromise_action(state, state.runtimes["p1"]) is None


def test_ordinary_discussion_limits_concern_opening_per_participant(monkeypatch):
    state = make_state(("A", "B", "C"))
    runtime = state.runtimes["p1"]
    runtime.opened_issue_keys.add("concern:B:first")
    monkeypatch.setattr(__import__("config_loader").cfg.conversation, "max_concerns_per_participant", 1)
    sim = UserSimulator(state.persona("p1"), random.Random(1))
    assert sim._concern_action(state, runtime) is None


def test_question_uses_configured_semantic_mode():
    from models import QuestionMode
    from tests.fixtures import make_state

    state = make_state(("A", "B", "C"))
    for participant_id, option_id in zip(("p1", "p2", "p3"), ("A", "B", "C")):
        state.runtimes[participant_id].public_preference = option_id
    simulator = UserSimulator(state.persona("p1"), random.Random(9))
    action = simulator._question_action(state, state.runtimes["p1"])
    assert action is not None
    assert action.question_mode in set(QuestionMode)


def test_opening_mode_reflects_visible_group_context():
    from models import OpeningMode

    state = make_state(("A", "B", "C"))
    simulator = UserSimulator(state.persona("p1"), random.Random(1))
    assert simulator.opening_action(state).opening_mode is OpeningMode.INITIAL

    state.runtimes["p2"].public_preference = "A"
    assert simulator.opening_action(state).opening_mode is OpeningMode.ALIGN

    state.runtimes["p2"].public_preference = "B"
    assert simulator.opening_action(state).opening_mode is OpeningMode.CONTRAST
