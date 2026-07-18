import random

from models import ActionType, BidPriority, DiscussionThread, ThreadKind, UserAction
from simulator import FloorManager, UserSimulator, bid_probability, movement_probability
from tests.fixtures import make_persona, make_state


def test_engagement_increases_bid_probability():
    values = [bid_probability(level) for level in range(1, 6)]
    assert values == sorted(values)


def test_stubbornness_reduces_movement_probability():
    values = [movement_probability(level) for level in range(1, 6)]
    assert values == sorted(values, reverse=True)
    assert values[-1] == 0


def test_floor_returns_one_intact_bid():
    state = make_state()
    bids = [
        UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="one"),
        UserAction("p2", True, BidPriority.NORMAL, ActionType.OBJECT, ("A",), reason="two"),
    ]
    selected = FloorManager(random.Random(1)).select(state, bids).action
    assert selected in bids
    assert selected.reason in {"one", "two"}


def test_required_bid_has_priority():
    state = make_state()
    bids = [
        UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",)),
        UserAction("p2", True, BidPriority.REQUIRED, ActionType.ANSWER, ("B",)),
    ]
    assert FloorManager(random.Random(2)).select(state, bids).action is bids[1]


def test_question_requires_another_visible_preference():
    state = make_state(("A", "B", "C"))
    simulator = UserSimulator(state.persona("p1"), random.Random(3))
    assert simulator._question_action(state) is None
    for runtime in state.runtimes.values():
        runtime.public_preference = runtime.preferred_option
    assert simulator._question_action(state) is not None


def test_active_thread_candidates_do_not_open_another_question():
    state = make_state(("A", "B", "C"))
    for runtime in state.runtimes.values():
        runtime.public_preference = runtime.preferred_option
    state.active_thread = DiscussionThread(
        "t1", ThreadKind.QUESTION, "p1", ("B",), ("B", "concern"),
        "Does background noise change whether Cafe works?", required_answer_pending=False,
        participants={"p1"},
    )
    simulator = UserSimulator(state.persona("p3"), random.Random(4))
    candidates = simulator._thread_candidates(state)
    assert candidates
    assert all(action.act is not ActionType.ASK for action, _ in candidates)


def test_hard_blocker_never_proposes_movement():
    state = make_state(("A", "B", "C"))
    blocker = make_persona("p1", "Nora", "A", hard_blocker=True)
    state.personas[0] = blocker
    state.runtimes["p1"].hard_rejected_options = {"B", "C", "D"}
    simulator = UserSimulator(blocker, random.Random(5))
    action = simulator.compromise_action(state, ("B",))
    assert action.stance_update is None


def test_recent_public_point_is_not_selected_again_for_ordinary_turn():
    state = make_state()
    simulator = UserSimulator(state.persona("p1"), random.Random(3))
    state.recent_point_keys = [("A", "upside")]
    state.public_point_counts[("A", "upside")] = 1
    state.runtimes["p1"].used_point_keys.add(("A", "cost"))
    source, _ = simulator._positive_point(state, "A")
    assert source is None
    neutral_source, _ = simulator._neutral_point(state, "A")
    assert neutral_source is not None
    assert neutral_source.point_key not in {("A", "upside"), ("A", "cost")}


def test_question_uses_only_a_publicly_unseen_point():
    state = make_state(("A", "B", "C"))
    for runtime in state.runtimes.values():
        runtime.public_preference = runtime.preferred_option
    state.public_point_counts[("B", "concern")] = 1
    simulator = UserSimulator(state.persona("p1"), random.Random(9))
    action = simulator._question_action(state)
    assert action is not None
    assert action.point_key is not None
    assert state.public_point_counts.get(action.point_key, 0) == 0


def test_question_is_not_reopened_when_all_target_points_are_public():
    state = make_state(("A", "B", "C"))
    for runtime in state.runtimes.values():
        runtime.public_preference = runtime.preferred_option
    option = state.scenario.option("B")
    state.public_point_counts[("B", "concern")] = 1
    for key in option.attrs:
        state.public_point_counts[("B", key.strip().lower())] = 1
    simulator = UserSimulator(state.persona("p1"), random.Random(10))
    # The other visible option C still has unseen points, so constrain the
    # floor to B by making p3 share p1's preference.
    state.runtimes["p3"].public_preference = "A"
    assert simulator._question_action(state) is None


def test_support_and_objection_use_polarity_specific_sources():
    state = make_state(("A", "B", "C"))
    simulator = UserSimulator(state.persona("p1"), random.Random(12))
    positive, _ = simulator._positive_point(state, "A", allow_used=True)
    negative, _ = simulator._negative_point(state, "B", allow_used=True)
    neutral, _ = simulator._neutral_point(state, "A", allow_used=True)
    assert positive is not None and positive.attribute_name == "upside"
    assert negative is not None and negative.attribute_name == "concern"
    assert neutral is not None and neutral.attribute_name in state.scenario.option("A").attrs


def test_unwilling_compromise_returns_silence(monkeypatch):
    state = make_state(("A", "B", "C"))
    simulator = UserSimulator(state.persona("p1"), random.Random(13))
    monkeypatch.setattr("simulator.movement_probability", lambda _level: 0.0)
    action = simulator.compromise_action(state, ("B",))
    assert action.wants_to_speak is False
    assert action.stance_update is None


def test_compare_action_carries_one_grounded_source_per_option():
    state = make_state(("A", "B", "C"))
    for runtime in state.runtimes.values():
        runtime.public_preference = runtime.preferred_option
        runtime.voluntary_turns = 1
    simulator = UserSimulator(state.persona("p1"), random.Random(21))
    action = simulator._compare_action(state, forced_target="B")
    assert action is not None
    assert action.act is ActionType.COMPARE
    assert len(action.comparison_sources) == 2
    assert {source.option_id for source in action.comparison_sources} == set(action.option_focus)
    assert len({source.attribute_name for source in action.comparison_sources}) == 1


def test_compare_is_suppressed_after_recent_comparison():
    from models import Phase, ReasonSource, TurnRecord

    state = make_state(("A", "B", "C"))
    state.runtimes["p1"].voluntary_turns = 1
    prior = UserAction(
        "p2",
        True,
        BidPriority.NORMAL,
        ActionType.COMPARE,
        ("B", "A"),
        comparison_sources=(
            ReasonSource("B", "upside", "relaxed atmosphere"),
            ReasonSource("A", "upside", "quiet and predictable"),
        ),
    )
    state.turns.append(
        TurnRecord(0, Phase.DISCUSSION, "p2", "Ben", "Cafe is relaxed, but Library is quieter.", action=prior)
    )
    simulator = UserSimulator(state.persona("p1"), random.Random(22))
    assert simulator._compare_action(state, forced_target="B") is None


def test_mid_discussion_acceptance_can_switch_when_it_builds_common_ground(monkeypatch):
    from models import Phase, StanceUpdateKind, TurnRecord

    state = make_state(("A", "B", "C"))
    for runtime in state.runtimes.values():
        runtime.public_preference = runtime.preferred_option
    state.runtimes["p3"].public_preference = "B"
    prior = UserAction(
        "p2", True, BidPriority.NORMAL, ActionType.SUPPORT, ("B",),
        reason="relaxed atmosphere",
    )
    state.turns.append(
        TurnRecord(0, Phase.DISCUSSION, "p2", "Ben", "That relaxed atmosphere sounds useful.", action=prior)
    )
    simulator = UserSimulator(state.persona("p1"), random.Random(1))
    monkeypatch.setattr("simulator.movement_probability", lambda _level: 1.0)
    action = simulator._accept_action(state, forced_target="B")
    assert action is not None
    assert action.stance_update is not None
    assert action.stance_update.kind is StanceUpdateKind.SWITCH_PREFERRED


def test_mid_discussion_acceptance_stays_noncommittal_without_recent_support(monkeypatch):
    from models import StanceUpdateKind

    state = make_state(("A", "B", "C"))
    for runtime in state.runtimes.values():
        runtime.public_preference = runtime.preferred_option
    simulator = UserSimulator(state.persona("p1"), random.Random(1))
    monkeypatch.setattr("simulator.movement_probability", lambda _level: 1.0)
    action = simulator._accept_action(state, forced_target="B")
    assert action is not None
    assert action.stance_update is not None
    assert action.stance_update.kind is StanceUpdateKind.MAKE_ACCEPTABLE
