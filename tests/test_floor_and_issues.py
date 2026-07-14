from __future__ import annotations

import inspect
import random

from config_loader import cfg
from dialogue import DialogueRunner
from models import (
    ActionType,
    IssueEffect,
    IssueKind,
    IssueStatus,
    Phase,
    TurnRecord,
    UserAction,
)
from simulator import FloorManager
from tests.fixtures import make_runner, make_state


def test_urgency_influences_open_floor_selection():
    state = make_state()
    floor = FloorManager(random.Random(42))
    low = UserAction("p1", True, 0.1, ActionType.SUPPORT, ("A",), reason="low")
    high = UserAction("p2", True, 0.9, ActionType.SUPPORT, ("B",), reason="high")
    counts = {"p1": 0, "p2": 0}
    for _ in range(2000):
        counts[floor.select(state, [low, high]).action.speaker_id] += 1
    assert counts["p2"] > counts["p1"] * 5


def test_maximum_consecutive_turn_protection_works():
    state = make_state()
    state.turns.extend([
        TurnRecord(1, Phase.DISCUSSION, "p1", "Nora", "First."),
        TurnRecord(2, Phase.DISCUSSION, "p1", "Nora", "Second."),
    ])
    floor = FloorManager(random.Random(3))
    dominant = UserAction("p1", True, 1.0, ActionType.SUPPORT, ("A",), reason="more")
    other = UserAction("p2", True, 0.01, ActionType.SUPPORT, ("B",), reason="other")
    assert floor.select(state, [dominant, other]).action is other


def test_two_consecutive_turns_are_allowed_before_cap():
    state = make_state()
    state.turns.append(TurnRecord(1, Phase.DISCUSSION, "p1", "Nora", "First."))
    floor = FloorManager(random.Random(0))
    dominant = UserAction("p1", True, 10.0, ActionType.SUPPORT, ("A",), reason="more")
    other = UserAction("p2", True, 0.001, ActionType.SUPPORT, ("B",), reason="other")
    assert floor.select(state, [dominant, other]).action is dominant


def test_expected_share_correction_does_not_exist():
    source = inspect.getsource(DialogueRunner) + inspect.getsource(FloorManager)
    lowered = source.casefold()
    assert "expected_turn_share" not in lowered
    assert "participation deficit" not in lowered
    assert "quota" not in lowered


def test_direct_question_opens_issue_and_response_obligation():
    runner = make_runner()
    action = UserAction(
        "p1", True, 0.8, ActionType.ASK, ("B",),
        addressee_id="p2", reason="ask about the noise", issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(action, "Ben, what makes Option B workable?", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is not None
    assert runner.state.active_issue.kind is IssueKind.QUESTION
    assert runner.state.response_obligation == "p2"


def test_group_question_opens_issue_without_obligation():
    runner = make_runner()
    action = UserAction(
        "p1", True, 0.8, ActionType.ASK, ("B",),
        reason="ask the group", issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(action, "What makes Option B workable?", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is not None
    assert runner.state.response_obligation is None


def _open_concern(runner):
    action = UserAction(
        "p1", True, 0.8, ActionType.CONCERN, ("B",),
        reason="background noise", issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(action, "I have a concern about Option B because of the background noise.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    return runner.state.active_issue


def test_concern_can_remain_open_after_response():
    runner = make_runner()
    issue = _open_concern(runner)
    response = UserAction(
        "p2", True, 0.7, ActionType.SUPPORT, ("B",),
        addressee_id="p1", reason="relaxed atmosphere", issue_id=issue.id, issue_effect=IssueEffect.CONTINUE,
    )
    runner._commit_action(response, "Option B still has a relaxed atmosphere.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is issue
    assert issue.status is IssueStatus.OPEN


def test_concern_owner_can_resolve_it_explicitly():
    runner = make_runner()
    issue = _open_concern(runner)
    resolution = UserAction(
        "p1", True, 0.9, ActionType.ACKNOWLEDGE, ("B",),
        reason="response addressed it", issue_id=issue.id, issue_effect=IssueEffect.RESOLVE,
    )
    runner._commit_action(resolution, "That addresses my concern about Option B.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is None
    assert runner.state.issue_history[-1].status is IssueStatus.RESOLVED


def test_concern_owner_can_maintain_it():
    runner = make_runner()
    issue = _open_concern(runner)
    maintain = UserAction(
        "p1", True, 0.9, ActionType.CONCERN, ("B",),
        reason="still matters", issue_id=issue.id, issue_effect=IssueEffect.MAINTAIN,
    )
    runner._commit_action(maintain, "My concern about Option B still matters.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is issue
    assert issue.status is IssueStatus.OPEN


def test_issue_becomes_stale_when_nobody_continues():
    runner = make_runner()
    issue = _open_concern(runner)
    runner._stale_active_issue("nobody continued")
    assert runner.state.active_issue is None
    assert runner.state.issue_history[-1] is issue
    assert issue.status is IssueStatus.STALE


def test_hard_follow_up_cap_stales_issue(monkeypatch):
    runner = make_runner()
    issue = _open_concern(runner)
    monkeypatch.setattr(cfg.conversation, "issue_follow_up_cap", 2)
    for participant_id in ("p2", "p3"):
        action = UserAction(
            participant_id, True, 0.7, ActionType.SUPPORT, ("B",),
            reason="response", issue_id=issue.id, issue_effect=IssueEffect.CONTINUE,
        )
        runner._commit_action(action, "Option B remains workable for me.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is None
    assert runner.state.issue_history[-1].status is IssueStatus.STALE


def test_only_one_issue_is_active():
    runner = make_runner()
    first = _open_concern(runner)
    second_action = UserAction(
        "p2", True, 0.9, ActionType.ASK, ("C",),
        reason="new urgent question", issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(second_action, "What makes Option C workable?", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is not None
    assert runner.state.active_issue.id != first.id
    assert first.status is IssueStatus.STALE
    assert first in runner.state.issue_history


def test_first_comparison_is_evidence_not_automatically_an_issue():
    runner = make_runner()
    action = UserAction("p1", True, 0.5, ActionType.COMPARE, ("A", "B"), reason="trade-off")
    runner._commit_action(action, "Option A fits better than Option B.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is None


def test_second_developed_comparison_can_open_issue():
    runner = make_runner()
    first = UserAction("p1", True, 0.5, ActionType.COMPARE, ("A", "B"), reason="trade-off")
    second = UserAction("p2", True, 0.6, ActionType.COMPARE, ("B", "A"), reason="develop trade-off")
    runner._commit_action(first, "Option A fits better than Option B.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    runner._commit_action(second, "Option B fits better than Option A.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    assert runner.state.active_issue is not None
    assert runner.state.active_issue.kind is IssueKind.COMPARISON


def test_comparison_issue_action_places_speakers_preferred_option_first():
    runner = make_runner()
    first = UserAction("p1", True, 0.5, ActionType.COMPARE, ("A", "B"), reason="trade-off")
    second = UserAction("p2", True, 0.6, ActionType.COMPARE, ("B", "A"), reason="develop trade-off")
    runner._commit_action(first, "Option A fits better than Option B.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    runner._commit_action(second, "Option B fits better than Option A.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0)
    issue = runner.state.active_issue
    assert issue is not None
    simulator = runner._simulators["p1"]
    actions = simulator._issue_actions(runner.state, runner.state.runtimes["p1"], issue)
    comparison = next(action for action in actions if action.act is ActionType.COMPARE)
    assert comparison.option_focus[0] == runner.state.runtimes["p1"].preferred_option


def test_concern_owner_follow_ups_count_toward_hard_cap(monkeypatch):
    runner = make_runner()
    issue = _open_concern(runner)
    monkeypatch.setattr(cfg.conversation, "issue_follow_up_cap", 2)
    for _ in range(2):
        action = UserAction(
            "p1", True, 0.8, ActionType.CONCERN, ("B",),
            reason="the concern remains", issue_id=issue.id, issue_effect=IssueEffect.MAINTAIN,
        )
        runner._commit_action(
            action,
            "My concern about Option B remains.",
            mandatory=False,
            voluntary=True,
            liveness_forced=False,
            repair_count=0,
        )
    assert runner.state.active_issue is None
    assert runner.state.issue_history[-1].status is IssueStatus.STALE


def test_group_question_author_does_not_answer_itself():
    runner = make_runner()
    question = UserAction(
        "p1", True, 0.8, ActionType.ASK, ("B",),
        reason="ask the group", issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(
        question,
        "What makes Option B workable?",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    issue = runner.state.active_issue
    assert issue is not None
    actions = runner._simulators["p1"]._issue_actions(
        runner.state, runner.state.runtimes["p1"], issue
    )
    assert all(action.act is not ActionType.ANSWER for action in actions)
