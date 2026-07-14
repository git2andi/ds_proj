from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config_loader import cfg
from dialogue import DialogueRunner
from eval.run_eval_suite import scenario_for
from models import (
    ActionType,
    IssueEffect,
    IssueResponseKind,
    IssueStatus,
    Phase,
    ReasonSource,
    StanceUpdateKind,
    StimulusKind,
    UserAction,
    VoteStatus,
)
from simulator import UserSimulator, action_cooldown_context, action_signature
from tests.fixtures import ActionRendererLLM, NullLogger, make_persona, make_runner
from validation import validate_realization


def _open_concern_with_response(runner: DialogueRunner) -> object:
    concern = UserAction(
        "p1", True, 0.8, ActionType.CONCERN, ("B",),
        reason="background noise",
        reason_source=ReasonSource("B", "noise", "moderate"),
        issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(
        concern,
        "I have a concern about Option B because of the background noise.",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    issue = runner.state.active_issue
    assert issue is not None
    response = UserAction(
        "p2", True, 0.8, ActionType.SUPPORT, ("B",),
        reason="relaxed atmosphere", issue_id=issue.id,
        issue_effect=IssueEffect.CONTINUE,
    )
    runner._commit_action(
        response,
        "Option B remains workable because of its relaxed atmosphere.",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    return issue


def test_low_rank_concern_can_be_resolved_with_atomic_acceptance():
    runner = make_runner()
    runtime = runner.state.runtimes["p1"]
    runtime.ranks["B"] = 2
    runtime.disliked_options.add("B")
    issue = _open_concern_with_response(runner)
    relevant = UserAction(
        "p3", True, 0.8, ActionType.SUPPORT, ("B",),
        reason="The public noise level is moderate",
        reason_source=ReasonSource("B", "noise", "moderate"),
        issue_id=issue.id,
        issue_effect=IssueEffect.CONTINUE,
        issue_response_kind=IssueResponseKind.MITIGATION,
    )
    runner._commit_action(
        relevant,
        "The listed noise level is moderate, which directly limits that concern.",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    actions = runner._simulators["p1"]._issue_actions(runner.state, runtime, issue)
    resolution = next(action for action in actions if action.issue_effect is IssueEffect.RESOLVE)
    assert resolution.stance_update is not None
    assert resolution.stance_update.kind is StanceUpdateKind.MAKE_ACCEPTABLE
    assert resolution.stance_update.option_id == "B"


def test_unrelated_upside_does_not_count_as_concern_resolution_evidence():
    runner = make_runner()
    issue = _open_concern_with_response(runner)
    assert issue.relevant_responder_ids == set()
    actions = runner._simulators["p1"]._concern_owner_reactions(
        runner.state, runner.state.runtimes["p1"], issue
    )
    assert all(action.issue_effect is not IssueEffect.RESOLVE for action in actions)


def test_matching_issue_attribute_creates_relevant_mitigation_action():
    runner = make_runner()
    issue = _open_concern_with_response(runner)
    actions = runner._simulators["p2"]._issue_actions(
        runner.state, runner.state.runtimes["p2"], issue
    )
    mitigation = next(
        action for action in actions
        if action.issue_response_kind is IssueResponseKind.MITIGATION
    )
    assert mitigation.reason_source == ReasonSource("B", "noise", "moderate")


def test_relevant_tradeoff_acknowledges_concern_without_claiming_it_is_solved():
    runner = make_runner()
    issue = _open_concern_with_response(runner)
    actions = runner._simulators["p2"]._issue_actions(
        runner.state, runner.state.runtimes["p2"], issue
    )
    tradeoff = next(
        action for action in actions
        if action.issue_response_kind is IssueResponseKind.TRADE_OFF
    )
    assert tradeoff.issue_effect is IssueEffect.CONTINUE
    assert "concern is real" in tradeoff.reason.casefold()


def test_high_stubbornness_concern_owner_only_maintains():
    runner = make_runner()
    persona = make_persona("p1", "Nora", "A", stubbornness=4)
    runner.state.personas[0] = persona
    runner._simulators["p1"] = UserSimulator(persona, random.Random(2))
    runtime = runner.state.runtimes["p1"]
    runtime.ranks["B"] = 2
    runtime.disliked_options.add("B")
    issue = _open_concern_with_response(runner)
    actions = runner._simulators["p1"]._issue_actions(runner.state, runtime, issue)
    assert actions
    assert {action.issue_effect for action in actions} == {IssueEffect.MAINTAIN}


def test_answered_question_finishes_resolved_not_stale():
    runner = make_runner()
    question = UserAction(
        "p1", True, 0.8, ActionType.ASK, ("B",), addressee_id="p2",
        reason="ask about B", issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(
        question,
        "Ben, what makes Option B workable?",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    issue = runner.state.active_issue
    answer = UserAction(
        "p2", True, 1.0, ActionType.ANSWER, ("B",), addressee_id="p1",
        reason="relaxed atmosphere", issue_id=issue.id,
        issue_effect=IssueEffect.ANSWERED,
    )
    runner._commit_action(
        answer,
        "Option B can work because its atmosphere is relaxed.",
        mandatory=True,
        voluntary=False,
        liveness_forced=False,
        repair_count=0,
    )
    runner._finish_exhausted_issue("nobody continued")
    assert runner.state.active_issue is None
    assert runner.state.issue_history[-1].status is IssueStatus.RESOLVED
    assert runner.state.issue_history[-1].outcome == "answered"


def test_coverage_prompt_creates_structured_simulator_stimulus():
    runner = make_runner()
    runner._set_group_stimulus(StimulusKind.COVERAGE, ("D",), "Discuss Online.")
    stimulus = runner.state.group_stimulus
    action = runner._simulators["p1"]._stimulus_action(
        runner.state, runner.state.runtimes["p1"], stimulus
    )
    assert action is not None
    assert action.stimulus_id == stimulus.id
    assert "D" in action.option_focus


def test_narrowing_considers_compromise_even_when_own_option_is_finalist():
    runner = make_runner()
    runner.state.phase = Phase.NARROWING
    runner.state.narrowing_options = ("A", "B")
    runner.state.public_supports["B"] = 6
    runner.state.runtimes["p1"].ranks["B"] = 3
    actions = runner._simulators["p1"]._narrowing_actions(
        runner.state, runner.state.runtimes["p1"]
    )
    assert any(action.stance_update is not None for action in actions)
    assert any(action.act is ActionType.COMPROMISE for action in actions)


def test_narrowing_does_not_generate_unrelated_ordinary_actions():
    runner = make_runner()
    runner.state.phase = Phase.NARROWING
    runner.state.narrowing_options = ("A", "B")
    actions = runner._simulators["p1"]._candidate_actions(
        runner.state, runner.state.runtimes["p1"]
    )
    assert actions
    assert all(
        set(action.option_focus).issubset({"A", "B"})
        for action in actions
        if action.option_focus
    )
    assert all(action.act is not ActionType.ASK for action in actions)


def test_structured_repetition_filter_suppresses_same_comparison():
    runner = make_runner()
    simulator = runner._simulators["p1"]
    runtime = runner.state.runtimes["p1"]
    action = simulator._compare_action(runner.state, runtime)
    assert action is not None
    runtime.action_signature_counts[action_signature(action)] = 1
    filtered = simulator._filter_repeated_candidates(runner.state, runtime, [action])
    assert filtered == []
    assert runner.state.stats.suppressed_repetitions == 1


def test_empty_floor_reaches_liveness_only_after_three_rounds(monkeypatch):
    runner = make_runner()
    runner._moderator_enabled = False
    rounds = {"count": 0}

    class Silent:
        def propose(self, _state, **_kwargs):
            rounds["count"] += 1
            return UserAction("p1", False, 0.0, ActionType.COMMENT)

    runner._simulators = {"p1": Silent()}
    liveness = []
    monkeypatch.setattr(runner, "_force_liveness", lambda _phase: liveness.append(True) or False)
    monkeypatch.setattr(cfg.conversation, "min_voluntary_turns", 8)
    runner._run_discussion()
    assert rounds["count"] == 3
    assert len(liveness) == 1


class _VoteFailureLLM(ActionRendererLLM):
    def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
        if "PRIVATE SPEAKER CARD — only for Ben:" in prompt or "Speaker: Ben" in prompt:
            self.prompts.append(prompt)
            self.profiles.append(profile)
            self.calls += 1
            self.last_tokens_in = max(1, len(prompt.split()))
            self.last_tokens_out = 8
            return "I vote for Option A or Option B."
        return super().generate(prompt, profile=profile)


def test_vote_round_records_every_participant_and_failure_status():
    runner = make_runner(llm=_VoteFailureLLM())
    runner._run_voting(revote=False)
    records = runner.state.vote_records[1]
    assert set(records) == {"p1", "p2", "p3"}
    assert records["p2"].status is VoteStatus.UNCLEAR
    assert records["p2"].option_id is None
    assert records["p2"].attempts == 2
    assert runner.state.vote_protocol_degraded
    assert any("p2" in error for error in runner.state.vote_protocol_errors)


def test_generation_attempt_log_keeps_raw_errors_and_repair():
    runner = make_runner(llm=ActionRendererLLM(scripted=["", ""]))
    action = runner._simulators["p1"].opening_action(runner.state)
    assert runner._realize_and_commit(action, mandatory=True, voluntary=False) is None
    attempt = runner.state.generation_attempts[-1]
    assert attempt.raw_text == ""
    assert attempt.validation_errors == ["empty output"]
    assert attempt.repair_text == ""
    assert attempt.repair_errors == ["empty output"]
    assert attempt.final_status == "dropped"


def test_unsupported_single_option_concrete_comparison_is_blocked():
    sc = scenario_for("flight")
    persona = make_persona("p1", "Nora", "A")
    runner = DialogueRunner(
        "", scenario=sc, personas=[persona], llm=ActionRendererLLM(),
        logger=NullLogger(), rng=random.Random(1), seed=1,
    )
    action = UserAction("p1", True, 0.7, ActionType.SUPPORT, ("A",), reason="direct flight")
    result = validate_realization(
        "Option A has a longer flight.", runner.state, persona, action
    )
    assert not result.ok
    assert "concrete comparison contradicts public values" in result.errors


def test_common_explicit_vote_wording_is_accepted():
    runner = make_runner()
    persona = runner.state.personas[0]
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("A",),
        reason="best fit", vote_option="A",
    )
    result = validate_realization(
        "Option A gets my vote.", runner.state, persona, action
    )
    assert result.ok, result.errors


def test_false_directional_comparison_is_blocked_even_with_public_shortest_reason():
    sc = scenario_for("flight")
    persona = make_persona("p1", "Nora", "A")
    runner = DialogueRunner(
        "", scenario=sc, personas=[persona], llm=ActionRendererLLM(),
        logger=NullLogger(), rng=random.Random(1), seed=1,
    )
    source = __import__("models").ReasonSource("A", "upside", "shortest travel time")
    action = UserAction(
        "p1", True, 0.7, ActionType.SUPPORT, ("A",),
        reason="shortest travel time", reason_source=source,
    )
    result = validate_realization(
        "Option A has a longer flight.", runner.state, persona, action
    )
    assert not result.ok
    assert "concrete comparison contradicts public values" in result.errors


def test_true_two_option_directional_comparison_is_allowed():
    sc = scenario_for("flight")
    persona = make_persona("p1", "Nora", "A")
    runner = DialogueRunner(
        "", scenario=sc, personas=[persona], llm=ActionRendererLLM(),
        logger=NullLogger(), rng=random.Random(1), seed=1,
    )
    action = UserAction(
        "p1", True, 0.7, ActionType.COMPARE, ("A", "B"),
        reason="compare public durations",
    )
    result = validate_realization(
        "Option A has a shorter flight than Option B.", runner.state, persona, action
    )
    assert result.ok, result.errors


def test_unanimous_public_preference_skips_soft_coverage_prompt(monkeypatch):
    runner = make_runner(("A", "A", "A"))
    for runtime in runner.state.runtimes.values():
        runtime.public_preference = "A"
    runner.state.phase = Phase.DISCUSSION
    monkeypatch.setattr(cfg.conversation, "min_voluntary_turns", 0)
    assert runner._publicly_converged()
    assert not runner._coverage_prompt_needed()


def test_concern_owner_followup_has_high_bid_probability():
    runner = make_runner()
    runtime = runner.state.runtimes["p1"]
    runtime.ranks["B"] = 2
    runtime.disliked_options.add("B")
    issue = _open_concern_with_response(runner)
    simulator = runner._simulators["p1"]
    bids = [simulator.propose(runner.state) for _ in range(40)]
    relevant = [bid for bid in bids if bid.wants_to_speak and bid.issue_id == issue.id]
    assert len(relevant) >= 30
    assert all(bid.issue_effect in {IssueEffect.MAINTAIN, IssueEffect.PARTIAL, IssueEffect.RESOLVE} for bid in relevant)


def test_action_cooldown_resets_after_phase_change():
    runner = make_runner()
    simulator = runner._simulators["p1"]
    runtime = runner.state.runtimes["p1"]
    action = simulator._compare_action(runner.state, runtime)
    assert action is not None
    signature = action_signature(action)
    runtime.action_signature_counts[signature] = 1
    runtime.action_signature_contexts[signature] = action_cooldown_context(
        runner.state, runtime, action
    )
    assert simulator._filter_repeated_candidates(runner.state, runtime, [action]) == []
    runner.state.phase = Phase.NARROWING
    assert simulator._filter_repeated_candidates(runner.state, runtime, [action]) == [action]


def test_action_cooldown_resets_when_another_participant_challenges_focus():
    runner = make_runner()
    simulator = runner._simulators["p1"]
    runtime = runner.state.runtimes["p1"]
    action = simulator._support_action(runner.state, runtime)
    assert action is not None
    signature = action_signature(action)
    runtime.action_signature_counts[signature] = 1
    runtime.action_signature_contexts[signature] = action_cooldown_context(
        runner.state, runtime, action
    )
    assert simulator._filter_repeated_candidates(runner.state, runtime, [action]) == []
    challenge = UserAction(
        "p2", True, 0.8, ActionType.CONCERN, action.option_focus,
        reason="a new public challenge",
    )
    runner._commit_action(
        challenge,
        f"I have a concern about Option {action.option_focus[0]}.",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    assert simulator._filter_repeated_candidates(runner.state, runtime, [action]) == [action]


def test_action_cooldown_does_not_reset_from_own_aggregate_counter_change():
    runner = make_runner()
    simulator = runner._simulators["p1"]
    runtime = runner.state.runtimes["p1"]
    action = simulator._support_action(runner.state, runtime)
    assert action is not None
    signature = action_signature(action)
    runtime.action_signature_counts[signature] = 1
    runtime.action_signature_contexts[signature] = action_cooldown_context(
        runner.state, runtime, action
    )
    runner.state.public_concerns[action.option_focus[0]] += 1
    assert simulator._filter_repeated_candidates(runner.state, runtime, [action]) == []


def test_repeated_support_from_one_speaker_counts_as_one_distinct_supporter():
    from consensus import candidate_standings
    from models import ActionType, UserAction
    from tests.fixtures import make_runner

    runner = make_runner(("A", "B", "C"))
    for _ in range(5):
        runner._commit_action(
            UserAction("p1", True, 0.6, ActionType.SUPPORT, ("A",), reason="quiet"),
            "Option A remains useful.",
            mandatory=False,
            voluntary=True,
            liveness_forced=False,
            repair_count=0,
        )
    standing = next(row for row in candidate_standings(runner.state) if row.option_id == "A")
    assert standing.supports == 1
    assert runner.state.public_supports["A"] == 5


def test_support_from_three_speakers_counts_as_three_distinct_supporters():
    from consensus import candidate_standings
    from models import ActionType, UserAction
    from tests.fixtures import make_runner

    runner = make_runner(("A", "B", "C"))
    for participant_id in ("p1", "p2", "p3"):
        runner._commit_action(
            UserAction(participant_id, True, 0.6, ActionType.SUPPORT, ("A",), reason="public reason"),
            "Option A remains useful.",
            mandatory=False,
            voluntary=True,
            liveness_forced=False,
            repair_count=0,
        )
    standing = next(row for row in candidate_standings(runner.state) if row.option_id == "A")
    assert standing.supports == 3


def test_duplicate_support_does_not_change_candidate_standing_score():
    from consensus import candidate_standings
    from models import ActionType, UserAction
    from tests.fixtures import make_runner

    runner = make_runner(("A", "B", "C"))
    action = UserAction("p1", True, 0.6, ActionType.SUPPORT, ("A",), reason="quiet")
    runner._commit_action(
        action,
        "Option A remains useful.",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    first = next(row for row in candidate_standings(runner.state) if row.option_id == "A")
    runner._commit_action(
        action.copy(),
        "Option A remains useful for the same reason.",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    second = next(row for row in candidate_standings(runner.state) if row.option_id == "A")
    assert second.score == first.score


def test_switch_hysteresis_blocks_immediate_return_without_new_external_evidence():
    from models import Phase, TurnRecord
    from tests.fixtures import make_runner

    runner = make_runner(("A", "B", "C"))
    simulator = runner._simulators["p1"]
    runtime = runner.state.runtimes["p1"]
    runtime.preferred_option = "B"
    runtime.last_switch_turn = 5
    runtime.last_switch_target = "B"
    runtime.last_switch_external_evidence_turn = 4
    runner.state.turns.extend([
        TurnRecord(index=5, phase=Phase.DISCUSSION, speaker_id="p1", speaker_name="Nora", text="I now prefer Option B."),
        TurnRecord(index=6, phase=Phase.DISCUSSION, speaker_id="p1", speaker_name="Nora", text="Another point."),
    ])
    allowed, reason = simulator._switch_gate(
        runner.state,
        runtime,
        "A",
        target_evidence=0.75,
        current_evidence=0.40,
    )
    assert not allowed
    assert reason in {"cooldown", "no_new_external_evidence"}


def test_switch_hysteresis_allows_later_switch_after_new_external_evidence():
    from models import ActionType, Phase, TurnRecord, UserAction
    from tests.fixtures import make_runner

    runner = make_runner(("A", "B", "C"))
    simulator = runner._simulators["p1"]
    runtime = runner.state.runtimes["p1"]
    runtime.preferred_option = "B"
    runtime.last_switch_turn = 2
    runtime.last_switch_target = "B"
    runtime.last_switch_external_evidence_turn = 1
    runner.state.turns.extend([
        TurnRecord(index=1, phase=Phase.DISCUSSION, speaker_id="p2", speaker_name="Ben", text="I support Option B."),
        TurnRecord(index=2, phase=Phase.DISCUSSION, speaker_id="p1", speaker_name="Nora", text="I now prefer Option B."),
        TurnRecord(index=3, phase=Phase.DISCUSSION, speaker_id="p3", speaker_name="Mira", text="I support Option A.", action=UserAction("p3", True, 0.6, ActionType.SUPPORT, ("A",), reason="quiet")),
        TurnRecord(index=4, phase=Phase.DISCUSSION, speaker_id="p2", speaker_name="Ben", text="Option A also works.", action=UserAction("p2", True, 0.6, ActionType.SUPPORT, ("A",), reason="free")),
        TurnRecord(index=5, phase=Phase.DISCUSSION, speaker_id="p3", speaker_name="Mira", text="The case for A is stronger.", action=UserAction("p3", True, 0.6, ActionType.SUPPORT, ("A",), reason="closing")),
    ])
    allowed, reason = simulator._switch_gate(
        runner.state,
        runtime,
        "A",
        target_evidence=0.75,
        current_evidence=0.40,
    )
    assert allowed, reason


def test_persona_distinctness_fixture_has_independent_private_priorities():
    from eval.run_eval_suite import EvalCase, personas_for, scenario_for

    case = EvalCase("persona_distinctness", "", ("C", "B", "A", "D"), 112)
    personas = personas_for(case, scenario_for("study"))
    assert len({persona.private_goal for persona in personas}) == 4
    assert len({persona.background for persona in personas}) == 4
    assert {persona.sim_params.directness for persona in personas} == {3}


def test_unanimous_openings_plus_one_confirmation_narrow_early(monkeypatch):
    runner = make_runner(("A", "A", "A"))
    for runtime in runner.state.runtimes.values():
        runtime.public_preference = "A"
    runner.state.phase = Phase.DISCUSSION
    monkeypatch.setattr(cfg.conversation, "min_voluntary_turns", 8)
    runner._commit_action(
        UserAction("p1", True, 0.6, ActionType.SUPPORT, ("A",), reason="quiet"),
        "Option A still works for me.", mandatory=False, voluntary=True,
        liveness_forced=False, repair_count=0,
    )
    assert runner._ready_to_narrow()


def test_unanimous_openings_with_active_concern_do_not_narrow(monkeypatch):
    runner = make_runner(("A", "A", "A"))
    for runtime in runner.state.runtimes.values():
        runtime.public_preference = "A"
    monkeypatch.setattr(cfg.conversation, "min_voluntary_turns", 0)
    runner._commit_action(
        UserAction(
            "p1", True, 0.8, ActionType.CONCERN, ("A",),
            reason="crowding", issue_effect=IssueEffect.OPEN,
        ),
        "Option A may be too crowded.", mandatory=False, voluntary=True,
        liveness_forced=False, repair_count=0,
    )
    assert runner.state.active_issue is not None
    assert not runner._ready_to_narrow()


def test_split_public_preferences_do_not_trigger_early_convergence(monkeypatch):
    runner = make_runner(("A", "B", "C"))
    for participant_id, option_id in zip(runner.state.runtimes, ("A", "B", "C")):
        runner.state.runtimes[participant_id].public_preference = option_id
    runner.state.phase = Phase.DISCUSSION
    monkeypatch.setattr(cfg.conversation, "min_voluntary_turns", 8)
    runner._commit_action(
        UserAction("p1", True, 0.6, ActionType.SUPPORT, ("A",), reason="quiet"),
        "Option A works for me.", mandatory=False, voluntary=True,
        liveness_forced=False, repair_count=0,
    )
    assert not runner._ready_to_narrow()


def test_concern_owner_never_submits_a_responder_action_to_own_issue():
    runner = make_runner()
    concern = UserAction(
        "p1", True, 0.8, ActionType.CONCERN, ("B",), reason="noise",
        reason_source=ReasonSource("B", "noise", "moderate"),
        issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(
        concern, "Option B raises a noise concern.", mandatory=False, voluntary=True,
        liveness_forced=False, repair_count=0,
    )
    issue = runner.state.active_issue
    assert issue is not None and issue.follow_up_count == 0
    runner.state.runtimes["p1"].ranks["B"] = 2
    runner.state.runtimes["p1"].disliked_options.add("B")
    actions = runner._simulators["p1"]._issue_actions(
        runner.state, runner.state.runtimes["p1"], issue
    )
    assert actions == []


def test_strong_relevant_evidence_drives_flexible_concern_owner_to_resolution():
    runner = make_runner()
    runtime = runner.state.runtimes["p1"]
    runner.state.personas[0].sim_params.stubbornness = 1
    runtime.ranks["B"] = 2
    runtime.disliked_options.add("B")
    issue = _open_concern_with_response(runner)
    for participant_id in ("p2", "p3"):
        runner._commit_action(
            UserAction(
                participant_id, True, 0.8, ActionType.SUPPORT, ("B",),
                reason="moderate noise directly limits the concern",
                reason_source=ReasonSource("B", "noise", "moderate"),
                issue_id=issue.id, issue_effect=IssueEffect.CONTINUE,
                issue_response_kind=IssueResponseKind.MITIGATION,
            ),
            "The moderate noise level directly limits that concern about Option B.",
            mandatory=False, voluntary=True, liveness_forced=False, repair_count=0,
        )
    actions = runner._simulators["p1"]._concern_owner_reactions(
        runner.state, runtime, issue
    )
    assert actions
    assert {action.issue_effect for action in actions} == {IssueEffect.RESOLVE}
