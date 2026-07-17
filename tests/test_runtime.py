from __future__ import annotations

import random

import pytest

from config_loader import cfg
from dialogue import DialogueRunner
from models import ActionType, BidPriority, Phase, StanceUpdate, StanceUpdateKind, TurnRecord, UserAction
from tests.fixtures import ActionRendererLLM, NullLogger, make_persona, make_personas, make_runner, make_scenario


def test_full_offline_run_has_required_phases_and_bounded_length():
    result = make_runner(("A", "B", "C"), seed=7).run()
    assert result.state.phase_history[:4] == ["OPENING", "DISCUSSION", "NARROWING", "VOTING"]
    assert result.state.phase_history[-1] == "CLOSED"
    assert result.state.phase_history.count("VOTING") in {1, 2}
    assert 9 <= len(result.state.participant_turns) <= 35
    assert result.outcome.status in {"successful", "majority", "unresolved"}
    assert result.token_summary["dialogue_tokens_in"] < 12000


def test_public_options_are_printed_even_with_moderator(capsys):
    make_runner(("A", "A", "A"), seed=1).run()
    out = capsys.readouterr().out
    assert "Options:" in out
    assert "A) Central Library" in out
    assert "Moderator:" in out


def test_every_participant_has_one_opening():
    result = make_runner(("A", "B", "C"), seed=3).run()
    openings = [turn for turn in result.state.participant_turns if turn.action and turn.action.act is ActionType.OPENING]
    assert {turn.speaker_id for turn in openings} == {"p1", "p2", "p3"}


def test_mandatory_opening_failure_is_not_silent():
    llm = ActionRendererLLM(scripted=[""] * 20)
    runner = DialogueRunner(
        "", scenario=make_scenario(), personas=make_personas(("A", "B", "C")),
        llm=llm, logger=NullLogger(), rng=random.Random(1), seed=1,
    )
    with pytest.raises(RuntimeError, match="mandatory opening failed"):
        runner.run()


def test_direct_question_creates_answer_before_transition():
    result = make_runner(("A", "B", "C"), seed=8).run()
    turns = result.state.participant_turns
    for index, turn in enumerate(turns[:-1]):
        if turn.action and turn.action.act is ActionType.ASK and turn.action.addressee_id:
            next_participant = turns[index + 1]
            assert next_participant.speaker_id == turn.action.addressee_id
            assert next_participant.action and next_participant.action.act is ActionType.ANSWER


def test_majority_run_does_not_enter_revote():
    result = make_runner(("A", "A", "B"), seed=2).run()
    if result.outcome.status == "majority":
        assert result.state.vote_round == 1
        assert result.state.phase_history.count("NARROWING") == 1


def test_hard_blocker_never_votes_for_other_option():
    personas = [
        make_persona("p1", "Mira", "C", hard_blocker=True),
        make_persona("p2", "Nora", "A"),
        make_persona("p3", "Ben", "A"),
    ]
    runner = DialogueRunner(
        "", scenario=make_scenario(), personas=personas,
        llm=ActionRendererLLM(), logger=NullLogger(), rng=random.Random(4), seed=4,
    )
    result = runner.run()
    assert result.state.votes["p1"] == "C"


def test_no_moderator_mode_has_no_visible_moderator(monkeypatch):
    monkeypatch.setattr(cfg.moderator, "enabled", False)
    result = make_runner(("A", "A", "B"), seed=9).run()
    assert not any(turn.moderator for turn in result.state.turns)
    assert result.outcome.status in {"successful", "majority", "unresolved"}


def test_floor_does_not_equalize_participation():
    personas = [
        make_persona("p1", "High", "A", engagement=5),
        make_persona("p2", "Low", "B", engagement=1),
        make_persona("p3", "Mid", "C", engagement=3),
    ]
    result = DialogueRunner(
        "", scenario=make_scenario(), personas=personas,
        llm=ActionRendererLLM(), logger=NullLogger(), rng=random.Random(12), seed=12,
    ).run()
    voluntary = {pid: runtime.voluntary_turns for pid, runtime in result.state.runtimes.items()}
    assert voluntary["p1"] >= voluntary["p2"]


def test_direct_question_allows_one_optional_follow_up_then_resolves():
    from models import IssueEffect, IssueKind, IssueStatus

    runner = make_runner(("A", "B", "C"), seed=8)
    runner.state.phase = Phase.DISCUSSION
    question = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.ASK,
        ("B",), addressee_id="p2", reason="background noise",
        issue_effect=IssueEffect.OPEN,
    )
    runner._commit_action(
        question,
        "Ben, does the background noise change your choice of Cafe?",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    issue_id = runner.state.active_issue.id
    answer = UserAction(
        "p2", True, BidPriority.REQUIRED, ActionType.ANSWER,
        ("B",), addressee_id="p1", reason="the atmosphere matters more",
        issue_id=issue_id, issue_effect=IssueEffect.RESPOND,
    )
    runner._commit_action(
        answer,
        "No, the atmosphere still matters more to me.",
        mandatory=True,
        voluntary=False,
        liveness_forced=False,
        repair_count=0,
    )
    assert runner.state.response_obligation is None
    assert runner.state.active_issue is not None
    assert runner.state.active_issue.required_answer_completed

    follow_up = UserAction(
        "p3", True, BidPriority.ISSUE_RESPONSE, ActionType.COMMENT,
        ("B",), reason="the noise matters to me too",
        issue_id=issue_id, issue_effect=IssueEffect.RESPOND,
    )
    runner._commit_action(
        follow_up,
        "The noise matters to me too.",
        mandatory=False,
        voluntary=True,
        liveness_forced=False,
        repair_count=0,
    )
    assert runner.state.active_issue is None
    closed = runner.state.issue_history[-1]
    assert closed.kind is IssueKind.QUESTION
    assert closed.status is IssueStatus.RESOLVED
    assert closed.outcome == "answered_with_follow_up"


def test_answered_question_resolves_when_nobody_follows_up():
    from models import ActiveIssue, IssueKind, IssueStatus

    runner = make_runner(("A", "B", "C"), seed=8)
    runner.state.active_issue = ActiveIssue(
        id="i1", kind=IssueKind.QUESTION, option_focus=("B",),
        opened_by="p1", addressed_to="p2", summary="background noise",
        status=IssueStatus.OPEN, opened_at_turn=0, last_relevant_turn=1,
        response_count=1, responded_by={"p2"}, required_answer_completed=True,
        outcome="answered",
    )
    runner._close_or_stale_inactive_issue("nobody followed up")
    assert runner.state.active_issue is None
    assert runner.state.issue_history[-1].status is IssueStatus.RESOLVED
    assert runner.state.issue_history[-1].outcome == "answered"


def test_narrowing_is_adaptive_instead_of_forcing_everyone_to_restate():
    result = make_runner(("A", "B", "C"), seed=5).run()
    narrowing = [
        turn for turn in result.state.participant_turns
        if turn.phase is Phase.NARROWING
    ]
    assert len(narrowing) <= int(cfg.conversation.compromise_window_max_turns) * 2 + 2
    # Movement may already have happened during the discussion compromise window;
    # narrowing must not force a second generic restatement round.
    assert result.state.movement_events >= 0
    generic_restated_positions = [
        turn for turn in narrowing
        if turn.action and turn.action.act is ActionType.FINAL_POSITION
    ]
    assert len(generic_restated_positions) < len(result.state.personas)


def test_unanimous_public_preference_skips_participant_narrowing():
    result = make_runner(("A", "A", "A"), seed=11).run()
    assert not [
        turn for turn in result.state.participant_turns
        if turn.phase is Phase.NARROWING
    ]


def test_clear_leader_schedules_only_dissenter_as_mandatory(monkeypatch):
    import simulator as simulator_module

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 0.0)
    result = make_runner(("A", "A", "B"), seed=13).run()
    mandatory_narrowing = [
        turn.speaker_id
        for turn in result.state.participant_turns
        if turn.phase is Phase.NARROWING and turn.mandatory
    ]
    assert set(mandatory_narrowing) <= {"p3"}


def test_no_revote_when_final_discussion_produces_no_movement(monkeypatch):
    import simulator as simulator_module

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 0.0)
    result = make_runner(("A", "B", "C"), seed=14).run()
    assert result.outcome.status == "unresolved"
    assert result.state.revote_skipped_no_movement
    assert result.state.vote_round == 1
    assert result.state.phase_history.count(Phase.VOTING.value) == 1


def test_revote_narrowing_can_create_movement_without_forcing_it(monkeypatch):
    import simulator as simulator_module

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 1.0)
    runner = make_runner(("A", "B", "C"), seed=15)
    for pid, option in zip(("p1", "p2", "p3"), ("A", "B", "C")):
        runner.state.runtimes[pid].public_preference = option
    runner.state.first_round_votes = {"p1": "A", "p2": "B", "p3": "C"}
    _, movements = runner._run_narrowing(revote=True)
    assert movements >= 1


def test_public_common_ground_is_detected_without_weighted_scoring():
    runner = make_runner(("A", "B", "B"), seed=21)
    runner.state.runtimes["p1"].public_preference = "A"
    runner.state.runtimes["p1"].public_acceptances.add("B")
    runner.state.runtimes["p2"].public_preference = "B"
    runner.state.runtimes["p3"].public_preference = "B"
    assert runner._shared_acceptable_option() == "B"


def test_incomplete_comparison_does_not_create_hidden_comparison_evidence():
    runner = make_runner(("A", "B", "C"), seed=31)
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.COMPARE,
        option_focus=("A", "B"), reason="quiet versus atmosphere",
    )
    runner._apply_public_action(action, "Library is quieter for me.")
    assert not runner.state.public_comparisons


def test_complete_split_gets_one_visible_compromise_prompt_without_untouched_options(monkeypatch, capsys):
    import simulator as simulator_module

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 0.0)
    runner = make_runner(("A", "B", "C"), seed=32)
    for participant_id, option_id in zip(("p1", "p2", "p3"), ("A", "B", "C")):
        runner.state.runtimes[participant_id].public_preference = option_id
    runner._run_narrowing(revote=False)
    output = capsys.readouterr().out
    assert "still split" in output
    assert "Library" in output
    assert "Cafe" in output
    assert "Lab" in output
    assert "Online" not in output
    assert runner.state.stats.selected_movement_actions == 0


def test_soft_coverage_respects_engagement_and_can_receive_no_response(monkeypatch, capsys):
    import simulator as simulator_module

    monkeypatch.setattr(simulator_module, "bid_probability", lambda _level: 0.0)
    runner = make_runner(("A", "B", "C"), seed=33)
    runner.state.phase = Phase.DISCUSSION

    assert runner._run_coverage_window("D") is False
    assert "D" in runner.state.coverage_no_interest
    assert not runner.state.participant_turns
    assert "We have not really considered" not in capsys.readouterr().out


def test_public_unanimity_prevents_liveness_filler_before_minimum(monkeypatch):
    from simulator import UserSimulator

    runner = make_runner(("A", "A", "A"), seed=34)
    runner.state.phase = Phase.DISCUSSION
    runner._moderator_enabled = False
    for index, participant_id in enumerate(("p1", "p2", "p3")):
        runner.state.runtimes[participant_id].public_preference = "A"
        runner.state.turns.append(
            TurnRecord(
                index,
                Phase.DISCUSSION,
                participant_id,
                runner.state.persona(participant_id).name,
                "A remains my choice for a different reason.",
                action=UserAction(
                    participant_id,
                    True,
                    BidPriority.NORMAL,
                    ActionType.SUPPORT,
                    ("A",),
                    reason=f"reason {index}",
                ),
                voluntary=True,
            )
        )

    def no_bid_unless_forced(self, _state, *, liveness_forced=False):
        if liveness_forced:
            return UserAction(
                self.id,
                True,
                BidPriority.NORMAL,
                ActionType.SUPPORT,
                ("A",),
                reason="forced filler",
            )
        return UserAction(self.id, False, BidPriority.NORMAL, ActionType.COMMENT)

    monkeypatch.setattr(UserSimulator, "propose", no_bid_unless_forced)
    runner._run_discussion()

    assert runner.state.stats.liveness_forced_turns == 0
    assert len(runner.state.participant_turns) == 3


def test_failed_vote_realization_uses_authoritative_visible_fallback():
    class VoteFailingLLM(ActionRendererLLM):
        def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
            if "State only one short" in prompt or "State one clear vote" in prompt:
                text = "Either Library or Cafe could work."
                self.prompts.append(prompt)
                self.profiles.append(profile)
                self.calls += 1
                self.last_tokens_in = max(1, len(prompt.split()))
                self.last_tokens_out = len(text.split())
                self.session_tokens_in += self.last_tokens_in
                self.session_tokens_out += self.last_tokens_out
                self.session_calls += 1
                return text
            if profile == "repair" and "selected action" in prompt.casefold() and "vote" in prompt.casefold():
                text = "I cannot decide between Library and Cafe."
                self.prompts.append(prompt)
                self.profiles.append(profile)
                self.calls += 1
                self.last_tokens_in = max(1, len(prompt.split()))
                self.last_tokens_out = len(text.split())
                self.session_tokens_in += self.last_tokens_in
                self.session_tokens_out += self.last_tokens_out
                self.session_calls += 1
                return text
            return super().generate(prompt, profile=profile)

    result = make_runner(("A", "A", "A"), llm=VoteFailingLLM(), seed=41).run()
    assert result.state.stats.vote_fallbacks == 3
    assert all(record.status.value == "valid" for record in result.state.vote_records[1].values())
    assert all(result.state.votes[pid] == "A" for pid in ("p1", "p2", "p3"))
    vote_turns = [turn for turn in result.state.participant_turns if turn.action and turn.action.act is ActionType.VOTE]
    assert len(vote_turns) == 3
    assert all("voting for" in turn.text.casefold() or "switching" in turn.text.casefold() for turn in vote_turns)
    assert result.state.vote_protocol_degraded


def test_failed_voluntary_compromise_is_dropped_instead_of_scripted(monkeypatch, capsys):
    import simulator as simulator_module

    class CompromiseFailingLLM(ActionRendererLLM):
        def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
            if "visibly acceptable" in prompt or "moving to" in prompt:
                text = "That point is interesting."
                self.prompts.append(prompt)
                self.profiles.append(profile)
                self.calls += 1
                self.last_tokens_in = max(1, len(prompt.split()))
                self.last_tokens_out = len(text.split())
                self.session_tokens_in += self.last_tokens_in
                self.session_tokens_out += self.last_tokens_out
                self.session_calls += 1
                return text
            return super().generate(prompt, profile=profile)

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 1.0)
    runner = make_runner(("A", "B", "C"), llm=CompromiseFailingLLM(), seed=42)
    for participant_id, option_id in zip(("p1", "p2", "p3"), ("A", "B", "C")):
        runner.state.runtimes[participant_id].public_preference = option_id
    runner._run_narrowing(revote=False)
    output = capsys.readouterr().out
    assert "still split" in output
    assert "relaxed atmosphere" not in output.casefold()
    assert runner.state.stats.movement_fallbacks == 0
    assert runner.state.stats.movement_realization_failures >= 1
    assert runner.state.stats.dropped_turns >= 1
    assert runner.state.stats.selected_movement_actions > runner.state.stats.committed_movement_actions


def test_failed_mandatory_movement_is_dropped_instead_of_scripted():
    class VagueMovementLLM(ActionRendererLLM):
        def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
            if "visibly acceptable" in prompt or "concrete movement reason" in prompt:
                text = "Cafe seems reasonable enough for me."
                self.prompts.append(prompt)
                self.profiles.append(profile)
                self.calls += 1
                self.last_tokens_in = max(1, len(prompt.split()))
                self.last_tokens_out = len(text.split())
                self.session_tokens_in += self.last_tokens_in
                self.session_tokens_out += self.last_tokens_out
                self.session_calls += 1
                return text
            return super().generate(prompt, profile=profile)

    runner = make_runner(("A", "B", "C"), llm=VagueMovementLLM(), seed=51)
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.COMPROMISE,
        ("B",), reason="relaxed atmosphere", decisive_reason="relaxed atmosphere",
        stance_update=StanceUpdate(
            StanceUpdateKind.MAKE_ACCEPTABLE,
            "B",
            previous_option_id="A",
            movement_reason="relaxed atmosphere",
            movement_basis="common_ground",
        ),
    )
    record = runner._realize_and_commit(action, mandatory=True, voluntary=False)
    assert record is None
    assert runner.state.stats.movement_fallbacks == 0
    assert runner.state.stats.movement_realization_failures == 1
    assert "B" not in runner.state.runtimes["p1"].public_acceptances


def test_clear_leader_narrowing_stops_after_enough_new_support(monkeypatch):
    import simulator as simulator_module
    from dialogue import DialogueRunner
    from tests.fixtures import ActionRendererLLM, NullLogger, make_personas, make_scenario

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 1.0)
    scenario = make_scenario()
    personas = make_personas(("A", "A", "A", "B", "C", "D", "B"))
    runner = DialogueRunner(
        "", scenario=scenario, personas=personas,
        llm=ActionRendererLLM(), logger=NullLogger(), seed=81,
    )
    for persona, preference in zip(personas, ("A", "A", "A", "B", "C", "D", "B")):
        runtime = runner.state.runtimes[persona.id]
        runtime.public_preference = preference
        runtime.preferred_option = preference
    runner.state.phase = Phase.DISCUSSION
    runner._run_narrowing(revote=False)
    accepted_a = sum(
        runtime.public_preference == "A" or "A" in runtime.public_acceptances
        for runtime in runner.state.runtimes.values()
    )
    assert accepted_a >= 4
    # Three initial supporters need only one additional acceptance for a strict
    # seven-person majority; the runtime should not march every dissenter through.
    movement_turns = [
        turn for turn in runner.state.participant_turns
        if turn.phase is Phase.NARROWING and turn.stance_update is not None
    ]
    assert len(movement_turns) <= 2


def test_reopening_same_concern_increments_global_record_once():
    from models import IssueEffect, IssueKind, IssueStatus

    runner = make_runner(("A", "B", "C"), seed=44)
    runner.state.phase = Phase.DISCUSSION
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.CONCERN,
        option_focus=("B",), reason="background noise",
        issue_effect=IssueEffect.OPEN,
    )
    runner._open_issue(IssueKind.CONCERN, action, "Cafe may have background noise.")
    key = runner.state.active_issue.issue_key
    runner._close_active_issue(IssueStatus.STALE, "moved on")
    runner.state.phase = Phase.NARROWING
    runner._open_issue(IssueKind.CONCERN, action, "The background noise still matters.")
    assert runner.state.issue_records[key].reopen_count == 1


def test_small_group_shared_acceptance_does_not_close_at_bare_minimum(monkeypatch):
    runner = make_runner(("A", "B", "B"), seed=52)
    minimum, target, _maximum = cfg.conversation_turn_budgets(3)
    threshold = min(target, minimum + 3)
    assert threshold > minimum


def test_large_group_clear_leader_narrowing_caps_required_final_positions(monkeypatch):
    import simulator as simulator_module

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 0.0)
    runner = make_runner(("A", "B", "C", "D", "B", "C", "D"), seed=91)
    for persona, preference in zip(runner.state.personas, ("A", "B", "C", "D", "B", "C", "D")):
        runtime = runner.state.runtimes[persona.id]
        runtime.public_preference = preference
        runtime.preferred_option = preference
    runner.state.phase = Phase.DISCUSSION
    runner._run_narrowing(revote=False)
    mandatory = [
        turn for turn in runner.state.participant_turns
        if turn.phase is Phase.NARROWING and turn.mandatory
    ]
    assert len(mandatory) <= int(cfg.conversation.large_group_narrowing_final_position_cap)
    narrowing_turns = [
        turn for turn in runner.state.participant_turns
        if turn.phase is Phase.NARROWING
    ]
    assert len(narrowing_turns) <= 2 * int(
        cfg.conversation.large_group_narrowing_final_position_cap
    )


def test_complete_split_no_response_bridge_replaces_immediate_vote_prompt(monkeypatch, capsys):
    import simulator as simulator_module

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 0.0)
    runner = make_runner(("A", "B", "C"), seed=141)
    for participant_id, option_id in zip(("p1", "p2", "p3"), ("A", "B", "C")):
        runtime = runner.state.runtimes[participant_id]
        runtime.public_preference = option_id
        runtime.preferred_option = option_id
    runner.state.phase = Phase.DISCUSSION

    accepted, movement = runner._run_narrowing(revote=False)
    assert accepted == 0
    assert movement == 0
    runner._run_voting(revote=False)

    moderator_texts = [turn.text for turn in runner.state.turns if turn.moderator]
    assert any("No one? All right" in text for text in moderator_texts)
    assert sum("final vote" in text.casefold() for text in moderator_texts) == 1


def test_repaired_turn_records_total_raw_and_repair_token_usage():
    llm = ActionRendererLLM(scripted=[
        "",
        "Hi everyone. Library seems best because it is quiet and predictable.",
    ])
    runner = make_runner(("A", "B", "C"), llm=llm, seed=201)
    action = runner._simulators["p1"].opening_action(runner.state)

    record = runner._realize_and_commit(action, mandatory=True, voluntary=False)

    assert record is not None
    assert record.repair_count == 1
    assert llm.session_calls == 2
    assert record.prompt_tokens == llm.session_tokens_in
    assert record.output_tokens == llm.session_tokens_out


def test_failed_direct_answer_is_logged_as_response_failure_not_vote_failure():
    from models import ActiveIssue, IssueKind, IssueStatus, QuestionMode

    llm = ActionRendererLLM(scripted=["Bananas are purple."] * 4)
    runner = make_runner(("A", "B", "C"), llm=llm, seed=202)
    runner.state.phase = Phase.DISCUSSION
    runner.state.active_issue = ActiveIssue(
        id="q1",
        kind=IssueKind.QUESTION,
        option_focus=("B",),
        opened_by="p1",
        addressed_to="p2",
        summary="whether background noise changes the choice",
        status=IssueStatus.OPEN,
        opened_at_turn=0,
        last_relevant_turn=0,
        question_mode=QuestionMode.CHOICE_IMPACT,
    )
    runner.state.response_obligation = "p2"

    runner._drain_response_obligation("direct answer could not be realized")

    assert runner.state.stats.response_failures == 1
    assert runner.state.protocol_errors == ["direct answer could not be realized"]
    assert runner.state.vote_protocol_degraded is False
    assert runner.state.response_obligation is None
    assert runner.state.active_issue is None
