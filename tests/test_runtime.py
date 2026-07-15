from __future__ import annotations

import random

import pytest

from config_loader import cfg
from dialogue import DialogueRunner
from models import ActionType, BidPriority, Phase, StanceUpdate, StanceUpdateKind, UserAction
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


def test_direct_question_issue_closes_after_required_answer():
    result = make_runner(("A", "B", "C"), seed=8).run()
    answered = [
        issue for issue in result.state.issue_history
        if issue.kind.value == "question" and issue.outcome == "answered"
    ]
    assert answered
    assert all(issue.status.value == "resolved" for issue in answered)


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


def test_tie_narrowing_prompt_is_hidden_when_no_simulator_wants_to_move(monkeypatch, capsys):
    import simulator as simulator_module

    monkeypatch.setattr(simulator_module, "movement_probability", lambda *_args, **_kwargs: 0.0)
    runner = make_runner(("A", "B", "C"), seed=32)
    for participant_id, option_id in zip(("p1", "p2", "p3"), ("A", "B", "C")):
        runner.state.runtimes[participant_id].public_preference = option_id
    runner._run_narrowing(revote=False)
    output = capsys.readouterr().out
    assert "There is no clear leader yet" not in output
    assert "We seem stuck" not in output
    assert runner.state.stats.selected_movement_actions == 0


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
    assert all("vote" in turn.text.casefold() or "switching" in turn.text.casefold() for turn in vote_turns)


def test_failed_compromise_realization_uses_fallback_and_keeps_nudge_answered(monkeypatch, capsys):
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
    assert "There is no clear leader yet" in output
    assert "I can accept" in output
    assert runner.state.stats.movement_fallbacks >= 1
    assert runner.state.stats.movement_realization_failures >= 1
    assert runner.state.stats.dropped_turns == 0
    assert (
        runner.state.stats.selected_movement_actions
        == runner.state.stats.committed_movement_actions
    )


def test_failed_mandatory_movement_uses_grounded_fallback():
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
    assert record is not None
    assert "relaxed atmosphere" in record.text.casefold()
    assert runner.state.stats.movement_fallbacks == 1
    assert "B" in runner.state.runtimes["p1"].public_acceptances
    assert runner.state.runtimes["p1"].acceptance_reasons["B"] == "relaxed atmosphere"
