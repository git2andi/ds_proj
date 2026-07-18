import random

import pytest

from config_loader import cfg
from dialogue import DialogueRunner
from models import ActionType, BidPriority, DiscussionThread, Phase, ReasonSource, ThreadKind, UserAction
from tests.fixtures import ActionRendererLLM, NullLogger, make_persona, make_personas, make_runner, make_scenario


def test_full_offline_run_has_bounded_phases_and_votes():
    result = make_runner(("A", "B", "C"), seed=7).run()
    assert result.state.phase_history[0] == "OPENING"
    assert result.state.phase_history[-1] == "CLOSED"
    assert {"DISCUSSION", "NARROWING", "VOTING"} <= set(result.state.phase_history)
    assert len(result.state.participant_turns) <= 40
    assert len(result.state.votes) == 3
    assert result.outcome.status in {"successful", "majority", "unresolved"}


def test_every_participant_has_one_opening():
    result = make_runner(seed=3).run()
    openings = [turn for turn in result.state.participant_turns if turn.action and turn.action.act is ActionType.OPENING]
    assert {turn.speaker_id for turn in openings} == {"p1", "p2", "p3"}


def test_direct_questions_are_answered_next():
    result = make_runner(seed=8).run()
    turns = result.state.participant_turns
    for index, turn in enumerate(turns[:-1]):
        if turn.action and turn.action.act is ActionType.ASK and turn.action.addressee_id:
            assert turns[index + 1].speaker_id == turn.action.addressee_id
            assert turns[index + 1].action.act is ActionType.ANSWER


def test_threads_are_bounded():
    result = make_runner(seed=9).run()
    cap = int(cfg.conversation.thread_turn_cap)
    # The active thread is always closed before voting; thread events let us
    # verify that no sequence grows beyond the configured cap.
    current = 0
    maximum = 0
    for turn in result.state.participant_turns:
        if turn.thread_event in {"opened_question", "opened_concern"}:
            current = 1
        elif turn.thread_event in {"answered_question", "thread_follow_up"}:
            current += 1
        elif turn.phase is not Phase.DISCUSSION:
            current = 0
        maximum = max(maximum, current)
    assert maximum <= cap


def test_no_moderator_mode(monkeypatch):
    monkeypatch.setattr(cfg.moderator, "enabled", False)
    result = make_runner(("A", "A", "B"), seed=10).run()
    assert not any(turn.moderator for turn in result.state.turns)


def test_decisive_majority_does_not_pressure_dissenter():
    result = make_runner(("A", "A", "A", "B"), seed=11, alternatives_acceptable=False).run()
    narrowing_turns = [turn for turn in result.state.participant_turns if turn.phase is Phase.NARROWING]
    assert not narrowing_turns


def test_hard_blocker_votes_for_own_option():
    personas = [
        make_persona("p1", "Nora", "C", hard_blocker=True),
        make_persona("p2", "Ben", "A"),
        make_persona("p3", "Mira", "A"),
    ]
    result = DialogueRunner(
        "", scenario=make_scenario(), personas=personas,
        llm=ActionRendererLLM(), logger=NullLogger(), rng=random.Random(4), seed=4,
    ).run()
    assert result.state.votes["p1"] == "C"


def test_required_answer_keeps_original_text_without_repair_or_fallback():
    llm = ActionRendererLLM(scripted=["Cafe has a relaxed atmosphere."])
    runner = make_runner(llm=llm, seed=12)
    runner.state.active_thread = DiscussionThread(
        "t1",
        ThreadKind.QUESTION,
        "p1",
        ("B",),
        ("B", "noise"),
        "Does background noise make Cafe unsuitable?",
        required_answer_pending=True,
    )
    action = UserAction(
        "p2",
        True,
        BidPriority.REQUIRED,
        ActionType.ANSWER,
        ("B",),
        reason="background noise",
        reason_source=ReasonSource("B", "noise", "moderate"),
    )
    record = runner._realize_and_commit(action, mandatory=True, voluntary=False)
    assert record is not None
    assert record.text == "Cafe has a relaxed atmosphere."
    assert runner.state.stats.repair_calls == 0
    assert runner.state.stats.fallback_turns == 0


def test_empty_opening_uses_last_resort_fallback():
    llm = ActionRendererLLM(scripted=["", ""])
    runner = DialogueRunner(
        "", scenario=make_scenario(), personas=make_personas(),
        llm=llm, logger=NullLogger(), rng=random.Random(1), seed=1,
    )
    result = runner.run()
    first_opening = next(
        turn for turn in result.state.participant_turns
        if turn.action and turn.action.act is ActionType.OPENING
    )
    assert first_opening.text.startswith("Hi, I prefer ")
    assert result.state.stats.fallback_turns >= 1


def test_only_one_formal_vote_round_is_visible():
    result = make_runner(("A", "B", "C"), seed=31).run()
    vote_turns = [
        turn for turn in result.state.participant_turns
        if turn.action and turn.action.act is ActionType.VOTE
    ]
    assert len(vote_turns) == len(result.state.personas)
    assert result.state.vote_round == 1
    assert set(result.state.vote_records) == {1}


def test_formal_votes_are_deterministic_and_use_no_llm_calls_for_voting():
    llm = ActionRendererLLM()
    result = make_runner(("A", "B", "C"), llm=llm, seed=41).run()
    vote_attempts = [
        attempt
        for attempt in result.state.generation_attempts
        if attempt.phase is Phase.VOTING
    ]
    assert len(vote_attempts) == len(result.state.personas)
    assert all(attempt.final_status == "deterministic" for attempt in vote_attempts)
    assert all(attempt.repair_text is None for attempt in vote_attempts)
    assert result.state.stats.repair_calls == sum(
        attempt.repair_text is not None for attempt in result.state.generation_attempts
    )


def test_compromise_prompt_waits_without_follow_up_moderator_commentary():
    result = make_runner(
        ("A", "B", "C"),
        seed=42,
        alternatives_acceptable=False,
    ).run()
    narrowing_moderator = [
        turn
        for turn in result.state.turns
        if turn.moderator and turn.phase is Phase.NARROWING
    ]
    narrowing_participants = [
        turn
        for turn in result.state.participant_turns
        if turn.phase is Phase.NARROWING
    ]
    assert len(narrowing_moderator) == 1
    assert "?" in narrowing_moderator[0].text
    assert not narrowing_participants
    index = result.state.turns.index(narrowing_moderator[0])
    assert result.state.turns[index + 1].action.act is ActionType.VOTE
