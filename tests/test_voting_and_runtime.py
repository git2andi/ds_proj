from __future__ import annotations

from collections import Counter

import pytest

from config_loader import cfg
from consensus import majority_threshold, outcome_from_votes
from models import ActionType, IssueEffect, Phase, UserAction
from tests.fixtures import ActionRendererLLM, make_persona, make_personas, make_runner, make_scenario, make_state


def test_unanimous_result_is_successful():
    state = make_state(("A", "A", "A"))
    outcome = outcome_from_votes(state, {"p1": "A", "p2": "A", "p3": "A"}, allow_unresolved=False)
    assert outcome is not None
    assert outcome.status == "successful"
    assert outcome.final_option == "A"


def test_majority_result_closes_as_majority():
    state = make_state(("A", "A", "B", "C"))
    outcome = outcome_from_votes(
        state,
        {"p1": "A", "p2": "A", "p3": "A", "p4": "B"},
        allow_unresolved=False,
    )
    assert outcome is not None
    assert outcome.status == "majority"
    assert outcome.final_option == "A"


def test_first_no_majority_requests_revote_instead_of_inventing_result():
    state = make_state(("A", "B", "C", "D"))
    votes = {"p1": "A", "p2": "B", "p3": "C", "p4": "D"}
    assert outcome_from_votes(state, votes, allow_unresolved=False) is None


def test_second_no_majority_closes_unresolved():
    state = make_state(("A", "B", "C", "D"))
    votes = {"p1": "A", "p2": "B", "p3": "C", "p4": "D"}
    outcome = outcome_from_votes(state, votes, allow_unresolved=True)
    assert outcome is not None
    assert outcome.status == "unresolved"
    assert outcome.final_option is None


def test_majority_threshold_is_strict_majority():
    assert majority_threshold(3, fraction=0.5) == 2
    assert majority_threshold(4, fraction=0.5) == 3
    assert majority_threshold(5, fraction=0.5) == 3


def test_hard_blocker_vote_is_always_its_preferred_option():
    scenario = make_scenario()
    personas = [
        make_persona("p1", "Nora", "C", hard_blocker=True),
        make_persona("p2", "Ben", "A"),
        make_persona("p3", "Mira", "B"),
    ]
    runner = make_runner()
    runner.state = __import__("dialogue").initialise_state(scenario, personas)
    blocker_simulator = __import__("simulator").UserSimulator(personas[0], __import__("random").Random(9))
    for revote in (False, True):
        action = blocker_simulator.decide_vote(runner.state, revote=revote)
        assert action.act is ActionType.VOTE
        assert action.vote_option == "C"
        assert action.stance_update is None


def test_run_permits_exactly_one_revote_after_no_majority(monkeypatch):
    runner = make_runner()
    monkeypatch.setattr(runner, "_run_opening", lambda: None)
    monkeypatch.setattr(runner, "_run_discussion", lambda: None)
    narrowing_calls: list[bool] = []
    monkeypatch.setattr(runner, "_run_narrowing", lambda *, revote: narrowing_calls.append(revote))
    outcomes = iter([None, None])
    vote_calls: list[bool] = []

    def fake_vote(*, revote: bool):
        vote_calls.append(revote)
        runner.state.votes = {persona.id: None for persona in runner.state.personas}
        return next(outcomes)

    monkeypatch.setattr(runner, "_run_voting", fake_vote)
    result = runner.run()
    assert result.outcome.status == "unresolved"
    assert narrowing_calls == [False, True]
    assert vote_calls == [False, True]
    assert runner.state.phase is Phase.CLOSED


def test_valid_majority_does_not_trigger_revote(monkeypatch):
    runner = make_runner(("A", "A", "B"))
    monkeypatch.setattr(runner, "_run_opening", lambda: None)
    monkeypatch.setattr(runner, "_run_discussion", lambda: None)
    narrowing_calls: list[bool] = []
    monkeypatch.setattr(runner, "_run_narrowing", lambda *, revote: narrowing_calls.append(revote))
    majority = outcome_from_votes(
        runner.state,
        {"p1": "A", "p2": "A", "p3": "B"},
        allow_unresolved=False,
    )
    assert majority is not None
    vote_calls: list[bool] = []

    def fake_vote(*, revote: bool):
        vote_calls.append(revote)
        return majority

    monkeypatch.setattr(runner, "_run_voting", fake_vote)
    result = runner.run()
    assert result.outcome.status == "majority"
    assert narrowing_calls == [False]
    assert vote_calls == [False]


def test_offline_end_to_end_has_opening_discussion_narrowing_voting_and_closed(monkeypatch):
    monkeypatch.setattr(cfg.conversation, "min_voluntary_turns", 3)
    monkeypatch.setattr(cfg.conversation, "soft_target_voluntary_turns", 5)
    monkeypatch.setattr(cfg.conversation, "hard_max_voluntary_turns", 7)
    monkeypatch.setattr(cfg.conversation, "narrowing_voluntary_turns", 2)
    monkeypatch.setattr(cfg.conversation, "revote_narrowing_voluntary_turns", 1)
    llm = ActionRendererLLM()
    runner = make_runner(("A", "A", "B"), llm=llm, seed=11)
    result = runner.run()

    participant_turns = [turn for turn in result.state.turns if not turn.moderator]
    openings = [turn for turn in participant_turns if turn.action and turn.action.act is ActionType.OPENING]
    votes = [turn for turn in participant_turns if turn.action and turn.action.act is ActionType.VOTE]
    assert len(openings) == 3
    assert len(votes) >= 3
    assert result.state.phase is Phase.CLOSED
    assert result.outcome.status in {"successful", "majority", "unresolved"}
    assert {"OPENING", "DISCUSSION", "NARROWING", "VOTING", "CLOSED"}.issubset(result.state.phase_history)
    assert result.state.stats.llm_calls == llm.calls
    assert all(turn.action is not None for turn in participant_turns)


def test_formal_votes_are_excluded_from_voluntary_engagement_counts(monkeypatch):
    monkeypatch.setattr(cfg.conversation, "min_voluntary_turns", 1)
    monkeypatch.setattr(cfg.conversation, "soft_target_voluntary_turns", 1)
    monkeypatch.setattr(cfg.conversation, "hard_max_voluntary_turns", 2)
    monkeypatch.setattr(cfg.conversation, "narrowing_voluntary_turns", 0)
    monkeypatch.setattr(cfg.conversation, "revote_narrowing_voluntary_turns", 0)
    result = make_runner(("A", "A", "A"), seed=3).run()
    counted = Counter(turn.speaker_id for turn in result.state.turns if turn.voluntary)
    for persona in result.state.personas:
        assert result.state.runtimes[persona.id].voluntary_turns == counted[persona.id]
    assert all(not turn.voluntary for turn in result.state.turns if turn.vote_option)


def test_last_allowed_discussion_question_is_answered_before_transition(monkeypatch):
    runner = make_runner(("A", "B", "C"), seed=19)
    runner.state.phase = Phase.DISCUSSION
    monkeypatch.setattr(cfg.conversation, "hard_max_voluntary_turns", 1)
    monkeypatch.setattr(cfg.conversation, "min_voluntary_turns", 1)
    monkeypatch.setattr(cfg.conversation, "soft_target_voluntary_turns", 1)

    question = UserAction(
        "p1", True, 0.9, ActionType.ASK, ("B",),
        addressee_id="p2",
        reason="Ask how the background noise affects Ben",
        issue_effect=IssueEffect.OPEN,
    )
    selected = False

    def select_once(_bids, *, phase):
        nonlocal selected
        assert phase is Phase.DISCUSSION
        if selected:
            return False
        selected = True
        runner._commit_action(
            question,
            "Ben, how does the background noise at Option B affect you?",
            mandatory=False,
            voluntary=True,
            liveness_forced=False,
            repair_count=0,
        )
        return True

    monkeypatch.setattr(runner, "_select_and_realize", select_once)
    original_realize = runner._realize_and_commit
    answer_calls: list[str] = []

    def record_answer(action, **kwargs):
        if action.act is ActionType.ANSWER:
            answer_calls.append(action.speaker_id)
        return original_realize(action, **kwargs)

    monkeypatch.setattr(runner, "_realize_and_commit", record_answer)
    runner._run_discussion()

    assert answer_calls == ["p2"]
    actions = [turn.action.act for turn in runner.state.participant_turns if turn.action]
    assert actions[-2:] == [ActionType.ASK, ActionType.ANSWER]
    assert runner.state.response_obligation is None
    assert runner.state.voluntary_turn_count == 1


def test_narrowing_does_not_remove_a_simulators_valid_vote_choice():
    import random

    from simulator import UserSimulator

    scenario = make_scenario()
    persona = make_persona("p1", "Nora", "B", stubbornness=4)
    state = __import__("dialogue").initialise_state(scenario, [persona])
    state.narrowing_options = ("A", "C")
    action = UserSimulator(persona, random.Random(4)).decide_vote(state)
    assert action.vote_option == "B"
    assert action.stance_update is None


def test_revote_flag_does_not_add_switch_pressure_without_new_evidence():
    import random

    from simulator import UserSimulator

    persona = make_persona("p1", "Nora", "A", stubbornness=1)
    state = __import__("dialogue").initialise_state(make_scenario(), [persona])
    simulator = UserSimulator(persona, random.Random(4))
    runtime = state.runtimes[persona.id]
    assert simulator._vote_score(state, runtime, "B", revote=False) == simulator._vote_score(
        state, runtime, "B", revote=True
    )


def test_invalid_final_vote_is_protocol_degradation_even_when_attempt_record_exists():
    from eval.run_eval_suite import vote_protocol_flags
    from models import VoteRecord, VoteStatus

    state = make_state(("A", "B"))
    state.vote_round = 1
    state.vote_records[1] = {
        "p1": VoteRecord("p1", 1, VoteStatus.VALID, option_id="A", attempts=1),
        "p2": VoteRecord("p2", 1, VoteStatus.GENERATION_FAILED, option_id=None, attempts=2),
    }
    attempts_complete, protocol_valid = vote_protocol_flags(state, 2)
    assert attempts_complete
    assert not protocol_valid


def test_explicit_abstention_is_protocol_valid_but_not_a_valid_vote():
    from eval.run_eval_suite import vote_protocol_flags
    from models import VoteRecord, VoteStatus

    state = make_state(("A", "B"))
    state.vote_round = 1
    state.vote_records[1] = {
        "p1": VoteRecord("p1", 1, VoteStatus.VALID, option_id="A", attempts=1),
        "p2": VoteRecord("p2", 1, VoteStatus.ABSTAINED, option_id=None, attempts=1),
    }
    attempts_complete, protocol_valid = vote_protocol_flags(state, 2)
    assert attempts_complete
    assert protocol_valid
