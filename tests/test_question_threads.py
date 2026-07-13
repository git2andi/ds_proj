"""Tests for thread-driven question handling (TODO 7)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

import simulator as sim_policy
from models import ActType, MoveIntent, Phase, ThreadStatus, ThreadType, TurnObligation

from tests.fixtures import append_turn, make_state
from tests.stubs import make_runner


def _question_threads(state):
    return [t for t in state.threads.values() if t.thread_type is ThreadType.QUESTION]


def _answer_intent(state, thread, respondent=None):
    """Build the simulator-owned answer intent for a question thread the way the
    flow does: a framework obligation fixes speaker/act/target; the simulator
    policy chooses the answer's option focus and direction."""
    respondent = respondent or thread.required_respondent
    ob = TurnObligation(
        kind="direct_answer", participant_id=respondent, act=ActType.ANSWER,
        respond_to_turn=thread.source_turn_index, thread_id=thread.thread_id,
        addressee_id=None if thread.started_by in {"moderator", ""} else thread.started_by,
        focus_options=list(thread.focus_options),
    )
    return sim_policy.decide_simulator_bid(state, respondent, obligation=ob).intent


def _observe(runner, state, speaker_id, text, *, intent=None):
    """Run one turn through the real generation pipeline with scripted text."""
    runner._llm.responses.append(text)
    intent = intent or MoveIntent(speaker_id=speaker_id, act=ActType.COMMENT, reason="acknowledge the exchange")
    return runner._generate_and_append(state, intent)


class QuestionThreadCreationTests(unittest.TestCase):
    def setUp(self):
        random.seed(21)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def test_direct_question_creates_hot_thread_with_explicit_respondent(self):
        _observe(
            self.runner, self.state, "p1",
            "Jonas, what do you think about the Museum?",
            intent=MoveIntent(
                speaker_id="p1", act=ActType.ASK, reason="ask",
                addressee_id="p2", option_focus=["A"],
            ),
        )
        threads = _question_threads(self.state)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        self.assertEqual(thread.status, ThreadStatus.HOT)
        self.assertEqual(thread.question_scope, "direct")
        self.assertEqual(thread.required_respondent, "p2")
        self.assertIn("A", thread.focus_options)

    def test_group_question_has_no_assigned_respondent(self):
        # Group questions are answered through self-selection: no controller-
        # assigned required respondent (todo 12).
        _observe(
            self.runner, self.state, "p1",
            "Which option is actually cheapest for all of us?",
            intent=MoveIntent(speaker_id="p1", act=ActType.ASK, reason="ask"),
        )
        threads = _question_threads(self.state)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        self.assertEqual(thread.question_scope, "group")
        self.assertIsNone(thread.required_respondent)

    def test_failed_turn_creates_no_thread(self):
        # A dropped decision turn (hard-blocked required vote) must not leave
        # question threads behind even if its text contained a question.
        self.state.runtimes["p1"].mark_rejected("C", reason_against="booked rooms")
        runner = make_runner(self.state, [
            "Jonas, should we do the Escape Room?",
            "Jonas, should we do the Escape Room?",
        ])
        record = runner._generate_and_append(
            self.state,
            MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="vote", option_focus=["C"], required_vote="C"),
        )
        self.assertTrue(record.state_mutation_blocked)
        self.assertEqual(_question_threads(self.state), [])

    def test_voting_and_repair_questions_open_no_threads(self):
        # Closeout 1: the bounded decision flow owns its own question/answer
        # exchanges — they never become ordinary question threads that could
        # sit falsely hot/cooling at closing.
        for phase in (Phase.VOTING, Phase.COMPROMISE_REPAIR):
            state = make_state()
            runner = make_runner(state)
            state.phase = phase
            _observe(
                runner, state, "p1",
                "Jonas, what would make the Escape Room workable for you?",
                intent=MoveIntent(
                    speaker_id="p1", act=ActType.ASK, reason="repair probe",
                    addressee_id="p2", option_focus=["C"],
                    route_source="majority_holdout_repair",
                ),
            )
            self.assertEqual(_question_threads(state), [])
            _observe(
                runner, state, "p2",
                "A later slot would make the Escape Room workable for me.",
                intent=MoveIntent(
                    speaker_id="p2", act=ActType.ANSWER, reason="repair answer",
                    route_source="majority_holdout_repair", option_focus=["C"],
                ),
            )
            active = [
                t for t in state.threads.values()
                if t.status in (ThreadStatus.HOT, ThreadStatus.COOLING)
            ]
            self.assertEqual(active, [], f"leftover active thread in {phase}")

    def test_discussion_questions_still_open_threads(self):
        self.state.phase = Phase.DISCUSSION
        _observe(
            self.runner, self.state, "p1",
            "Jonas, what do you think about the Museum?",
            intent=MoveIntent(
                speaker_id="p1", act=ActType.ASK, reason="ask",
                addressee_id="p2", option_focus=["A"],
            ),
        )
        self.assertEqual(len(_question_threads(self.state)), 1)


class QuestionRoutingTests(unittest.TestCase):
    def setUp(self):
        random.seed(22)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def test_direct_question_creates_mandatory_answer_obligation(self):
        _observe(
            self.runner, self.state, "p1",
            "Jonas, what do you think about the Museum?",
            intent=MoveIntent(
                speaker_id="p1", act=ActType.ASK, reason="ask",
                addressee_id="p2", option_focus=["A"],
            ),
        )
        obligation = self.runner._pending_answer_obligation(self.state)
        self.assertIsNotNone(obligation)
        self.assertEqual(obligation.participant_id, "p2")
        self.assertEqual(obligation.act, ActType.ANSWER)
        thread = _question_threads(self.state)[0]
        self.assertEqual(obligation.respond_to_turn, thread.source_turn_index)

    def test_answer_obligation_takes_priority_over_open_floor(self):
        # A pending direct-answer obligation forces the named respondent before
        # any open-floor bidding round (the discussion loop checks it first).
        self.state.coverage["C"].mentions = 0
        for _ in range(4):
            append_turn(self.state, "p1", "I still like the Museum best.")
            append_turn(self.state, "p2", "The Bike Ride keeps cost low.")
        _observe(
            self.runner, self.state, "p3",
            "Mira, does the Museum really fill a whole Saturday?",
            intent=MoveIntent(
                speaker_id="p3", act=ActType.ASK, reason="ask",
                addressee_id="p1", option_focus=["A"],
            ),
        )
        for seed in range(15):
            random.seed(seed)
            obligation = self.runner._pending_answer_obligation(self.state)
            self.assertIsNotNone(obligation)
            self.assertEqual(obligation.participant_id, "p1")
            self.assertEqual(obligation.act, ActType.ANSWER)


class QuestionResolutionTests(unittest.TestCase):
    def setUp(self):
        random.seed(23)
        self.state = make_state()
        self.runner = make_runner(self.state)
        _observe(
            self.runner, self.state, "p1",
            "Jonas, what do you think about the Museum?",
            intent=MoveIntent(
                speaker_id="p1", act=ActType.ASK, reason="ask",
                addressee_id="p2", option_focus=["A"],
            ),
        )
        self.thread = _question_threads(self.state)[0]

    def test_relevant_answer_moves_thread_to_cooling(self):
        answer_intent = _answer_intent(self.state, self.thread)
        _observe(self.runner, self.state, "p2", "The Museum seems fine for the calm pace.", intent=answer_intent)
        self.assertEqual(self.thread.status, ThreadStatus.COOLING)

    def test_unrelated_turn_by_respondent_does_not_resolve(self):
        _observe(self.runner, self.state, "p2", "The Escape Room is at least memorable.")
        self.assertEqual(self.thread.status, ThreadStatus.HOT)

    def test_other_speaker_does_not_resolve(self):
        _observe(self.runner, self.state, "p3", "The Museum keeps things simple, honestly.")
        self.assertEqual(self.thread.status, ThreadStatus.HOT)

    def test_one_answer_does_not_close_other_assigned_questions(self):
        _observe(
            self.runner, self.state, "p3",
            "Jonas, would the Escape Room booking bother you?",
            intent=MoveIntent(
                speaker_id="p3", act=ActType.ASK, reason="ask",
                addressee_id="p2", option_focus=["C"],
            ),
        )
        threads = sorted(_question_threads(self.state), key=lambda t: t.created_turn)
        self.assertEqual(len(threads), 2)
        first, second = threads
        answer_intent = _answer_intent(self.state, first)
        _observe(self.runner, self.state, "p2", "The Museum seems fine for the calm pace.", intent=answer_intent)
        self.assertEqual(first.status, ThreadStatus.COOLING)
        self.assertEqual(second.status, ThreadStatus.HOT)

    def test_answered_question_resolves_after_quiet_cooling_window(self):
        answer_intent = _answer_intent(self.state, self.thread)
        _observe(self.runner, self.state, "p2", "The Museum seems fine for the calm pace.", intent=answer_intent)
        _observe(self.runner, self.state, "p3", "Fair enough, I can see that.")
        _observe(self.runner, self.state, "p1", "Good, that settles my worry.")
        self.assertEqual(self.thread.status, ThreadStatus.RESOLVED)

    def test_routed_answer_with_unrelated_evidence_does_not_resolve(self):
        # Closeout 2 / item 5: the routed act alone never cools a question.
        # Whether the answer addressed its target is the VALIDATOR's judgment,
        # carried in the accepted evidence — an accepted line whose answer
        # evidence says "unrelated" (and that names no focus option) leaves
        # the thread hot.
        from models import AnswerEvidence, EvidenceSpan, VisibleEvidence

        answer_intent = _answer_intent(self.state, self.thread)
        text = "Weekends always fill up fast around here."
        record = append_turn(self.state, "p2", text, intent=answer_intent)
        record.evidence = VisibleEvidence(
            utterance=text,
            answers=[AnswerEvidence(
                completeness="unrelated", span=EvidenceSpan(text=text, start=0),
                addresses_target=False,
            )],
        )
        self.runner._apply_semantics(self.state, record)
        self.assertEqual(self.thread.status, ThreadStatus.HOT)

    def test_addressed_answer_resolves_focusless_question(self):
        # A group question with no parseable option focus is still answerable:
        # the routed reply's accepted answer evidence addresses the target.
        random.seed(24)
        state = make_state()
        runner = make_runner(state)
        _observe(
            runner, state, "p1",
            "Which of these actually stays inside our budget?",
            intent=MoveIntent(speaker_id="p1", act=ActType.ASK, reason="ask"),
        )
        thread = _question_threads(state)[0]
        self.assertEqual(thread.focus_options, [])
        self.assertIsNone(thread.required_respondent)  # group question: self-selection
        # Any eligible sim may answer a group question; p2 self-selects here.
        respondent = "p2"
        answer_intent = _answer_intent(state, thread, respondent=respondent)
        answer_intent.option_focus = []
        from models import AnswerEvidence, EvidenceSpan, VisibleEvidence
        answer = VisibleEvidence(answers=[AnswerEvidence(
            completeness="full", addresses_target=True,
            span=EvidenceSpan(text="everything here stays under the sixty euro budget", start=10),
        )])
        record = append_turn(
            state, respondent, "Cost-wise everything here stays under the sixty euro budget.",
            intent=answer_intent, evidence=answer,
        )
        runner._apply_semantics(state, record)
        self.assertEqual(thread.status, ThreadStatus.COOLING)


if __name__ == "__main__":
    unittest.main()
