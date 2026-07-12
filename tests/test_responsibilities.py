"""Tests for the parser/observer/validation responsibility split (TODO 15)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent, ThreadStatus, ThreadType
from controller.threads import open_thread

from tests.fixtures import make_persona, make_state, make_resolver, parse_text
from tests.stubs import make_runner


class ThreadAwareValidationTests(unittest.TestCase):
    def setUp(self):
        random.seed(101)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def _report(self, speaker_id, text, intent):
        from tests.evidence_adapter import derive_evidence
        persona = self.state.persona_by_id(speaker_id)
        assessment = self.runner._assess_candidate(
            text=text, state=self.state, persona=persona, intent=intent,
            evidence=derive_evidence(
                text, self.runner._resolver,
                speaker_id=speaker_id,
                participant_names={p.id: p.name for p in self.state.personas},
                intent=intent,
            ),
        )

        class _Report:
            issues = [i.code for i in assessment.issues]
            block_state_mutation = any(i.blocking for i in assessment.issues)

        return _Report()

    def test_evasive_routed_answer_is_flagged(self):
        intent = MoveIntent(
            speaker_id="p2", act=ActType.ANSWER, reason="answer",
            route_source="answer_required", option_focus=["A"], respond_to_turn=1,
        )
        report = self._report("p2", "Well, what does everyone else think about all this?", intent)
        self.assertIn("ANSWER_DOES_NOT_ADDRESS_QUESTION", report.issues)

    def test_real_answer_passes(self):
        intent = MoveIntent(
            speaker_id="p2", act=ActType.ANSWER, reason="answer",
            route_source="answer_required", option_focus=["A"], respond_to_turn=1,
        )
        report = self._report("p2", "The Museum honestly fits a tired Saturday best.", intent)
        self.assertNotIn("ANSWER_DOES_NOT_ADDRESS_QUESTION", report.issues)

    def test_thread_response_missing_option_is_flagged(self):
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="defend",
            route_source="thread_hot", option_focus=["A"], respond_to_turn=1,
        )
        report = self._report("p1", "The Escape Room is at least something different.", intent)
        self.assertIn("THREAD_RESPONSE_MISSES_OPTION", report.issues)

    def test_thread_comparison_missing_pair_is_flagged(self):
        intent = MoveIntent(
            speaker_id="p1", act=ActType.COMPARE, reason="compare",
            route_source="thread_hot", option_focus=["A", "B"], respond_to_turn=1,
        )
        report = self._report("p1", "The Museum keeps things calm and simple.", intent)
        self.assertIn("COMPARISON_MISSES_OPTIONS", report.issues)
        report = self._report("p1", "The Museum is calmer than the Bike Ride.", intent)
        self.assertNotIn("COMPARISON_MISSES_OPTIONS", report.issues)

    def test_validation_never_touches_threads(self):
        thread = open_thread(
            self.state, thread_type=ThreadType.QUESTION, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1, required_respondent="p2", question_scope="direct",
        )
        intent = MoveIntent(
            speaker_id="p2", act=ActType.ANSWER, reason="answer",
            route_source="answer_required", option_focus=["A"], respond_to_turn=1,
        )
        self._report("p2", "The Museum honestly fits a tired Saturday best.", intent)
        self.assertEqual(thread.status, ThreadStatus.HOT)  # only the observer moves threads


class FallbackAnswerTests(unittest.TestCase):
    def test_unanswerable_question_drops_instead_of_faking_an_answer(self):
        # The question thread points at a turn that does not exist, so no
        # truthful deterministic answer can be built: the turn is dropped
        # (item 11) and the question stays hot — never resolved by a
        # fabricated line (item 12).
        random.seed(102)
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=["C"], issue_key="cost",
            started_by="p1", source_turn_index=0, required_respondent="p2", question_scope="direct",
        )
        runner = make_runner(state)
        answer_intent = runner._answer_intent_for_thread(state, thread)
        # Both generation attempts evade the question entirely; validation
        # flags them and no truthful fallback exists.
        runner._llm.responses.extend([
            "Well, what does everyone else think about all this?",
            "Honestly, who can say anything about any of this?",
        ])
        record = runner._generate_and_append(state, answer_intent)
        self.assertFalse(record.used_fallback)
        self.assertEqual(record.text, "")
        self.assertTrue(record.state_mutation_blocked)
        self.assertEqual(thread.status, ThreadStatus.HOT)  # never resolved by a dropped turn

    def test_listed_answer_fallback_resolves_the_question(self):
        # When the asked attribute IS on the card, the deterministic answer
        # fallback is a genuine answer and may resolve the thread (item 11/12).
        random.seed(103)
        state = make_state()
        runner = make_runner(state)
        from tests.fixtures import append_turn
        question = append_turn(state, "p1", "Jonas, how much does the Escape Room cost?")
        thread = open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=["C"], issue_key="cost",
            started_by="p1", source_turn_index=question.index,
            required_respondent="p2", question_scope="direct",
        )
        answer_intent = runner._answer_intent_for_thread(state, thread)
        runner._llm.responses.extend([
            "Well, what does everyone else think about all this?",
            "Honestly, who can say anything about any of this?",
        ])
        record = runner._generate_and_append(state, answer_intent)
        self.assertTrue(record.used_fallback)
        self.assertEqual(record.fallback_family, "answer_listed")
        self.assertIn("32 euros", record.text)
        self.assertEqual(thread.status, ThreadStatus.COOLING)


class ParserPurityTests(unittest.TestCase):
    def test_parse_does_not_mutate_state(self):
        state = make_state()
        before = (len(state.threads), state.turn_index, len(state.turns))
        parse_text(state, "p1", "Jonas, what do you think about the Museum?")
        self.assertEqual((len(state.threads), state.turn_index, len(state.turns)), before)


if __name__ == "__main__":
    unittest.main()
