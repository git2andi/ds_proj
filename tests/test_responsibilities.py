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
        act = parse_text(self.state, speaker_id, text, intent=intent)
        persona = self.state.persona_by_id(speaker_id)
        return self.runner._validate_turn_text(text, self.state, persona, intent, act)

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
    def test_fallback_answer_does_not_resolve_question_thread(self):
        random.seed(102)
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=["C"], issue_key="cost",
            started_by="p1", source_turn_index=0, required_respondent="p2", question_scope="direct",
        )
        runner = make_runner(state)
        answer_intent = runner._answer_intent_for_thread(state, thread)
        # Both generation attempts evade the question entirely; validation
        # flags them and the deterministic fallback replaces the line.
        runner._llm.responses.extend([
            "Well, what does everyone else think about all this?",
            "Honestly, who can say anything about any of this?",
        ])
        record = runner._generate_and_append(state, answer_intent)
        self.assertTrue(record.used_fallback)
        self.assertEqual(thread.status, ThreadStatus.HOT)  # not resolved by fallback


class ParserPurityTests(unittest.TestCase):
    def test_parse_does_not_mutate_state(self):
        state = make_state()
        before = (len(state.threads), state.turn_index, len(state.turns))
        parse_text(state, "p1", "Jonas, what do you think about the Museum?")
        self.assertEqual((len(state.threads), state.turn_index, len(state.turns)), before)


if __name__ == "__main__":
    unittest.main()
