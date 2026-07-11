"""Tests for the deterministic progress signature and coverage gating (TODO 12)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent
from controller.threads import mark_response, open_thread
from models import ThreadType

from tests.fixtures import make_state
from tests.stubs import make_runner


def _observe(runner, state, speaker_id, text, *, intent=None):
    runner._llm.responses.append(text)
    intent = intent or MoveIntent(speaker_id=speaker_id, act=ActType.COMMENT, reason="light beat")
    return runner._generate_and_append(state, intent)


class ProgressSignatureTests(unittest.TestCase):
    def setUp(self):
        random.seed(71)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def test_generic_comment_does_not_reset_progress(self):
        self.state.no_progress_count = 3
        _observe(self.runner, self.state, "p1", "Right, that's roughly where we are, I guess.")
        self.assertEqual(self.state.no_progress_count, 4)

    def test_new_concern_thread_resets_progress(self):
        self.state.no_progress_count = 3
        _observe(
            self.runner, self.state, "p2",
            "The Museum cost worries me a bit.",
            intent=MoveIntent(speaker_id="p2", act=ActType.CONCERN, reason="push back", option_focus=["A"]),
        )
        self.assertEqual(self.state.no_progress_count, 0)

    def test_answered_question_resets_progress(self):
        thread = open_thread(
            self.state, thread_type=ThreadType.QUESTION, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=0, required_respondent="p2", question_scope="direct",
        )
        self.state.no_progress_count = 2
        answer_intent = self.runner._answer_intent_for_thread(self.state, thread)
        _observe(self.runner, self.state, "p2", "The Museum cost is fine by me.", intent=answer_intent)
        self.assertEqual(self.state.no_progress_count, 0)

    def test_visible_vote_resets_progress(self):
        self.state.no_progress_count = 2
        _observe(
            self.runner, self.state, "p1",
            "I vote for the Museum.",
            intent=MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="vote", option_focus=["A"], length_hint="short"),
        )
        self.assertEqual(self.state.no_progress_count, 0)


class CoverageGatingTests(unittest.TestCase):
    def test_coverage_skipped_near_hard_cap(self):
        state = make_state()
        runner = make_runner(state)
        state.hard_max_turns = 10
        # Enough turns that coverage is normally eligible, but at the cap edge.
        for i in range(9):
            speaker = ("p1", "p2", "p3")[i % 3]
            from tests.fixtures import append_turn
            append_turn(state, speaker, "Still talking about the Museum against the Bike Ride.")
        self.assertIsNone(runner._coverage_gap_option(state))

    def test_coverage_eligible_when_turns_remain(self):
        state = make_state()
        runner = make_runner(state)
        state.hard_max_turns = 30
        state.coverage["A"].mentions = 2
        state.coverage["B"].mentions = 2
        from tests.fixtures import append_turn
        for i in range(5):
            speaker = ("p1", "p2", "p3")[i % 3]
            append_turn(state, speaker, "Still talking about the Museum against the Bike Ride.")
        self.assertEqual(runner._coverage_gap_option(state), "C")


if __name__ == "__main__":
    unittest.main()
