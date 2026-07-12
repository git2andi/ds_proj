"""Tests for the simplified normal act set and `comment` semantics (TODO 11)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent

from tests.fixtures import append_turn, make_state
from tests.stubs import make_runner


class NormalActSamplerTests(unittest.TestCase):
    def test_sampler_limited_to_defined_normal_set(self):
        state = make_state()
        runner = make_runner(state)
        append_turn(state, "p1", "I like the Museum for the calm pace.")
        allowed = {ActType.SUPPORT, ActType.CONCERN, ActType.ASK, ActType.COMPARE, ActType.COMMENT}
        seen = set()
        for seed in range(300):
            random.seed(seed)
            act = runner._choose_discussion_act(state, state.personas[seed % 3])
            self.assertIn(act, allowed)
            seen.add(act)
        # The sampler actually uses the breadth of the set, including comment.
        self.assertIn(ActType.COMMENT, seen)
        self.assertGreaterEqual(len(seen), 4)

    def test_no_random_roll_selects_softening_compromise_or_process(self):
        state = make_state()
        runner = make_runner(state)
        for i in range(6):
            append_turn(state, ("p1", "p2", "p3")[i % 3], "Still weighing the Museum against the Bike Ride here.")
        # SOFTEN_TOWARD no longer exists as an act at all; softening is only
        # an observed stance effect (act.softens_toward).
        forbidden = {ActType.COMPROMISE, ActType.PROCESS, ActType.ANSWER, ActType.VOTE}
        for seed in range(200):
            random.seed(seed)
            intent = runner._route_discussion_turn(state)
            if intent.route_source == "normal":
                self.assertNotIn(intent.act, forbidden)


class CommentSemanticsTests(unittest.TestCase):
    def test_comment_creates_no_thread_and_no_support(self):
        state = make_state()
        runner = make_runner(state, ["Honestly, ease matters more to me than excitement with the Museum."])
        random.seed(61)
        intent = MoveIntent(
            speaker_id="p2", act=ActType.COMMENT, reason="state a priority", option_focus=["A"]
        )
        runner._generate_and_append(state, intent)
        self.assertEqual(state.threads, {})
        self.assertEqual(state.coverage["A"].reasons, 0)       # not support
        self.assertEqual(state.coverage["A"].acceptances, 0)
        self.assertGreaterEqual(state.coverage["A"].mentions, 1)  # still a mention
        self.assertIsNone(state.runtimes["p2"].explicit_vote)

    def test_comment_with_parsed_question_still_opens_question_thread(self):
        state = make_state()
        runner = make_runner(state, ["Fair point — but which of us actually checked the Museum hours?"])
        random.seed(62)
        intent = MoveIntent(speaker_id="p2", act=ActType.COMMENT, reason="light beat", option_focus=["A"])
        runner._generate_and_append(state, intent)
        # Parsing independently detected a genuine group question.
        self.assertEqual(len(state.threads), 1)


class CoverageEvidenceTests(unittest.TestCase):
    """Closeout 5: coverage updates from per-option semantic evidence, not
    only the dominant realized act label."""

    def setUp(self):
        random.seed(63)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def _apply(self, pid, text, intent):
        from tests.fixtures import append_turn

        record = append_turn(self.state, pid, text, intent=intent)
        self.runner._apply_semantics(self.state, record)
        return record

    def test_comparative_question_counts_comparison_evidence(self):
        intent = MoveIntent(speaker_id="p2", act=ActType.ASK, reason="ask", option_focus=["A", "B"])
        record = self._apply(
            "p2", "Back to the Museum versus the Bike Ride: which actually fits a tired group?", intent
        )
        self.assertEqual(record.realized_act(), ActType.ASK)  # question wins the label
        self.assertEqual(self.state.coverage["A"].reasons, 1)
        self.assertEqual(self.state.coverage["B"].reasons, 1)

    def test_answer_containing_objection_counts_objection(self):
        intent = MoveIntent(speaker_id="p2", act=ActType.ANSWER, reason="answer", option_focus=["C"])
        self._apply("p2", "Honestly the Escape Room cost bothers me.", intent)
        self.assertEqual(self.state.coverage["C"].objections, 1)
        self.assertEqual(self.state.coverage["C"].reasons, 0)

    def test_opening_with_support_and_concern_splits_evidence(self):
        intent = MoveIntent(speaker_id="p1", act=ActType.OPENING, reason="open", option_focus=["A"])
        self._apply(
            "p1", "The Museum seems too expensive for what it offers, while the Bike Ride fits us much better.", intent
        )
        self.assertEqual(self.state.coverage["A"].objections, 1)
        self.assertEqual(self.state.coverage["A"].reasons, 0)
        self.assertEqual(self.state.coverage["B"].reasons, 1)
        self.assertEqual(self.state.coverage["B"].objections, 0)

    def test_neutral_mention_counts_mention_only(self):
        intent = MoveIntent(speaker_id="p2", act=ActType.COMMENT, reason="light beat", option_focus=["A"])
        self._apply("p2", "The Museum is on the list, sure.", intent)
        self.assertEqual(self.state.coverage["A"].mentions, 1)
        self.assertEqual(self.state.coverage["A"].reasons, 0)
        self.assertEqual(self.state.coverage["A"].objections, 0)

    def test_ordinary_support_and_comparison_turns_count_reasons(self):
        self._apply(
            "p1", "The Museum keeps the day easy to adjust.",
            MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support", option_focus=["A"]),
        )
        self.assertEqual(self.state.coverage["A"].reasons, 1)
        self._apply(
            "p2", "The Museum beats the Bike Ride on flexibility for us.",
            MoveIntent(speaker_id="p2", act=ActType.COMPARE, reason="compare", option_focus=["A", "B"]),
        )
        self.assertEqual(self.state.coverage["A"].reasons, 2)
        self.assertEqual(self.state.coverage["B"].reasons, 1)


if __name__ == "__main__":
    unittest.main()
