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


if __name__ == "__main__":
    unittest.main()
