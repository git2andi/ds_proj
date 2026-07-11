"""Mismatch tests: when routed intent and final text differ, text wins (cleanup 2).

Each case routes one act but scripts a final line that visibly realizes a
different one. State (threads, leans, votes, coverage) must follow the parsed
text, never the routed intent.
"""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent, ThreadType

from tests.fixtures import append_turn, make_state, vote_intent
from tests.stubs import make_runner


def _threads_of(state, *types):
    return [t for t in state.threads.values() if t.thread_type in types]


class IntentTextMismatchTests(unittest.TestCase):
    def setUp(self):
        random.seed(71)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def _apply(self, speaker_id, text, intent):
        record = append_turn(self.state, speaker_id, text, intent=intent)
        self.runner._apply_semantics(self.state, record)
        return record

    def test_routed_concern_without_visible_objection_opens_nothing(self):
        intent = MoveIntent(speaker_id="p2", act=ActType.CONCERN, reason="push back", option_focus=["A"])
        record = self._apply("p2", "The Museum is on everyone's list already anyway.", intent)
        self.assertEqual(_threads_of(self.state, ThreadType.CONCERN, ThreadType.BLOCKER), [])
        self.assertNotEqual(record.act.act_type, ActType.CONCERN)
        self.assertEqual(self.state.coverage["A"].objections, 0)

    def test_routed_support_with_visible_objection_opens_concern(self):
        intent = MoveIntent(speaker_id="p2", act=ActType.SUPPORT, reason="back it", option_focus=["C"])
        record = self._apply("p2", "The Escape Room cost worries me.", intent)
        threads = _threads_of(self.state, ThreadType.CONCERN)
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].focus_options, ["C"])
        self.assertEqual(record.act.act_type, ActType.CONCERN)

    def test_routed_compare_without_visible_contrast_opens_no_pair_thread(self):
        intent = MoveIntent(speaker_id="p2", act=ActType.COMPARE, reason="weigh", option_focus=["A", "B"])
        record = self._apply("p2", "The Museum would be a calm start to the day.", intent)
        self.assertEqual(_threads_of(self.state, ThreadType.COMPARISON), [])
        self.assertNotEqual(record.act.act_type, ActType.COMPARE)

    def test_plain_mention_without_benefit_claim_stays_comment(self):
        intent = MoveIntent(speaker_id="p2", act=ActType.SUPPORT, reason="back it", option_focus=["A"])
        record = self._apply("p2", "The Museum is on the list, sure.", intent)
        self.assertEqual(record.act.act_type, ActType.COMMENT)
        self.assertEqual(self.state.coverage["A"].reasons, 0)

    def test_opening_lean_follows_named_option_not_routed_focus(self):
        intent = MoveIntent(speaker_id="p1", act=ActType.OPENING, reason="open", option_focus=["A"])
        self._apply("p1", "First thought: the Bike Ride keeps things active and cheap.", intent)
        self.assertEqual(self.state.runtimes["p1"].top_option(), "B")

    def test_ambiguous_opening_naming_moves_no_lean(self):
        intent = MoveIntent(speaker_id="p3", act=ActType.OPENING, reason="open", option_focus=["C"])
        self._apply("p3", "The Museum and the Bike Ride both have something going for them.", intent)
        self.assertEqual(self.state.runtimes["p3"].top_option(), "C")  # unchanged initial preference

    def test_routed_vote_without_visible_commitment_records_no_vote(self):
        self._apply("p2", "I'm honestly still torn on this one.", vote_intent("p2", "B"))
        self.assertIsNone(self.state.runtimes["p2"].explicit_vote)

    def test_visible_softening_moves_discussion_lean(self):
        # p2 prefers B; the softening wording alone moves the latent lean to A.
        intent = MoveIntent(speaker_id="p2", act=ActType.SUPPORT, reason="react")
        self._apply("p2", "Honestly, the Museum is growing on me after that.", intent)
        self.assertEqual(self.state.runtimes["p2"].top_option(), "A")
        self.assertEqual(self.state.discussion_lean_shifts, 1)

    def test_visible_conditional_support_can_move_discussion_lean(self):
        # A fully flexible sim (stubbornness 0) always follows a parsed
        # conditional-support signal in discussion.
        from tests.fixtures import make_persona

        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A"),
            make_persona("p2", "Jonas", preferred="B", stubbornness=0.0),
            make_persona("p3", "Lea", preferred="C"),
        ])
        runner = make_runner(state)
        intent = MoveIntent(speaker_id="p2", act=ActType.SUPPORT, reason="react")
        record = append_turn(state, "p2", "I can support the Museum if we keep the day short.", intent=intent)
        self.assertEqual(record.act.conditional_support, "A")
        runner._apply_semantics(state, record)
        self.assertEqual(state.runtimes["p2"].top_option(), "A")
        self.assertEqual(state.discussion_lean_shifts, 1)


if __name__ == "__main__":
    unittest.main()
