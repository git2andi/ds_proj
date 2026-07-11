"""Anti-repetition flags: adaptive, evidence-based, at most one per turn (item 8)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent

from tests.fixtures import append_turn, make_state
from tests.stubs import make_runner


def _lexical_flags(intent: MoveIntent) -> list[str]:
    flags = []
    if intent.avoid_pattern:
        flags.append("avoid_pattern")
    for name in (
        "vary_opening",
        "suppress_option_opening",
        "suppress_name_prefix",
        "suppress_i_opening",
        "suppress_we_opening",
    ):
        if getattr(intent, name):
            flags.append(name)
    return flags


class StyleFlagTests(unittest.TestCase):
    def setUp(self):
        random.seed(3)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def _intent(self, act=ActType.SUPPORT, **kwargs) -> MoveIntent:
        return MoveIntent(speaker_id="p1", act=act, reason="say it", **kwargs)

    def test_fresh_conversation_sets_no_variation_flags(self):
        # No recent turns -> no evidence -> no notes (no proactive random damping).
        for _ in range(50):
            intent = self._intent()
            self.runner._apply_style_flags(self.state, intent)
            self.assertEqual(_lexical_flags(intent), [])
            self.assertFalse(intent.suppress_name_prefix)

    def test_multiple_tripwires_yield_exactly_one_note(self):
        # Every recent turn: starts with "I", opens on the same word, and is
        # an "I worry ... but" shape -> several tripwires fire at once.
        for speaker in ("p1", "p2", "p3", "p1"):
            append_turn(
                self.state, speaker,
                "I worry the Escape Room is pricey, but the vibe is great.",
            )
        intent = self._intent()
        self.runner._apply_style_flags(self.state, intent)
        flags = _lexical_flags(intent)
        self.assertEqual(len(flags), 1, flags)
        # The rhetorical-shape pattern outranks opening-word notes.
        self.assertEqual(flags, ["avoid_pattern"])

    def test_repeated_opening_word_gets_the_vary_opening_note(self):
        for speaker, text in (
            ("p1", "Honestly the Museum day is fine."),
            ("p2", "Honestly the cost matters more."),
            ("p3", "Honestly we should just decide."),
        ):
            append_turn(self.state, speaker, text)
        intent = self._intent(act=ActType.COMMENT)
        self.runner._apply_style_flags(self.state, intent)
        self.assertEqual(_lexical_flags(intent), ["vary_opening"])

    def test_functional_naming_is_never_suppressed(self):
        for speaker, text in (
            ("p1", "Lea, the Museum keeps it easy."),
            ("p2", "Mira, the cost matters."),
            ("p3", "Jonas, what do you prefer here."),
            ("p1", "Lea, that works."),
        ):
            append_turn(self.state, speaker, text)
        intent = self._intent(act=ActType.ASK, addressee_id="p2")
        self.runner._apply_style_flags(self.state, intent)
        self.assertFalse(intent.suppress_name_prefix)

    def test_tail_question_flag_is_independent_flow_control(self):
        for speaker, text in (
            ("p1", "Is the Museum too quiet for us?"),
            ("p2", "Would the Bike Ride tire anyone out?"),
        ):
            append_turn(self.state, speaker, text)
        intent = self._intent(act=ActType.COMMENT)
        self.runner._apply_style_flags(self.state, intent)
        self.assertTrue(intent.suppress_tail_question)


if __name__ == "__main__":
    unittest.main()
