"""Stance movement: rank map only, moved by visible accepted text alone
(todo_prompt item 5), with discussion-phase lean shifts observable."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent, Phase, STANCE_ACCEPTABLE, STANCE_PREFERRED, STANCE_REJECTED

from tests.fixtures import make_persona, make_state
from tests.stubs import make_runner


def _discussion_state(**persona_kwargs):
    state = make_state(personas=[
        make_persona("p1", "Mira", preferred="A", **persona_kwargs),
        make_persona("p2", "Jonas", preferred="B"),
        make_persona("p3", "Lea", preferred="C"),
    ])
    state.phase = Phase.DISCUSSION
    return state


class DiscussionLeanShifts(unittest.TestCase):
    def test_visible_softening_promotes_and_counts_a_shift(self):
        state = _discussion_state()
        runner = make_runner(state, ["The Bike Ride is really growing on me, cost-wise."])
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="react", option_focus=["B"])
        )
        self.assertEqual(state.runtimes["p1"].top_option(), "B")
        self.assertEqual(state.discussion_lean_shifts, 1)

    def test_visible_discussion_commitment_that_moves_the_top_option_counts(self):
        # This path was previously invisible to the metric: a clear visible
        # commitment during discussion changed the lean without counting.
        state = _discussion_state()
        runner = make_runner(state, [
            "Let's go with the Bike Ride — I liked the Museum, but the lower cost wins for me."
        ])
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support", option_focus=["B"])
        )
        self.assertEqual(state.runtimes["p1"].top_option(), "B")
        self.assertEqual(state.discussion_lean_shifts, 1)

    def test_mere_acceptance_keeps_the_lean_and_counts_nothing(self):
        state = _discussion_state()
        runner = make_runner(state, ["The Bike Ride works for me too, for what it's worth."])
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="react", option_focus=["B"])
        )
        rt = state.runtimes["p1"]
        self.assertEqual(rt.top_option(), "A")                      # lean unchanged
        self.assertGreaterEqual(rt.rank("B"), STANCE_ACCEPTABLE)    # acceptance recorded
        self.assertEqual(state.discussion_lean_shifts, 0)

    def test_promotion_demotes_the_former_preferred_option(self):
        state = _discussion_state()
        rt = state.runtimes["p1"]
        rt.promote_to_preferred("B")
        self.assertEqual(rt.rank("B"), STANCE_PREFERRED)
        self.assertEqual(rt.rank("A"), STANCE_ACCEPTABLE)


class NoHiddenMovement(unittest.TestCase):
    def test_blocked_vote_for_rejected_option_moves_nothing(self):
        random.seed(4)
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", rejection="B", rejection_reason="too risky"),
            make_persona("p2", "Jonas", preferred="B"),
        ])
        state.phase = Phase.DISCUSSION
        runner = make_runner(state, [
            "I vote for the Bike Ride.",   # accepts own rejected option -> blocked
            "I vote for the Bike Ride.",   # repair attempt, still blocked
        ])
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support", option_focus=["A"])
        )
        rt = state.runtimes["p1"]
        self.assertEqual(rt.rank("B"), STANCE_REJECTED)   # hard rejection intact
        self.assertNotEqual(rt.explicit_vote, "B")
        self.assertEqual(state.discussion_lean_shifts, 0)

    def test_acceptance_of_a_rejected_option_is_ignored_without_resolution(self):
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", rejection="C", rejection_reason="too rigid"),
            make_persona("p2", "Jonas", preferred="B"),
        ])
        state.phase = Phase.DISCUSSION
        runner = make_runner(state, ["The Escape Room works for me after all."])
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="react", option_focus=["C"])
        )
        self.assertEqual(state.runtimes["p1"].rank("C"), STANCE_REJECTED)

    def test_controller_intent_alone_never_changes_a_rank(self):
        state = _discussion_state()
        before = dict(state.runtimes["p1"].option_ranks)
        runner = make_runner(state, ["Fair enough, that all sounds reasonable to me."])
        # Routed as a compromise toward B, but the text shows no visible signal.
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p1", act=ActType.COMPROMISE, reason="propose B", option_focus=["B"])
        )
        self.assertEqual(state.runtimes["p1"].option_ranks, before)
        self.assertEqual(state.discussion_lean_shifts, 0)


if __name__ == "__main__":
    unittest.main()
