"""Baseline tests for the option-rank stance model (single source of truth)."""

from __future__ import annotations

import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import (
    ParticipantRuntime,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
)

from tests.fixtures import make_state


class RankTableTests(unittest.TestCase):
    def _runtime(self) -> ParticipantRuntime:
        return ParticipantRuntime(
            persona_id="p1",
            option_ranks={"A": STANCE_PREFERRED, "B": STANCE_NEUTRAL, "C": STANCE_NEUTRAL},
        )

    def test_set_rank_clips_to_valid_range(self):
        rt = self._runtime()
        rt.set_rank("A", 99)
        self.assertEqual(rt.rank("A"), STANCE_PREFERRED)
        rt.set_rank("A", -3)
        self.assertEqual(rt.rank("A"), STANCE_REJECTED)

    def test_top_option_prefers_fallback_on_tie(self):
        rt = ParticipantRuntime(persona_id="p1", option_ranks={"A": 4, "B": 4})
        self.assertEqual(rt.top_option(fallback="B"), "B")
        self.assertEqual(rt.top_option(), "A")  # deterministic tie-break

    def test_bucket_helpers_read_ranks(self):
        rt = ParticipantRuntime(
            persona_id="p1",
            option_ranks={
                "A": STANCE_PREFERRED,
                "B": STANCE_ACCEPTABLE,
                "C": STANCE_DISLIKED,
                "D": STANCE_REJECTED,
            },
        )
        self.assertEqual(rt.top_option(), "A")
        self.assertEqual(rt.acceptable_options(), {"B"})
        self.assertEqual(rt.disliked_options(), {"C"})
        self.assertEqual(rt.rejected_options(), {"D"})

    def test_promote_to_preferred_demotes_old_favorite(self):
        rt = self._runtime()
        rt.promote_to_preferred("B")
        self.assertEqual(rt.rank("B"), STANCE_PREFERRED)
        self.assertEqual(rt.rank("A"), STANCE_ACCEPTABLE)

    def test_rejected_option_cannot_become_acceptable_silently(self):
        rt = self._runtime()
        rt.mark_rejected("C", reason_against="hard constraint")
        rt.mark_acceptable("C")
        self.assertEqual(rt.rank("C"), STANCE_REJECTED)


class InitialiseStateTests(unittest.TestCase):
    def test_initial_ranks_follow_persona_setup(self):
        state = make_state()
        self.assertEqual(state.runtimes["p1"].top_option(), "A")
        self.assertEqual(state.runtimes["p2"].top_option(), "B")
        self.assertEqual(state.runtimes["p3"].top_option(), "C")
        for rt in state.runtimes.values():
            for rank in rt.option_ranks.values():
                self.assertTrue(STANCE_REJECTED <= rank <= STANCE_PREFERRED)

    def test_rejection_initialises_rank_one(self):
        from tests.fixtures import make_persona

        state = make_state(
            personas=[
                make_persona("p1", "Mira", preferred="A", rejection="C", rejection_reason="booked rooms are inflexible"),
                make_persona("p2", "Jonas", preferred="B"),
            ]
        )
        self.assertEqual(state.runtimes["p1"].rank("C"), STANCE_REJECTED)
        self.assertIn("C", state.runtimes["p1"].rejected_options())


if __name__ == "__main__":
    unittest.main()
