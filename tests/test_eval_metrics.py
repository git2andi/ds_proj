"""Tests for the baseline evaluation-metric fixes (todo section 16.3)."""

from __future__ import annotations

import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from eval.eval import metrics_for
from models import ActType, MoveIntent, RunOutcome

from tests.fixtures import append_turn, make_state


def _outcome(state) -> RunOutcome:
    return RunOutcome("unresolved", None, "test", len(state.turns))


def _coverage_intent(speaker_id: str, option_id: str) -> MoveIntent:
    # Mirrors the router's coverage intent (policy._route_discussion_turn).
    return MoveIntent(
        speaker_id=speaker_id,
        act=ActType.COMPARE,
        reason="briefly bring in an option that has not yet been socially processed, then compare it with the current lean",
        route_source="coverage",
        option_focus=[option_id],
    )


class StanceRankDistributionTests(unittest.TestCase):
    def test_distribution_covers_ranks_one_to_five(self):
        state = make_state()
        state.runtimes["p1"].set_rank("B", 1)
        metrics = metrics_for(state, _outcome(state))
        distribution = metrics["stance_rank_distribution"]
        self.assertEqual(set(distribution.keys()), {"1", "2", "3", "4", "5"})
        # Three personas each prefer one option (rank 5); p1 rejects B (rank 1).
        self.assertEqual(distribution["5"], 3)
        self.assertEqual(distribution["1"], 1)
        total_ranks = sum(distribution.values())
        self.assertEqual(total_ranks, sum(len(rt.option_ranks) for rt in state.runtimes.values()))


class CoverageSelectedVsRealizedTests(unittest.TestCase):
    def test_selected_and_realized_counted_separately(self):
        state = make_state()
        # Router selected two coverage routes for option C.
        state.coverage["C"].coverage_attempts = 2
        # First coverage turn realized the option visibly.
        append_turn(state, "p1", "We haven't talked about the Escape Room; it is memorable.", intent=_coverage_intent("p1", "C"))
        # Second coverage turn drifted and never named option C.
        append_turn(state, "p2", "I still think the Museum is the easy pick.", intent=_coverage_intent("p2", "C"))
        metrics = metrics_for(state, _outcome(state))
        self.assertEqual(metrics["coverage_routes_selected"], 2)
        self.assertEqual(metrics["coverage_turns_realized"], 1)

    def test_blocked_coverage_turn_does_not_realize(self):
        state = make_state()
        state.coverage["C"].coverage_attempts = 1
        append_turn(
            state,
            "p1",
            "We haven't talked about the Escape Room yet.",
            intent=_coverage_intent("p1", "C"),
            blocked=True,
        )
        metrics = metrics_for(state, _outcome(state))
        self.assertEqual(metrics["coverage_routes_selected"], 1)
        self.assertEqual(metrics["coverage_turns_realized"], 0)


if __name__ == "__main__":
    unittest.main()
