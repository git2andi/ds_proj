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


class QuestionAnswerCompletionTests(unittest.TestCase):
    """Closeout 6: the answer search must include the immediately following turn."""

    def _ask(self, state, asker, text, addressee):
        return append_turn(
            state, asker, text,
            intent=MoveIntent(speaker_id=asker, act=ActType.ASK, reason="ask", addressee_id=addressee),
        )

    def test_immediate_answer_counts(self):
        state = make_state()
        self._ask(state, "p1", "Jonas, what do you think about the Museum?", "p2")
        append_turn(state, "p2", "The Museum works fine for a calm day.")
        metrics = metrics_for(state, _outcome(state))
        self.assertEqual(metrics["question_answer_completion"], 1.0)
        self.assertEqual(metrics["open_questions_at_end"], 0)

    def test_earlier_turns_and_question_turn_do_not_count(self):
        state = make_state()
        append_turn(state, "p2", "The Bike Ride keeps the cost low.")  # BEFORE the question
        self._ask(state, "p1", "Jonas, what do you think about the Museum?", "p2")
        metrics = metrics_for(state, _outcome(state))
        self.assertEqual(metrics["question_answer_completion"], 0.0)
        self.assertEqual(metrics["open_questions_at_end"], 1)

    def test_answer_outside_window_is_not_prompt(self):
        state = make_state()
        self._ask(state, "p1", "Jonas, what do you think about the Museum?", "p2")
        for _ in range(5):  # window is 2 * question_answer_window_turns = 4
            append_turn(state, "p3", "Still weighing the options here.")
        append_turn(state, "p2", "The Museum works fine for a calm day.")
        metrics = metrics_for(state, _outcome(state))
        self.assertEqual(metrics["question_answer_completion"], 0.0)  # late
        self.assertEqual(metrics["open_questions_at_end"], 0)         # but answered


if __name__ == "__main__":
    unittest.main()
