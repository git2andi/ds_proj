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


class SemanticContractMetricsTests(unittest.TestCase):
    """Item 15: metrics measure semantic correctness of the evidence contract."""

    def test_realized_function_and_action_metrics_from_pipeline_turns(self):
        from tests.stubs import make_runner
        state = make_state()
        runner = make_runner(state, [
            "The Museum keeps the day easy for everyone.",   # realized support
            "Which is calmer, the Museum or the Bike Ride?",  # ask realizing compare? (ask intent)
        ])
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="say it", option_focus=["A"]))
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p2", act=ActType.ASK, reason="ask", option_focus=["A", "B"]))
        metrics = metrics_for(state, _outcome(state))
        self.assertEqual(metrics["intended_function_realized_rate"], 1.0)
        self.assertEqual(metrics["intended_focus_agreement_rate"], 1.0)
        self.assertGreaterEqual(metrics["assessment_action_counts"].get("accept", 0), 1)
        self.assertEqual(metrics["validator_failure_turns"], 0)
        self.assertEqual(metrics["dropped_turn_count"], 0)
        self.assertEqual(metrics["unsupported_printed_turns"], 0)

    def test_validation_path_summary_metrics(self):
        # Item 14: per-run validator call/skip accounting and the public-
        # evidence/observer consistency check are first-class metrics.
        from tests.stubs import make_runner
        state = make_state()
        runner = make_runner(state, ["The Museum keeps the day easy for everyone."])
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="say it", option_focus=["A"]))
        metrics = metrics_for(state, _outcome(state))
        self.assertIn("validator_calls", metrics)
        self.assertIn("validator_calls_per_accepted_turn", metrics)
        self.assertIn("validation_fast_path_rate", metrics)
        self.assertIn("validator_input_share", metrics)
        self.assertEqual(metrics["vote_state_consistency_failures"], 0)
        self.assertEqual(metrics["discussion_lean_shift_turns"], [])

    def test_fallback_family_and_drop_metrics(self):
        from tests.stubs import make_runner
        state = make_state()
        runner = make_runner(state, [
            "Just to be clear.", "Just to be clear.",   # -> comparison fallback
            "Just to be clear.", "Just to be clear.",   # comment intent -> drop
        ])
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p1", act=ActType.COMPARE, reason="compare", option_focus=["A", "B"],
            route_source="thread_hot"))
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p2", act=ActType.COMMENT, reason="beat"))
        metrics = metrics_for(state, _outcome(state))
        self.assertEqual(metrics["fallback_by_family"], {"comparison": 1})
        self.assertEqual(metrics["dropped_turn_count"], 1)

    def test_unsupported_printed_turns_checks_final_accepted_claims(self):
        from models import EvidenceSpan, GroundingClaim
        state = make_state()
        record = append_turn(state, "p1", "The Museum has free entry on Saturdays.")
        record.evidence.claims.append(GroundingClaim(
            span=EvidenceSpan("free entry on Saturdays", 15), kind="invented_detail",
            option_id="A", supported=False, reason="not in the scenario",
        ))
        metrics = metrics_for(state, _outcome(state))
        self.assertEqual(metrics["unsupported_printed_turns"], 1)


if __name__ == "__main__":
    unittest.main()
