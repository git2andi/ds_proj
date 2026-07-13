from __future__ import annotations

import unittest

import tests  # noqa: F401
from eval.eval import flat_metrics_for, metrics_for
from models import ActType, MoveIntent, Phase, RepairState, RunOutcome
from tests.fixtures import append_turn, make_state


def outcome(status: str = "unresolved", final: str | None = None) -> RunOutcome:
    return RunOutcome(status, final, "test", 0, {})


class MinimalMetricSchemaTests(unittest.TestCase):
    def test_grouped_schema_and_zero_denominators(self):
        state = make_state()
        metrics = metrics_for(state, outcome())
        self.assertEqual(metrics["metric_schema_version"], "3.1")
        self.assertIsNone(metrics["run_structure"]["question_density"])
        self.assertIsNone(metrics["interaction"]["question_completion_rate"])
        self.assertIsNone(metrics["interaction"]["concern_response_rate"])
        self.assertIsNone(metrics["decision_behavior"]["compromise_success_rate"])

    def test_participation_fields_are_distinct(self):
        state = make_state()
        append_turn(state, "p1", "The Museum seems practical.")
        append_turn(state, "p1", "It also keeps the day simple.")
        append_turn(state, "p2", "The Bike Ride is my preference.")
        metrics = metrics_for(state, outcome())
        participation = metrics["participation"]
        self.assertIn("expected_engagement", participation)
        self.assertIn("expected_turn_share", participation)
        self.assertIn("realized_turn_count", participation)
        self.assertIn("realized_turn_share", participation)
        self.assertNotIn("realized_engagement", participation)
        self.assertEqual(participation["realized_turn_count"]["Mira"], 2)
        self.assertAlmostEqual(sum(v for v in participation["realized_turn_share"].values() if v is not None), 1.0, places=2)

    def test_validation_rates_use_attempt_denominator(self):
        state = make_state()
        first = append_turn(state, "p1", "I vote for the Museum.", phase=Phase.VOTING)
        first.repaired = True
        first.used_fallback = True
        state.controller_trace.append({"type": "turn", "result": {"appended": False}})
        metrics = metrics_for(state, outcome())
        validation = metrics["validation_grounding"]
        self.assertEqual(validation["repaired_turns"], 1)
        self.assertEqual(validation["fallback_turns"], 1)
        self.assertEqual(validation["dropped_turns"], 1)
        self.assertEqual(validation["repair_rate"], 0.5)
        self.assertEqual(validation["drop_rate"], 0.5)

    def test_vote_repair_metrics_do_not_crash_and_count_opportunity(self):
        state = make_state()
        append_turn(
            state, "p1", "I vote for the Museum.",
            intent=MoveIntent(
                "p1", ActType.VOTE, "final choice",
                route_source="repair_protocol", option_focus=["A"],
                required_vote="A",
            ),
            phase=Phase.COMPROMISE_REPAIR,
        )
        metrics = metrics_for(state, outcome("majority", "A"))
        self.assertEqual(metrics["traits"]["switch_opportunities"]["Mira"], 1)

    def test_token_categories_are_mutually_exclusive(self):
        state = make_state()
        state.token_usage_by_call_type = {
            "setup": {"in": 10, "out": 2, "calls": 1},
            "utterance": {"in": 20, "out": 4, "calls": 2},
            "moderator": {"in": 5, "out": 1, "calls": 1},
            "repair": {"in": 7, "out": 2, "calls": 1},
            "validator": {"in": 0, "out": 0, "calls": 0},
        }
        token_usage = metrics_for(state, outcome())["token_usage"]
        self.assertEqual(token_usage["total"]["input_tokens"], 42)
        self.assertEqual(token_usage["total"]["output_tokens"], 9)
        self.assertEqual(token_usage["total"]["api_calls"], 5)
        self.assertEqual(token_usage["runtime_validation"]["api_calls"], 0)

    def test_flat_metrics_remain_small_and_stable(self):
        state = make_state()
        row = flat_metrics_for("run", state, outcome())
        self.assertEqual(row["metric_schema_version"], "3.1")
        self.assertIn("runtime_validator_calls", row)
        self.assertLess(len(row), 55)


    def test_social_metrics_separate_functional_address_from_name_mentions(self):
        state = make_state()
        append_turn(
            state, "p1", "Jonas, what worries you about the Bike Ride?",
            intent=MoveIntent(
                "p1", ActType.ASK, "ask the public concern owner",
                addressee_id="p2", option_focus=["B"],
            ),
            phase=Phase.DISCUSSION,
        )
        interaction = metrics_for(state, outcome())["interaction"]
        self.assertEqual(interaction["direct_address_turn_count"], 1)
        self.assertEqual(interaction["participant_reference_turn_count"], 1)
        self.assertEqual(interaction["unique_directed_participant_pairs"], 1)

    def test_compromise_success_requires_visible_split_repair_switch(self):
        state = make_state()
        state.repair_history.append(RepairState(repair_reason="split_vote", status="resolved"))
        without_switch = metrics_for(state, outcome("majority", "A"))["decision_behavior"]
        self.assertEqual(without_switch["compromise_attempt_count"], 1)
        self.assertEqual(without_switch["compromise_success_count"], 0)
        self.assertEqual(without_switch["compromise_success_rate"], 0.0)

        state.runtimes["p1"].switch_events.append({
            "from": "B", "to": "A", "route_source": "repair_protocol"
        })
        with_switch = metrics_for(state, outcome("majority", "A"))["decision_behavior"]
        self.assertEqual(with_switch["compromise_success_count"], 1)
        self.assertEqual(with_switch["compromise_success_rate"], 1.0)

    def test_verbosity_metrics_compare_assigned_budget_to_realized_words(self):
        state = make_state()
        turn = append_turn(state, "p1", "The Museum is practical and easy to adjust.")
        turn.assigned_min_words = 6
        turn.assigned_max_words = 10
        traits = metrics_for(state, outcome())["traits"]
        self.assertEqual(traits["assigned_avg_word_budget"]["Mira"], 8.0)
        self.assertEqual(traits["word_budget_adherence"]["Mira"], 1.0)


if __name__ == "__main__":
    unittest.main()
