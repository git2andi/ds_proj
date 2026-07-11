"""Tests for the per-turn controller trace (TODO 2): immutable pre/post data,
route-source metadata, and selected-vs-realized act separation."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent, Phase

from tests.fixtures import make_state
from tests.stubs import make_runner


def _support_intent(speaker_id: str, option_id: str) -> MoveIntent:
    return MoveIntent(
        speaker_id=speaker_id,
        act=ActType.SUPPORT,
        reason="add a grounded reason",
        option_focus=[option_id],
    )


class ControllerTraceTests(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        self.state = make_state()

    def test_turn_entry_has_pre_and_result(self):
        runner = make_runner(self.state, ["The Museum keeps the day easy to adjust."])
        record = runner._generate_and_append(self.state, _support_intent("p1", "A"))
        entries = [e for e in self.state.controller_trace if e["type"] == "turn"]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["turn_index"], record.index)
        self.assertEqual(entry["pre"]["route_source"], "normal")
        self.assertEqual(entry["pre"]["selected_act"], "support")
        self.assertEqual(entry["pre"]["speaker_id"], "p1")
        self.assertTrue(entry["result"]["appended"])
        self.assertIn("realized_act", entry["result"])
        self.assertIn("tokens_in", entry["result"])

    def test_selected_and_realized_acts_logged_separately(self):
        # A support-routed line that visibly commits parses as a decision act;
        # the trace must show both without corrupting state.
        runner = make_runner(self.state, ["The Museum works for me."])
        runner._generate_and_append(self.state, _support_intent("p1", "A"))
        entry = [e for e in self.state.controller_trace if e["type"] == "turn"][0]
        self.assertEqual(entry["pre"]["selected_act"], "support")
        self.assertEqual(entry["result"]["realized_act"], "vote")
        self.assertTrue(entry["result"]["act_mismatch"])

    def test_pre_snapshot_is_immutable(self):
        runner = make_runner(self.state, ["The Museum keeps the day easy to adjust."])
        runner._generate_and_append(self.state, _support_intent("p1", "A"))
        entry = [e for e in self.state.controller_trace if e["type"] == "turn"][0]
        # The turn itself covered option A, but the pre-turn snapshot must keep
        # showing the coverage gap that existed when the route was selected.
        self.assertIn("A", entry["pre"]["coverage_gaps"])
        self.assertGreater(self.state.coverage["A"].mentions, 0)

    def test_route_source_recorded_for_special_routes(self):
        runner = make_runner(self.state, ["We could compare the Escape Room with the Museum."])
        intent = MoveIntent(
            speaker_id="p2",
            act=ActType.COMPARE,
            reason="briefly bring in an option that has not yet been socially processed, then compare it with the current lean",
            route_source="coverage",
            option_focus=["C", "B"],
        )
        runner._generate_and_append(self.state, intent)
        entry = [e for e in self.state.controller_trace if e["type"] == "turn"][0]
        self.assertEqual(entry["pre"]["route_source"], "coverage")
        self.assertTrue(entry["result"]["coverage_realized"])

    def test_phase_transition_traced(self):
        runner = make_runner(self.state)
        runner._mark_phase(self.state, Phase.DISCUSSION, "test transition")
        transitions = [e for e in self.state.controller_trace if e["type"] == "phase_transition"]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["from_phase"], "opening")
        self.assertEqual(transitions[0]["to_phase"], "discussion")
        self.assertEqual(transitions[0]["reason"], "test transition")


if __name__ == "__main__":
    unittest.main()
