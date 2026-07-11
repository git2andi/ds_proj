"""Tests for the consolidated voting/repair state machine (TODO 14)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from consensus import ConsensusManager
from models import Phase, RepairState

from tests.fixtures import append_turn, make_persona, make_state, vote_intent
from tests.stubs import make_runner


def _formal_vote(state, pid, option, text=None):
    append_turn(
        state, pid, text or f"I vote for option {option}.",
        intent=vote_intent(pid, option), phase=Phase.VOTING,
    )
    state.runtimes[pid].explicit_vote = option


def _vote_text(option_name: str) -> str:
    return f"I vote for the {option_name}."


_NAMES = {"A": "Museum", "B": "Bike Ride", "C": "Escape Room"}


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        random.seed(91)
        self.state = make_state()
        self.state.phase = Phase.VOTING
        self.runner = make_runner(self.state)

    def _classify(self):
        provisional = ConsensusManager.finalize(self.state)
        return self.runner._classify_repair(self.state, provisional)

    def test_unclear_vote_has_top_priority(self):
        _formal_vote(self.state, "p1", "A", _vote_text("Museum"))
        _formal_vote(self.state, "p2", "B", _vote_text("Bike Ride"))
        # p3 never produced a clear formal vote.
        repair = self._classify()
        self.assertEqual(repair.repair_reason, "unclear_vote")
        self.assertEqual(repair.participants_involved, ["p3"])

    def test_majority_classifies_holdout_repair(self):
        _formal_vote(self.state, "p1", "A", _vote_text("Museum"))
        _formal_vote(self.state, "p2", "A", _vote_text("Museum"))
        _formal_vote(self.state, "p3", "B", _vote_text("Bike Ride"))
        repair = self._classify()
        self.assertEqual(repair.repair_reason, "majority_holdout")
        self.assertEqual(repair.candidate_or_pair, ["A"])
        self.assertEqual(repair.participants_involved, ["p3"])

    def test_three_way_split_classifies_split_repair(self):
        _formal_vote(self.state, "p1", "A", _vote_text("Museum"))
        _formal_vote(self.state, "p2", "B", _vote_text("Bike Ride"))
        _formal_vote(self.state, "p3", "C", _vote_text("Escape Room"))
        repair = self._classify()
        self.assertEqual(repair.repair_reason, "split_vote")
        self.assertEqual(repair.max_attempts, 2)

    def test_each_reason_runs_at_most_once(self):
        _formal_vote(self.state, "p1", "A", _vote_text("Museum"))
        _formal_vote(self.state, "p2", "B", _vote_text("Bike Ride"))
        _formal_vote(self.state, "p3", "C", _vote_text("Escape Room"))
        self.state.repair_history.append(
            RepairState(repair_reason="split_vote", status="exhausted")
        )
        self.assertIsNone(self._classify())

    def test_holdout_repair_skipped_after_split_repair(self):
        _formal_vote(self.state, "p1", "A", _vote_text("Museum"))
        _formal_vote(self.state, "p2", "A", _vote_text("Museum"))
        _formal_vote(self.state, "p3", "B", _vote_text("Bike Ride"))
        self.state.repair_history.append(
            RepairState(repair_reason="split_vote", status="resolved")
        )
        self.assertIsNone(self._classify())

    def test_two_person_deadlock_specialization(self):
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A"),
            make_persona("p2", "Jonas", preferred="B"),
        ])
        state.phase = Phase.VOTING
        runner = make_runner(state)
        _formal_vote(state, "p1", "A", _vote_text("Museum"))
        _formal_vote(state, "p2", "B", _vote_text("Bike Ride"))
        repair = runner._classify_repair(state, ConsensusManager.finalize(state))
        self.assertEqual(repair.repair_reason, "two_person_deadlock")


class RunRepairTests(unittest.TestCase):
    def test_only_one_repair_objective_active_and_history_recorded(self):
        random.seed(92)
        state = make_state()
        state.phase = Phase.VOTING
        runner = make_runner(state)
        _formal_vote(state, "p1", "A", _vote_text("Museum"))
        _formal_vote(state, "p2", "A", _vote_text("Museum"))
        _formal_vote(state, "p3", "B", _vote_text("Bike Ride"))
        seen_active = []
        original = runner._repair_majority_holdout
        def spy(s, repair):
            seen_active.append(s.active_repair)
            return original(s, repair)
        runner._repair_majority_holdout = spy
        repair = runner._classify_repair(state, ConsensusManager.finalize(state))
        runner._run_repair(state, repair)
        self.assertEqual(len(seen_active), 1)
        self.assertIs(seen_active[0], repair)          # active while running
        self.assertIsNone(state.active_repair)          # cleared afterwards
        self.assertEqual([r.repair_reason for r in state.repair_history], ["majority_holdout"])
        self.assertIn(state.repair_history[0].status, {"resolved", "exhausted"})
        trace = [e for e in state.controller_trace if e["type"] == "repair"]
        self.assertEqual(len(trace), 1)


class DeadlockEndToEndTests(unittest.TestCase):
    def test_deadlock_terminates_unresolved_when_neither_moves(self):
        random.seed(93)
        state = make_state(personas=[
            make_persona(
                "p1", "Mira", preferred="A", switch_resistance=0.95,
                rejection="B", rejection_reason="cannot accept B",
            ),
            make_persona(
                "p2", "Jonas", preferred="B", switch_resistance=0.95,
                rejection="A", rejection_reason="cannot accept A",
            ),
        ])
        state.min_discussion_turns = 0
        state.force_narrow_turns = 1
        state.hard_max_turns = 6
        state.phase = Phase.VOTING
        runner = make_runner(state)
        runner._decision_loop(state)
        self.assertEqual(state.phase, Phase.CLOSING)
        self.assertIsNotNone(state.outcome)
        self.assertEqual(state.outcome.status, "unresolved")
        reasons = [r.repair_reason for r in state.repair_history]
        self.assertIn("two_person_deadlock", reasons)
        self.assertEqual(reasons.count("two_person_deadlock"), 1)
        # Hard rejections survived the repair.
        self.assertIn("B", state.runtimes["p1"].rejected_options())
        self.assertIn("A", state.runtimes["p2"].rejected_options())


if __name__ == "__main__":
    unittest.main()
