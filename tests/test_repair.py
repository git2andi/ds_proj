"""Tests for the consolidated voting/repair state machine (TODO 14)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

import prompts
from consensus import ConsensusManager
from models import ActType, MoveIntent, Phase, RepairState, ValidationIssue

from tests.fixtures import append_turn, make_persona, make_state, vote_intent
from tests.stubs import make_runner


def _repair_prompt(state, intent, issues, original="The Museum tour includes free lunch."):
    return prompts.repair_utterance(
        original_text=original, issues=issues,
        persona=state.persona_by_id(intent.speaker_id),
        state=state, recent_lines=[], intent=intent, max_words=15,
    )


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
        self.assertEqual(repair.max_attempts, 1)

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
        self.assertNotIn("two_person_deadlock", reasons)
        # Hard rejections survived the repair.
        self.assertIn("B", state.runtimes["p1"].rejected_options())
        self.assertIn("A", state.runtimes["p2"].rejected_options())


# Merged from test_repair_prompting.py (item 8): repair-prompt content and
# severity-based selection belong with the repair state machine.
class RepairPromptCarriesActionableEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.state = make_state()

    def test_issue_explanations_are_itemized_not_just_codes(self) -> None:
        intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="vote",
                            option_focus=["B"], required_vote="B")
        issues = [ValidationIssue(
            code="REQUIRED_VOTE_MISMATCH", explanation="committed to A, required B",
            option_id="B", blocking=True,
        )]
        text = _repair_prompt(self.state, intent, issues, original="I vote for the Museum.")
        self.assertIn("- REQUIRED_VOTE_MISMATCH: committed to A, required B", text)

    def test_preservation_instruction_is_explicit(self) -> None:
        intent = MoveIntent(speaker_id="p1", act=ActType.CONCERN, reason="push back", option_focus=["C"])
        text = _repair_prompt(self.state, intent, [ValidationIssue(code="CONCERN_NOT_REALIZED")])
        self.assertIn("Preserve everything not flagged below", text)
        self.assertIn("stance and commitment target", text)

    def test_wrong_option_focus_names_the_intended_option(self) -> None:
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="defend", option_focus=["A"])
        text = _repair_prompt(self.state, intent, [ValidationIssue(code="WRONG_OPTION_FOCUS", option_id="A")])
        self.assertIn("make it about Museum", text)


class RepairSelectionBySeverity(unittest.TestCase):
    def test_unrealized_original_and_worse_repair_are_dropped(self) -> None:
        state = make_state()
        runner = make_runner(state, [
            "The Museum has been discussed a lot today.",
            "Just to be clear.",
        ])
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="defend", option_focus=["A"])
        record = runner._generate_and_append(state, intent)
        self.assertEqual(record.text, "")
        self.assertTrue(record.state_mutation_blocked)
        self.assertIn("SUPPORT_NOT_REALIZED", record.validation_issues)

    def test_strictly_better_repair_is_taken(self) -> None:
        state = make_state()
        runner = make_runner(state, [
            "The Museum has been discussed a lot today.",
            "The Escape Room is pricier than the Museum, worth noting.",
        ])
        intent = MoveIntent(
            speaker_id="p1", act=ActType.COMPARE, reason="cover it",
            route_source="coverage", option_focus=["C", "A"],
        )
        record = runner._generate_and_append(state, intent)
        self.assertTrue(record.repaired)
        self.assertEqual(record.text, "The Escape Room is pricier than the Museum, worth noting.")
        self.assertNotIn("MISSING_REQUIRED_OPTION_FOCUS", record.validation_issues)


if __name__ == "__main__":
    unittest.main()
