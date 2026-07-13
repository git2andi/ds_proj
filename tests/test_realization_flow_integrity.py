"""Regressions for realization, obligation, repetition, and flow accounting."""

from __future__ import annotations

import unittest

from models import ActType, MoveIntent, Phase, SimulatorBid, TurnObligation
from tests.evidence_adapter import derive_evidence
from tests.fixtures import append_turn, make_resolver, make_state
from tests.stubs import make_runner


class NaturalRealizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = make_resolver()

    def evidence(self, text: str, act: ActType, focus: str):
        return derive_evidence(
            text,
            self.resolver,
            intent=MoveIntent("p1", act, "test", option_focus=[focus]),
        )

    def test_natural_support_effect_verbs(self):
        for text in (
            "The Museum removes the daily coordination headache.",
            "The Museum cuts our planning stress significantly.",
            "The Museum broadens access for the group.",
        ):
            with self.subTest(text=text):
                evidence = self.evidence(text, ActType.SUPPORT, "A")
                self.assertIn("A", {item.option_id for item in evidence.supports})

    def test_natural_concern_consequence_verbs(self):
        examples = (
            "The Bike Ride risks delaying the rest of the day.",
            "The Bike Ride might stretch the budget too far.",
            "The Bike Ride could leave too little recovery time.",
        )
        for text in examples:
            with self.subTest(text=text):
                evidence = self.evidence(text, ActType.CONCERN, "B")
                self.assertIn("B", {item.option_id for item in evidence.concerns})

    def test_natural_personal_conditional_acceptance(self):
        examples = (
            "I'm open to the Bike Ride if we keep a relaxed pace.",
            "The Bike Ride could work if we shorten the route.",
            "I could consider the Bike Ride if everyone is comfortable.",
        )
        for text in examples:
            with self.subTest(text=text):
                evidence = self.evidence(text, ActType.COMPROMISE, "B")
                self.assertIn("B", {item.option_id for item in evidence.proposals})


class FloorAccountingTests(unittest.TestCase):
    def test_valid_bids_that_all_fail_are_not_counted_as_no_claim(self):
        state = make_state()
        state.phase = Phase.DISCUSSION
        runner = make_runner(state)
        intent = MoveIntent("p1", ActType.SUPPORT, "support", option_focus=["A"])
        bid = SimulatorBid("p1", True, 0.9, intent)
        runner._collect_bids = lambda *_: [bid]
        runner._ranked_valid_bids = lambda *_: [bid]
        dropped = append_turn(state, "p1", "", intent=intent, blocked=True)
        state.turns.pop()
        state.turn_index = 0
        runner._generate_and_append = lambda *_: dropped
        self.assertIsNone(runner._run_open_floor_turn(state, runner._discussion_stimulus(state)))
        self.assertEqual(state.no_bid_round_count, 0)
        self.assertEqual(state.generation_failure_round_count, 1)

    def test_no_valid_claim_is_counted_as_true_no_bid(self):
        state = make_state()
        state.phase = Phase.DISCUSSION
        runner = make_runner(state)
        runner._collect_bids = lambda *_: []
        runner._ranked_valid_bids = lambda *_: []
        self.assertIsNone(runner._run_open_floor_turn(state, runner._discussion_stimulus(state)))
        self.assertEqual(state.no_bid_round_count, 1)
        self.assertEqual(state.generation_failure_round_count, 0)


class ObligationAndRepetitionTests(unittest.TestCase):
    def test_protocol_obligation_retries_until_accepted(self):
        state = make_state()
        runner = make_runner(state)
        calls = []
        blocked = append_turn(state, "p1", "", blocked=True)
        state.turns.pop(); state.turn_index = 0
        accepted = append_turn(state, "p1", "The Museum is my current choice.")
        state.turns.pop(); state.turn_index = 0

        def generate(_state, _intent):
            calls.append(1)
            return blocked if len(calls) < 3 else accepted

        runner._generate_and_append = generate
        runner._emit = lambda *_: None
        record = runner._run_obligation_turn(
            state, TurnObligation("opening", "p1", ActType.OPENING)
        )
        self.assertEqual(record.text, accepted.text)
        self.assertEqual(len(calls), 3)
        self.assertEqual(state.protocol_obligation_failures, 0)

    def test_nonconsecutive_contribution_key_is_rejected(self):
        state = make_state()
        runner = make_runner(state)
        state.phase = Phase.DISCUSSION
        first = MoveIntent(
            "p1", ActType.CONCERN, "budget concern", option_focus=["A"],
            contribution_key="concern:A:budget",
        )
        append_turn(state, "p1", "The Museum may exceed the budget.", intent=first)
        append_turn(state, "p2", "The Bike Ride works for me.")
        repeated = SimulatorBid(
            "p1", True, 0.8,
            MoveIntent(
                "p1", ActType.CONCERN, "same concern", option_focus=["A"],
                contribution_key="concern:A:budget",
            ),
        )
        self.assertIn("repeats", runner._validate_bid(state, repeated, obligation=None))
        self.assertEqual(state.repeated_bid_rejections, 1)


class OpeningRealizationTests(unittest.TestCase):
    def test_opening_requires_visible_initial_position(self):
        state = make_state()
        runner = make_runner(state)
        intent = MoveIntent(
            "p1", ActType.OPENING, "state initial preference", option_focus=["A"]
        )
        evidence = derive_evidence(
            "I am ready to discuss the options.",
            runner._resolver,
            speaker_id="p1",
            participant_names={p.id: p.name for p in state.personas},
            intent=intent,
        )
        assessment = runner._assess_candidate(
            text="I am ready to discuss the options.",
            state=state,
            persona=state.persona_by_id("p1"),
            intent=intent,
            evidence=evidence,
        )
        self.assertIn(
            "OPENING_POSITION_NOT_REALIZED",
            {issue.code for issue in assessment.issues},
        )
        self.assertTrue(any(issue.blocking for issue in assessment.issues))

    def test_opening_with_visible_support_for_focus_passes_contract(self):
        state = make_state()
        runner = make_runner(state)
        intent = MoveIntent(
            "p1", ActType.OPENING, "state initial preference", option_focus=["A"]
        )
        text = "The Museum works best for me because it is easy to adjust."
        evidence = derive_evidence(
            text,
            runner._resolver,
            speaker_id="p1",
            participant_names={p.id: p.name for p in state.personas},
            intent=intent,
        )
        assessment = runner._assess_candidate(
            text=text, state=state, persona=state.persona_by_id("p1"),
            intent=intent, evidence=evidence,
        )
        self.assertNotIn(
            "OPENING_POSITION_NOT_REALIZED",
            {issue.code for issue in assessment.issues},
        )
        self.assertTrue(assessment.intended_act_realized)


if __name__ == "__main__":
    unittest.main()
