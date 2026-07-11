"""Offline end-to-end flow test: opening → discussion → narrowing → voting →
repair → closing, driven through the real controller with a scripted LLM.

This is the deterministic stand-in for a live LLM run when no provider is
available: it exercises the full migrated flow (routing, threads, phases,
votes, repair machine, traces) with the FakeLLM, whose generic lines force the
deterministic fallbacks on decision turns.
"""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from consensus import ConsensusManager
from models import Phase

from tests.fixtures import make_persona, make_state
from tests.stubs import make_runner


def _run_full_flow(state, runner):
    runner._derive_pacing(state)
    runner._opening_round(state)
    runner._discussion_loop(state)
    runner._narrowing_phase(state)
    runner._decision_loop(state)


class EndToEndFlowTests(unittest.TestCase):
    def test_three_way_run_reaches_a_legal_outcome(self):
        random.seed(7)
        state = make_state()
        runner = make_runner(state)
        _run_full_flow(state, runner)

        self.assertEqual(state.phase, Phase.CLOSING)
        outcome = state.outcome or ConsensusManager.finalize(state)
        self.assertIn(outcome.status, {"successful", "majority", "unresolved"})
        # Phase walk is legal by construction (illegal transitions raise), and
        # every participant turn carries a routed intent with a route source.
        for turn in state.turns:
            if turn.speaker_id == "moderator" or turn.intent is None:
                continue
            self.assertTrue(turn.intent.route_source)
        turn_entries = [e for e in state.controller_trace if e["type"] == "turn"]
        self.assertGreater(len(turn_entries), len(state.personas))
        transitions = [e for e in state.controller_trace if e["type"] == "phase_transition"]
        self.assertGreaterEqual(len(transitions), 3)  # opening->...->closing
        self.assertEqual(transitions[0]["from_phase"], "opening")
        self.assertEqual(transitions[-1]["to_phase"], "closing")

    def test_run_is_bounded_by_the_hard_cap_plus_decision_beats(self):
        random.seed(8)
        state = make_state()
        runner = make_runner(state)
        _run_full_flow(state, runner)
        participant_turns = sum(1 for t in state.turns if t.speaker_id != "moderator")
        # Discussion is capped by hard_max_turns; narrowing/voting/repair add a
        # bounded number of beats on top (narrowing 2, votes n, repair ~2n+3).
        n = len(state.personas)
        self.assertLessEqual(participant_turns, state.hard_max_turns + 2 + n + (2 * n + 4))

    def test_two_person_run_with_rejections_ends_unresolved(self):
        random.seed(9)
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", switch_resistance=0.95,
                         rejection="B", rejection_reason="cannot accept B"),
            make_persona("p2", "Jonas", preferred="B", switch_resistance=0.95,
                         rejection="A", rejection_reason="cannot accept A"),
        ])
        runner = make_runner(state)
        _run_full_flow(state, runner)
        self.assertEqual(state.phase, Phase.CLOSING)
        outcome = state.outcome or ConsensusManager.finalize(state)
        # Mutually hard-blocked picks with maximal switch resistance: the
        # bounded deadlock repair must terminate without fake agreement.
        self.assertEqual(outcome.status, "unresolved")
        self.assertIn("B", state.runtimes["p1"].rejected_options())
        self.assertIn("A", state.runtimes["p2"].rejected_options())


if __name__ == "__main__":
    unittest.main()
