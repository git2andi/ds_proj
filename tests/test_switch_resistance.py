"""Tests for switch_resistance (TODO 5): derivation, override, and the
stubbornness/switch_resistance responsibility split."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import TraitProfile
from simulator import derive_simulator_parameters, expected_turn_share

from tests.fixtures import make_persona, make_state
from tests.stubs import make_runner


class DerivationTests(unittest.TestCase):
    def test_switch_resistance_derived_in_range(self):
        for traits in (
            TraitProfile(1, 1, 1, 1, 1),
            TraitProfile(5, 5, 5, 5, 5),
            TraitProfile(3, 4, 2, 1, 4),
        ):
            params = derive_simulator_parameters(traits)
            self.assertGreaterEqual(params.switch_resistance, 0.0)
            self.assertLessEqual(params.switch_resistance, 1.0)

    def test_disagreeable_traits_raise_switch_resistance(self):
        low = derive_simulator_parameters(TraitProfile(5, 2, 3, 5, 1))
        high = derive_simulator_parameters(TraitProfile(1, 5, 3, 1, 5))
        self.assertLess(low.switch_resistance, high.switch_resistance)

    def test_existing_formulas_unchanged(self):
        # Regression guard: adding switch_resistance must not move the four
        # existing parameters (values pinned from the pre-change formulas).
        params = derive_simulator_parameters(TraitProfile(3, 4, 2, 5, 1))
        self.assertAlmostEqual(params.engagement, 0.25 + 0.60 * 0.25 + 0.15 * 0.75)
        self.assertAlmostEqual(params.verbosity, 0.20 + 0.55 * 0.25 + 0.25 * 0.50)
        self.assertAlmostEqual(params.directness, 0.25 + 0.35 * 0.75 + 0.25 * 0.25 + 0.15 * 0.0)
        self.assertAlmostEqual(params.stubbornness, 0.45 * 0.0 + 0.25 * 0.0 + 0.20 * 0.50 + 0.10 * 0.75)

    def test_manual_override_like_other_parameters(self):
        params = derive_simulator_parameters(TraitProfile(3, 3, 3, 3, 3))
        overridden = replace(params, **{"switch_resistance": 0.93}).clipped()
        self.assertAlmostEqual(overridden.switch_resistance, 0.93)
        self.assertAlmostEqual(overridden.engagement, params.engagement)


class ResponsibilitySplitTests(unittest.TestCase):
    def test_switch_resistance_gates_final_moves_not_discussion_moves(self):
        # Same persona, extreme switch_resistance, low stubbornness: final
        # movement is blocked, discussion lean movement stays open.
        persona = make_persona("p1", "Mira", preferred="A", stubbornness=0.1, switch_resistance=0.99)
        state = make_state(personas=[persona, make_persona("p2", "Jonas", preferred="B")])
        runner = make_runner(state)
        random.seed(9)
        final_allowed = sum(
            1 for _ in range(200) if runner._can_shift_to(state, persona, "B", final_decision=True)
        )
        discussion_allowed = sum(
            1 for _ in range(200) if runner._can_shift_to(state, persona, "B")
        )
        self.assertLess(final_allowed, 30)          # near-hard final gate
        self.assertEqual(discussion_allowed, 200)   # discussion movement unaffected

    def test_stubbornness_gates_discussion_moves_not_final_moves(self):
        persona = make_persona("p1", "Mira", preferred="A", stubbornness=0.99, switch_resistance=0.1)
        state = make_state(personas=[persona, make_persona("p2", "Jonas", preferred="B")])
        runner = make_runner(state)
        random.seed(9)
        final_allowed = sum(
            1 for _ in range(200) if runner._can_shift_to(state, persona, "B", final_decision=True)
        )
        discussion_allowed = sum(
            1 for _ in range(200) if runner._can_shift_to(state, persona, "B")
        )
        self.assertEqual(final_allowed, 200)
        self.assertLess(discussion_allowed, 30)

    def test_switch_resistance_does_not_change_turn_share(self):
        flexible = make_persona("p1", "Mira", switch_resistance=0.05)
        resistant = make_persona("p2", "Jonas", switch_resistance=0.95)
        shares = expected_turn_share([flexible, resistant])
        self.assertAlmostEqual(shares["p1"], shares["p2"])

    def test_candidate_resistance_follows_switch_resistance(self):
        soft = make_persona("p1", "Mira", preferred="A", stubbornness=0.9, switch_resistance=0.1)
        hard = make_persona("p2", "Jonas", preferred="A", stubbornness=0.1, switch_resistance=0.9)
        state = make_state(personas=[soft, hard, make_persona("p3", "Lea", preferred="B")])
        runner = make_runner(state)
        self.assertLess(
            runner._candidate_resistance(state, soft, "B"),
            runner._candidate_resistance(state, hard, "B"),
        )


if __name__ == "__main__":
    unittest.main()
