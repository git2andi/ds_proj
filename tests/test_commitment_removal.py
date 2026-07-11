"""No hidden commitment state: decisions run on ranks + traits + threads only
(todo_prompt items 2/3)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent, ParticipantRuntime, Phase

from tests.fixtures import make_persona, make_state
from tests.stubs import make_runner


class FieldsAreGone(unittest.TestCase):
    def test_participant_runtime_has_no_commitment_fields(self):
        rt = ParticipantRuntime(persona_id="p1")
        self.assertFalse(hasattr(rt, "commitment_strength"))
        self.assertFalse(hasattr(rt, "commitment_min"))

    def test_initialised_state_has_no_commitment_fields(self):
        state = make_state()
        for rt in state.runtimes.values():
            self.assertFalse(hasattr(rt, "commitment_strength"))


class HoldoutUsesExplicitSignals(unittest.TestCase):
    def test_high_switch_resistance_holds_out_when_candidate_not_acceptable(self):
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", switch_resistance=0.9),
            make_persona("p2", "Jonas", preferred="B"),
        ])
        runner = make_runner(state)
        # Candidate B is neutral (rank 3) for Mira -> valid holdout.
        self.assertTrue(runner._valid_holdout_against(state, state.persona_by_id("p1"), "B"))

    def test_acceptable_candidate_defeats_trait_resistance(self):
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", switch_resistance=0.9),
            make_persona("p2", "Jonas", preferred="B"),
        ])
        state.runtimes["p1"].set_rank("B", 4)
        runner = make_runner(state)
        self.assertFalse(runner._valid_holdout_against(state, state.persona_by_id("p1"), "B"))


class CandidateResistanceOrdering(unittest.TestCase):
    def test_resistance_orders_by_switch_resistance_and_rank(self):
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", switch_resistance=0.9),
            make_persona("p2", "Jonas", preferred="A", switch_resistance=0.2),
            make_persona("p3", "Lea", preferred="A", switch_resistance=0.2),
        ])
        state.runtimes["p3"].set_rank("C", 4)  # candidate already acceptable to Lea
        runner = make_runner(state)
        r1 = runner._candidate_resistance(state, state.persona_by_id("p1"), "C")
        r2 = runner._candidate_resistance(state, state.persona_by_id("p2"), "C")
        r3 = runner._candidate_resistance(state, state.persona_by_id("p3"), "C")
        self.assertGreater(r1, r2)   # trait resistance raises the score
        self.assertGreater(r2, r3)   # an acceptable rank lowers it


class DiscussionLeanMovement(unittest.TestCase):
    def test_visible_compromise_offer_by_low_stubbornness_sim_shifts_lean(self):
        random.seed(1)
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", stubbornness=0.0),
            make_persona("p2", "Jonas", preferred="B"),
            make_persona("p3", "Lea", preferred="C"),
        ])
        state.phase = Phase.DISCUSSION
        runner = make_runner(state)
        runner._llm.responses.append(
            "What if we went with the Bike Ride as a compromise for everyone?"
        )
        intent = MoveIntent(speaker_id="p1", act=ActType.COMPROMISE, reason="propose common ground", option_focus=["B"])
        runner._generate_and_append(state, intent)
        self.assertEqual(state.runtimes["p1"].top_option(), "B")
        self.assertEqual(state.discussion_lean_shifts, 1)

    def test_max_stubbornness_sim_mostly_keeps_its_lean(self):
        shifts = 0
        for seed in range(30):
            random.seed(seed)
            state = make_state(personas=[
                make_persona("p1", "Mira", preferred="A", stubbornness=1.0),
                make_persona("p2", "Jonas", preferred="B"),
                make_persona("p3", "Lea", preferred="C"),
            ])
            runner = make_runner(state)
            runner._llm.responses.append(
                "What if we went with the Bike Ride as a compromise for everyone?"
            )
            intent = MoveIntent(speaker_id="p1", act=ActType.COMPROMISE, reason="propose common ground", option_focus=["B"])
            runner._generate_and_append(state, intent)
            if state.runtimes["p1"].top_option() == "B":
                shifts += 1
        self.assertLess(shifts, 15)  # resist = 0.8 -> shifts ~20% of the time


if __name__ == "__main__":
    unittest.main()
