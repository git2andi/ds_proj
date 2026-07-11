"""Exclusive hard-blocker model and visible-evidence-only stance updates
(docs/todo_blocker.md items 1-4)."""

from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import tests  # noqa: F401  # puts src/ on sys.path before src imports

import prompts
from builders import SetupBuilder, _normalise_initial_stances
from models import (
    ActType,
    MoveIntent,
    OptionCard,
    OptionStance,
    Persona,
    Phase,
    Scenario,
    SimulatorParameters,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_PREFERRED,
    STANCE_REJECTED,
    TraitProfile,
)

from tests.fixtures import make_persona, make_state
from tests.stubs import make_runner


def _scenario4() -> Scenario:
    return Scenario(
        topic="Choose a weekend activity",
        shared_context=["Only Saturday is available."],
        options=[
            OptionCard("A", "Museum and Cafe Day", {"cost": "24 euros"}, "low effort", "may feel quiet", "Museum"),
            OptionCard("B", "Lake Bike Ride", {"cost": "12 euros"}, "active and cheap", "tiring for some", "Bike Ride"),
            OptionCard("C", "Escape Room", {"cost": "32 euros"}, "memorable", "inflexible booking", "Escape Room"),
            OptionCard("D", "Home Cooking Night", {"cost": "18 euros"}, "cheapest", "may feel ordinary", "Cooking"),
        ],
    )


def _persona4(pid: str, preferred: str, stances: dict[str, OptionStance], *, hard: bool, rejection=None) -> Persona:
    return Persona(
        id=pid,
        name=f"Sim{pid}",
        traits=TraitProfile(2, 4, 3, 1 if hard else 3, 3),
        sim_params=SimulatorParameters(0.5, 0.5, 0.5, 0.9 if hard else 0.5, 0.9 if hard else 0.5),
        background=f"Sim{pid} is a 33 year old planner with one firm requirement.",
        private_goal="wants the only option that meets their requirement",
        preferred_options=[preferred],
        age=33,
        speech_style="relaxed practical wording",
        rejection=rejection,
        rejection_reason="conflicts with the requirement" if rejection else "",
        option_stances=stances,
        hard_blocker=hard,
    )


def _exclusive_stances(preferred: str) -> dict[str, OptionStance]:
    out = {}
    for oid in "ABCD":
        if oid == preferred:
            out[oid] = OptionStance(oid, STANCE_PREFERRED, "meets the requirement", "")
        else:
            out[oid] = OptionStance(oid, STANCE_REJECTED, "", "fails the requirement")
    return out


class ExclusiveNormalization(unittest.TestCase):
    def test_sampled_blocker_gets_one_preferred_and_all_alternatives_rejected(self):
        scenario = _scenario4()
        raw = {oid: OptionStance(oid) for oid in "ABCD"}
        stances = _normalise_initial_stances(
            scenario, raw, ["B"], None, "", exclusive_blocker=True
        )
        self.assertEqual(stances["B"].rank, STANCE_PREFERRED)
        for oid in "ACD":
            self.assertEqual(stances[oid].rank, STANCE_REJECTED, oid)
            self.assertTrue(stances[oid].reason_against, oid)

    def test_normal_participant_never_gets_accidental_hard_rejections(self):
        scenario = _scenario4()
        raw = {oid: OptionStance(oid, STANCE_REJECTED, "", "llm went hard") for oid in "ABCD"}
        stances = _normalise_initial_stances(scenario, raw, ["B"], None, "")
        for oid in "ACD":
            self.assertGreaterEqual(stances[oid].rank, STANCE_DISLIKED, oid)  # clamped, movable


class ContractValidation(unittest.TestCase):
    def _validate(self, personas):
        builder = SetupBuilder.__new__(SetupBuilder)
        builder._validate_world(_scenario4(), personas)

    def test_correct_exclusive_blocker_passes(self):
        blocker = _persona4("p1", "B", _exclusive_stances("B"), hard=True)
        self._validate([blocker])

    def test_blocker_with_an_acceptable_alternative_is_rejected(self):
        stances = _exclusive_stances("B")
        stances["C"] = OptionStance("C", STANCE_ACCEPTABLE, "also fine", "")
        with self.assertRaises(ValueError):
            self._validate([_persona4("p1", "B", stances, hard=True)])

    def test_blocker_rejection_without_reason_is_rejected(self):
        stances = _exclusive_stances("B")
        stances["C"] = OptionStance("C", STANCE_REJECTED, "", "")
        with self.assertRaises(ValueError):
            self._validate([_persona4("p1", "B", stances, hard=True)])

    def test_non_blocker_with_exclusive_pattern_is_rejected(self):
        with self.assertRaises(ValueError):
            self._validate([_persona4("p1", "B", _exclusive_stances("B"), hard=False)])

    def test_manual_single_rejection_persona_passes(self):
        stances = {
            "A": OptionStance("A", STANCE_PREFERRED, "fits", ""),
            "B": OptionStance("B", STANCE_REJECTED, "", "vendor lock-in"),
            "C": OptionStance("C"),
            "D": OptionStance("D"),
        }
        self._validate([_persona4("p1", "A", stances, hard=False, rejection="B")])


class GroupLevelSampling(unittest.TestCase):
    def _builder(self):
        builder = SetupBuilder.__new__(SetupBuilder)
        builder._profiles = []
        builder._hard_blocker_id = None
        return builder

    def test_sampled_event_marks_exactly_one_blocker_row(self):
        builder = self._builder()
        with patch("builders.random.random", return_value=0.0):
            random.seed(7)
            rows = builder._trait_rows(4)
        self.assertIsNotNone(builder._hard_blocker_id)
        blocker_rows = [r for r in rows if r["traits"]["agreeableness"] == 1]
        self.assertEqual(len(blocker_rows), 1)
        self.assertEqual(f"p{rows.index(blocker_rows[0]) + 1}", builder._hard_blocker_id)

    def test_unsampled_group_has_no_blocker(self):
        builder = self._builder()
        with patch("builders.random.random", return_value=0.99):
            random.seed(7)
            rows = builder._trait_rows(4)
        self.assertIsNone(builder._hard_blocker_id)
        self.assertFalse([r for r in rows if r["traits"]["agreeableness"] == 1])


class BlockerPrompts(unittest.TestCase):
    def test_persona_prompt_states_the_exclusive_contract_when_sampled(self):
        rows = [{"id": f"p{i}", "name": f"Sim{i}", "traits": {}} for i in (1, 2, 3)]
        text = prompts.setup_personas(
            "topic", 3, rows, {"p1": "A", "p2": "B", "p3": "C"}, [], [], hard_blocker_id="p2"
        )
        self.assertIn("p2 (Sim2) is this group's ONE exclusive hard blocker", text)
        self.assertIn("EVERY other option rank 1", text)
        self.assertIn("Every other participant must remain movable", text)

    def test_persona_prompt_keeps_the_old_rules_when_not_sampled(self):
        rows = [{"id": "p1", "name": "Sim1", "traits": {}}]
        text = prompts.setup_personas("topic", 1, rows, {"p1": "A"}, [], [])
        self.assertIn("For agreeableness=1 only", text)
        self.assertNotIn("exclusive hard blocker", text)

    def test_utterance_prompt_shows_the_exclusive_constraint(self):
        state = make_state()  # three options A-C
        rt = state.runtimes["p1"]
        rt.mark_rejected("B", reason_against="fails the requirement")
        rt.mark_rejected("C", reason_against="fails the requirement")
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support it")
        persona = state.persona_by_id("p1")
        text = prompts.sim_utterance(
            persona=persona, state=state, intent=intent, recent_lines=[],
            focus_options=[], addressee_name=None, max_words=15,
        )
        self.assertIn("Hard constraint: only Museum is acceptable to them", text)
        self.assertIn("reject every other option", text)


class ShortAliasInflection(unittest.TestCase):
    """Setup reliability fix found during the todo_blocker verification runs:
    the setup LLM reliably pluralizes aliases ("Board Games" for "Board Game
    Night"), which used to reject the whole scenario attempt three times."""

    def test_trivial_singular_plural_inflection_is_accepted(self):
        from aliases import validated_short_alias

        self.assertEqual(
            validated_short_alias("Board Game Tournament Afternoon", "Board Games"),
            "Board Games",
        )
        self.assertEqual(
            validated_short_alias("Guided Nature Walks at the Park", "Nature Walk"),
            "Nature Walk",
        )

    def test_invented_aliases_are_still_rejected(self):
        from aliases import validated_short_alias

        self.assertEqual(validated_short_alias("Lake Bike Ride", "Kayak Tour"), "")
        self.assertEqual(validated_short_alias("Lake Bike Ride", "Biker"), "")


class VisibleEvidenceOnlyStanceUpdates(unittest.TestCase):
    """Other participants' turns never directly change someone else's ranks."""

    def test_support_and_criticism_leave_other_participants_ranks_unchanged(self):
        random.seed(6)
        state = make_state()
        state.phase = Phase.DISCUSSION
        before = {pid: dict(rt.option_ranks) for pid, rt in state.runtimes.items()}
        runner = make_runner(state, [
            "I vote for the Bike Ride — my earlier doubts are gone, the cost wins.",
            "The Museum worries me, the cost is high for what it offers.",
        ])
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p2", act=ActType.SUPPORT, reason="support", option_focus=["B"])
        )
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p3", act=ActType.CONCERN, reason="push back", option_focus=["A"])
        )
        # p2 supported B and p3 attacked A (p1's favorite): p1's private ranks
        # must be untouched — pressure works only through routing/opportunity.
        self.assertEqual(state.runtimes["p1"].option_ranks, before["p1"])
        # p3's own visible objection may move p3's own rank; p2's may move p2's.
        self.assertEqual(state.runtimes["p2"].rank("B"), STANCE_PREFERRED)

    def test_own_visible_acceptance_still_updates_own_rank(self):
        random.seed(6)
        state = make_state()
        state.phase = Phase.DISCUSSION
        runner = make_runner(state, ["The Bike Ride works for me too."])
        runner._generate_and_append(
            state, MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="react", option_focus=["B"])
        )
        self.assertGreaterEqual(state.runtimes["p1"].rank("B"), STANCE_ACCEPTABLE)


if __name__ == "__main__":
    unittest.main()
