"""Baseline tests for visible-text parsing: commitments, blockers, questions."""

from __future__ import annotations

import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent
from parsing import (
    active_blocker_option,
    blocker_resolution_option,
    softening_option,
    visible_commitment,
)

from tests.fixtures import make_resolver, make_state, parse_text


class VisibleCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.resolver = make_resolver()

    def test_direct_vote_parses(self):
        self.assertEqual(visible_commitment("I vote for the Museum.", self.resolver), ("vote", "A"))

    def test_acceptance_parses_as_accept(self):
        self.assertEqual(visible_commitment("The Bike Ride works for me.", self.resolver), ("accept", "B"))

    def test_hedged_line_is_not_a_commitment(self):
        self.assertIsNone(visible_commitment("Maybe the Museum could work for me.", self.resolver))

    def test_conditional_support_is_not_a_commitment(self):
        self.assertIsNone(
            visible_commitment("I can support the Escape Room, but only if we book early.", self.resolver)
        )

    def test_question_is_not_a_commitment(self):
        self.assertIsNone(visible_commitment("Should we go with the Museum?", self.resolver))

    def test_rejection_parses_as_reject(self):
        self.assertEqual(
            visible_commitment("I can't support the Bike Ride.", self.resolver), ("reject", "B")
        )

    def test_multi_option_line_uses_commitment_object(self):
        stance = visible_commitment(
            "The Museum is cheaper than the Escape Room, so I vote for the Museum.", self.resolver
        )
        self.assertEqual(stance, ("vote", "A"))


class BlockerParsingTests(unittest.TestCase):
    def setUp(self):
        self.resolver = make_resolver()

    def test_personal_dealbreaker_binds(self):
        self.assertEqual(
            active_blocker_option("The Escape Room is a dealbreaker for me.", self.resolver), "C"
        )

    def test_negated_dealbreaker_does_not_bind(self):
        self.assertIsNone(
            active_blocker_option("The Escape Room is not a dealbreaker for me.", self.resolver)
        )

    def test_speculative_other_directed_blocker_does_not_bind(self):
        self.assertIsNone(
            active_blocker_option(
                "The Bike Ride might be a dealbreaker for some of the others.", self.resolver
            )
        )

    def test_resolution_detected_without_conditionals(self):
        self.assertEqual(
            blocker_resolution_option(
                "That fixes my concern; I can live with the Escape Room.", self.resolver
            ),
            "C",
        )

    def test_softening_moves_toward_named_option(self):
        self.assertEqual(
            softening_option("The Bike Ride is starting to make more sense to me.", self.resolver), "B"
        )


class ParseDialogueActTests(unittest.TestCase):
    def test_named_question_targets_that_participant(self):
        state = make_state()
        act = parse_text(state, "p1", "Jonas, what do you think about the Museum?")
        self.assertEqual(act.addressee_id, "p2")
        self.assertEqual(act.question_scope, "direct")
        self.assertEqual(act.question_target_id, "p2")

    def test_group_question_has_no_parser_assigned_respondent(self):
        # Contract 4.4: a group question carries scope only; the controller,
        # not the parser, picks who answers it.
        state = make_state()
        act = parse_text(state, "p1", "Which option is cheapest for all of us?", previous_speaker_id="p3")
        self.assertEqual(act.question_scope, "group")
        self.assertIsNone(act.question_target_id)

    def test_you_question_is_direct_at_previous_speaker(self):
        state = make_state()
        act = parse_text(state, "p1", "Do you actually mind the longer ride?", previous_speaker_id="p2")
        self.assertEqual(act.question_scope, "direct")
        self.assertEqual(act.question_target_id, "p2")

    def test_statement_has_no_question_scope(self):
        state = make_state()
        act = parse_text(state, "p1", "The Museum keeps the day easy to adjust.")
        self.assertIsNone(act.question_scope)
        self.assertIsNone(act.question_target_id)

    def test_vote_line_sets_explicit_vote(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p2", act=ActType.VOTE, reason="vote", option_focus=["B"])
        act = parse_text(state, "p2", "I vote for the Bike Ride.", intent=intent)
        self.assertEqual(act.explicit_vote, "B")

    def test_statement_keeps_intent_act_type(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support", option_focus=["A"])
        act = parse_text(state, "p1", "The Museum keeps the day easy to adjust.", intent=intent)
        self.assertEqual(act.act_type, ActType.SUPPORT)
        self.assertIn("A", act.option_refs)


if __name__ == "__main__":
    unittest.main()
