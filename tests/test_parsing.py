"""Baseline tests for visible-text parsing: commitments, blockers, questions."""

from __future__ import annotations

import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent
from parsing import (
    active_blocker_option,
    blocker_resolution_option,
    visible_commitment,
    visible_comparison,
)

from tests.evidence_adapter import softening_option
from tests.fixtures import make_resolver, make_state, parse_text


class VisibleCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.resolver = make_resolver()

    def test_direct_vote_parses(self):
        self.assertEqual(visible_commitment("I vote for the Museum.", self.resolver), ("vote", "A"))

    def test_acceptance_parses_as_accept(self):
        self.assertEqual(visible_commitment("The Bike Ride works for me.", self.resolver), ("accept", "B"))

    def test_switching_from_to_binds_the_vote_to_the_target(self):
        # The natural switch idiom must parse as a vote for the NEW option
        # (todo_prompt item 7): the old option is the bridge, not the object.
        self.assertEqual(
            visible_commitment(
                "I'm switching from the Museum to the Bike Ride for the lower cost.", self.resolver
            ),
            ("vote", "B"),
        )
        self.assertEqual(
            visible_commitment(
                "Switching from the Museum to the Escape Room — the group energy wins.", self.resolver
            ),
            ("vote", "C"),
        )

    def test_natural_vote_wordings_parse(self):
        # Menu-less vote prompts produce these shapes (todo_prompt item 7).
        self.assertEqual(
            visible_commitment("I'm voting Museum — easy day for everyone.", self.resolver),
            ("vote", "A"),
        )
        self.assertEqual(
            visible_commitment("The Bike Ride is the right choice — cheap and active.", self.resolver),
            ("vote", "B"),
        )
        self.assertEqual(
            visible_commitment("I'm firmly with the Escape Room on this one.", self.resolver),
            ("accept", "C"),
        )

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

    def test_vote_line_gets_vote_display_label(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p2", act=ActType.VOTE, reason="vote", option_focus=["B"])
        act = parse_text(state, "p2", "I vote for the Bike Ride.", intent=intent)
        self.assertEqual(act.act_type, ActType.VOTE)

    def test_plain_statement_display_label_is_comment(self):
        # Soft semantics (support claims etc.) are the validator's job; the
        # deterministic display label only reflects critical signals (item 6).
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support", option_focus=["A"])
        act = parse_text(state, "p1", "The Museum keeps the day easy to adjust.", intent=intent)
        self.assertEqual(act.act_type, ActType.COMMENT)
        self.assertIn("A", act.option_refs)


class GrammaticalQuestionDetectorTests(unittest.TestCase):
    """Item 2: a small grammatical detector for aux-led / WH / choice questions,
    replacing the narrow pronoun-specific catalog. Representative boundaries only."""

    # (text, expected question scope or None)
    CASES = [
        # aux-led question with a full noun phrase (the reported false-negatives)
        ("Does the Park Pavilion have a backup plan if it rains?", "group"),
        ("Does the Moccamaster allow half-pot brewing?", "group"),
        ("How does the Ninja's footprint compare?", "group"),
        # ordinary WH-question
        ("How reliable is the weather forecast?", "group"),
        ("What makes the Museum the safer pick here?", "group"),
        # short option-choice question
        ("Moccamaster or Ninja?", "group"),
        # rhetorical / tag check-in — NOT a question thread
        ("The Ninja is the cheaper one, right?", None),
        ("It's a solid pick, isn't it?", None),
        # conditional statement that is not a question
        ("If it rains we can move indoors.", None),
        # ambiguous / plain statement
        ("The Museum keeps the day easy to adjust.", None),
    ]

    def test_boundaries(self):
        state = make_state()
        for text, expected in self.CASES:
            with self.subTest(text=text):
                act = parse_text(state, "p1", text)
                self.assertEqual(act.question_scope, expected)

    def test_direct_addressee_question(self):
        state = make_state()
        act = parse_text(state, "p1", "Jonas, does the longer ride bother you?")
        self.assertEqual(act.question_scope, "direct")
        self.assertEqual(act.question_target_id, "p2")

    def test_question_survives_alongside_a_comparison(self):
        # A line may simultaneously be a comparison and a question (item 2/3).
        state = make_state()
        act = parse_text(state, "p1", "The Museum is cheaper than the Bike Ride, but is it worth the wait?")
        self.assertEqual(act.question_scope, "group")


class BasicComparisonDetectorTests(unittest.TestCase):
    """Item 3: deterministic ComparisonEvidence from canonical option spans and
    grammatical comparison structures — no growing endpoint phrase catalogue."""

    def setUp(self):
        self.resolver = make_resolver()  # A=Museum, B=Bike Ride, C=Escape Room

    # (text, expected option-id pair or None)
    CASES = [
        ("The Museum versus the Bike Ride for cost.", ["A", "B"]),
        ("The Museum is cheaper, while the Bike Ride has more variety.", ["A", "B"]),
        ("The Museum's cost is almost double the Bike Ride's—worth it?", ["A", "B"]),
        ("The Museum costs more but the Bike Ride takes longer.", ["A", "B"]),
        ("The Escape Room is smaller and cheaper compared to the Museum.", ["C", "A"]),
        ("The Museum is riskier than the Bike Ride because it might rain.", ["A", "B"]),
        ("Which is cheaper, the Museum or the Bike Ride?", ["A", "B"]),
        # two mentions, no comparison construction
        ("I like the Museum and the Bike Ride.", None),
        # one option with comparative wording — not a two-option comparison
        ("The Museum is cheaper.", None),
    ]

    def test_boundaries(self):
        for text, expected in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(visible_comparison(text, self.resolver), expected)


if __name__ == "__main__":
    unittest.main()
