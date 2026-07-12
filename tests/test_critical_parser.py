"""Item 6 (todo_validation.md): the narrow deterministic critical parser.

commitment_post_checks is the gate every claimed public commitment must pass,
whether the strict regex layer found it or the validator LLM proposed it.
Strict vote precision is never weakened: these tests pin target naming,
prerequisite protection, question-vote separation, conflict detection,
required-target alignment, and rejected-option safety.
"""

from __future__ import annotations

import unittest

from parsing import commitment_post_checks, visible_commitment
from tests import semantic_fixtures as sf
from tests.fixtures import make_resolver


class CommitmentPostChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = make_resolver()

    def check(self, text: str, option_id: str, **kwargs) -> list[str]:
        return commitment_post_checks(text, option_id, self.resolver, **kwargs)

    def test_plain_vote_passes(self) -> None:
        self.assertEqual(self.check("I vote for the Bike Ride.", "B"), [])

    def test_vote_target_must_be_named(self) -> None:
        self.assertIn("COMMITMENT_TARGET_NOT_NAMED", self.check("That one gets my vote.", "B"))

    def test_unresolved_prerequisite_voids(self) -> None:
        issues = self.check("I'd go with the Escape Room only if we can move the booking.", "C")
        self.assertIn("CONDITIONAL_COMMITMENT", issues)

    def test_concrete_prerequisite_question_voids(self) -> None:
        issues = self.check("I can support the Museum, but are we okay with the higher cost?", "A")
        self.assertIn("CONDITIONAL_COMMITMENT", issues)

    def test_question_masquerading_as_vote_voids(self) -> None:
        issues = self.check("Should we just go with the Bike Ride?", "B")
        self.assertIn("QUESTION_NOT_COMMITMENT", issues)

    def test_trailing_group_checkin_does_not_void(self) -> None:
        # The canonical multi-function fixture: commitment + trailing group
        # question must survive the deterministic post-checks.
        fixture = sf.by_id("switch_with_reason_multi")
        self.assertEqual(self.check(fixture.text, "B"), [])

    def test_conflicting_commitment_is_rejected(self) -> None:
        issues = self.check(
            "I vote for the Bike Ride. Actually, the Museum works for me.", "B"
        )
        self.assertIn("CONFLICTING_COMMITMENT", issues)

    def test_required_target_alignment(self) -> None:
        issues = self.check("I vote for the Museum.", "A", required_vote="B")
        self.assertIn("REQUIRED_VOTE_MISMATCH", issues)

    def test_rejected_option_protection(self) -> None:
        issues = self.check("I vote for the Escape Room.", "C", rejected_options=["C"])
        self.assertIn("BLOCKED_OPTION_ACCEPTED", issues)

    def test_rejected_option_with_same_line_resolution_passes(self) -> None:
        issues = self.check(
            "That fixes my concern; I can live with the Escape Room.", "C",
            kind="accept", rejected_options=["C"], resolves_blocker="C",
        )
        self.assertEqual(issues, [])

    def test_sanctioned_switch_allows_concessive_rider(self) -> None:
        text = "I can live with the Museum, as long as we keep the morning free."
        self.assertIn("CONDITIONAL_COMMITMENT", self.check(text, "A"))
        self.assertEqual(self.check(text, "A", kind="accept", sanctioned_switch=True), [])

    def test_sanctioned_switch_still_blocks_genuine_prerequisites(self) -> None:
        text = "I can live with the Museum, but only if we can reschedule."
        self.assertIn(
            "CONDITIONAL_COMMITMENT",
            self.check(text, "A", kind="accept", sanctioned_switch=True),
        )


class StrictVotePrecisionUnchanged(unittest.TestCase):
    """The conservative regex layer keeps exactly its current precision."""

    def setUp(self) -> None:
        self.resolver = make_resolver()

    def test_direct_vote_still_parses(self) -> None:
        self.assertEqual(
            visible_commitment("I vote for the Bike Ride.", self.resolver), ("vote", "B")
        )

    def test_hedged_support_still_refuses(self) -> None:
        self.assertIsNone(
            visible_commitment("Maybe I could live with the Museum.", self.resolver)
        )

    def test_preference_wording_still_refuses(self) -> None:
        for fixture_id in ("preference_worth_considering", "preference_lean", "conditional_not_vote"):
            fixture = sf.by_id(fixture_id)
            self.assertIsNone(
                visible_commitment(fixture.text, self.resolver), fixture_id
            )


if __name__ == "__main__":
    unittest.main()
