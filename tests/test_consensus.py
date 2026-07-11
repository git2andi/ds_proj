"""Baseline tests for the exact outcome definitions (successful/majority/unresolved)."""

from __future__ import annotations

import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from consensus import ConsensusManager, public_evidence, public_support, visible_votes_from_transcript
from models import ActType, MoveIntent, Phase

from tests.fixtures import append_turn, make_state, vote_intent


def _vote(state, pid, text, option, *, phase=Phase.VOTING, blocked=False):
    append_turn(state, pid, text, intent=vote_intent(pid, option), phase=phase, blocked=blocked)


class ConsensusTests(unittest.TestCase):
    def test_no_visible_votes_is_unresolved(self):
        state = make_state()
        outcome = ConsensusManager.finalize(state)
        self.assertEqual(outcome.status, "unresolved")
        self.assertIsNone(outcome.final_option)

    def test_unanimity_is_successful(self):
        state = make_state()
        for pid in ("p1", "p2", "p3"):
            _vote(state, pid, "I vote for the Museum.", "A")
        outcome = ConsensusManager.finalize(state)
        self.assertEqual(outcome.status, "successful")
        self.assertEqual(outcome.final_option, "A")

    def test_two_of_three_is_majority(self):
        state = make_state()
        _vote(state, "p1", "I vote for the Museum.", "A")
        _vote(state, "p2", "I vote for the Museum.", "A")
        _vote(state, "p3", "I vote for the Bike Ride.", "B")
        outcome = ConsensusManager.finalize(state)
        self.assertEqual(outcome.status, "majority")
        self.assertEqual(outcome.final_option, "A")

    def test_three_way_split_is_unresolved(self):
        state = make_state()
        _vote(state, "p1", "I vote for the Museum.", "A")
        _vote(state, "p2", "I vote for the Bike Ride.", "B")
        _vote(state, "p3", "I vote for the Escape Room.", "C")
        outcome = ConsensusManager.finalize(state)
        self.assertEqual(outcome.status, "unresolved")
        self.assertIsNone(outcome.final_option)

    def test_blocked_turn_does_not_count_as_vote(self):
        state = make_state()
        _vote(state, "p1", "I vote for the Museum.", "A", blocked=True)
        self.assertEqual(visible_votes_from_transcript(state), {})

    def test_discussion_commitment_is_not_a_formal_vote(self):
        # 13.1: opening leans and discussion support never silently become
        # final votes; only voting/compromise_repair turns count.
        state = make_state()
        _vote(state, "p1", "The Museum works for me.", "A", phase=Phase.DISCUSSION)
        self.assertEqual(visible_votes_from_transcript(state), {})

    def test_repair_phase_concession_replaces_formal_vote(self):
        state = make_state()
        _vote(state, "p1", "The Museum works for me.", "A")
        _vote(
            state, "p1", "Actually, I vote for the Bike Ride.", "B",
            phase=Phase.COMPROMISE_REPAIR,
        )
        votes = visible_votes_from_transcript(state)
        self.assertEqual(votes["p1"], "B")


class PublicEvidenceTests(unittest.TestCase):
    """Cleanup 3: private stance, public support, and formal votes stay distinct."""

    def test_opening_lean_is_neither_support_nor_vote(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.OPENING, reason="open", option_focus=["A"])
        append_turn(state, "p1", "My first thought is the Museum for the calm pace.", intent=intent, phase=Phase.OPENING)
        ev = public_evidence(state)
        self.assertEqual(ev.backing["A"], set())
        self.assertEqual(ev.formal_votes, {})
        self.assertEqual(ev.candidate_leaders, ())

    def test_private_acceptable_rank_does_not_count_publicly(self):
        state = make_state()
        state.runtimes["p2"].mark_acceptable("A")  # private rank only, never spoken
        ev = public_evidence(state)
        self.assertEqual(ev.backing["A"], set())
        self.assertEqual(ev.candidate_leaders, ())

    def test_discussion_acceptance_backs_narrowing_but_not_consensus(self):
        state = make_state()
        append_turn(
            state, "p2", "The Museum works for me.",
            intent=MoveIntent(speaker_id="p2", act=ActType.SUPPORT, reason="accept", option_focus=["A"]),
            phase=Phase.DISCUSSION,
        )
        ev = public_evidence(state)
        self.assertEqual(ev.backing["A"], {"p2"})           # visible discussion support
        self.assertEqual(ev.candidate_leaders, ("A",))
        self.assertEqual(ev.formal_votes, {})               # but no formal vote
        self.assertEqual(ConsensusManager.finalize(state).status, "unresolved")

    def test_blocked_turn_produces_no_public_backing(self):
        state = make_state()
        _vote(state, "p2", "The Museum works for me.", "A", phase=Phase.DISCUSSION, blocked=True)
        self.assertEqual(public_support(state)["A"], set())

    def test_later_hard_rejection_withdraws_earlier_backing(self):
        state = make_state()
        append_turn(
            state, "p2", "The Museum works for me.",
            intent=MoveIntent(speaker_id="p2", act=ActType.SUPPORT, reason="accept", option_focus=["A"]),
            phase=Phase.DISCUSSION,
        )
        state.runtimes["p2"].mark_rejected("A", reason_against="visible dealbreaker")
        self.assertEqual(public_support(state)["A"], set())

    def test_repair_acceptance_replaces_formal_vote_in_counts(self):
        state = make_state()
        _vote(state, "p1", "I vote for the Museum.", "A")
        _vote(state, "p2", "I vote for the Bike Ride.", "B")
        append_turn(
            state, "p2", "Fine — I can live with the Museum since it keeps the day easy.",
            intent=MoveIntent(
                speaker_id="p2", act=ActType.VOTE, reason="repair acceptance",
                option_focus=["A"], allow_vote_change=True,
            ),
            phase=Phase.COMPROMISE_REPAIR,
        )
        ev = public_evidence(state)
        self.assertEqual(ev.formal_votes["p2"], "A")
        self.assertEqual(ev.formal_counts["A"], 2)


if __name__ == "__main__":
    unittest.main()
