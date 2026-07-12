"""Item 12 (todo_validation.md): the observer consumes accepted evidence.

The evidence used to accept the turn is the evidence used to mutate state —
the observer runs no independent broad text interpretation, updates are
option-specific, and only the speaker's own accepted visible utterance may
change that speaker's private ranks or vote.
"""

from __future__ import annotations

import unittest

from models import (
    ActType,
    CommitmentEvidence,
    ConcernEvidence,
    EvidenceSpan,
    MoveIntent,
    SupportEvidence,
    VisibleEvidence,
)
from tests.fixtures import append_turn, make_state
from tests.stubs import make_runner


def _span(text: str) -> EvidenceSpan:
    return EvidenceSpan(text=text, start=0)


class ObserverReadsEvidenceNotText(unittest.TestCase):
    def test_state_follows_the_attached_evidence_object(self) -> None:
        # The text is a neutral aside the regex layer reads as a comment; the
        # attached (validator-accepted) evidence carries support for A. State
        # must follow the evidence — proof the observer does not reparse text.
        state = make_state()
        runner = make_runner(state)
        text = "The Museum feels like the easiest day for everyone."
        record = append_turn(
            state, "p1", text,
            intent=MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say it"),
        )
        record.evidence = VisibleEvidence(
            utterance=text,
            mentions=runner._resolver.mentions(text),
            supports=[SupportEvidence("A", "weak", _span(text))],
            primary_act=ActType.SUPPORT,
        )
        runner._apply_semantics(state, record)
        self.assertEqual(state.coverage["A"].reasons, 1)
        self.assertEqual(state.coverage["A"].objections, 0)

    def test_updates_are_option_specific_never_global(self) -> None:
        state = make_state()
        runner = make_runner(state)
        text = "The Museum keeps it simple, the Bike Ride exists, and the Escape Room price worries me."
        record = append_turn(
            state, "p1", text,
            intent=MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="beat"),
        )
        record.evidence = VisibleEvidence(
            utterance=text,
            mentions=runner._resolver.mentions(text),
            supports=[SupportEvidence("A", "weak", _span("keeps it simple"))],
            concerns=[ConcernEvidence("C", "ordinary", _span("price worries me"))],
            primary_act=ActType.COMMENT,
        )
        runner._apply_semantics(state, record)
        self.assertEqual(state.coverage["A"].reasons, 1)
        self.assertEqual(state.coverage["B"].reasons, 0)   # mere mention: no credit
        self.assertEqual(state.coverage["B"].mentions, 1)
        self.assertEqual(state.coverage["C"].objections, 1)
        self.assertEqual(state.coverage["C"].reasons, 0)
        # Concern binds only to C in the speaker's private ranks.
        self.assertEqual(state.runtimes["p1"].disliked_options(), {"C"})


class SpeakerLocalMutation(unittest.TestCase):
    def test_another_participants_statement_never_moves_private_ranks(self) -> None:
        state = make_state()
        runner = make_runner(state)
        before_p2 = dict(state.runtimes["p2"].option_ranks)
        before_p3 = dict(state.runtimes["p3"].option_ranks)
        text = "I vote for the Museum because it keeps the day simple."
        record = append_turn(
            state, "p1", text,
            intent=MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="vote", option_focus=["A"]),
        )
        record.evidence = VisibleEvidence(
            utterance=text,
            mentions=runner._resolver.mentions(text),
            commitments=[CommitmentEvidence("vote", "A", _span(text))],
            primary_act=ActType.VOTE,
        )
        runner._apply_semantics(state, record)
        self.assertEqual(state.runtimes["p1"].explicit_vote, "A")
        self.assertEqual(state.runtimes["p2"].option_ranks, before_p2)
        self.assertEqual(state.runtimes["p3"].option_ranks, before_p3)
        self.assertIsNone(state.runtimes["p2"].explicit_vote)
        self.assertIsNone(state.runtimes["p3"].explicit_vote)

    def test_dropped_turn_mutates_nothing(self) -> None:
        state = make_state()
        runner = make_runner(state, ["Just to be clear.", "Just to be clear."])
        coverage_before = {oid: cov.mentions for oid, cov in state.coverage.items()}
        ranks_before = {pid: dict(rt.option_ranks) for pid, rt in state.runtimes.items()}
        record = runner._generate_and_append(
            state, MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="beat")
        )
        self.assertTrue(record.state_mutation_blocked)
        self.assertEqual({oid: cov.mentions for oid, cov in state.coverage.items()}, coverage_before)
        self.assertEqual({pid: dict(rt.option_ranks) for pid, rt in state.runtimes.items()}, ranks_before)


if __name__ == "__main__":
    unittest.main()
