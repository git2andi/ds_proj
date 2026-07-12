"""VisibleEvidence as the sole public semantic authority (todo_validation item 4).

Every public semantic consumer — public support, proposal counts, formal
votes, observer state — reads the SAME accepted evidence object. There is no
second semantic authority: a turn with no supporting evidence produces no
public support/vote/proposal, and evidence alone (whatever the routed intent
was) is what counts.
"""

from __future__ import annotations

import unittest

import tests  # noqa: F401  # puts src/ on sys.path

from consensus import public_evidence, public_support, visible_votes_from_transcript
from models import (
    CommitmentEvidence,
    EvidenceSpan,
    OptionMention,
    Phase,
    ProposalEvidence,
    SupportEvidence,
    VisibleEvidence,
)
from tests.fixtures import append_turn, make_state
from tests.stubs import make_runner


def _span(text: str) -> EvidenceSpan:
    return EvidenceSpan(text=text, start=0)


def _turn_with(state, pid: str, text: str, evidence: VisibleEvidence,
               phase: Phase = Phase.DISCUSSION):
    """Append a turn and pin its accepted evidence explicitly (the validated
    object is the ONLY semantic authority every consumer reads)."""
    record = append_turn(state, pid, text, phase=phase)
    record.evidence = evidence
    return record


class EvidenceIsSoleAuthorityTests(unittest.TestCase):
    def test_support_evidence_counts_as_public_support(self):
        state = make_state()
        text = "The quiet morning slot makes the Museum genuinely workable for us."
        evidence = VisibleEvidence(
            utterance=text,
            mentions=[OptionMention("A", _span("Museum"))],
            supports=[SupportEvidence("A", "firm", _span(text))],
        )
        _turn_with(state, "p2", text, evidence)
        support = public_support(state, include_support_acts=True)
        self.assertIn("p2", support["A"])
        self.assertNotIn("p2", support["B"])

    def test_no_evidence_creates_no_support(self):
        state = make_state()
        text = "The Museum came up earlier, hard to say more."
        _turn_with(state, "p2", text, VisibleEvidence(utterance=text))
        support = public_support(state, include_support_acts=True)
        self.assertNotIn("p2", support["A"])

    def test_acceptance_evidence_counts_as_public_support(self):
        state = make_state()
        text = "Honestly the Museum would be fine by me."
        evidence = VisibleEvidence(
            utterance=text,
            commitments=[CommitmentEvidence("accept", "A", _span(text))],
        )
        _turn_with(state, "p3", text, evidence)
        self.assertIn("p3", public_support(state)["A"])

    def test_no_commitment_evidence_creates_no_vote(self):
        state = make_state()
        text = "The Museum, hmm."
        _turn_with(state, "p1", text, VisibleEvidence(utterance=text), phase=Phase.VOTING)
        self.assertEqual(visible_votes_from_transcript(state), {})
        self.assertNotIn("p1", public_support(state)["A"])

    def test_proposal_evidence_counts(self):
        state = make_state()
        text = "Suppose we settle this with the Museum, would everyone manage?"
        evidence = VisibleEvidence(
            utterance=text,
            proposals=[ProposalEvidence("A", _span(text))],
        )
        _turn_with(state, "p1", text, evidence)
        self.assertIn("A", public_evidence(state).proposals)

    def test_no_proposal_evidence_creates_no_proposal(self):
        state = make_state()
        text = "Could we live with something simpler?"
        _turn_with(state, "p1", text, VisibleEvidence(utterance=text))
        self.assertNotIn("A", public_evidence(state).proposals)


class ObserverAndConsensusAgreeTests(unittest.TestCase):
    def test_observer_and_public_support_consume_the_same_evidence(self):
        """Observer coverage and public support read the SAME accepted evidence
        object — support for A credits A's coverage and public support, nothing
        else, and invents no vote/acceptance."""
        state = make_state()
        runner = make_runner(state)
        text = "The quiet morning slot makes the Museum genuinely workable."
        evidence = VisibleEvidence(
            utterance=text,
            mentions=[OptionMention("A", _span("Museum"))],
            supports=[SupportEvidence("A", "firm", _span(text))],
        )
        record = _turn_with(state, "p2", text, evidence)
        runner._apply_semantics(state, record)
        self.assertEqual(state.coverage["A"].reasons, 1)
        self.assertIn("p2", public_support(state, include_support_acts=True)["A"])
        self.assertEqual(state.coverage["B"].reasons, 0)
        self.assertIsNone(state.runtimes["p2"].explicit_vote)

    def test_moderator_turns_carry_no_public_semantic_signal(self):
        state = make_state()
        record = append_turn(state, "p1", "I can live with the Museum.")
        record.speaker_id = "moderator"  # simulate an evidence-less system line
        record.evidence = None
        support = public_support(state)
        self.assertEqual(support["A"], set())


if __name__ == "__main__":
    unittest.main()
