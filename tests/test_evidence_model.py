"""Item 3 (todo_validation.md): the typed visible-evidence contract.

Proves the new multi-label model can express every semantic category in the
item-1 fixture corpus without destructive precedence, keeps option-specific
bindings and exact spans, serializes through the existing logger path, and
attaches to TurnRecord without behavior changes.
"""

from __future__ import annotations

import json
import unittest

from logger import _to_jsonable
from models import (
    ANSWER_COMPLETENESS,
    AnswerEvidence,
    AssessmentAction,
    BlockerEvidence,
    CLAIM_KINDS,
    COMMITMENT_KINDS,
    CommitmentEvidence,
    ComparisonEvidence,
    CONCERN_SEVERITIES,
    ConcernEvidence,
    EvidenceSpan,
    GroundingClaim,
    ProposalEvidence,
    QUESTION_SCOPES,
    QuestionEvidence,
    SofteningEvidence,
    SUPPORT_STRENGTHS,
    SupportEvidence,
    SwitchEvidence,
    TurnAssessment,
    TurnRecord,
    ValidationIssue,
    VisibleEvidence,
)
from tests import semantic_fixtures as sf


def _span(fixture: sf.SemanticFixture, text: str | None = None) -> EvidenceSpan:
    """Locate a span inside the fixture utterance (whole utterance by default)."""
    if text is None:
        return EvidenceSpan(text=fixture.text, start=0)
    start = fixture.text.find(text)
    return EvidenceSpan(text=text, start=start)


def evidence_from_fixture(fixture: sf.SemanticFixture) -> VisibleEvidence:
    """Map a labelled fixture's expected semantics into the typed contract."""
    evidence = VisibleEvidence(utterance=fixture.text)
    for option, strength in fixture.support:
        evidence.supports.append(SupportEvidence(option, strength, _span(fixture)))
    for option, severity in fixture.concerns:
        evidence.concerns.append(ConcernEvidence(option, severity, _span(fixture)))
    for comparison in fixture.comparisons:
        evidence.comparisons.append(ComparisonEvidence(
            option_ids=list(comparison.options), span=_span(fixture),
            favored=comparison.favored, dimension=comparison.dimension,
        ))
    for question in fixture.questions:
        evidence.questions.append(QuestionEvidence(
            scope=question.scope, span=_span(fixture),
            addressee_id=question.addressee, option_ids=list(question.options),
        ))
    if fixture.answer:
        evidence.answers.append(AnswerEvidence(
            completeness=fixture.answer.completeness, span=_span(fixture),
            addresses_target=fixture.answer.addresses_target,
        ))
    if fixture.softens_toward or fixture.concession:
        evidence.softenings.append(SofteningEvidence(
            span=_span(fixture), option_id=fixture.softens_toward,
            concession=fixture.concession,
        ))
    if fixture.proposes:
        evidence.proposals.append(ProposalEvidence(fixture.proposes, _span(fixture)))
    if fixture.commitment:
        kind, option = fixture.commitment
        evidence.commitments.append(CommitmentEvidence(kind, option, _span(fixture)))
    if fixture.switch:
        evidence.switches.append(SwitchEvidence(
            target=fixture.switch.target, span=_span(fixture),
            source=fixture.switch.source,
            reason_span=_span(fixture) if fixture.switch.has_visible_reason else None,
        ))
    if fixture.blocker_raised:
        evidence.blockers.append(BlockerEvidence(fixture.blocker_raised, "raised", _span(fixture)))
    if fixture.blocker_resolved:
        evidence.blockers.append(BlockerEvidence(fixture.blocker_resolved, "resolved", _span(fixture)))
    for claim in fixture.claims:
        evidence.claims.append(GroundingClaim(
            span=_span(fixture, claim.span), kind=claim.kind, option_id=claim.option,
        ))
    if fixture.ambiguous_reference:
        evidence.ambiguous_references.append(_span(fixture))
    return evidence


class ContractExpressiveness(unittest.TestCase):
    def test_every_fixture_maps_losslessly(self) -> None:
        for fixture in sf.FIXTURES:
            evidence = evidence_from_fixture(fixture)
            self.assertEqual(
                evidence.evidence_kinds(), fixture.evidence_kinds(), fixture.fixture_id
            )
            self.assertEqual(len(evidence.claims), len(fixture.claims), fixture.fixture_id)
            self.assertEqual(
                bool(evidence.ambiguous_references), fixture.ambiguous_reference, fixture.fixture_id
            )
            for claim in evidence.claims:
                self.assertGreaterEqual(claim.span.start, 0, fixture.fixture_id)
                self.assertIn(claim.span.text, fixture.text, fixture.fixture_id)

    def test_vocabularies_agree_with_fixture_corpus(self) -> None:
        self.assertEqual(
            {sf.SUPPORT_WEAK, sf.SUPPORT_CONDITIONAL, sf.SUPPORT_FIRM}, set(SUPPORT_STRENGTHS)
        )
        self.assertEqual({sf.CONCERN_ORDINARY, sf.CONCERN_HARD}, set(CONCERN_SEVERITIES))
        self.assertEqual({sf.Q_DIRECT, sf.Q_GROUP, sf.Q_RHETORICAL}, set(QUESTION_SCOPES))
        self.assertEqual(
            {sf.ANSWER_FULL, sf.ANSWER_PARTIAL, sf.ANSWER_EVASIVE, sf.ANSWER_UNRELATED},
            set(ANSWER_COMPLETENESS),
        )
        self.assertEqual({sf.COMMIT_VOTE, sf.COMMIT_ACCEPT}, set(COMMITMENT_KINDS))
        fixture_claim_kinds = {
            sf.CLAIM_LISTED_FACT, sf.CLAIM_ARITHMETIC, sf.CLAIM_OPINION, sf.CLAIM_INFERENCE,
            sf.CLAIM_UNCERTAINTY, sf.CLAIM_INVENTED, sf.CLAIM_CROSS_OPTION,
        }
        self.assertLessEqual(fixture_claim_kinds, set(CLAIM_KINDS))

    def test_multi_function_turn_keeps_all_evidence(self) -> None:
        # The canonical example: concern + switch commitment + reason + group
        # question in one line — nothing erased by precedence.
        fixture = sf.by_id("switch_with_reason_multi")
        evidence = evidence_from_fixture(fixture)
        self.assertEqual([c.option_id for c in evidence.concerns], ["A"])
        self.assertEqual([c.option_id for c in evidence.commitments], ["B"])
        self.assertEqual(evidence.switches[0].source, "A")
        self.assertIsNotNone(evidence.switches[0].reason_span)
        self.assertEqual(evidence.questions[0].scope, sf.Q_GROUP)

    def test_option_specific_binding_not_global(self) -> None:
        fixture = sf.by_id("multi_option_split")
        evidence = evidence_from_fixture(fixture)
        self.assertEqual({s.option_id for s in evidence.supports}, {"A"})
        self.assertEqual({c.option_id for c in evidence.concerns}, {"C"})

    def test_sole_commitment_rejects_conflicts(self) -> None:
        span = EvidenceSpan("x", 0)
        single = VisibleEvidence(commitments=[CommitmentEvidence("vote", "A", span)])
        self.assertEqual(single.sole_commitment().option_id, "A")
        conflicting = VisibleEvidence(commitments=[
            CommitmentEvidence("vote", "A", span), CommitmentEvidence("accept", "B", span),
        ])
        self.assertIsNone(conflicting.sole_commitment())
        self.assertIsNone(VisibleEvidence().sole_commitment())


class SerializationAndTurnRecord(unittest.TestCase):
    def test_evidence_serializes_through_logger_path(self) -> None:
        evidence = evidence_from_fixture(sf.by_id("switch_with_reason_multi"))
        payload = _to_jsonable(evidence)
        text = json.dumps(payload)
        self.assertIn("commitments", payload)
        self.assertIn("reason_span", text)
        self.assertEqual(payload["concerns"][0]["option_id"], "A")

    def test_assessment_serializes_with_action_value(self) -> None:
        assessment = TurnAssessment(
            action=AssessmentAction.REPAIR,
            issues=[ValidationIssue(code="UNSUPPORTED_FACT", explanation="invented price",
                                    span="costs 5 euros", option_id="A")],
            intended_act_realized=False,
        )
        payload = _to_jsonable(assessment)
        json.dumps(payload)
        self.assertEqual(payload["action"], "repair")
        self.assertEqual(payload["issues"][0]["code"], "UNSUPPORTED_FACT")

    def test_turn_record_fields_default_to_none_and_serialize(self) -> None:
        from models import Phase
        record = TurnRecord(
            index=1, speaker_id="p1", speaker_name="Mira", text="hi",
            phase=Phase.DISCUSSION,
        )
        self.assertIsNone(record.evidence)
        self.assertIsNone(record.assessment)
        payload = _to_jsonable(record)
        json.dumps(payload)
        self.assertIsNone(payload["evidence"])
        record.evidence = VisibleEvidence(utterance="hi")
        record.assessment = TurnAssessment()
        payload = _to_jsonable(record)
        json.dumps(payload)
        self.assertEqual(payload["assessment"]["action"], "accept")


if __name__ == "__main__":
    unittest.main()
