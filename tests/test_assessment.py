"""Item 9 (todo_validation.md): explicit action-oriented assessment.

The assessment decides ACCEPT / ACCEPT_WITH_METRIC / REPAIR / FALLBACK / DROP
from validated visible evidence — never from raw issue counts. Every intended
move gets the same realization check on every route, harmless deviations stay
metric-only, and candidate ordering follows assessment severity.
"""

from __future__ import annotations

import unittest

from models import (
    ActType,
    AssessmentAction,
    CommitmentEvidence,
    ComparisonEvidence,
    EvidenceSpan,
    GroundingClaim,
    MoveIntent,
    QuestionEvidence,
    SupportEvidence,
    TurnAssessment,
    ValidationIssue,
    VisibleEvidence,
)
from validation import assessment_severity
from tests.fixtures import make_state
from tests.stubs import make_runner


def _span(text: str) -> EvidenceSpan:
    return EvidenceSpan(text=text, start=0)


class AssessmentActions(unittest.TestCase):
    def setUp(self) -> None:
        self.state = make_state()
        self.runner = make_runner(self.state)

    def assess(self, text, intent, evidence, **kwargs) -> TurnAssessment:
        persona = self.state.persona_by_id(intent.speaker_id)
        return self.runner._assess_candidate(
            text=text, state=self.state, persona=persona, intent=intent,
            evidence=evidence, **kwargs,
        )

    def test_clean_realized_support_is_accepted(self) -> None:
        text = "The Museum keeps the day easy for everyone."
        evidence = VisibleEvidence(
            utterance=text,
            mentions=self.runner._resolver.mentions(text),
            supports=[SupportEvidence("A", "weak", _span(text))],
            primary_act=ActType.SUPPORT,
        )
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say it", option_focus=["A"])
        assessment = self.assess(text, intent, evidence)
        self.assertIs(assessment.action, AssessmentAction.ACCEPT)
        self.assertTrue(assessment.intended_act_realized)
        self.assertTrue(assessment.intended_focus_realized)

    def test_realized_function_with_different_primary_is_metric_only(self) -> None:
        # Intended COMPARE realized by a comparative question: primary ASK,
        # comparison evidence present — a safe deviation, never a repair.
        text = "Which is easier to plan, the Museum or the Escape Room?"
        evidence = VisibleEvidence(
            utterance=text,
            mentions=self.runner._resolver.mentions(text),
            comparisons=[ComparisonEvidence(option_ids=["A", "C"], span=_span(text))],
            questions=[QuestionEvidence(scope="group", span=_span(text))],
            primary_act=ActType.ASK,
        )
        intent = MoveIntent(speaker_id="p1", act=ActType.COMPARE, reason="compare", option_focus=["A", "C"])
        assessment = self.assess(text, intent, evidence)
        self.assertIs(assessment.action, AssessmentAction.ACCEPT_WITH_METRIC)
        self.assertTrue(assessment.intended_act_realized)
        self.assertEqual(assessment.issues, [])

    def test_missing_intended_support_is_metric_only_on_every_route(self) -> None:
        # Item 11: an unrealized soft function is telemetry, never a repair —
        # a neutral aside under a SUPPORT intent is safe to print. The
        # realization result stays visible for metrics.
        text = "The Museum has been discussed a lot today."
        evidence = VisibleEvidence(
            utterance=text, mentions=self.runner._resolver.mentions(text),
            primary_act=ActType.COMMENT,
        )
        for route in ("normal", "thread_hot", "coverage_adjacent"):
            intent = MoveIntent(
                speaker_id="p1", act=ActType.SUPPORT, reason="defend",
                route_source=route, option_focus=["A"],
            )
            assessment = self.assess(text, intent, evidence)
            self.assertIs(assessment.action, AssessmentAction.ACCEPT_WITH_METRIC, route)
            codes = [i.code for i in assessment.issues]
            self.assertIn("SUPPORT_NOT_REALIZED", codes, route)
            self.assertFalse(any(i.blocking for i in assessment.issues), route)
            self.assertFalse(assessment.intended_act_realized)

    def test_wrong_intended_option_is_a_blocking_mismatch(self) -> None:
        text = "The Bike Ride keeps the day active."
        evidence = VisibleEvidence(
            utterance=text, mentions=self.runner._resolver.mentions(text),
            supports=[SupportEvidence("B", "weak", _span(text))],
            primary_act=ActType.SUPPORT,
        )
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="defend", option_focus=["A"])
        assessment = self.assess(text, intent, evidence)
        issue = next(i for i in assessment.issues if i.code == "WRONG_OPTION_FOCUS")
        self.assertTrue(issue.blocking)
        self.assertIs(assessment.action, AssessmentAction.REPAIR)

    def test_vote_without_commitment_blocks(self) -> None:
        text = "I really like the Bike Ride for this."
        evidence = VisibleEvidence(
            utterance=text, mentions=self.runner._resolver.mentions(text),
            supports=[SupportEvidence("B", "firm", _span(text))],
            primary_act=ActType.SUPPORT,
        )
        intent = MoveIntent(speaker_id="p2", act=ActType.VOTE, reason="vote", option_focus=["B"])
        assessment = self.assess(text, intent, evidence)
        codes = {i.code: i.blocking for i in assessment.issues}
        self.assertTrue(codes.get("UNCLEAR_VISIBLE_COMMITMENT"))
        self.assertFalse(assessment.intended_act_realized)

    def test_required_vote_mismatch_blocks(self) -> None:
        text = "I vote for the Museum."
        evidence = VisibleEvidence(
            utterance=text, mentions=self.runner._resolver.mentions(text),
            commitments=[CommitmentEvidence("vote", "A", _span(text))],
            primary_act=ActType.VOTE,
        )
        intent = MoveIntent(
            speaker_id="p1", act=ActType.VOTE, reason="vote",
            option_focus=["B"], required_vote="B",
        )
        assessment = self.assess(text, intent, evidence)
        issue = next(i for i in assessment.issues if i.code == "REQUIRED_VOTE_MISMATCH")
        self.assertTrue(issue.blocking)

    def test_unsupported_claim_is_blocking_with_span_and_reason(self) -> None:
        text = "The Museum has free entry on Saturdays."
        claim = GroundingClaim(
            span=EvidenceSpan("free entry on Saturdays", 15), kind="invented_detail",
            option_id="A", supported=False, reason="concrete detail not present in the scenario",
        )
        evidence = VisibleEvidence(
            utterance=text, mentions=self.runner._resolver.mentions(text),
            claims=[claim], primary_act=ActType.COMMENT,
        )
        intent = MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="say it")
        assessment = self.assess(text, intent, evidence)
        issue = next(i for i in assessment.issues if i.code.startswith("UNSUPPORTED_CLAIM"))
        self.assertTrue(issue.blocking)
        self.assertEqual(issue.span, "free entry on Saturdays")
        self.assertIn("not present", issue.explanation)

    def test_operational_validator_failure_falls_back_not_open(self) -> None:
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say it")
        assessment = self.assess(
            "Perfectly fine text.", intent, None, operational_failure=True
        )
        self.assertIs(assessment.action, AssessmentAction.FALLBACK)
        self.assertEqual(assessment.issues[0].code, "VALIDATOR_UNAVAILABLE")
        self.assertTrue(assessment.issues[0].blocking)


class SeverityOrdering(unittest.TestCase):
    def test_one_blocking_issue_outweighs_many_metric_issues(self) -> None:
        one_blocking = TurnAssessment(
            action=AssessmentAction.REPAIR,
            issues=[ValidationIssue("UNSUPPORTED_CLAIM:invented_detail", blocking=True)],
        )
        many_soft = TurnAssessment(
            action=AssessmentAction.REPAIR,
            issues=[ValidationIssue(f"CODE_{i}") for i in range(5)],
        )
        self.assertGreater(assessment_severity(one_blocking), assessment_severity(many_soft))

    def test_actions_order_correctly(self) -> None:
        ordered = [
            TurnAssessment(action=AssessmentAction.ACCEPT),
            TurnAssessment(action=AssessmentAction.ACCEPT_WITH_METRIC),
            TurnAssessment(action=AssessmentAction.REPAIR),
            TurnAssessment(action=AssessmentAction.FALLBACK),
            TurnAssessment(action=AssessmentAction.DROP),
        ]
        severities = [assessment_severity(a) for a in ordered]
        self.assertEqual(severities, sorted(severities))


class PipelineUsesAssessment(unittest.TestCase):
    def test_final_record_carries_evidence_and_assessment(self) -> None:
        state = make_state()
        runner = make_runner(state, ["The Museum keeps the day easy for everyone."])
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say it", option_focus=["A"])
        record = runner._generate_and_append(state, intent)
        self.assertIsNotNone(record.evidence)
        self.assertIsNotNone(record.assessment)
        self.assertIs(record.assessment.action, AssessmentAction.ACCEPT)
        self.assertEqual([s.option_id for s in record.evidence.supports], ["A"])

    def test_unrealized_soft_function_prints_without_any_repair_call(self) -> None:
        # Item 11: a safe line with only a non-blocking realization miss
        # prints immediately — no repair round is spent on it.
        state = make_state()
        runner = make_runner(state, [
            "The Museum has been discussed a lot today.",
        ])
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="defend", option_focus=["A"])
        record = runner._generate_and_append(state, intent)
        self.assertFalse(record.repaired)
        self.assertIs(record.assessment.action, AssessmentAction.ACCEPT_WITH_METRIC)
        self.assertFalse(record.state_mutation_blocked)
        self.assertEqual(record.text, "The Museum has been discussed a lot today.")


if __name__ == "__main__":
    unittest.main()
