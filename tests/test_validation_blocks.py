"""Focused tests: every semantic block reason yields one validation code and
one block decision through `_validate_turn_text` (cleanup 8 — `_semantic_block`
is gone; `report.block_state_mutation` is the single mutation-blocking path).
"""

from __future__ import annotations

import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent

from tests.fixtures import make_state, parse_text
from tests.stubs import make_runner


class BlockReasonTests(unittest.TestCase):
    def setUp(self):
        self.state = make_state()
        self.runner = make_runner(self.state)

    def _report(self, pid, text, intent):
        persona = self.state.persona_by_id(pid)
        act = parse_text(self.state, pid, text, intent=intent)
        return self.runner._validate_turn_text(text, self.state, persona, intent, act)

    def _assert_blocks(self, code, pid, text, intent):
        report = self._report(pid, text, intent)
        self.assertIn(code, report.issues)
        self.assertTrue(report.block_state_mutation, f"{code} must block state mutation")
        return report

    def test_empty_blocks(self):
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say it")
        self._assert_blocks("EMPTY", "p1", "", intent)

    def test_malformed_utterance_blocks(self):
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say it")
        self._assert_blocks("MALFORMED_UTTERANCE", "p1", "Just to be clear.", intent)

    def test_invalid_option_reference_blocks(self):
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say it")
        self._assert_blocks("INVALID_OPTION_REFERENCE", "p1", "Option D looks best to me.", intent)

    def test_missing_required_coverage_focus_blocks(self):
        # The coverage ROUTE requires the focus option, not a magic reason string.
        intent = MoveIntent(
            speaker_id="p1", act=ActType.COMPARE,
            reason="bring Escape Room into the discussion and compare it with your current lean",
            route_source="coverage",
            option_focus=["C"],
        )
        self._assert_blocks("MISSING_REQUIRED_OPTION_FOCUS", "p1", "The Museum keeps things simple.", intent)

    def test_implicit_thread_reply_without_any_option_name_is_not_flagged(self):
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="defend it",
            route_source="thread_hot", option_focus=["A"],
        )
        report = self._report("p1", "That cost is fair for what we get, honestly.", intent)
        self.assertNotIn("THREAD_RESPONSE_MISSES_OPTION", report.issues)

    def test_thread_support_realized_as_comment_gets_one_repair_flag(self):
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="defend it",
            route_source="thread_hot", option_focus=["A"],
        )
        # Mentions the option but neither supports nor objects: a neutral aside.
        report = self._report("p1", "The Museum has been discussed a lot today.", intent)
        self.assertIn("SUPPORT_NOT_REALIZED", report.issues)
        self.assertFalse(report.block_state_mutation)

    def test_thread_concern_without_objection_gets_one_repair_flag(self):
        intent = MoveIntent(
            speaker_id="p2", act=ActType.CONCERN, reason="push back",
            route_source="participant_narrowing", option_focus=["A"],
        )
        report = self._report("p2", "The Museum keeps the day easy to adjust.", intent)
        self.assertIn("CONCERN_NOT_REALIZED", report.issues)
        self.assertFalse(report.block_state_mutation)

    def test_concession_first_support_is_not_flagged(self):
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="defend it",
            route_source="thread_hot", option_focus=["A"],
        )
        # Acknowledges the worry, then supports: realizes CONCERN via the
        # soft-objection wording but is a legitimate concession-first defense.
        report = self._report(
            "p1", "I get the cost worry, but the Museum keeps the day easy for everyone.", intent
        )
        self.assertNotIn("SUPPORT_NOT_REALIZED", report.issues)

    def test_decision_fallback_always_parses_to_the_target(self):
        # A stored reason can smuggle in a second commitment phrase, another
        # option, or a question that voids the composed line's parse; the
        # self-check must then emit the minimal guaranteed-parseable form.
        from parsing import visible_commitment
        intent = MoveIntent(
            speaker_id="p1", act=ActType.VOTE, reason="final decision",
            option_focus=["B"], required_vote="B", old_preference="A",
            allow_vote_change=True,
            allowed_reason="the Museum works for me too, doesn't it?",
        )
        from validation import ValidationReport
        report = ValidationReport(["UNBRIDGED_SWITCH"], True)
        persona = self.state.persona_by_id("p1")
        text = self.runner._safe_fallback_text(self.state, persona, intent, report)
        commit = visible_commitment(text, self.runner._resolver, sanctioned_switch=True)
        self.assertIsNotNone(commit)
        self.assertEqual(commit[1], "B")

    def test_normal_route_support_comment_is_not_flagged(self):
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="add a reason",
            route_source="normal", option_focus=["A"],
        )
        report = self._report("p1", "The Museum has been discussed a lot today.", intent)
        self.assertNotIn("SUPPORT_NOT_REALIZED", report.issues)

    def test_unclear_visible_commitment_blocks(self):
        intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="vote", option_focus=["A"])
        self._assert_blocks("UNCLEAR_VISIBLE_COMMITMENT", "p1", "I'm honestly torn on this.", intent)

    def test_required_vote_mismatch_blocks(self):
        intent = MoveIntent(
            speaker_id="p1", act=ActType.VOTE, reason="vote", option_focus=["B"], required_vote="B",
        )
        self._assert_blocks("REQUIRED_VOTE_MISMATCH", "p1", "I vote for the Museum.", intent)

    def test_hard_blocker_accepted_rejected_option_blocks(self):
        self.state.runtimes["p1"].mark_rejected("C", reason_against="booked out")
        intent = MoveIntent(
            speaker_id="p1", act=ActType.VOTE, reason="vote", option_focus=["C"],
            allow_vote_change=True,
        )
        report = self._assert_blocks(
            "HARD_BLOCKER_ACCEPTED_REJECTED_OPTION", "p1", "I vote for the Escape Room.", intent
        )
        self.assertIn("BLOCKED_OPTION_ACCEPTED", report.issues)

    def test_hybrid_compromise_blocks(self):
        intent = MoveIntent(speaker_id="p1", act=ActType.COMPROMISE, reason="middle ground")
        self._assert_blocks(
            "HYBRID_COMPROMISE", "p1",
            "What if we go with the Museum and also the Bike Ride?", intent,
        )

    def test_continuation_repeat_blocks(self):
        self.state.runtimes["p1"].already_said.append("The Museum keeps the day easy to adjust.")
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="add on",
            option_focus=["A"], continuation=True,
        )
        self._assert_blocks(
            "CONTINUATION_REPEATS", "p1", "The Museum keeps the day easy to adjust.", intent
        )

    def test_continuation_topic_jump_blocks(self):
        self.state.runtimes["p1"].already_said.append("The Museum keeps the day easy to adjust.")
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="add on",
            option_focus=["A"], continuation=True,
        )
        self._assert_blocks(
            "CONTINUATION_TOPIC_JUMP", "p1", "Oh, and the Bike Ride is cheap too.", intent
        )

    def test_evasive_required_answer_blocks(self):
        intent = MoveIntent(
            speaker_id="p1", act=ActType.ANSWER, reason="answer the question",
            route_source="answer_required", option_focus=["A"],
        )
        self._assert_blocks(
            "ANSWER_DOES_NOT_ADDRESS_QUESTION", "p1", "What do you all think, though?", intent
        )

    def test_off_target_switch_blocks(self):
        # p2 prefers B; the sanctioned switch offers A, but the line lands on C.
        intent = MoveIntent(
            speaker_id="p2", act=ActType.VOTE, reason="switch or stay",
            option_focus=["A"], allow_vote_change=True,
        )
        self._assert_blocks("OFF_TARGET_SWITCH", "p2", "Actually I choose the Escape Room.", intent)

    def test_unbridged_switch_blocks(self):
        # p2 prefers B and votes A without linking the old stance to the new pick.
        intent = MoveIntent(speaker_id="p2", act=ActType.VOTE, reason="vote", option_focus=["A"])
        self._assert_blocks("UNBRIDGED_SWITCH", "p2", "I vote for the Museum.", intent)

    def test_unsupported_fact_blocks_via_deterministic_grounding(self):
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say it", option_focus=["A"])
        persona = self.state.persona_by_id("p1")
        text = "The Museum tour covers 3 miles of exhibits."
        act = parse_text(self.state, "p1", text, intent=intent)
        report, _ti, _to = self.runner._collect_report(text, self.state, persona, intent, act, [])
        self.assertIn("UNSUPPORTED_FACT", report.issues)
        self.assertTrue(report.block_state_mutation)

    def test_thread_realization_misses_are_reported_without_blocking(self):
        # THREAD_RESPONSE_MISSES_OPTION / COMPARISON_MISSES_OPTIONS stay
        # telemetry: one repair chance, but no mutation block by themselves.
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="defend it",
            route_source="thread_hot", option_focus=["C"],
        )
        report = self._report("p1", "The Museum keeps the day easy to adjust.", intent)
        self.assertIn("THREAD_RESPONSE_MISSES_OPTION", report.issues)
        self.assertFalse(report.block_state_mutation)


if __name__ == "__main__":
    unittest.main()
