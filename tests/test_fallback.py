"""Item 12 (todo_validation.md): truthful, state-safe fallback only.

Retained families: explicit vote / grounded vote switch, hard-blocker
restatement, coverage request, exact factual comparison, and exact listed /
does-not-say answers. Generic support/concern/compromise stand-ins are gone;
every retained fallback passes through the complete candidate pipeline, and
unsafe cases drop the turn instead of printing false evidence.
"""

from __future__ import annotations

import unittest

from models import ActType, AssessmentAction, MoveIntent
from parsing import visible_commitment
from tests.fixtures import make_persona, make_state
from tests.stubs import make_runner


def _intent(act: ActType, speaker="p1", **kwargs) -> MoveIntent:
    return MoveIntent(speaker_id=speaker, act=act, reason="fallback test", **kwargs)


class FallbackFamilies(unittest.TestCase):
    def setUp(self) -> None:
        self.state = make_state()
        self.runner = make_runner(self.state)

    def fallback(self, intent, issue_codes=()):
        persona = self.state.persona_by_id(intent.speaker_id)
        return self.runner._fallback_candidate(self.state, persona, intent, list(issue_codes))

    def test_generic_discussion_fallback_is_gone(self) -> None:
        text, family = self.fallback(_intent(ActType.COMMENT))
        self.assertIsNone(text)
        self.assertEqual(family, "")
        for act in (ActType.PROCESS, ActType.ASK):
            text, _family = self.fallback(_intent(act))
            self.assertIsNone(text, act)

    def test_minimal_vote_fallback_parses_to_target(self) -> None:
        text, family = self.fallback(
            _intent(ActType.VOTE, option_focus=["B"], required_vote="B"),
            ["UNCLEAR_VISIBLE_COMMITMENT"],
        )
        self.assertEqual(family, "vote")
        self.assertEqual(visible_commitment(text, self.runner._resolver), ("vote", "B"))

    def test_support_fallback_is_gone(self) -> None:
        # Item 12: no fallback may fabricate a supportive stance on the
        # sim's behalf — an unsafe support turn drops instead.
        text, family = self.fallback(_intent(ActType.SUPPORT, option_focus=["A"]))
        self.assertIsNone(text)
        self.assertEqual(family, "")

    def test_ordinary_concern_fallback_is_gone(self) -> None:
        # A non-blocker concern is a stance; only the grounded hard-blocker
        # restatement remains a truthful deterministic form.
        text, family = self.fallback(_intent(ActType.CONCERN, speaker="p2", option_focus=["A"]))
        self.assertIsNone(text)
        self.assertEqual(family, "")

    def test_compromise_fallback_is_gone(self) -> None:
        text, family = self.fallback(_intent(ActType.COMPROMISE, option_focus=["B"]))
        self.assertIsNone(text)
        self.assertEqual(family, "")

    def test_blocker_restatement_uses_stored_grounded_reason(self) -> None:
        self.state.runtimes["p1"].mark_rejected("C", reason_against="the booking cannot move")
        text, family = self.fallback(_intent(ActType.CONCERN, option_focus=["C"]))
        self.assertEqual(family, "blocker_restate")
        self.assertIn("doesn't work for me", text)
        self.assertIn("booking cannot move", text)

    def test_comparison_fallback_uses_exact_listed_attributes(self) -> None:
        text, family = self.fallback(_intent(ActType.COMPARE, option_focus=["A", "B"]))
        self.assertEqual(family, "comparison")
        self.assertIn("24 euros", text)
        self.assertIn("12 euros", text)
        self.assertIn("versus", text)

    def test_answer_fallback_answers_listed_attribute(self) -> None:
        from tests.fixtures import append_turn
        record = append_turn(self.state, "p2", "How long does the Museum take?")
        intent = _intent(
            ActType.ANSWER, option_focus=["A"],
            route_source="answer_required", respond_to_turn=record.index,
        )
        text, family = self.fallback(intent)
        self.assertEqual(family, "answer_listed")
        self.assertIn("duration: 4 hours", text)

    def test_answer_fallback_says_not_listed_when_unknown(self) -> None:
        from tests.fixtures import append_turn
        record = append_turn(self.state, "p2", "Is there parking at the Museum?")
        intent = _intent(
            ActType.ANSWER, option_focus=["A"],
            route_source="answer_required", respond_to_turn=record.index,
        )
        text, family = self.fallback(intent)
        self.assertEqual(family, "answer_unknown")
        self.assertIn("don't say", text)

    def test_continuation_fallback_is_dropped(self) -> None:
        text, family = self.fallback(_intent(ActType.SUPPORT, option_focus=["A"], continuation=True))
        self.assertIsNone(text)

    def test_compromise_fallback_never_offers_a_rejected_option(self) -> None:
        self.state.runtimes["p1"].mark_rejected("A", reason_against="cannot do it")
        text, _family = self.fallback(_intent(ActType.COMPROMISE, option_focus=["A"]))
        self.assertIsNone(text)

    # ---- Item 4: minimal, public, truthful decision fallback ----

    def test_vote_fallback_never_leaks_controller_rationale(self) -> None:
        # allowed_reason / old_preference are controller-facing; none of that
        # wording may reach the public vote text.
        text, family = self.fallback(_intent(
            ActType.VOTE, option_focus=["B"], required_vote="B",
            allow_vote_change=True, old_preference="A",
            allowed_reason="B has the clearest visible support now",
        ))
        self.assertEqual(family, "vote")
        self.assertNotIn("support now", text)
        self.assertNotIn("preferred", text.lower())
        self.assertNotIn("defensible", text.lower())
        self.assertNotIn("visible discussion", text.lower())

    def test_vote_fallback_is_minimal_without_a_prior_public_commitment(self) -> None:
        # No accepted public vote for this sim yet -> the fallback text is a plain
        # vote, never a fabricated "switch"/"I preferred A", even though a private
        # preference and controller old_preference differ from the target. (Whether
        # such a pushed switch is ultimately accepted or dropped by the shared
        # UNBRIDGED_SWITCH guard is a separate pipeline decision.)
        self.assertIsNone(self.state.runtimes["p1"].explicit_vote)
        text, _family = self.fallback(_intent(
            ActType.VOTE, option_focus=["B"], required_vote="B",
            allow_vote_change=True, old_preference="A",
        ))
        self.assertNotIn("switching", text.lower())
        self.assertNotIn("preferred", text.lower())
        self.assertEqual(visible_commitment(text, self.runner._resolver), ("vote", "B"))

    def test_vote_fallback_states_a_switch_only_from_a_public_commitment(self) -> None:
        # A prior accepted public vote to A exists; switching to B may name A as
        # the visible bridge, with no fabricated reason clause.
        self.state.runtimes["p1"].explicit_vote = "A"
        text, _family = self.fallback(_intent(
            ActType.VOTE, option_focus=["B"], required_vote="B",
            allow_vote_change=True, old_preference="A",
        ))
        self.assertIn("switching", text.lower())
        commit = visible_commitment(text, self.runner._resolver, sanctioned_switch=True)
        self.assertEqual(commit, ("vote", "B"))
        # No invented reason wording.
        self.assertNotIn("because", text.lower())
        self.assertNotIn("but", text.lower())


class FallbackIsFullyValidated(unittest.TestCase):
    def test_fallback_goes_through_the_complete_pipeline(self) -> None:
        # Both generation and repair return malformed fragments; the factual
        # comparison fallback replaces them and is itself interpreted and
        # assessed like any generated candidate.
        state = make_state()
        runner = make_runner(state, ["Just to be clear.", "Just to be clear."])
        intent = _intent(ActType.COMPARE, option_focus=["A", "B"], route_source="thread_hot")
        record = runner._generate_and_append(state, intent)
        self.assertTrue(record.used_fallback)
        self.assertEqual(record.fallback_family, "comparison")
        self.assertIsNotNone(record.evidence)
        self.assertIn(record.assessment.action,
                      {AssessmentAction.ACCEPT, AssessmentAction.ACCEPT_WITH_METRIC})
        self.assertEqual(
            [set(c.option_ids) for c in record.evidence.comparisons], [{"A", "B"}]
        )
        self.assertEqual(record.evidence.commitments, [])
        self.assertFalse(record.state_mutation_blocked)

    def test_unsafe_case_drops_instead_of_printing(self) -> None:
        # A COMMENT intent has no truthful act-specific fallback: after a
        # malformed generation and repair, the turn is dropped, consuming the
        # route attempt without fabricating evidence.
        state = make_state()
        runner = make_runner(state, ["Just to be clear.", "Just to be clear."])
        record = runner._generate_and_append(state, _intent(ActType.COMMENT))
        self.assertEqual(record.text, "")
        self.assertTrue(record.state_mutation_blocked)
        self.assertIs(record.assessment.action, AssessmentAction.DROP)
        self.assertFalse(record.used_fallback)

    def test_blocker_restate_fallback_creates_no_unintended_semantics(self) -> None:
        state = make_state()
        state.runtimes["p2"].mark_rejected("A", reason_against="the queue times are unmanageable")
        runner = make_runner(state, ["Just to be clear.", "Just to be clear."])
        intent = _intent(ActType.CONCERN, speaker="p2", option_focus=["A"], route_source="thread_hot")
        record = runner._generate_and_append(state, intent)
        self.assertTrue(record.used_fallback)
        self.assertEqual(record.fallback_family, "blocker_restate")
        evidence = record.evidence
        self.assertEqual(evidence.commitments, [])
        self.assertEqual(evidence.questions, [])
        self.assertEqual([c.option_id for c in evidence.concerns], ["A"])
        # No rank movement for anyone else, no vote for the speaker.
        self.assertIsNone(state.runtimes["p2"].explicit_vote)

    def test_unsafe_concern_drops_instead_of_fabricating_a_worry(self) -> None:
        state = make_state()
        runner = make_runner(state, ["Just to be clear.", "Just to be clear."])
        intent = _intent(ActType.CONCERN, speaker="p2", option_focus=["A"], route_source="thread_hot")
        record = runner._generate_and_append(state, intent)
        self.assertFalse(record.used_fallback)
        self.assertEqual(record.text, "")
        self.assertTrue(record.state_mutation_blocked)


class HardBlockerVoteSafety(unittest.TestCase):
    def test_decision_fallback_never_accepts_a_rejected_option(self) -> None:
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", rejection="B",
                         rejection_reason="cannot accept B"),
            make_persona("p2", "Jonas", preferred="B"),
            make_persona("p3", "Lea", preferred="C"),
        ])
        runner = make_runner(state)
        intent = _intent(ActType.VOTE, option_focus=["B"], allow_vote_change=True)
        text, family = runner._fallback_candidate(
            state, state.persona_by_id("p1"), intent, ["BLOCKED_OPTION_ACCEPTED"]
        )
        self.assertEqual(family, "vote")
        commit = visible_commitment(text, runner._resolver, sanctioned_switch=True)
        self.assertIsNotNone(commit)
        self.assertNotEqual(commit[1], "B")


if __name__ == "__main__":
    unittest.main()
