"""Focused regressions for the post-authority-split stabilization TODO."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401
import simulator as sim_policy
from consensus import public_evidence, public_participant_ledger
from models import (
    ActType,
    ConcernEvidence,
    EvidenceSpan,
    MoveIntent,
    Phase,
    ProposalEvidence,
    SupportEvidence,
    VisibleEvidence,
)
from tests.evidence_adapter import derive_evidence
from tests.fixtures import append_turn, make_persona, make_resolver, make_state
from tests.stubs import make_runner


class PublicEvidenceBoundaryTests(unittest.TestCase):
    def test_private_preferences_cannot_create_candidate_or_top_pair(self):
        state = make_state()
        state.runtimes["p1"].option_ranks.update({"A": 1, "B": 5, "C": 1})
        state.runtimes["p2"].option_ranks.update({"A": 5, "B": 1, "C": 1})
        state.personas[2].private_goal = "secretly force the Escape Room"
        evidence = public_evidence(state)
        self.assertEqual(evidence.candidate_leaders, ())
        self.assertEqual(evidence.top_pair, ())
        self.assertEqual(evidence.candidate_scores, {})

    def test_visible_single_option_proposal_affects_candidate_scoring(self):
        state = make_state()
        text = "Could we choose the Bike Ride as common ground?"
        append_turn(
            state,
            "p1",
            text,
            intent=MoveIntent(
                "p1", ActType.COMPROMISE, "offer public common ground", option_focus=["B"]
            ),
            evidence=VisibleEvidence(
                utterance=text,
                proposals=[ProposalEvidence("B", EvidenceSpan(text, 0))],
            ),
        )
        evidence = public_evidence(state)
        self.assertEqual(evidence.proposals, {"B"})
        self.assertEqual(evidence.candidate_leaders, ("B",))

    def test_public_objection_breaks_an_otherwise_equal_candidate_tie(self):
        state = make_state()
        for pid, option, text in (
            ("p1", "A", "The Museum works well for me."),
            ("p2", "B", "The Bike Ride works well for me."),
        ):
            append_turn(
                state,
                pid,
                text,
                intent=MoveIntent(pid, ActType.SUPPORT, "support", option_focus=[option]),
                evidence=VisibleEvidence(
                    utterance=text,
                    supports=[SupportEvidence(option, "firm", EvidenceSpan(text, 0))],
                ),
            )
        concern = "The Museum feels too quiet for a full day."
        append_turn(
            state,
            "p3",
            concern,
            intent=MoveIntent("p3", ActType.CONCERN, "object", option_focus=["A"]),
            evidence=VisibleEvidence(
                utterance=concern,
                concerns=[ConcernEvidence("A", "ordinary", EvidenceSpan(concern, 0))],
            ),
        )
        evidence = public_evidence(state)
        self.assertGreater(evidence.objection_counts["A"], evidence.objection_counts["B"])
        self.assertEqual(evidence.candidate_leaders, ("B",))

    def test_public_social_ledger_contains_no_other_sim_private_state(self):
        state = make_state()
        text = "The Bike Ride works for me because it is inexpensive."
        append_turn(
            state,
            "p2",
            text,
            intent=MoveIntent("p2", ActType.SUPPORT, "support", option_focus=["B"]),
            evidence=VisibleEvidence(
                utterance=text,
                supports=[SupportEvidence("B", "firm", EvidenceSpan(text, 0))],
            ),
        )
        state.runtimes["p2"].reasons_for["A"] = "hidden reason"
        state.personas[1].private_goal = "hidden private goal"
        ledger = public_participant_ledger(state)["p2"]
        self.assertEqual(ledger.public_position, "B")
        self.assertIn("B", ledger.supported_options)
        serialized = repr(ledger)
        self.assertNotIn("hidden reason", serialized)
        self.assertNotIn("hidden private goal", serialized)


class StallPacingTests(unittest.TestCase):
    def test_repeated_empty_rounds_do_not_narrow_before_hard_cap_when_below_minimum(self):
        state = make_state()
        state.phase = Phase.DISCUSSION
        state.min_discussion_turns = 5
        state.force_narrow_turns = 8
        state.hard_max_turns = 6
        runner = make_runner(state)
        runner._ready_to_narrow = lambda _state: False
        runner._pending_answer_obligation = lambda _state: None
        runner._maybe_moderator_nudge = lambda _state: None
        runner._handle_stall = lambda _state: False

        def silent_round(s, _stimulus):
            s.bid_round_count += 1
            s.no_bid_round_count += 1
            return None

        transitions: list[tuple[Phase, int]] = []
        runner._run_open_floor_turn = silent_round

        def mark(s, phase, _reason):
            transitions.append((phase, s.no_bid_round_count))
            s.phase = phase

        runner._mark_phase = mark
        runner._discussion_loop(state)
        self.assertEqual(transitions, [(Phase.NARROWING, state.hard_max_turns)])
        self.assertLess(0, state.min_discussion_turns)


class InterpretationContractTests(unittest.TestCase):
    def setUp(self):
        self.state = make_state()
        self.resolver = make_resolver(self.state.scenario)
        self.names = {p.id: p.name for p in self.state.personas}

    def evidence(self, text: str, intent: MoveIntent):
        return derive_evidence(
            text,
            self.resolver,
            speaker_id=intent.speaker_id,
            participant_names=self.names,
            intent=intent,
        )

    def test_natural_support_predicate_binds_locally(self):
        intent = MoveIntent("p1", ActType.SUPPORT, "support", option_focus=["A"])
        ev = self.evidence("The Museum clearly addresses our need for a low-effort day.", intent)
        self.assertEqual([item.option_id for item in ev.supports], ["A"])

    def test_negated_positive_predicate_does_not_create_support(self):
        intent = MoveIntent("p1", ActType.CONCERN, "object", option_focus=["A"])
        ev = self.evidence("The Museum does not work for me because it feels too quiet.", intent)
        self.assertEqual(ev.supports, [])
        self.assertEqual([item.option_id for item in ev.concerns], ["A"])

    def test_natural_single_option_compromise_is_visible(self):
        intent = MoveIntent("p1", ActType.COMPROMISE, "offer", option_focus=["B"])
        ev = self.evidence("Could we choose the Bike Ride as common ground?", intent)
        self.assertEqual([item.option_id for item in ev.proposals], ["B"])


class SocialPolicyTests(unittest.TestCase):
    def test_ask_targets_visible_owner_of_public_position(self):
        state = make_state()
        text = "The Bike Ride works best for me."
        append_turn(
            state,
            "p2",
            text,
            intent=MoveIntent("p2", ActType.SUPPORT, "support", option_focus=["B"]),
            evidence=VisibleEvidence(
                utterance=text,
                supports=[SupportEvidence("B", "firm", EvidenceSpan(text, 0))],
            ),
        )
        view = sim_policy.build_view(state, "p1")
        intent = sim_policy._build_open_intent(state, view, ActType.ASK)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.addressee_id, "p2")

    def test_comment_has_no_unconditional_baseline(self):
        state = make_state()
        view = sim_policy.build_view(state, "p1")
        scores = sim_policy._score_acts(state, view)
        self.assertEqual(scores[ActType.COMMENT], 0.0)

    def test_ask_concern_and_compromise_are_available_in_suitable_public_states(self):
        state = make_state([
            make_persona("p1", "Mira", preferred="A", switch_resistance=0.1),
            make_persona("p2", "Jonas", preferred="B"),
            make_persona("p3", "Lea", preferred="B"),
        ])
        state.runtimes["p1"].mark_acceptable("B", reason_for="still workable")
        for pid in ("p2", "p3"):
            text = "The Bike Ride works well for me."
            append_turn(
                state,
                pid,
                text,
                intent=MoveIntent(pid, ActType.SUPPORT, "support", option_focus=["B"]),
                evidence=VisibleEvidence(
                    utterance=text,
                    supports=[SupportEvidence("B", "firm", EvidenceSpan(text, 0))],
                ),
            )
        view = sim_policy.build_view(state, "p1")
        scores = sim_policy._score_acts(state, view)
        self.assertGreaterEqual(scores[ActType.ASK], sim_policy._MIN_ACT_SCORE)
        self.assertGreaterEqual(scores[ActType.CONCERN], sim_policy._MIN_ACT_SCORE)
        self.assertGreaterEqual(scores[ActType.COMPROMISE], sim_policy._MIN_ACT_SCORE)

    def test_contribution_key_remains_used_after_it_leaves_prompt_window(self):
        state = make_state()
        view = sim_policy.build_view(state, "p1")
        first = sim_policy._build_open_intent(state, view, ActType.SUPPORT)
        self.assertIsNotNone(first)
        append_turn(
            state,
            "p1",
            "The Museum works well for me because it is easy to adjust.",
            intent=first,
            evidence=VisibleEvidence(
                utterance="The Museum works well for me because it is easy to adjust.",
                supports=[SupportEvidence("A", "firm", EvidenceSpan("The Museum works well for me because it is easy to adjust.", 0))],
            ),
        )
        for index in range(12):
            pid = "p2" if index % 2 == 0 else "p3"
            append_turn(state, pid, f"Public filler contribution {index}.")
        later = sim_policy.build_view(state, "p1")
        self.assertFalse(sim_policy._contribution_available(later, ActType.SUPPORT, ["A"], "support"))


class OpeningBudgetTests(unittest.TestCase):
    def test_opening_word_ranges_follow_verbosity(self):
        state = make_state([
            make_persona("p1", "Low", verbosity=0.1),
            make_persona("p2", "Medium", verbosity=0.5),
            make_persona("p3", "High", verbosity=0.9),
        ])
        runner = make_runner(state)
        intent = MoveIntent("p1", ActType.OPENING, "open", option_focus=["A"])
        self.assertEqual(runner._word_bounds(intent, state.persona_by_id("p1")), (10, 18))
        self.assertEqual(runner._word_bounds(intent, state.persona_by_id("p2")), (18, 28))
        self.assertEqual(runner._word_bounds(intent, state.persona_by_id("p3")), (25, 38))


class ProcedureOwnershipTests(unittest.TestCase):
    def test_framework_peer_procedure_helpers_are_absent(self):
        from dialogue import DialogueRunner

        for name in (
            "_append_peer_procedure",
            "_procedural_speaker",
            "_emit_narrowing_reaction",
            "_emit_peer_closing",
            "_emit_unresolved_acknowledgement",
        ):
            self.assertFalse(hasattr(DialogueRunner, name), name)


if __name__ == "__main__":
    unittest.main()
