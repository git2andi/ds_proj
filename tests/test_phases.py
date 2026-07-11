"""Tests for phase transitions and narrowing readiness (TODO 13)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, BlockingStrength, MoveIntent, Phase, ThreadType
from controller.threads import open_thread

from tests.fixtures import append_turn, make_state, vote_intent
from tests.stubs import make_runner


def _prepped_state(runner=None):
    """A state that satisfies the mandatory narrowing conditions."""
    state = make_state()
    state.min_discussion_turns = 3
    state.force_narrow_turns = 8
    state.hard_max_turns = 30
    state.phase = Phase.DISCUSSION
    for oid in state.scenario.option_ids:
        state.coverage[oid].mentions = 2
    open_thread(
        state, thread_type=ThreadType.COMPARISON, focus_options=["A", "B"], issue_key="pair",
        started_by="p1", source_turn_index=1,
    )
    return state


def _add_discussion_support(state, pid: str, text: str, option: str):
    append_turn(
        state, pid, text,
        intent=MoveIntent(speaker_id=pid, act=ActType.SUPPORT, reason="support", option_focus=[option]),
        phase=Phase.DISCUSSION,
    )


class TransitionGraphTests(unittest.TestCase):
    def test_allowed_transitions_pass_and_illegal_raise(self):
        state = make_state()
        runner = make_runner(state)
        runner._mark_phase(state, Phase.DISCUSSION, "opening done")
        runner._mark_phase(state, Phase.NARROWING, "ready")
        runner._mark_phase(state, Phase.DISCUSSION, "collapse, once")
        runner._mark_phase(state, Phase.NARROWING, "ready again")
        runner._mark_phase(state, Phase.VOTING, "vote")
        runner._mark_phase(state, Phase.COMPROMISE_REPAIR, "split")
        runner._mark_phase(state, Phase.VOTING, "revote")
        runner._mark_phase(state, Phase.CLOSING, "done")
        with self.assertRaises(ValueError):
            runner._mark_phase(state, Phase.DISCUSSION, "illegal")

    def test_same_phase_mark_is_a_note_not_a_transition(self):
        state = make_state()
        runner = make_runner(state)
        runner._mark_phase(state, Phase.OPENING, "still opening")
        self.assertEqual(
            [e for e in state.controller_trace if e["type"] == "phase_transition"], []
        )
        self.assertEqual(state.phase, Phase.OPENING)


class NarrowingReadinessTests(unittest.TestCase):
    def setUp(self):
        random.seed(81)

    def test_opening_leans_alone_do_not_narrow(self):
        state = _prepped_state()
        runner = make_runner(state)
        for pid, text in (("p1", "Museum for me."), ("p2", "Bike Ride."), ("p3", "Escape Room.")):
            append_turn(state, pid, text, phase=Phase.OPENING)
        append_turn(state, "p1", "Anyway.", phase=Phase.DISCUSSION)
        self.assertFalse(runner._ready_to_narrow(state))  # no discussion support

    def test_discussion_support_enables_candidate_trigger(self):
        state = _prepped_state()
        runner = make_runner(state)
        for i in range(6):
            append_turn(state, ("p1", "p2", "p3")[i % 3], "Weighing things.", phase=Phase.DISCUSSION)
        # Public support must be visible turns, never manufactured runtime state.
        _add_discussion_support(state, "p2", "The Museum works for me.", "A")
        _add_discussion_support(state, "p1", "I'm fine with the Museum as well.", "A")
        self.assertTrue(runner._ready_to_narrow(state))

    def test_required_answer_blocks_even_at_hard_cap(self):
        state = _prepped_state()
        runner = make_runner(state)
        state.hard_max_turns = 2
        for i in range(4):
            append_turn(state, ("p1", "p2")[i % 2], "Talking.", phase=Phase.DISCUSSION)
        open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1, required_respondent="p2", question_scope="direct",
        )
        self.assertFalse(runner._ready_to_narrow(state))

    def test_hot_hard_blocker_against_candidate_blocks_narrowing(self):
        state = _prepped_state()
        runner = make_runner(state)
        for i in range(6):
            append_turn(state, ("p1", "p2", "p3")[i % 3], "Weighing.", phase=Phase.DISCUSSION)
        # Public support must be visible turns, never manufactured runtime state.
        _add_discussion_support(state, "p2", "The Museum works for me.", "A")
        _add_discussion_support(state, "p1", "I'm fine with the Museum as well.", "A")
        self.assertTrue(runner._ready_to_narrow(state))
        open_thread(
            state, thread_type=ThreadType.BLOCKER, focus_options=["A"], issue_key="cost",
            started_by="p3", source_turn_index=state.turn_index,
            blocking_strength=BlockingStrength.HARD,
        )
        self.assertFalse(runner._ready_to_narrow(state))

    def test_stale_soft_concern_does_not_block(self):
        state = _prepped_state()
        runner = make_runner(state)
        for i in range(6):
            append_turn(state, ("p1", "p2", "p3")[i % 3], "Weighing.", phase=Phase.DISCUSSION)
        # Public support must be visible turns, never manufactured runtime state.
        _add_discussion_support(state, "p2", "The Museum works for me.", "A")
        _add_discussion_support(state, "p1", "I'm fine with the Museum as well.", "A")
        concern = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p3", source_turn_index=1,
        )
        from models import ThreadStatus
        concern.status = ThreadStatus.STALE  # aged out
        self.assertTrue(runner._ready_to_narrow(state))

    def test_stable_top_pair_triggers_narrowing(self):
        state = _prepped_state()
        runner = make_runner(state)
        for i in range(4):
            append_turn(state, ("p1", "p2", "p3")[i % 3], "Weighing.", phase=Phase.DISCUSSION)
        _add_discussion_support(state, "p1", "The Museum works for me.", "A")
        state.top_pair_history.extend([("A", "B"), ("A", "B")])
        self.assertTrue(runner._ready_to_narrow(state))

    def test_hard_cap_needs_a_viable_candidate(self):
        state = _prepped_state()
        runner = make_runner(state)
        state.hard_max_turns = 2
        for i in range(4):
            append_turn(state, ("p1", "p2")[i % 2], "Talking.", phase=Phase.DISCUSSION)
        # Personas have latent leans, so a viable candidate exists.
        self.assertTrue(runner._ready_to_narrow(state))
        # Strip every lean and all visible support: nothing viable, no narrowing.
        for rt in state.runtimes.values():
            rt.option_ranks = {}
        self.assertFalse(runner._ready_to_narrow(state))


class NarrowingPhaseTests(unittest.TestCase):
    def test_narrowing_runs_bounded_beats_then_votes(self):
        random.seed(82)
        state = _prepped_state()
        runner = make_runner(state)
        state.phase = Phase.NARROWING
        for pid in ("p1", "p2"):
            state.runtimes[pid].mark_acceptable("A")
            append_turn(state, pid, "The Museum works for me.", intent=vote_intent(pid, "A"))
            state.runtimes[pid].explicit_vote = "A"
        turns_before = len(state.turns)
        runner._narrowing_phase(state)
        self.assertEqual(state.phase, Phase.VOTING)
        self.assertLessEqual(len(state.turns) - turns_before, 3)  # bounded
        self.assertFalse(state.narrowing_returned)

    def test_collapse_returns_to_discussion_at_most_once(self):
        random.seed(83)
        state = _prepped_state()
        runner = make_runner(state)
        state.phase = Phase.NARROWING
        state.min_discussion_turns = 0
        state.force_narrow_turns = 1
        state.hard_max_turns = 40
        # No visible support at all: the tested candidate collapses immediately.
        phases_seen = []
        original_mark = runner._mark_phase
        def spy(s, phase, reason):
            phases_seen.append(phase)
            original_mark(s, phase, reason)
        runner._mark_phase = spy
        runner._narrowing_phase(state)
        self.assertEqual(state.phase, Phase.VOTING)
        self.assertTrue(state.narrowing_returned)
        self.assertEqual(phases_seen.count(Phase.DISCUSSION), 1)  # exactly one fallback


if __name__ == "__main__":
    unittest.main()
