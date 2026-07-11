"""Tests for the Section 8 routing order and thread speaker selection (TODO 10)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, BlockingStrength, ThreadStatus, ThreadType
from controller.threads import mark_response, open_thread

from tests.fixtures import append_turn, make_persona, make_state
from tests.stubs import make_runner


def _question(state, respondent="p2", turn=1):
    return open_thread(
        state, thread_type=ThreadType.QUESTION, focus_options=["A"], issue_key="cost",
        started_by="p1", source_turn_index=turn, required_respondent=respondent,
        question_scope="direct",
    )


class RoutingOrderTests(unittest.TestCase):
    def setUp(self):
        random.seed(51)
        self.state = make_state()
        self.runner = make_runner(self.state)
        append_turn(self.state, "p3", "The Escape Room is at least memorable.")

    def test_required_answer_outranks_hot_threads_and_coverage(self):
        open_thread(
            self.state, thread_type=ThreadType.CONCERN, focus_options=["B"], issue_key="risk",
            started_by="p3", source_turn_index=1,
        )
        _question(self.state, respondent="p2", turn=1)
        for seed in range(10):
            random.seed(seed)
            intent = self.runner._route_discussion_turn(self.state)
            self.assertEqual(intent.route_source, "answer_required")
            self.assertEqual(intent.speaker_id, "p2")

    def test_hot_thread_outranks_coverage(self):
        # Option C untouched (coverage gap) plus one hot concern: the concern wins.
        for _ in range(4):
            append_turn(self.state, "p1", "I still like the Museum best.")
            append_turn(self.state, "p2", "The Bike Ride keeps cost low.")
        open_thread(
            self.state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p2", source_turn_index=self.state.turn_index,
        )
        for seed in range(10):
            random.seed(seed)
            intent = self.runner._route_discussion_turn(self.state)
            self.assertEqual(intent.route_source, "thread_hot")

    def test_coverage_runs_only_in_quiet_state(self):
        for _ in range(4):
            append_turn(self.state, "p1", "I still like the Museum best.")
            append_turn(self.state, "p2", "The Bike Ride keeps cost low.")
        self.state.coverage["A"].mentions = 3
        self.state.coverage["B"].mentions = 3
        # Quiet state: no threads at all -> coverage for untouched option C.
        random.seed(3)
        intent = self.runner._route_discussion_turn(self.state)
        self.assertEqual(intent.route_source, "coverage")
        self.assertEqual(intent.option_focus[0], "C")

    def test_blocker_thread_routes_bounded_probe_then_mitigation(self):
        self.state.runtimes["p1"].mark_rejected("B", reason_against="too tiring")
        thread = open_thread(
            self.state, thread_type=ThreadType.BLOCKER, focus_options=["B"], issue_key="risk",
            started_by="p1", source_turn_index=1, blocking_strength=BlockingStrength.HARD,
        )
        random.seed(4)
        intent = self.runner._route_discussion_turn(self.state)
        self.assertEqual(intent.route_source, "thread_hot")
        self.assertEqual(intent.act, ActType.ASK)          # first: probe the blocker
        self.assertEqual(intent.addressee_id, "p1")
        thread.probe_count += 1                             # probe consumed (per thread)
        intent = self.runner._route_discussion_turn(self.state)
        self.assertEqual(intent.act, ActType.SUPPORT)       # then: honest mitigation
        self.assertNotEqual(intent.speaker_id, "p1")

    def test_second_participants_blocker_on_same_option_gets_own_probe(self):
        # One sim's consumed probe must not suppress another sim's separate
        # blocker against the same option (cleanup 5: per-thread probe state).
        self.state.runtimes["p1"].mark_rejected("B", reason_against="too tiring")
        first = open_thread(
            self.state, thread_type=ThreadType.BLOCKER, focus_options=["B"], issue_key="risk",
            started_by="p1", source_turn_index=1, blocking_strength=BlockingStrength.HARD,
        )
        first.probe_count += 1                              # p1's blocker already probed
        self.state.runtimes["p2"].mark_rejected("B", reason_against="too pricey")
        open_thread(
            self.state, thread_type=ThreadType.BLOCKER, focus_options=["B"], issue_key="cost",
            started_by="p2", source_turn_index=2, blocking_strength=BlockingStrength.HARD,
        )
        random.seed(5)
        intent = self.runner._route_discussion_turn(self.state)
        self.assertEqual(intent.act, ActType.ASK)           # fresh probe for the new blocker
        self.assertEqual(intent.addressee_id, "p2")

    def test_cooling_continuation_is_probabilistic_and_bounded(self):
        thread = open_thread(
            self.state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p2", source_turn_index=1,
        )
        mark_response(self.state, thread, responder_id="p1", turn_index=self.state.turn_index)
        sources = set()
        for seed in range(40):
            random.seed(seed)
            sources.add(self.runner._route_discussion_turn(self.state).route_source)
        self.assertIn("thread_cooling", sources)   # sometimes continues
        self.assertGreater(len(sources), 1)        # never scripted
        # Bounded: an old cooling thread (not freshly touched) never continues.
        append_turn(self.state, "p1", "Anyway, weekends fill up fast.")
        append_turn(self.state, "p3", "True, and budgets are budgets.")
        for seed in range(40):
            random.seed(seed)
            intent = self.runner._route_discussion_turn(self.state)
            self.assertNotEqual(intent.route_source, "thread_cooling")

    def test_continuation_cannot_override_required_answer(self):
        _question(self.state, respondent="p2", turn=1)
        for seed in range(50):
            random.seed(seed)
            intent = self.runner._route_discussion_turn(self.state)
            self.assertFalse(intent.continuation)
            self.assertEqual(intent.route_source, "answer_required")


class ThreadSpeakerTests(unittest.TestCase):
    def test_relevance_beats_engagement(self):
        # p2 is highly engaged but neutral on A; p3 is quiet but backs A.
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="B", engagement=0.5),
            make_persona("p2", "Jonas", preferred="C", engagement=0.95),
            make_persona("p3", "Lea", preferred="A", engagement=0.15),
        ])
        runner = make_runner(state)
        thread = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1,
        )
        candidates = [p for p in state.personas if p.id != "p1"]
        speaker = runner._thread_speaker(state, candidates, thread)
        self.assertEqual(speaker.id, "p3")

    def test_just_spoke_penalty_moves_the_floor(self):
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="B"),
            make_persona("p2", "Jonas", preferred="A", engagement=0.6),
            make_persona("p3", "Lea", preferred="A", engagement=0.55),
        ])
        runner = make_runner(state)
        thread = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1,
        )
        append_turn(state, "p2", "The Museum keeps things simple.")
        candidates = [p for p in state.personas if p.id != "p1"]
        speaker = runner._thread_speaker(state, candidates, thread)
        self.assertEqual(speaker.id, "p3")


if __name__ == "__main__":
    unittest.main()
