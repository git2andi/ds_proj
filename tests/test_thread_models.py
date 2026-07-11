"""Tests for the six-phase model, thread state models, and trace fields."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from config_loader import cfg
from models import (
    ActType,
    BlockingStrength,
    MoveIntent,
    Phase,
    RepairState,
    ThreadState,
    ThreadStatus,
    ThreadType,
)

from tests.fixtures import make_state
from tests.stubs import make_runner


class PhaseModelTests(unittest.TestCase):
    def test_six_phases_with_closing_terminology(self):
        self.assertEqual(
            [p.value for p in Phase],
            ["opening", "discussion", "narrowing", "voting", "compromise_repair", "closing"],
        )
        self.assertFalse(hasattr(Phase, "CLOSURE"))


class ThreadModelTests(unittest.TestCase):
    def test_thread_state_defaults(self):
        thread = ThreadState(thread_id="t1", thread_type=ThreadType.CONCERN)
        self.assertEqual(thread.status, ThreadStatus.HOT)
        self.assertEqual(thread.blocking_strength, BlockingStrength.NONE)
        self.assertIsNone(thread.resolution_reason)
        self.assertIsNone(thread.question_scope)

    def test_state_has_empty_thread_storage(self):
        state = make_state()
        self.assertEqual(state.threads, {})
        self.assertIsNone(state.active_repair)

    def test_thread_config_defaults_load(self):
        threads_cfg = cfg.threads
        self.assertEqual(int(threads_cfg.cooling_turns), 2)
        self.assertEqual(int(threads_cfg.stale_after_turns), 4)
        self.assertEqual(int(threads_cfg.hard_blocker_stale_after_turns), 6)
        self.assertEqual(int(threads_cfg.max_thread_turns_hard), 5)
        self.assertTrue(bool(threads_cfg.allow_reactivation))

    def test_thread_state_serializes_to_json(self):
        from logger import _to_jsonable

        thread = ThreadState(
            thread_id="t1",
            thread_type=ThreadType.QUESTION,
            focus_options=["A"],
            question_scope="direct",
            blocking_strength=BlockingStrength.SOFT,
        )
        data = _to_jsonable(thread)
        self.assertEqual(data["thread_type"], "question")
        self.assertEqual(data["status"], "hot")
        self.assertEqual(data["blocking_strength"], "soft")
        self.assertEqual(data["question_scope"], "direct")
        repair = RepairState(repair_reason="split_vote", candidate_or_pair=["A", "B"])
        data = _to_jsonable(repair)
        self.assertEqual(data["repair_reason"], "split_vote")
        self.assertEqual(data["status"], "active")

    def test_trace_records_the_thread_that_routed_the_turn(self):
        state = make_state()
        state.threads["t1"] = ThreadState(
            thread_id="t1", thread_type=ThreadType.CONCERN, focus_options=["A"]
        )
        runner = make_runner(state, ["The Museum keeps the day easy to adjust."])
        random.seed(11)
        runner._generate_and_append(
            state,
            MoveIntent(
                speaker_id="p1", act=ActType.SUPPORT, reason="support",
                option_focus=["A"], route_source="thread_hot", thread_id="t1",
            ),
        )
        entry = [e for e in state.controller_trace if e["type"] == "turn"][0]
        self.assertEqual(entry["pre"]["routed_thread_id"], "t1")
        self.assertEqual(entry["pre"]["routed_thread_type"], "concern")
        self.assertEqual(entry["pre"]["routed_thread_status"], "hot")
        self.assertEqual(entry["pre"]["hot_thread_count"], 1)

    def test_trace_routed_thread_is_none_for_normal_turns(self):
        state = make_state()
        runner = make_runner(state, ["The Museum keeps the day easy to adjust."])
        random.seed(12)
        runner._generate_and_append(
            state,
            MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support", option_focus=["A"]),
        )
        entry = [e for e in state.controller_trace if e["type"] == "turn"][0]
        self.assertIsNone(entry["pre"]["routed_thread_id"])


if __name__ == "__main__":
    unittest.main()
