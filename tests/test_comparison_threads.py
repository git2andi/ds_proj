"""Tests for comparison threads (TODO 9): normalized pairs, bounded lifecycle."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent, Phase, ThreadStatus, ThreadType

from tests.fixtures import make_state
from tests.stubs import make_runner


def _comparison_threads(state):
    return [t for t in state.threads.values() if t.thread_type is ThreadType.COMPARISON]


def _observe(runner, state, speaker_id, text, *, intent=None):
    runner._llm.responses.append(text)
    intent = intent or MoveIntent(speaker_id=speaker_id, act=ActType.SUPPORT, reason="say it")
    return runner._generate_and_append(state, intent)


def _compare_intent(speaker_id: str, pair: list[str]) -> MoveIntent:
    return MoveIntent(speaker_id=speaker_id, act=ActType.COMPARE, reason="compare", option_focus=pair)


class ComparisonCreationTests(unittest.TestCase):
    def setUp(self):
        random.seed(41)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def test_realized_comparison_creates_pair_thread(self):
        _observe(
            self.runner, self.state, "p1",
            "The Museum is calmer than the Bike Ride, but also less active.",
            intent=_compare_intent("p1", ["A", "B"]),
        )
        threads = _comparison_threads(self.state)
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].status, ThreadStatus.HOT)
        self.assertEqual(threads[0].focus_options, ["A", "B"])

    def test_pair_order_is_normalized_to_one_thread(self):
        _observe(
            self.runner, self.state, "p1",
            "The Museum is calmer than the Bike Ride.",
            intent=_compare_intent("p1", ["A", "B"]),
        )
        _observe(
            self.runner, self.state, "p2",
            "Sure, but the Bike Ride beats the Museum on cost.",
            intent=_compare_intent("p2", ["B", "A"]),
        )
        self.assertEqual(len(_comparison_threads(self.state)), 1)

    def test_single_option_compare_turn_creates_nothing(self):
        _observe(
            self.runner, self.state, "p1",
            "The Museum keeps the day easy to adjust.",
            intent=_compare_intent("p1", ["A", "B"]),
        )
        self.assertEqual(_comparison_threads(self.state), [])

    def test_unrouted_comparative_wording_creates_thread(self):
        _observe(
            self.runner, self.state, "p2",
            "I'd take the Bike Ride over the Museum for the price alone.",
        )
        self.assertEqual(len(_comparison_threads(self.state)), 1)

    def test_two_mentions_without_comparative_wording_create_nothing(self):
        _observe(
            self.runner, self.state, "p2",
            "The Museum sounds nice. The Bike Ride also sounds nice.",
        )
        self.assertEqual(_comparison_threads(self.state), [])


class ComparisonLifecycleTests(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.state = make_state()
        self.runner = make_runner(self.state)
        _observe(
            self.runner, self.state, "p1",
            "The Museum is calmer than the Bike Ride, but also less active.",
            intent=_compare_intent("p1", ["A", "B"]),
        )
        self.thread = _comparison_threads(self.state)[0]

    def test_relevant_pair_response_moves_hot_to_cooling(self):
        _observe(
            self.runner, self.state, "p2",
            "For me the Bike Ride still wins over the Museum on cost.",
        )
        self.assertEqual(self.thread.status, ThreadStatus.COOLING)

    def test_single_option_response_does_not_cool(self):
        _observe(self.runner, self.state, "p2", "The Bike Ride keeps the budget happy.")
        self.assertEqual(self.thread.status, ThreadStatus.HOT)

    def test_comparison_goes_stale_after_quiet_timeout(self):
        for i in range(4):
            speaker = ("p2", "p3")[i % 2]
            _observe(self.runner, self.state, speaker, "The Escape Room keeps coming back to my mind.")
        self.assertEqual(self.thread.status, ThreadStatus.STALE)

    def test_stale_comparison_reactivates_on_revisit(self):
        for i in range(4):
            speaker = ("p2", "p3")[i % 2]
            _observe(self.runner, self.state, speaker, "The Escape Room keeps coming back to my mind.")
        self.assertEqual(self.thread.status, ThreadStatus.STALE)
        _observe(
            self.runner, self.state, "p2",
            "Back to the Museum versus the Bike Ride: which actually fits a tired group?",
            intent=_compare_intent("p2", ["A", "B"]),
        )
        self.assertEqual(self.thread.status, ThreadStatus.HOT)

    def test_narrowing_transition_resolves_open_comparisons(self):
        from controller import threads as threads_engine

        threads_engine.resolve_comparison_threads(self.state, reason="left discussion for narrowing")
        self.assertEqual(self.thread.status, ThreadStatus.RESOLVED)
        self.assertEqual(self.thread.resolution_reason, "left discussion for narrowing")


if __name__ == "__main__":
    unittest.main()
