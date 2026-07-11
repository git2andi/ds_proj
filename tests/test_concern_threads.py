"""Tests for thread-driven concern/blocker handling (TODO 8)."""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, BlockingStrength, MoveIntent, ThreadStatus, ThreadType

from tests.fixtures import make_persona, make_state
from tests.stubs import make_runner


def _concern_threads(state):
    return [
        t for t in state.threads.values()
        if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
    ]


def _observe(runner, state, speaker_id, text, *, intent=None):
    runner._llm.responses.append(text)
    intent = intent or MoveIntent(speaker_id=speaker_id, act=ActType.SUPPORT, reason="say it")
    return runner._generate_and_append(state, intent)


def _concern_intent(speaker_id: str, option_id: str) -> MoveIntent:
    return MoveIntent(
        speaker_id=speaker_id, act=ActType.CONCERN, reason="push back", option_focus=[option_id]
    )


class ConcernThreadCreationTests(unittest.TestCase):
    def setUp(self):
        random.seed(31)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def test_visible_objection_opens_option_specific_concern_thread(self):
        _observe(
            self.runner, self.state, "p1",
            "The Escape Room worries me — the cost is the highest of the three.",
            intent=_concern_intent("p1", "C"),
        )
        threads = _concern_threads(self.state)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        self.assertEqual(thread.thread_type, ThreadType.CONCERN)
        self.assertEqual(thread.status, ThreadStatus.HOT)
        self.assertEqual(thread.focus_options, ["C"])
        self.assertEqual(thread.issue_key, "cost")
        self.assertEqual(thread.blocking_strength, BlockingStrength.SOFT)

    def test_hard_blocker_opens_blocker_thread_and_rank(self):
        _observe(
            self.runner, self.state, "p1",
            "The Escape Room is a dealbreaker for me — the cost is too high.",
            intent=_concern_intent("p1", "C"),
        )
        threads = [t for t in _concern_threads(self.state) if t.thread_type is ThreadType.BLOCKER]
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].blocking_strength, BlockingStrength.HARD)
        self.assertIn("C", self.state.runtimes["p1"].rejected_options())

    def test_same_issue_on_two_options_gets_two_threads_shared_key(self):
        _observe(
            self.runner, self.state, "p1",
            "The Escape Room cost worries me.",
            intent=_concern_intent("p1", "C"),
        )
        _observe(
            self.runner, self.state, "p2",
            "Honestly the Museum cost bothers me too.",
            intent=_concern_intent("p2", "A"),
        )
        threads = sorted(_concern_threads(self.state), key=lambda t: t.created_turn)
        self.assertEqual(len(threads), 2)
        self.assertEqual(threads[0].issue_key, threads[1].issue_key)
        self.assertNotEqual(threads[0].focus_options, threads[1].focus_options)

    def test_repeated_concern_touches_instead_of_duplicating(self):
        _observe(
            self.runner, self.state, "p1",
            "The Escape Room cost worries me.",
            intent=_concern_intent("p1", "C"),
        )
        _observe(
            self.runner, self.state, "p2",
            "Yes, the Escape Room cost bothers me as well.",
            intent=_concern_intent("p2", "C"),
        )
        self.assertEqual(len(_concern_threads(self.state)), 1)


class ConcernResolutionTests(unittest.TestCase):
    def setUp(self):
        random.seed(32)
        self.state = make_state()
        self.runner = make_runner(self.state)
        _observe(
            self.runner, self.state, "p2",
            "The Escape Room cost worries me.",
            intent=_concern_intent("p2", "C"),
        )
        self.thread = _concern_threads(self.state)[0]

    def test_option_mention_alone_does_not_cool_concern(self):
        # Same option, different issue: flexibility, not cost.
        _observe(self.runner, self.state, "p3", "The Escape Room is at least something we all remember.")
        self.assertEqual(self.thread.status, ThreadStatus.HOT)

    def test_issue_relevant_response_cools_concern(self):
        _observe(
            self.runner, self.state, "p3",
            "The Escape Room cost still fits the sixty euro budget, though.",
        )
        self.assertEqual(self.thread.status, ThreadStatus.COOLING)

    def test_raiser_visible_acceptance_resolves_concern(self):
        _observe(
            self.runner, self.state, "p3",
            "The Escape Room cost still fits the sixty euro budget, though.",
        )
        # The acceptance must bridge the old lean (p2 preferred the Bike Ride),
        # otherwise validation rejects it as an unexplained switch.
        _observe(
            self.runner, self.state, "p2",
            "Fair enough, I can live with the Escape Room since the cost fits after all.",
        )
        self.assertEqual(self.thread.status, ThreadStatus.RESOLVED)

    def test_stale_blocker_preserves_rejection(self):
        random.seed(33)
        state = make_state()
        runner = make_runner(state)
        _observe(
            runner, state, "p1",
            "The Escape Room is a dealbreaker for me — the cost is too high.",
            intent=_concern_intent("p1", "C"),
        )
        blocker = [t for t in _concern_threads(state) if t.thread_type is ThreadType.BLOCKER][0]
        self.assertIn("C", state.runtimes["p1"].rejected_options())
        for i in range(7):  # hard blocker timeout is 6 participant turns
            speaker = ("p2", "p3")[i % 2]
            _observe(runner, state, speaker, "Let's keep weighing the Museum against the Bike Ride.")
        self.assertEqual(blocker.status, ThreadStatus.STALE)
        self.assertIn("C", state.runtimes["p1"].rejected_options())

    def test_repeated_resolved_issue_is_suppressed(self):
        _observe(
            self.runner, self.state, "p3",
            "The Escape Room cost still fits the sixty euro budget, though.",
        )
        _observe(
            self.runner, self.state, "p2",
            "Fair enough, I can live with the Escape Room since the cost fits after all.",
        )
        self.assertEqual(self.thread.status, ThreadStatus.RESOLVED)
        # A third party repeating the settled cost objection opens nothing new.
        _observe(
            self.runner, self.state, "p3",
            "The Escape Room cost is a bit steep.",
            intent=_concern_intent("p3", "C"),
        )
        self.assertEqual(len(_concern_threads(self.state)), 1)
        self.assertEqual(self.thread.status, ThreadStatus.RESOLVED)

    def test_settled_issue_reaches_prompt_suppression_from_threads(self):
        # Prompt context for settled issues derives from resolved/stale
        # concern/blocker threads (cleanup 5) — no separate issue ledger.
        _observe(
            self.runner, self.state, "p3",
            "The Escape Room cost still fits the sixty euro budget, though.",
        )
        _observe(
            self.runner, self.state, "p2",
            "Fair enough, I can live with the Escape Room since the cost fits after all.",
        )
        self.assertEqual(self.thread.status, ThreadStatus.RESOLVED)
        _observe(self.runner, self.state, "p1", "Anything else we should weigh?")
        prompt = self.runner._llm.prompts[-1]
        self.assertIn("Already raised and settled in this chat: cost", prompt)


class ConcernRoutingTests(unittest.TestCase):
    def test_hot_concern_routes_advocate_defense(self):
        state = make_state(personas=[
            make_persona("p1", "Mira", preferred="A", engagement=0.9),
            make_persona("p2", "Jonas", preferred="B"),
            make_persona("p3", "Lea", preferred="C"),
        ])
        runner = make_runner(state)
        random.seed(35)
        _observe(
            runner, state, "p2",
            "The Museum cost seems high for what it is.",
            intent=_concern_intent("p2", "A"),
        )
        # A hot concern is the primary thread: routing is deterministic.
        for seed in range(10):
            random.seed(seed)
            intent = runner._route_discussion_turn(state)
            self.assertEqual(intent.route_source, "thread_hot")
            self.assertEqual(intent.speaker_id, "p1")  # the option's advocate
            self.assertIn("A", intent.option_focus)

    def test_unanswered_own_concern_blocks_vote_compromise(self):
        state = make_state()
        runner = make_runner(state)
        random.seed(36)
        _observe(
            runner, state, "p2",
            "The Museum cost seems high for what it is.",
            intent=_concern_intent("p2", "A"),
        )
        persona = state.personas[1]  # p2
        self.assertFalse(runner._should_compromise_to_candidate(state, persona, "A"))


if __name__ == "__main__":
    unittest.main()
