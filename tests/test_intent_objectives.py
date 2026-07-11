"""Each routed turn carries ONE coherent realization objective (todo_new item 5).

The controller decides the stance direction from existing state; the prompt
never offers the LLM a defend-or-concede / live-with-or-block menu.
"""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent, ThreadType

from tests.fixtures import make_persona, make_state
from tests.stubs import make_runner


class NarrowingReactionDirection(unittest.TestCase):
    """The candidate test names one objective, decided from rank/resistance."""

    def _run(self, candidate_rank: int, switch_resistance: float) -> str:
        random.seed(5)
        state = make_state(
            personas=[
                make_persona("p1", "Mira", preferred="A"),
                make_persona("p2", "Jonas", preferred="B", switch_resistance=switch_resistance),
                make_persona("p3", "Lea", preferred="A"),
            ]
        )
        state.runtimes["p2"].option_ranks["A"] = candidate_rank
        runner = make_runner(state)
        runner._emit_narrowing_reaction(state, "A")
        intent = state.turns[-1].intent
        self.assertEqual(intent.speaker_id, "p2")
        self._last_act = intent.act
        return intent.reason

    def test_acceptable_candidate_gets_a_live_with_objective(self):
        reason = self._run(candidate_rank=4, switch_resistance=0.8)
        self.assertIn("could live with", reason)
        self.assertNotIn("or name", reason)
        self.assertEqual(self._last_act, ActType.SUPPORT)

    def test_disliked_candidate_gets_a_blocker_objective(self):
        reason = self._run(candidate_rank=2, switch_resistance=0.8)
        self.assertIn("still blocks", reason)
        self.assertNotIn("live with", reason)
        self.assertEqual(self._last_act, ActType.CONCERN)


class ConcernThreadResponseDirection(unittest.TestCase):
    def test_committed_advocate_is_told_to_defend_not_defend_or_concede(self):
        random.seed(9)
        state = make_state()  # p1 prefers A (Museum), p2 B, p3 C
        runner = make_runner(state)
        runner._llm.responses.append(
            "The Museum worries me — the cost is on the high side for what it is."
        )
        concern_intent = MoveIntent(
            speaker_id="p2", act=ActType.CONCERN, reason="push back", option_focus=["A"]
        )
        runner._generate_and_append(state, concern_intent)
        thread = next(
            t for t in state.threads.values() if t.thread_type is ThreadType.CONCERN
        )
        intent = runner._thread_intent(state, thread)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.speaker_id, "p1")  # the option's advocate
        self.assertIn("defend it with one grounded reason", intent.reason)
        self.assertNotIn(" or concede", intent.reason)


class ActMatchesObjective(unittest.TestCase):
    """The selected act always agrees with the stated objective (item 4)."""

    def _concern_thread_intent(self, *, advocate_stubbornness: float, bystander_rank: int | None = None):
        random.seed(9)
        personas = [
            make_persona("p1", "Mira", preferred="A", stubbornness=advocate_stubbornness),
            make_persona("p2", "Jonas", preferred="B"),
            make_persona("p3", "Lea", preferred="C"),
        ]
        state = make_state(personas=personas)
        if bystander_rank is not None:
            # No advocates: nobody's top option is A; p3 gets the given rank.
            state.runtimes["p1"].promote_to_preferred("B")
            state.runtimes["p1"].option_ranks["A"] = 3
            state.runtimes["p3"].option_ranks["A"] = bystander_rank
        runner = make_runner(state)
        runner._llm.responses.append(
            "The Museum worries me — the cost is on the high side for what it is."
        )
        runner._generate_and_append(
            state,
            MoveIntent(speaker_id="p2", act=ActType.CONCERN, reason="push back", option_focus=["A"]),
        )
        thread = next(t for t in state.threads.values() if t.thread_type is ThreadType.CONCERN)
        return runner._thread_intent(state, thread)

    def test_conceding_advocate_gets_concern_act(self):
        intent = self._concern_thread_intent(advocate_stubbornness=0.1)
        self.assertEqual(intent.act, ActType.CONCERN)
        self.assertIn("concede the point honestly", intent.reason)

    def test_defending_advocate_gets_support_act(self):
        intent = self._concern_thread_intent(advocate_stubbornness=0.9)
        self.assertEqual(intent.act, ActType.SUPPORT)
        self.assertIn("defend it with one grounded reason", intent.reason)

    def test_neutral_bystander_gets_comment_act(self):
        intent = self._concern_thread_intent(advocate_stubbornness=0.5, bystander_rank=3)
        self.assertEqual(intent.act, ActType.COMMENT)
        self.assertIn("without taking a side", intent.reason)

    def test_cooling_raiser_reaction_act_matches_direction(self):
        # Whatever branch the stubbornness draw picks, act and objective agree.
        for seed in range(25):
            random.seed(seed)
            state = make_state()
            runner = make_runner(state)
            runner._llm.responses += [
                "The Museum worries me — the cost is on the high side.",
                "The Museum cost still covers the calm setting we want.",
            ]
            runner._generate_and_append(
                state,
                MoveIntent(speaker_id="p2", act=ActType.CONCERN, reason="push back", option_focus=["A"]),
            )
            thread = next(t for t in state.threads.values() if t.thread_type is ThreadType.CONCERN)
            runner._generate_and_append(
                state,
                MoveIntent(
                    speaker_id="p1", act=ActType.SUPPORT, reason="defend",
                    route_source="thread_hot", thread_id=thread.thread_id, option_focus=["A"],
                ),
            )
            intent = runner._maybe_cooling_continuation(state, thread)
            if intent is None:
                continue
            if "push back once" in intent.reason:
                self.assertEqual(intent.act, ActType.CONCERN)
            elif "it lands" in intent.reason:
                self.assertEqual(intent.act, ActType.SUPPORT)


if __name__ == "__main__":
    unittest.main()
