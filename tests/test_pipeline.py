"""Tests for the pure-routing / realized-turn pipeline (TODO 3).

Route selection must be read-only; persistent dialogue state changes only after
a final accepted turn is appended and observed; repaired/fallback text is
re-parsed before observation; failed turns consume their bounded attempt but
never fake semantic effects.
"""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import ActType, MoveIntent, ThreadStatus, ThreadType
from controller.threads import open_thread

from tests.fixtures import append_turn, make_persona, make_state
from tests.stubs import make_runner


def _fingerprint(state) -> tuple:
    """Everything routing could illegitimately mutate."""
    return (
        len(state.turns),
        state.turn_index,
        tuple(sorted((oid, c.mentions, c.reasons, c.objections, c.acceptances, c.coverage_attempts) for oid, c in state.coverage.items())),
        state.procedural_move_count,
        state.no_progress_count,
        tuple(sorted(
            (t.thread_id, t.status.value, t.last_touched_turn, t.contribution_count, t.probe_count)
            for t in state.threads.values()
        )),
        tuple(sorted((pid, rt.explicit_vote, tuple(sorted(rt.option_ranks.items()))) for pid, rt in state.runtimes.items())),
        state.phase.value,
    )


def _controller_fingerprint(runner) -> tuple:
    """Controller instance fields that could hide routing memory (cleanup 6).

    Everything not starting with '_' plus the known bookkeeping counters; a new
    mutable instance attribute created during route selection shows up here.
    """
    return (
        runner._intervention_count,
        runner._last_intervention_turn,
        tuple(sorted(vars(runner))),
    )


class RoutingIsReadOnlyTests(unittest.TestCase):
    def _routable_state(self):
        state = make_state()
        runner = make_runner(state)
        # Enough turns that every reactive branch is reachable, plus an open
        # concern and a partially covered board.
        append_turn(state, "p1", "I like the Museum for the easy pace.")
        append_turn(state, "p2", "The Bike Ride keeps the cost low.")
        append_turn(state, "p3", "The Escape Room is at least memorable.")
        append_turn(state, "p1", "The Bike Ride worries me for the tired ones.")
        open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["B"], issue_key="risk",
            started_by="p1", source_turn_index=4,
        )
        state.coverage["A"].mentions = 2
        state.coverage["B"].mentions = 1
        return state, runner

    def test_route_selection_never_mutates_state(self):
        state, runner = self._routable_state()
        before = _fingerprint(state)
        controller_before = _controller_fingerprint(runner)
        for seed in range(40):
            random.seed(seed)
            intent = runner._route_discussion_turn(state)
            self.assertIsNotNone(intent)
            self.assertEqual(_fingerprint(state), before, f"route selection mutated state (seed {seed})")
            self.assertEqual(
                _controller_fingerprint(runner), controller_before,
                f"route selection mutated controller state (seed {seed})",
            )

    def test_repeated_route_selection_is_reproducible(self):
        # The old _last_target_speaker memory made a second identical selection
        # differ from the first; over identical accepted history the same seed
        # must now yield the same route every time.
        state, runner = self._routable_state()
        for seed in range(20):
            random.seed(seed)
            first = runner._route_discussion_turn(state)
            random.seed(seed)
            second = runner._route_discussion_turn(state)
            self.assertEqual(
                (first.speaker_id, first.act, first.route_source, first.respond_to_turn, first.addressee_id),
                (second.speaker_id, second.act, second.route_source, second.respond_to_turn, second.addressee_id),
                f"route selection not reproducible (seed {seed})",
            )

    def test_ready_to_narrow_check_is_read_only(self):
        state = make_state()
        runner = make_runner(state)
        state.min_discussion_turns = 3
        state.force_narrow_turns = 5
        state.hard_max_turns = 8
        open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1, required_respondent="p2", question_scope="direct",
        )
        state.turn_index = 6
        before = _fingerprint(state)
        for seed in range(10):
            random.seed(seed)
            runner._ready_to_narrow(state)
            self.assertEqual(_fingerprint(state), before)

    def test_required_answer_thread_lookup_is_read_only(self):
        state = make_state()
        runner = make_runner(state)
        thread = open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=0, required_respondent="p2", question_scope="direct",
        )
        before = _fingerprint(state)
        self.assertIs(runner._required_answer_thread(state), thread)
        self.assertEqual(_fingerprint(state), before)


class FailedTurnTests(unittest.TestCase):
    def test_dropped_turn_mutates_nothing_semantic(self):
        personas = [
            make_persona("p1", "Mira", preferred="A", rejection="C", rejection_reason="booked rooms are inflexible"),
            make_persona("p2", "Jonas", preferred="B"),
        ]
        state = make_state(personas=personas)
        # Force an impossible decision: required vote on p1's hard-blocked option.
        # Generation, repair, and fallback all fail validation -> dropped turn.
        runner = make_runner(state, [
            "I vote for the Escape Room.",
            "I vote for the Escape Room.",
        ])
        random.seed(1)
        intent = MoveIntent(
            speaker_id="p1",
            act=ActType.VOTE,
            reason="vote",
            option_focus=["C"],
            required_vote="C",
        )
        turns_before = len(state.turns)
        record = runner._generate_and_append(state, intent)
        self.assertTrue(record.state_mutation_blocked)
        self.assertEqual(record.text, "")
        self.assertEqual(len(state.turns), turns_before)  # never appended
        self.assertIsNone(state.runtimes["p1"].explicit_vote)
        self.assertEqual(state.coverage["C"].mentions, 0)
        trace = [e for e in state.controller_trace if e["type"] == "turn"][-1]
        self.assertFalse(trace["result"]["appended"])
        self.assertEqual(trace["result"]["candidate_texts"]["initial"], "I vote for the Escape Room.")
        self.assertEqual(trace["result"]["candidate_texts"]["repairs"], ["I vote for the Escape Room."])
        self.assertTrue(trace["result"]["candidate_texts"]["final_rejected"])
        self.assertTrue(state.failed_route_counts)

    def test_repeated_failed_route_changes_speaker_then_simplifies(self):
        state = make_state()
        runner = make_runner(state)
        intent = MoveIntent(
            speaker_id="p1", act=ActType.COMPARE, reason="compare",
            route_source="thread_hot", option_focus=["A", "B"], thread_id="t001",
        )
        runner._record_failed_route(state, intent)
        adapted = runner._adapt_failed_route(state, intent)
        self.assertNotEqual(adapted.speaker_id, "p1")
        self.assertEqual(adapted.act, ActType.COMPARE)

        runner._record_failed_route(state, adapted)
        simplified = runner._adapt_failed_route(state, intent)
        self.assertEqual(simplified.act, ActType.COMMENT)
        self.assertEqual(simplified.route_source, "failed_route_recovery")

    def test_coverage_attempt_charged_once_post_turn(self):
        state = make_state()
        # Generation ignores the routed option twice; the deterministic fallback
        # then names it (that is the fallback's job), so coverage is realized by
        # the final accepted text — and exactly one attempt is charged, after
        # the turn, not at route selection.
        runner = make_runner(state, [
            "I still think my current pick is fine.",
            "I still think my current pick is fine.",
        ])
        random.seed(2)
        intent = MoveIntent(
            speaker_id="p1",
            act=ActType.COMPARE,
            reason="briefly bring in an option that has not yet been socially processed, then compare it with the current lean",
            route_source="coverage",
            option_focus=["C"],
        )
        record = runner._generate_and_append(state, intent)
        self.assertTrue(record.used_fallback)
        self.assertEqual(state.coverage["C"].coverage_attempts, 1)
        self.assertIn("C", record.mentioned_options())
        trace = [e for e in state.controller_trace if e["type"] == "turn"][-1]
        self.assertTrue(trace["result"]["coverage_realized"])


class ReparseTests(unittest.TestCase):
    def test_repaired_text_is_reparsed_before_observation(self):
        state = make_state()
        runner = make_runner(state, [
            "Hmm, hard to say anything definite yet.",   # unclear vote -> repair
            "I vote for the Museum.",                     # repaired, must be re-parsed
        ])
        random.seed(4)
        intent = MoveIntent(
            speaker_id="p1", act=ActType.VOTE, reason="vote", option_focus=["A"], length_hint="short",
        )
        record = runner._generate_and_append(state, intent)
        self.assertTrue(record.repaired)
        self.assertEqual(record.visible_vote(), "A")
        self.assertEqual(state.runtimes["p1"].explicit_vote, "A")

    def test_fallback_text_is_reparsed_before_observation(self):
        state = make_state()
        runner = make_runner(state, [
            "Hmm, hard to say anything definite yet.",
            "Still nothing definite from me.",
        ])
        random.seed(5)
        intent = MoveIntent(
            speaker_id="p1", act=ActType.VOTE, reason="vote", option_focus=["A"], length_hint="short",
        )
        record = runner._generate_and_append(state, intent)
        self.assertTrue(record.used_fallback)
        self.assertEqual(record.visible_vote(), "A")
        self.assertEqual(state.runtimes["p1"].explicit_vote, "A")


class ConcernObservationTests(unittest.TestCase):
    def test_concern_cools_only_via_realized_response(self):
        state = make_state()
        append_turn(state, "p2", "The Museum seems too quiet for a whole day.")
        thread = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="sig:quiet",
            started_by="p2", source_turn_index=1,
        )
        runner = make_runner(state, ["The Museum still gives us room to adjust the pace."])
        random.seed(6)
        # Routing alone must not mark the concern addressed.
        self.assertEqual(thread.status, ThreadStatus.HOT)
        runner._generate_and_append(
            state,
            MoveIntent(
                speaker_id="p1",
                act=ActType.SUPPORT,
                reason="defend your pick",
                route_source="thread_hot",
                option_focus=["A"],
                respond_to_turn=thread.source_turn_index,
            ),
        )
        self.assertEqual(thread.status, ThreadStatus.COOLING)


if __name__ == "__main__":
    unittest.main()
