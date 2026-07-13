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


class FloorArbitrationReadOnlyTests(unittest.TestCase):
    """Collecting and arbitrating simulator bids must be read-only over dialogue
    state and reproducible under a fixed seed (todo 8, 18)."""

    def _bidding_state(self):
        state = make_state()
        runner = make_runner(state)
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

    def test_bid_collection_never_mutates_state(self):
        state, runner = self._bidding_state()
        before = _fingerprint(state)
        controller_before = _controller_fingerprint(runner)
        for seed in range(40):
            random.seed(seed)
            stimulus = runner._discussion_stimulus(state)
            bids = runner._collect_bids(state, stimulus)
            runner._ranked_valid_bids(state, bids)
            self.assertEqual(_fingerprint(state), before, f"bidding mutated state (seed {seed})")
            self.assertEqual(
                _controller_fingerprint(runner), controller_before,
                f"bidding mutated controller state (seed {seed})",
            )

    def test_bids_reproducible_under_seed(self):
        state, runner = self._bidding_state()
        for seed in range(20):
            random.seed(seed)
            first = runner._collect_bids(state, runner._discussion_stimulus(state))
            random.seed(seed)
            second = runner._collect_bids(state, runner._discussion_stimulus(state))
            self.assertEqual(
                [(b.participant_id, b.wants_to_speak, b.intent.act if b.intent else None,
                  tuple(b.intent.option_focus) if b.intent else ()) for b in first],
                [(b.participant_id, b.wants_to_speak, b.intent.act if b.intent else None,
                  tuple(b.intent.option_focus) if b.intent else ()) for b in second],
                f"bids not reproducible under fixed seed {seed}",
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

    def test_failed_winner_falls_back_to_next_best_bid(self):
        # The floor tries ranked bids in order; when the top bid's realization
        # drops after bounded repair, the next-best submitted bid is used
        # unchanged (todo 10). No bid is rewritten.
        from models import SimulatorBid

        state = make_state()
        runner = make_runner(state)
        top = SimulatorBid(
            "p1", True, 0.9,
            MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="r", option_focus=["A"]),
        )
        nxt = SimulatorBid(
            "p2", True, 0.5,
            MoveIntent(speaker_id="p2", act=ActType.SUPPORT, reason="r", option_focus=["B"]),
        )
        runner._collect_bids = lambda s, stim: [top, nxt]
        runner._ranked_valid_bids = lambda s, bids: [top, nxt]

        from models import TurnRecord
        seen: list[str] = []

        def fake_generate(s, intent):
            seen.append(intent.speaker_id)
            s.turn_index += 1
            blocked = intent.speaker_id == "p1"      # top bid always drops
            rec = TurnRecord(
                index=s.turn_index, speaker_id=intent.speaker_id,
                speaker_name=s.name_for(intent.speaker_id),
                text="" if blocked else "The Bike Ride keeps costs low.",
                phase=s.phase, intent=intent, state_mutation_blocked=blocked,
            )
            if not blocked:
                s.turns.append(rec)
            return rec

        runner._generate_and_append = fake_generate
        record = runner._run_open_floor_turn(state, runner._discussion_stimulus(state))
        self.assertIsNotNone(record)
        self.assertEqual(seen, ["p1", "p2"])          # top tried first, then next-best
        self.assertEqual(record.speaker_id, "p2")
        self.assertEqual(record.intent.option_focus, ["B"])  # intent preserved unchanged


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
