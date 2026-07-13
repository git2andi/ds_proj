"""Deterministic tests for the simulator-authority split (todo 23).

These verify the final authority contract: ordinary participant behavior comes
from complete simulator bids, the floor manager arbitrates access without
rewriting bids, protocol obligations constrain only speaker/act, engagement
affects willingness (not double-counted), relevance can outweigh engagement,
private information stays private, and no removed routing function survives.
"""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

import simulator as sim_policy
from models import (
    ActType,
    DiscussionStimulus,
    MoveIntent,
    Phase,
    SimulatorBid,
    TurnObligation,
)

from tests.fixtures import append_turn, make_persona, make_state
from tests.stubs import make_runner


def _vote_intent(state, pid, candidate, *, kind="vote"):
    ob = TurnObligation(kind=kind, participant_id=pid, act=ActType.VOTE, candidate=candidate)
    return sim_policy.decide_simulator_bid(state, pid, obligation=ob).intent


class OpenFloorBiddingTests(unittest.TestCase):
    def test_every_eligible_simulator_is_asked_for_a_bid(self):
        state = make_state()
        runner = make_runner(state)
        append_turn(state, "p1", "The Museum is calm and easy.")
        random.seed(3)
        bids = runner._collect_bids(state, runner._discussion_stimulus(state))
        self.assertEqual({b.participant_id for b in bids}, {"p1", "p2", "p3"})

    def test_winning_bid_reaches_generation_unchanged(self):
        # The floor must not alter act/focus/target/reason/addressee of the bid
        # it selects (todo 8). We capture the intent handed to generation.
        state = make_state()
        runner = make_runner(state)
        chosen = MoveIntent(
            speaker_id="p2", act=ActType.CONCERN, reason="push back on the Museum",
            option_focus=["A"], addressee_id="p1", route_source="self_selection",
        )
        bid = SimulatorBid("p2", True, 0.9, chosen)
        runner._collect_bids = lambda s, stim: [bid]
        runner._ranked_valid_bids = lambda s, bids: [bid]
        seen = {}

        def capture(s, intent):
            seen["intent"] = intent
            from models import TurnRecord
            s.turn_index += 1
            rec = TurnRecord(index=s.turn_index, speaker_id=intent.speaker_id,
                             speaker_name=s.name_for(intent.speaker_id),
                             text="The Museum feels too quiet for a full day.",
                             phase=s.phase, intent=intent)
            s.turns.append(rec)
            return rec

        runner._generate_and_append = capture
        runner._run_open_floor_turn(state, runner._discussion_stimulus(state))
        got = seen["intent"]
        self.assertIs(got, chosen)  # exact same object, never rebuilt
        self.assertEqual(got.act, ActType.CONCERN)
        self.assertEqual(got.option_focus, ["A"])
        self.assertEqual(got.addressee_id, "p1")

    def test_floor_cannot_replace_concern_with_comment_or_alter_focus(self):
        state = make_state()
        state.phase = Phase.DISCUSSION
        runner = make_runner(state)
        bid = SimulatorBid("p1", True, 0.8, MoveIntent(
            speaker_id="p1", act=ActType.CONCERN, reason="r", option_focus=["B"]))
        ranked = runner._ranked_valid_bids(state, [bid])
        self.assertEqual(ranked[0].intent.act, ActType.CONCERN)
        self.assertEqual(ranked[0].intent.option_focus, ["B"])

    def test_next_best_used_only_after_failure(self):
        state = make_state()
        runner = make_runner(state)
        top = SimulatorBid("p1", True, 0.9, MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="r", option_focus=["A"]))
        nxt = SimulatorBid("p2", True, 0.5, MoveIntent(
            speaker_id="p2", act=ActType.SUPPORT, reason="r", option_focus=["B"]))
        runner._collect_bids = lambda s, stim: [top, nxt]
        runner._ranked_valid_bids = lambda s, bids: [top, nxt]
        from models import TurnRecord
        order = []

        def gen(s, intent):
            order.append(intent.speaker_id)
            s.turn_index += 1
            ok = intent.speaker_id == "p2"
            rec = TurnRecord(index=s.turn_index, speaker_id=intent.speaker_id,
                             speaker_name=s.name_for(intent.speaker_id),
                             text="Bike ride keeps cost low." if ok else "",
                             phase=s.phase, intent=intent, state_mutation_blocked=not ok)
            if ok:
                s.turns.append(rec)
            return rec

        runner._generate_and_append = gen
        rec = runner._run_open_floor_turn(state, runner._discussion_stimulus(state))
        self.assertEqual(order, ["p1", "p2"])   # top tried first
        self.assertEqual(rec.speaker_id, "p2")


class WillingnessAndTraitsTests(unittest.TestCase):
    def _claim_rate(self, engagement, seeds=200):
        claims = 0
        for seed in range(seeds):
            random.seed(seed)
            state = make_state([
                make_persona("p1", "P1", preferred="A", engagement=engagement),
                make_persona("p2", "P2", preferred="B", engagement=0.5),
                make_persona("p3", "P3", preferred="C", engagement=0.5),
            ])
            append_turn(state, "p2", "The Bike Ride keeps cost low and is active.")
            bid = sim_policy.decide_simulator_bid(state, "p1")
            claims += int(bid.wants_to_speak)
        return claims / seeds

    def test_engagement_changes_long_run_claim_frequency(self):
        low = self._claim_rate(0.1)
        high = self._claim_rate(0.95)
        self.assertGreater(high, low)

    def test_low_engagement_relevant_beats_high_engagement_irrelevant(self):
        # A low-engagement sim whose preferred option was just challenged should
        # out-bid a high-engagement sim with nothing new to add.
        wins_relevant = 0
        for seed in range(120):
            random.seed(seed)
            state = make_state([
                make_persona("p1", "Relevant", preferred="A", engagement=0.15, stubbornness=0.7),
                make_persona("p2", "Idle", preferred="B", engagement=0.95),
            ])
            # p2 (idle) already spoke twice; p1's option A is visibly challenged.
            append_turn(state, "p2", "I still think the Bike Ride is best overall.")
            from models import ConcernEvidence, EvidenceSpan, VisibleEvidence
            text = "The Museum is honestly too quiet for a whole day."
            append_turn(state, "p2", text, evidence=VisibleEvidence(
                utterance=text,
                concerns=[ConcernEvidence("A", "ordinary", EvidenceSpan(text, 0))],
            ))
            b1 = sim_policy.decide_simulator_bid(state, "p1")
            b2 = sim_policy.decide_simulator_bid(state, "p2")
            if b1.willingness > b2.willingness:
                wins_relevant += 1
        self.assertGreater(wins_relevant, 60)  # relevance wins most of the time


class TraitBehaviorTests(unittest.TestCase):
    def _support_concern_scores(self, *, stubbornness):
        from models import ConcernEvidence, EvidenceSpan, SupportEvidence, VisibleEvidence
        # p1 (prefers A) has A challenged and rival B gaining visible support.
        state = make_state([
            make_persona("p1", "P1", preferred="A", stubbornness=stubbornness),
            make_persona("p2", "P2", preferred="B"),
            make_persona("p3", "P3", preferred="B"),
        ])
        state.runtimes["p1"].mark_disliked("B", reason_against="too tiring")
        ch = "The Museum is too quiet for a whole day."
        append_turn(state, "p2", ch, evidence=VisibleEvidence(
            utterance=ch, concerns=[ConcernEvidence("A", "ordinary", EvidenceSpan(ch, 0))]))
        sup = "The Bike Ride is active and great."
        append_turn(state, "p3", sup, evidence=VisibleEvidence(
            utterance=sup, supports=[SupportEvidence("B", "firm", EvidenceSpan(sup, 0))]))
        state.runtimes["p3"].public_lean = "B"
        view = sim_policy.build_view(state, "p1")
        return sim_policy._score_acts(state, view)

    def test_stubbornness_increases_defense_and_pushback(self):
        low = self._support_concern_scores(stubbornness=0.1)
        high = self._support_concern_scores(stubbornness=0.9)
        # Defense of a challenged own option (SUPPORT) and rival pushback
        # (CONCERN) both rise with stubbornness.
        self.assertGreater(high[ActType.SUPPORT], low[ActType.SUPPORT])
        self.assertGreater(high[ActType.CONCERN], low[ActType.CONCERN])

    def test_switch_resistance_lowers_vote_switching(self):
        from tests.test_simplified_runtime import formal_vote

        def switches(sr):
            count = 0
            for seed in range(80):
                random.seed(seed)
                state = make_state([
                    make_persona("p1", "P1", preferred="A", switch_resistance=sr),
                    make_persona("p2", "P2", preferred="B", switch_resistance=0.5),
                    make_persona("p3", "P3", preferred="B", switch_resistance=0.5),
                ])
                for pid, opt in (("p1", "A"), ("p2", "B"), ("p3", "B")):
                    formal_vote(state, pid, opt)  # B is the visible plurality
                ob = TurnObligation(kind="final_decision", participant_id="p1",
                                    act=ActType.VOTE, candidate="B")
                intent = sim_policy.decide_simulator_bid(state, "p1", obligation=ob).intent
                count += int(intent.required_vote == "B")
            return count

        self.assertGreater(switches(0.1), switches(0.9))


class DirectQuestionTests(unittest.TestCase):
    def test_direct_question_forces_named_respondent(self):
        state = make_state()
        runner = make_runner(state)
        runner._llm.responses.append("Jonas, what do you think about the Museum?")
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p1", act=ActType.ASK, reason="ask", addressee_id="p2", option_focus=["A"]))
        obligation = runner._pending_answer_obligation(state)
        self.assertIsNotNone(obligation)
        self.assertEqual(obligation.participant_id, "p2")
        self.assertEqual(obligation.act, ActType.ANSWER)

    def test_forced_respondent_answer_direction_is_simulator_chosen(self):
        # The obligation fixes speaker/act; the answer's focus/reason come from
        # the simulator policy, not the controller.
        state = make_state()
        runner = make_runner(state)
        runner._llm.responses.append("Jonas, is the Escape Room too rigid?")
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p1", act=ActType.ASK, reason="ask", addressee_id="p2", option_focus=["C"]))
        ob = runner._pending_answer_obligation(state)
        bid = sim_policy.decide_simulator_bid(state, "p2", obligation=ob)
        self.assertTrue(bid.wants_to_speak)
        self.assertEqual(bid.intent.act, ActType.ANSWER)
        self.assertEqual(bid.intent.speaker_id, "p2")

    def test_group_question_gets_no_required_respondent(self):
        state = make_state()
        runner = make_runner(state)
        runner._llm.responses.append("Which option is cheapest for all of us?")
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p1", act=ActType.ASK, reason="ask"))
        self.assertIsNone(runner._pending_answer_obligation(state))
        q = next(t for t in state.threads.values())
        self.assertEqual(q.question_scope, "group")
        self.assertIsNone(q.required_respondent)

    def test_group_question_answerable_by_any_self_selecting_sim(self):
        from models import AnswerEvidence, EvidenceSpan, VisibleEvidence
        state = make_state()
        runner = make_runner(state)
        runner._llm.responses.append("Which option is cheapest for all of us?")
        runner._generate_and_append(state, MoveIntent(
            speaker_id="p1", act=ActType.ASK, reason="ask"))
        thread = next(t for t in state.threads.values())
        text = "The Bike Ride is the cheapest at twelve euros."
        # p3 self-selects an ANSWER to the group question (its bid carries the
        # respond_to_turn back to the question source).
        answer = MoveIntent(speaker_id="p3", act=ActType.ANSWER, reason="answer",
                            respond_to_turn=thread.source_turn_index)
        rec = append_turn(state, "p3", text, intent=answer, evidence=VisibleEvidence(
            utterance=text, answers=[AnswerEvidence("full", EvidenceSpan(text, 0), True)],
            mentions=runner._resolver.mentions(text)))
        runner._apply_semantics(state, rec)
        from models import ThreadStatus
        self.assertEqual(thread.status, ThreadStatus.COOLING)  # p3 (not asker) answered


class ThreadStimulusTests(unittest.TestCase):
    def test_concern_thread_does_not_force_defend_or_concede(self):
        # A hot concern thread influences bids but the advocate's reaction (if
        # any) is the simulator's choice, not a controller prescription.
        from controller.threads import open_thread
        from models import ThreadType
        state = make_state()
        append_turn(state, "p2", "The Museum feels too quiet for a full day.")
        open_thread(state, thread_type=ThreadType.CONCERN, focus_options=["A"],
                    issue_key="sig:quiet", started_by="p2", source_turn_index=1)
        random.seed(1)
        bid = sim_policy.decide_simulator_bid(state, "p1")  # p1 backs A
        # The bid, if any, is a complete simulator intent; no act was imposed.
        if bid.wants_to_speak:
            self.assertEqual(bid.intent.speaker_id, "p1")
            self.assertIn(bid.intent.act, sim_policy._OPEN_FLOOR_ACTS)


class CoverageTests(unittest.TestCase):
    def test_coverage_gap_does_not_force_a_compare_turn(self):
        state = make_state()
        state.phase = Phase.DISCUSSION
        runner = make_runner(state)
        for _ in range(2):
            append_turn(state, "p1", "The Museum is calm.")
            append_turn(state, "p2", "The Bike Ride is cheap.")
        # A and B are discussed; C is untouched -> a coverage gap exists, but no
        # participant is forced to compare it.
        state.coverage["A"].mentions = 2
        state.coverage["B"].mentions = 2
        gap = runner._coverage_gap_option(state)
        self.assertEqual(gap, "C")
        stimulus = runner._discussion_stimulus(state)
        random.seed(0)
        bids = runner._collect_bids(state, stimulus)
        for b in bids:
            if b.wants_to_speak and b.intent.act is ActType.COMPARE:
                # A compare bid may exist, but only from a sim that actually
                # cares about the gap option — never forced onto a random sim.
                self.assertNotEqual(b.intent.option_focus, [])


class HardBlockerTests(unittest.TestCase):
    def test_hard_blocker_never_bids_compromise_or_votes_rejected(self):
        state = make_state([
            make_persona("p1", "Blocker", preferred="B", rejection="A",
                         rejection_reason="cannot accept A"),
            make_persona("p2", "P2", preferred="A"),
            make_persona("p3", "P3", preferred="A"),
        ])
        runner = make_runner(state)
        # A is the public candidate with visible backing.
        append_turn(state, "p2", "I vote for the Museum.", phase=Phase.VOTING)
        append_turn(state, "p3", "The Museum works for me.", phase=Phase.VOTING)
        random.seed(2)
        # Open-floor: the blocker never proposes compromise toward A.
        for seed in range(30):
            random.seed(seed)
            bid = sim_policy.decide_simulator_bid(state, "p1")
            if bid.wants_to_speak and bid.intent.act is ActType.COMPROMISE:
                self.assertNotIn("A", bid.intent.option_focus)
        # Vote obligation testing A: the blocker never votes for A.
        intent = _vote_intent(state, "p1", "A", kind="final_decision")
        self.assertNotEqual(intent.required_vote, "A")

    def test_floor_rejects_a_hard_blocker_bid_targeting_rejected_option(self):
        state = make_state([
            make_persona("p1", "Blocker", preferred="B", rejection="A",
                         rejection_reason="cannot accept A"),
            make_persona("p2", "P2", preferred="A"),
        ])
        runner = make_runner(state)
        bad = SimulatorBid("p1", True, 0.9, MoveIntent(
            speaker_id="p1", act=ActType.COMPROMISE, reason="r", option_focus=["A"]))
        reason = runner._validate_bid(state, bad, obligation=None)
        self.assertTrue(reason)


class VoteAuthorityTests(unittest.TestCase):
    def test_formal_vote_target_comes_from_simulator(self):
        state = make_state()
        runner = make_runner(state)  # noqa: F841
        intent = _vote_intent(state, "p1", "B")   # candidate B, but p1 prefers A
        self.assertEqual(intent.required_vote, "A")

    def test_repair_does_not_set_switch_directly(self):
        # The framework only asks for a re-vote; the switch (if any) is the
        # simulator's decision, expressed in required_vote/allow_vote_change.
        state = make_state()
        from tests.test_simplified_runtime import formal_vote  # reuse helper
        formal_vote(state, "p1", "A")
        formal_vote(state, "p2", "B")
        formal_vote(state, "p3", "C")
        intent = _vote_intent(state, "p2", "A", kind="final_decision")
        self.assertIn(intent.required_vote, {"A", "B"})  # candidate or current


class PrivacyAndReproducibilityTests(unittest.TestCase):
    def test_other_sims_hidden_state_does_not_change_this_bid(self):
        # Changing another participant's hidden goal/ranks while keeping the
        # public transcript fixed must not change this sim's bid (todo 2).
        def build():
            state = make_state()
            append_turn(state, "p2", "The Bike Ride keeps cost low and active.")
            return state

        random.seed(9)
        base = sim_policy.decide_simulator_bid(build(), "p1")

        state2 = build()
        # Mutate p2 and p3 private ranks/goal — public transcript unchanged.
        state2.runtimes["p2"].option_ranks["A"] = 1
        state2.runtimes["p2"].option_ranks["B"] = 5
        state2.runtimes["p3"].option_ranks["C"] = 1
        state2.personas[1].private_goal = "totally different secret goal"
        random.seed(9)
        other = sim_policy.decide_simulator_bid(state2, "p1")

        self.assertEqual(base.wants_to_speak, other.wants_to_speak)
        self.assertAlmostEqual(base.willingness, other.willingness, places=9)
        self.assertEqual(
            base.intent.act if base.intent else None,
            other.intent.act if other.intent else None,
        )

    def test_same_seed_reproduces_bids(self):
        def run():
            random.seed(123)
            state = make_state()
            append_turn(state, "p1", "The Museum is calm and easy.")
            runner = make_runner(state)
            return [
                (b.participant_id, b.wants_to_speak, b.intent.act if b.intent else None)
                for b in runner._collect_bids(state, runner._discussion_stimulus(state))
            ]
        self.assertEqual(run(), run())

    def test_different_seeds_can_vary_bids_within_constraints(self):
        outcomes = set()
        for seed in range(40):
            random.seed(seed)
            state = make_state()
            append_turn(state, "p2", "The Bike Ride is cheap and active.")
            bid = sim_policy.decide_simulator_bid(state, "p1")
            if bid.wants_to_speak:
                self.assertIn(bid.intent.act, sim_policy._OPEN_FLOOR_ACTS)
                self.assertEqual(bid.intent.speaker_id, "p1")
            outcomes.add((bid.wants_to_speak, bid.intent.act if bid.intent else None))
        # Different seeds produce different valid bids (some speak, some don't;
        # acts vary) without ever violating the open-floor act constraint.
        self.assertGreaterEqual(len(outcomes), 2)


class NoRemovedRoutingSurvivesTests(unittest.TestCase):
    def test_removed_controller_routing_functions_are_gone(self):
        from dialogue import DialogueRunner
        removed = [
            "_route_discussion_turn", "_normal_intent", "_choose_speaker",
            "_choose_discussion_act", "_choose_target_turn", "_focus_options",
            "_reason_for_act", "_thread_intent", "_maybe_cooling_continuation",
            "_thread_speaker", "_maybe_continuation_intent",
            "_speaker_for_option_coverage", "_vote_intent",
            "_stance_consistent_vote_target", "_can_shift_to",
            "_should_switch_after_reservation", "_answer_intent_for_thread",
            "_pick_group_respondent", "_adapt_failed_route",
            "_append_final_decision", "_maybe_participant_procedural",
        ]
        for name in removed:
            self.assertFalse(
                hasattr(DialogueRunner, name),
                f"removed routing function {name} still present",
            )

    def test_select_primary_thread_removed_from_thread_engine(self):
        import controller.threads as threads
        self.assertFalse(hasattr(threads, "select_primary_thread"))


if __name__ == "__main__":
    unittest.main()
