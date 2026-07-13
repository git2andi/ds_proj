from __future__ import annotations

import unittest

import tests  # noqa: F401
import simulator as sim_policy
from consensus import ConsensusManager, public_support, visible_votes_from_transcript
from interpreter import TurnInterpreter
from models import ActType, MoveIntent, Phase, RunOutcome, TurnObligation
from parsing import OptionResolver


def _vote_decision(runner, state, pid, candidate, *, kind="vote"):
    """The simulator's own vote intent for a framework vote obligation."""
    ob = TurnObligation(kind=kind, participant_id=pid, act=ActType.VOTE, candidate=candidate)
    return sim_policy.decide_simulator_bid(state, pid, obligation=ob).intent
import prompts
from tests.fixtures import append_turn, make_persona, make_scenario, make_state, vote_intent
from tests.stubs import FakeLLM, make_runner


def formal_vote(state, pid: str, option: str) -> None:
    record = append_turn(
        state, pid, f"I vote for option {option}.",
        intent=vote_intent(pid, option), phase=Phase.VOTING,
    )
    state.runtimes[pid].explicit_vote = option
    state.runtimes[pid].current_acceptance = option
    state.runtimes[pid].public_lean = option
    assert record.visible_vote() == option


class CriticalInterpreterTests(unittest.TestCase):
    def test_critical_mode_never_calls_validator(self):
        scenario = make_scenario()
        llm = FakeLLM()
        interpreter = TurnInterpreter(
            llm, OptionResolver(scenario.options), scenario,
            {"p1": "Mira", "p2": "Jonas"}, mode="critical",
        )
        result = interpreter.interpret(
            text="The Museum looks better to me now.", speaker_id="p1",
            intent=MoveIntent("p1", ActType.SUPPORT, "support", option_focus=["A"]),
        )
        self.assertTrue(result.fast_path)
        self.assertEqual(llm.prompts, [])
        self.assertEqual(result.tokens_in, 0)
        self.assertEqual(result.tokens_out, 0)

    def test_explicit_warming_updates_visible_softening(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="I'm warming to the Bike Ride.", speaker_id="p1",
            intent=MoveIntent("p1", ActType.COMMENT, "react", option_focus=["B"]),
        )
        self.assertEqual([s.option_id for s in result.evidence.softenings], ["B"])

    def test_settles_it_is_visible_acceptance_or_lean(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="That settles it for me—the Museum is the workable choice.",
            speaker_id="p1",
            intent=MoveIntent("p1", ActType.COMMENT, "react", option_focus=["A"]),
        )
        commitment = result.evidence.sole_commitment()
        self.assertIsNotNone(commitment)
        self.assertEqual((commitment.kind, commitment.option_id), ("accept", "A"))
        self.assertEqual([s.option_id for s in result.evidence.softenings], ["A"])

    def test_conditional_settles_it_moves_lean_but_is_not_commitment(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="If the Museum is smoother for us, that settles it for me.",
            speaker_id="p1",
            intent=MoveIntent("p1", ActType.COMMENT, "react", option_focus=["A"]),
        )
        self.assertIsNone(result.evidence.sole_commitment())
        self.assertEqual([s.option_id for s in result.evidence.softenings], ["A"])

    def test_acknowledged_or_negated_concern_does_not_open_new_concern(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        for text, act, focus in (
            ("I get the worry about the Museum, but it still works for us.", ActType.SUPPORT, "A"),
            ("The Museum is easy to adjust, so timing won't be an issue.", ActType.COMMENT, "A"),
            ("The Museum's risk is manageable, so I still support it.", ActType.SUPPORT, "A"),
            ("The Museum's upside outweighs the risk.", ActType.SUPPORT, "A"),
        ):
            with self.subTest(text=text):
                result = interpreter.interpret(
                    text=text, speaker_id="p1",
                    intent=MoveIntent("p1", act, "react", option_focus=[focus]),
                )
                self.assertEqual(result.evidence.concerns, [])

    def test_direct_negative_still_creates_concern(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="The Museum won't solve the timing problem.", speaker_id="p1",
            intent=MoveIntent("p1", ActType.CONCERN, "object", option_focus=["A"]),
        )
        self.assertEqual([c.option_id for c in result.evidence.concerns], ["A"])

    def test_vote_reason_concern_applies_only_to_rejected_alternative(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="I vote for the Bike Ride since the Museum can't keep us active.",
            speaker_id="p1",
            intent=MoveIntent(
                "p1", ActType.VOTE, "vote", option_focus=["B", "A"], required_vote="B"
            ),
        )
        self.assertEqual([c.option_id for c in result.evidence.concerns], ["A"])
        self.assertEqual(result.evidence.sole_commitment().option_id, "B")

    def test_comparison_negative_is_locally_attributed(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="The Bike Ride is active, but the Museum is too quiet.",
            speaker_id="p1",
            intent=MoveIntent("p1", ActType.COMPARE, "compare", option_focus=["B", "A"]),
        )
        self.assertEqual([c.option_id for c in result.evidence.concerns], ["A"])

    def test_explicit_lean_binds_to_option_inside_lean_clause(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text=(
                "I'm leaning toward the Museum instead, since the Bike Ride "
                "still looks too tiring."
            ),
            speaker_id="p1",
            intent=MoveIntent(
                "p1", ActType.COMMENT, "react", option_focus=["B", "A"]
            ),
        )
        self.assertEqual([s.option_id for s in result.evidence.softenings], ["A"])
        self.assertEqual([c.option_id for c in result.evidence.concerns], ["B"])

    def test_low_risk_is_positive_not_a_concern(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="The Museum's low risk makes it easier to trust.",
            speaker_id="p1",
            intent=MoveIntent("p1", ActType.SUPPORT, "support", option_focus=["A"]),
        )
        self.assertEqual(result.evidence.concerns, [])

    def test_negative_predicate_binds_to_subject_not_later_comparison_option(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="The Museum doesn't keep us active like the Bike Ride.",
            speaker_id="p1",
            intent=MoveIntent(
                "p1", ActType.COMPARE, "compare", option_focus=["A", "B"]
            ),
        )
        self.assertEqual([c.option_id for c in result.evidence.concerns], ["A"])

    def test_definite_acceptance_remains_visible_acceptance_evidence(self):
        scenario = make_scenario()
        interpreter = TurnInterpreter(
            None, OptionResolver(scenario.options), scenario,
            {"p1": "Mira"}, mode="critical",
        )
        result = interpreter.interpret(
            text="The Museum works for me.", speaker_id="p1",
            intent=vote_intent("p1", "A"),
        )
        commitment = result.evidence.sole_commitment()
        self.assertIsNotNone(commitment)
        self.assertEqual(commitment.kind, "accept")
        self.assertEqual(commitment.option_id, "A")


class CurrentPublicStateTests(unittest.TestCase):
    def test_formal_acceptance_updates_runtime_vote_and_transcript_consistently(self):
        state = make_state()
        runner = make_runner(state)
        intent = vote_intent("p1", "A")
        interpreted = TurnInterpreter(
            None, OptionResolver(state.scenario.options), state.scenario,
            {p.id: p.name for p in state.personas}, mode="critical",
        ).interpret(
            text="The Museum works for me.", speaker_id="p1", intent=intent,
            rejected_options=tuple(state.runtimes["p1"].rejected_options()),
        )
        record = append_turn(
            state, "p1", "The Museum works for me.", intent=intent,
            phase=Phase.VOTING, evidence=interpreted.evidence,
        )
        runner._apply_semantics(state, record)
        self.assertEqual(state.runtimes["p1"].explicit_vote, "A")
        self.assertEqual(visible_votes_from_transcript(state), {"p1": "A"})

    def test_visible_switch_withdraws_old_active_backing(self):
        state = make_state()
        formal_vote(state, "p1", "A")
        formal_vote(state, "p2", "A")
        formal_vote(state, "p3", "B")
        # A later visible switch replaces p1's old vote in transcript and runtime.
        record = append_turn(
            state, "p1", "I'm switching from the Museum to the Bike Ride.",
            intent=MoveIntent(
                "p1", ActType.VOTE, "switch", route_source="split_vote_repair",
                option_focus=["B", "A"], required_vote="B", old_preference="A",
                allow_vote_change=True,
            ),
            phase=Phase.COMPROMISE_REPAIR,
        )
        state.runtimes["p1"].explicit_vote = "B"
        self.assertEqual(record.visible_vote(), "B")
        self.assertEqual(visible_votes_from_transcript(state)["p1"], "B")
        backing = public_support(state)
        self.assertNotIn("p1", backing["A"])
        self.assertIn("p1", backing["B"])

    def test_first_vote_is_not_switch_from_private_preference(self):
        state = make_state()
        state.runtimes["p1"].promote_to_preferred("A")
        record = append_turn(
            state, "p1", "I vote for the Bike Ride.",
            intent=vote_intent("p1", "B"), phase=Phase.VOTING,
        )
        self.assertEqual(record.visible_vote(), "B")
        self.assertEqual(state.runtimes["p1"].switch_events, [])

    def test_first_formal_vote_does_not_follow_group_candidate_without_visible_movement(self):
        state = make_state()
        runner = make_runner(state)
        # Other participants visibly favor B, but p1 has not visibly moved from A.
        state.runtimes["p2"].current_acceptance = "B"
        state.runtimes["p3"].current_acceptance = "B"
        intent = _vote_decision(runner, state, "p1", "B")
        self.assertEqual(intent.required_vote, "A")
        self.assertFalse(intent.allow_vote_change)
        self.assertIsNone(intent.old_preference)

    def test_visible_warming_updates_public_and_private_lean(self):
        state = make_state()
        runner = make_runner(state)
        intent = MoveIntent("p1", ActType.COMMENT, "react", option_focus=["B"])
        interpreted = TurnInterpreter(
            None, OptionResolver(state.scenario.options), state.scenario,
            {p.id: p.name for p in state.personas}, mode="critical",
        ).interpret(
            text="I'm warming to the Bike Ride.", speaker_id="p1", intent=intent,
            rejected_options=tuple(state.runtimes["p1"].rejected_options()),
        )
        record = append_turn(
            state, "p1", "I'm warming to the Bike Ride.",
            intent=intent, phase=Phase.DISCUSSION, evidence=interpreted.evidence,
        )
        runner._apply_semantics(state, record)
        self.assertEqual(state.runtimes["p1"].public_lean, "B")
        self.assertEqual(state.runtimes["p1"].top_option(), "B")
        self.assertEqual(state.discussion_lean_shifts, 1)

    def test_clause_bound_lean_updates_the_visible_target_only(self):
        state = make_state()
        runner = make_runner(state)
        intent = MoveIntent("p1", ActType.COMMENT, "react", option_focus=["B", "A"])
        text = "I'm leaning toward the Museum instead, since the Bike Ride still looks too tiring."
        interpreted = TurnInterpreter(
            None, OptionResolver(state.scenario.options), state.scenario,
            {p.id: p.name for p in state.personas}, mode="critical",
        ).interpret(
            text=text, speaker_id="p1", intent=intent,
            rejected_options=tuple(state.runtimes["p1"].rejected_options()),
        )
        record = append_turn(
            state, "p1", text, intent=intent, phase=Phase.DISCUSSION,
            evidence=interpreted.evidence,
        )
        runner._apply_semantics(state, record)
        self.assertEqual(state.runtimes["p1"].public_lean, "A")
        self.assertEqual(state.runtimes["p1"].top_option(), "A")

    def test_conditional_settles_it_updates_lean_without_becoming_vote(self):
        state = make_state()
        runner = make_runner(state)
        intent = MoveIntent("p1", ActType.COMMENT, "react", option_focus=["B"])
        interpreted = TurnInterpreter(
            None, OptionResolver(state.scenario.options), state.scenario,
            {p.id: p.name for p in state.personas}, mode="critical",
        ).interpret(
            text="If the Bike Ride is easier for everyone, that settles it for me.",
            speaker_id="p1", intent=intent,
            rejected_options=tuple(state.runtimes["p1"].rejected_options()),
        )
        record = append_turn(
            state, "p1", "If the Bike Ride is easier for everyone, that settles it for me.",
            intent=intent, phase=Phase.DISCUSSION, evidence=interpreted.evidence,
        )
        runner._apply_semantics(state, record)
        self.assertIsNone(state.runtimes["p1"].explicit_vote)
        self.assertEqual(state.runtimes["p1"].public_lean, "B")
        self.assertEqual(state.runtimes["p1"].top_option(), "B")


class ComparisonFocusTests(unittest.TestCase):
    def test_normal_comparison_requires_exactly_current_pick_and_one_rival(self):
        state = make_state()
        runner = make_runner(state)
        target = append_turn(
            state,
            "p2",
            "The Bike Ride is active, while the Escape Room is memorable.",
        )
        view = sim_policy.build_view(state, "p1")
        focus = sim_policy._compare_pair(view)
        self.assertEqual(len(focus), 2)
        self.assertEqual(focus[0], "A")
        self.assertIn(focus[1], {"B", "C"})



class MajorityClassificationTests(unittest.TestCase):
    def make_group(self, n: int):
        personas = [make_persona(f"p{i}", f"P{i}", preferred="A" if i <= (n // 2 + 1) else "B") for i in range(1, n + 1)]
        state = make_state(personas=personas)
        state.phase = Phase.VOTING
        return state, make_runner(state)

    def test_bare_majorities_get_one_round(self):
        for votes in (["A", "A", "B"], ["A", "A", "A", "B", "B"], ["A", "A", "A", "A", "B", "B", "B"]):
            state, runner = self.make_group(len(votes))
            for i, option in enumerate(votes, 1):
                formal_vote(state, f"p{i}", option)
            repair = runner._classify_repair(state, ConsensusManager.finalize(state))
            self.assertIsNotNone(repair)
            self.assertEqual(repair.repair_reason, "majority_holdout")
            self.assertEqual(repair.max_attempts, 1)

    def test_clear_majorities_close_without_repair(self):
        for votes in (["A", "A", "A", "B"], ["A", "A", "A", "A", "B"], ["A", "A", "A", "A", "B", "B"]):
            state, runner = self.make_group(len(votes))
            for i, option in enumerate(votes, 1):
                formal_vote(state, f"p{i}", option)
            self.assertEqual(ConsensusManager.finalize(state).status, "majority")
            self.assertIsNone(runner._classify_repair(state, ConsensusManager.finalize(state)))

    def test_split_has_one_candidate_attempt(self):
        state, runner = self.make_group(3)
        for i, option in enumerate(["A", "B", "C"], 1):
            formal_vote(state, f"p{i}", option)
        repair = runner._classify_repair(state, ConsensusManager.finalize(state))
        self.assertEqual(repair.repair_reason, "split_vote")
        self.assertEqual(repair.max_attempts, 1)


class BlockerOutcomeTests(unittest.TestCase):
    def test_majority_with_dissenting_blocker_is_valid(self):
        personas = [
            make_persona("p1", "A1", preferred="A"),
            make_persona("p2", "A2", preferred="A"),
            make_persona("p3", "B", preferred="B", rejection="A", rejection_reason="cannot accept A"),
        ]
        state = make_state(personas=personas)
        formal_vote(state, "p1", "A")
        formal_vote(state, "p2", "A")
        formal_vote(state, "p3", "B")
        self.assertEqual(ConsensusManager.finalize(state).status, "majority")

    def test_active_blocker_prevents_false_unanimity(self):
        personas = [
            make_persona("p1", "A1", preferred="A"),
            make_persona("p2", "B", preferred="B", rejection="A", rejection_reason="cannot accept A"),
        ]
        state = make_state(personas=personas)
        formal_vote(state, "p1", "A")
        formal_vote(state, "p2", "A")
        outcome = ConsensusManager.finalize(state)
        self.assertEqual(outcome.status, "unresolved")
        self.assertIn("active blocker", outcome.reason.lower())


class SplitCandidateSelectionTests(unittest.TestCase):
    def test_equal_vote_tie_uses_most_positive_discussion_mentions(self):
        from models import EvidenceSpan, SupportEvidence, VisibleEvidence

        state = make_state()
        runner = make_runner(state)
        # B receives two visible positive discussion turns, A one, C none.
        for pid, option, text in (
            ("p1", "A", "The Museum is easy to adjust."),
            ("p2", "B", "The Bike Ride is active and inexpensive."),
            ("p3", "B", "The Bike Ride also keeps us outside."),
        ):
            evidence = VisibleEvidence(utterance=text)
            evidence.supports.append(SupportEvidence(option, "firm", EvidenceSpan(text=text, start=0)))
            append_turn(state, pid, text, evidence=evidence)
        for pid, option in (("p1", "A"), ("p2", "B"), ("p3", "C")):
            formal_vote(state, pid, option)

        ranked = runner._rank_split_candidates(
            state, visible_votes_from_transcript(state)
        )
        self.assertTrue(ranked)
        candidate, dissenters, movers, meta = ranked[0]
        # The framework tests B: tied on formal votes, most positive visible
        # mentions. Whether each dissenter moves is the simulators' own call, so
        # movers is exactly the set of visible dissenters.
        self.assertEqual(candidate, "B")
        self.assertEqual(meta["positive_mentions"], 2)
        self.assertEqual({p.id for p in movers}, {p.id for p in dissenters})
        self.assertEqual({p.id for p in dissenters}, {"p1", "p3"})

    def test_split_candidate_is_visible_plurality(self):
        personas = [
            make_persona("p1", "P1", preferred="A", switch_resistance=0.2),
            make_persona("p2", "P2", preferred="B", switch_resistance=0.2),
            make_persona("p3", "P3", preferred="C", switch_resistance=0.2),
            make_persona("p4", "P4", preferred="D", switch_resistance=0.2),
        ]
        state = make_state(personas=personas, scenario=make_scenario())
        runner = make_runner(state)
        for pid, option in (("p1", "A"), ("p2", "A"), ("p3", "B"), ("p4", "C")):
            formal_vote(state, pid, option)
        ranked = runner._rank_split_candidates(state, visible_votes_from_transcript(state))
        # A is the visible plurality (2 votes); it is tested first.
        self.assertEqual(ranked[0][0], "A")
        self.assertEqual(ranked[0][3]["votes"], 2)



class MoverSelectionTests(unittest.TestCase):
    def test_tie_candidate_broken_by_visible_positive_mention(self):
        from models import EvidenceSpan, SupportEvidence, VisibleEvidence

        personas = [
            make_persona("p1", "P1", preferred="A", switch_resistance=0.3),
            make_persona("p2", "P2", preferred="B", switch_resistance=0.3),
            make_persona("p3", "P3", preferred="C", switch_resistance=0.3),
        ]
        state = make_state(personas=personas)
        runner = make_runner(state)
        span = EvidenceSpan("The Museum could work for me.", 0)
        append_turn(
            state, "p3", span.text, phase=Phase.DISCUSSION,
            evidence=VisibleEvidence(
                utterance=span.text,
                mentions=runner._resolver.mentions(span.text),
                supports=[SupportEvidence("A", "firm", span)],
            ),
        )
        for pid, option in (("p1", "A"), ("p2", "B"), ("p3", "C")):
            formal_vote(state, pid, option)
        ranked = runner._rank_split_candidates(state, visible_votes_from_transcript(state))
        candidate, _dissenters, _movers, meta = ranked[0]
        # 1-1-1 formal tie broken by A's single visible positive mention.
        self.assertEqual(candidate, "A")
        self.assertEqual(meta["positive_mentions"], 1)

    def test_one_ordinary_concern_does_not_make_flexible_mover_immovable(self):
        persona = make_persona("p1", "P1", preferred="B", switch_resistance=0.2)
        state = make_state(personas=[persona, make_persona("p2", "P2", preferred="A")])
        state.runtimes["p1"].mark_disliked("A", reason_against="one reservation")
        # Movement is now the simulator's own judgment (sim_policy.valid_holdout).
        self.assertFalse(sim_policy.valid_holdout(state, persona, "A"))

    def test_final_decision_is_candidate_or_current_vote_only(self):
        # A repair re-vote is simulator-owned: its target is either the tested
        # candidate or the mover's own current vote — never a third option, and
        # never a controller-computed can_move (todo 17/18).
        state = make_state()
        runner = make_runner(state)
        formal_vote(state, "p1", "A")
        formal_vote(state, "p2", "B")
        formal_vote(state, "p3", "C")
        ob = runner._final_decision_obligation(state.persona_by_id("p2"), "A")
        intent = sim_policy.decide_simulator_bid(state, "p2", obligation=ob).intent
        self.assertIn(intent.required_vote, {"A", "B"})
        self.assertNotEqual(intent.required_vote, "C")


class ClosingLineTests(unittest.TestCase):
    def test_majority_closure_never_claims_unanimity(self):
        state = make_state()
        state.runtimes["p1"].explicit_vote = "A"
        state.runtimes["p2"].explicit_vote = "A"
        state.runtimes["p3"].explicit_vote = "B"
        outcome = RunOutcome(
            status="majority", final_option="A", reason="2-1", turns=3,
            metadata={"visible_votes": {"p1": "A", "p2": "A", "p3": "B"}},
        )
        line = prompts.moderator_closure_line(outcome, state.scenario, state)
        self.assertIn("majority", line.lower())
        self.assertIn(state.persona_by_id("p3").name, line)
        self.assertNotIn("all agreed", line.lower())

    def test_unresolved_closure_announces_no_choice(self):
        state = make_state()
        outcome = RunOutcome(
            status="unresolved", final_option=None, reason="tie", turns=3, metadata={}
        )
        line = prompts.moderator_closure_line(outcome, state.scenario, state)
        self.assertIn("unresolved", line.lower())
        self.assertNotIn("group decision", line.lower())


if __name__ == "__main__":
    unittest.main()
