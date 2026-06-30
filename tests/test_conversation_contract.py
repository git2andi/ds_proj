from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import prompts  # noqa: E402
from config_loader import cfg  # noqa: E402
from dialogue import ConsensusManager, Orchestrator, _repair_regresses_state, initialise_state  # noqa: E402
from models import (  # noqa: E402
    ActType,
    DialogueAct,
    MoveIntent,
    OptionCard,
    Persona,
    Phase,
    RunOutcome,
    Scenario,
    TraitProfile,
    TurnRecord,
)
from parsing import OptionResolver, TurnMove, parse_dialogue_act  # noqa: E402
from router import TurnRouter  # noqa: E402
from scoring import visible_candidate_status, visible_preference_concentration  # noqa: E402
from validation import MessageValidator  # noqa: E402


def _persona(pid: str, name: str, traits: TraitProfile, preference: str, background: str = "Private history") -> Persona:
    return Persona(
        id=pid,
        name=name,
        traits=traits,
        background=background,
        private_goal=f"A private goal favoring option {preference}",
        preferred_options=[preference],
    )


def _state():
    options = [
        OptionCard(
            id=option_id,
            name=name,
            short_name=name,
            attrs={"cost": cost, "time": time},
            upside=upside,
            tradeoff=tradeoff,
            concern=tradeoff,
            best_for=upside,
        )
        for option_id, name, cost, time, upside, tradeoff in [
            ("A", "Alpha Trail", "$10", "2 hours", "quiet views", "steep finish"),
            ("B", "Beta Trail", "$15", "1 hour", "easy access", "busy path"),
            ("C", "Gamma Trail", "$20", "3 hours", "long hike", "higher cost"),
            ("D", "Delta Trail", "$12", "2 hours", "shaded route", "fewer views"),
        ]
    ]
    traits = TraitProfile(3, 3, 3, 4, 2)
    p1 = _persona("p1", "Ana", traits, "A", "Ana hikes every weekend")
    p2 = _persona("p2", "Bo", traits, "B", "Bo is recovering from an injury")
    state = initialise_state(
        Scenario("Pick a trail", "activity_choice", "What matters?", options, ["Two friends are choosing."]),
        [p1, p2],
    )
    state.phase = Phase.DISCUSSION
    state.turns.append(TurnRecord(
        index=1,
        speaker_id="p1",
        speaker_name="Ana",
        text="The steep finish worries me because we only have two hours before sunset.",
        phase=Phase.DISCUSSION,
        act=DialogueAct("p1", "", ActType.OBJECT, option_refs=["A"]),
        intent=MoveIntent("p1", ActType.OBJECT, "raise concern", option_focus=["A"]),
    ))
    return state, p1, p2


def _three_person_state():
    base, p1, p2 = _state()
    p3 = _persona("p3", "Cy", TraitProfile(3, 3, 3, 3, 2), "C")
    state = initialise_state(base.scenario, [p1, p2, p3])
    state.phase = Phase.CONFIRMATION
    return state, p1, p2, p3


def _record_vote(state, persona: Persona, option_id: str) -> None:
    state.runtimes[persona.id].explicit_vote = option_id
    state.runtimes[persona.id].current_preference = option_id
    state.turns.append(TurnRecord(
        index=len(state.turns) + 1,
        speaker_id=persona.id,
        speaker_name=persona.name,
        text=f"I choose {state.scenario.option(option_id).name}.",
        phase=state.phase,
        act=DialogueAct(persona.id, "", ActType.VOTE, option_refs=[option_id], explicit_vote=option_id),
        intent=MoveIntent(persona.id, ActType.VOTE, "vote", option_focus=[option_id]),
    ))


class ConversationContractTests(unittest.TestCase):
    def test_local_response_prefers_focus_then_falls_back_to_latest_other_speaker(self) -> None:
        state, _, _ = _state()
        router = TurnRouter()
        self.assertEqual(router._local_response_turn_for(state, "p2", ["A"]), 1)
        self.assertEqual(router._local_response_turn_for(state, "p2", ["D"]), 1)

    def test_targeted_prompt_leads_with_exact_point_and_omits_biography(self) -> None:
        state, _, p2 = _state()
        intent = MoveIntent(
            "p2",
            ActType.REACT,
            "reply locally",
            option_focus=["B"],
            respond_to_turn=1,
        )
        prompt = prompts.sim_utterance(
            persona=p2,
            state=state,
            recent_lines=[],
            intent=intent,
            focus_options=[state.scenario.option("B")],
            addressee_name=None,
            max_words=24,
        )
        self.assertIn("Reply to this exact point", prompt)
        self.assertIn("The steep finish worries me because we only have two hours before sunset.", prompt)
        self.assertNotIn(p2.background, prompt)
        self.assertNotIn(p2.private_goal, prompt)
        self.assertIn("Don't restart your case or repeat your background", prompt)

    def test_opening_prompt_keeps_personal_context(self) -> None:
        state, _, p2 = _state()
        intent = MoveIntent("p2", ActType.OPENING, "open", option_focus=["B"])
        card = prompts.runtime_speaker_card(p2, state, intent)
        self.assertIn(p2.background, card)
        self.assertIn(p2.private_goal, card)

    def test_traits_materially_change_act_weights(self) -> None:
        base = dict(cfg.routing.act_probabilities.items())
        router = TurnRouter()
        midpoint = TraitProfile(3, 3, 3, 3, 3)

        def weights(traits: TraitProfile):
            return router._trait_adjusted_act_probabilities(_persona("p1", "A", traits, "A"), base)

        open_high, open_low = weights(TraitProfile(5, 3, 3, 3, 3)), weights(TraitProfile(1, 3, 3, 3, 3))
        self.assertGreater(open_high["ask"], open_low["ask"] * 2)
        self.assertGreater(open_high["compare"], open_low["compare"] * 2)

        agreeable, blunt = weights(TraitProfile(3, 3, 3, 5, 3)), weights(TraitProfile(3, 3, 3, 1, 3))
        self.assertGreater(agreeable["react"], blunt["react"] * 2)
        self.assertGreater(blunt["push_back"], agreeable["push_back"] * 2)

        cautious, calm = weights(TraitProfile(3, 3, 3, 3, 5)), weights(TraitProfile(3, 3, 3, 3, 1))
        self.assertGreater(cautious["object"], calm["object"] * 2)
        self.assertGreater(calm["support"], cautious["support"] * 2)

        careful, loose = weights(TraitProfile(3, 5, 3, 3, 3)), weights(TraitProfile(3, 1, 3, 3, 3))
        self.assertGreater(careful["ask"], loose["ask"] * 2)
        self.assertGreater(careful["object"], loose["object"] * 2)

        outgoing, reserved = weights(TraitProfile(3, 3, 5, 3, 3)), weights(TraitProfile(3, 3, 1, 3, 3))
        self.assertGreater(outgoing["react"], reserved["react"] * 2)
        self.assertGreater(outgoing["propose_compromise"], reserved["propose_compromise"] * 2)

        self.assertEqual(weights(midpoint)["compare"], float(base["compare"]))

    def test_turn_behavior_is_act_specific_and_trait_derived(self) -> None:
        curious = _persona("p1", "A", TraitProfile(5, 5, 2, 4, 2), "A")
        blunt = _persona("p2", "B", TraitProfile(2, 2, 5, 1, 4), "B")
        ask = MoveIntent("p1", ActType.ASK, "ask")
        push = MoveIntent("p2", ActType.PUSH_BACK, "push")
        self.assertIn("less obvious trade-off", prompts._turn_behavior(curious, ask))
        self.assertIn("concrete constraint", prompts._turn_behavior(curious, ask))
        self.assertIn("challenge the weak point plainly", prompts._turn_behavior(blunt, push))
        self.assertIn("unresolved risk", prompts._turn_behavior(blunt, push))

    def test_repair_keeps_response_target_and_avoids_procedural_now(self) -> None:
        state, _, p2 = _state()
        intent = MoveIntent(
            "p2",
            ActType.ACCEPT,
            "accept",
            option_focus=["A"],
            respond_to_turn=1,
        )
        prompt = prompts.repair_utterance(
            original_text="Alpha Trail seems fine.",
            issue_codes=["UNCLEAR_ACCEPT"],
            persona=p2,
            state=state,
            recent_lines=[],
            intent=intent,
            max_words=20,
        )
        self.assertIn("Keep this as a direct reply", prompt)
        self.assertIn("The steep finish worries me", prompt)
        self.assertIn("explicit first-person choice or acceptance", prompt)
        self.assertNotIn("selecting it now", prompt)

    def test_decision_guidance_uses_plain_chat_not_ballot_templates(self) -> None:
        state, _, p2 = _state()
        vote = MoveIntent("p2", ActType.VOTE, "vote", option_focus=["B"])
        accept = MoveIntent("p2", ActType.ACCEPT, "accept", option_focus=["A"])
        vote_guidance = prompts._move_guidance(state, p2, vote)
        accept_guidance = prompts._move_guidance(state, p2, accept)
        self.assertIn("first-person wording", vote_guidance)
        self.assertIn("explicitly accept", accept_guidance)
        self.assertIn("no hedging", vote_guidance.lower())
        self.assertIn("no hedging", accept_guidance.lower())

    def test_visible_commitment_handles_typographic_contractions_and_acceptance(self) -> None:
        state, _, _ = _state()
        validator = MessageValidator(OptionResolver(state.scenario.options), {"p1": "Ana", "p2": "Bo"})
        self.assertTrue(validator._has_visible_commitment("I’m voting for Beta Trail.", "B", ActType.VOTE))
        self.assertTrue(validator._has_visible_commitment("Beta Trail works for me.", "B", ActType.VOTE))
        self.assertTrue(validator._has_visible_commitment("Beta Trail is my choice.", "B", ActType.VOTE))
        self.assertTrue(validator._has_visible_commitment("I’m ready to commit to Beta Trail.", "B", ActType.ACCEPT))

    def test_visible_commitment_rejects_preference_and_hedges(self) -> None:
        state, _, _ = _state()
        validator = MessageValidator(OptionResolver(state.scenario.options), {"p1": "Ana", "p2": "Bo"})
        self.assertFalse(validator._has_visible_commitment("Beta Trail is the best option.", "B", ActType.VOTE))
        self.assertFalse(validator._has_visible_commitment("Beta Trail is not my choice.", "B", ActType.VOTE))
        self.assertFalse(validator._has_visible_commitment("Beta Trail isn’t my top pick, but I can handle the extra effort.", "B", ActType.ACCEPT))
        self.assertFalse(validator._has_visible_commitment("Beta Trail works for me if the path is quiet.", "B", ActType.ACCEPT))
        self.assertFalse(validator._has_visible_commitment("I’m still deciding, but I vote for Beta Trail.", "B", ActType.VOTE))

    def test_hedged_vote_does_not_mutate_binding_state(self) -> None:
        state, _, p2 = _state()
        intent = MoveIntent("p2", ActType.VOTE, "vote", option_focus=["B"])
        act = parse_dialogue_act(
            speaker_id=p2.id,
            speaker_name=p2.name,
            text="I’m still deciding, but I vote for Beta Trail.",
            resolver=OptionResolver(state.scenario.options),
            participant_names={"p1": "Ana", "p2": "Bo"},
            move=TurnMove(act=ActType.VOTE, option="B", stance="vote", present=True),
            intent=intent,
        )
        self.assertIsNone(act.explicit_vote)

    def test_repair_cannot_replace_valid_vote_with_blocked_rewrite(self) -> None:
        self.assertTrue(_repair_regresses_state(["REPEATED_START"], ["UNCLEAR_VOTE"]))
        self.assertFalse(_repair_regresses_state(["UNCLEAR_VOTE"], ["UNCLEAR_VOTE"]))
        self.assertFalse(_repair_regresses_state(["UNCLEAR_VOTE"], []))

    def test_outcome_uses_visible_support_instead_of_controller_candidate(self) -> None:
        state, p1, p2, p3 = _three_person_state()
        state.candidate_option = "A"
        _record_vote(state, p1, "B")
        _record_vote(state, p2, "B")
        _record_vote(state, p3, "A")
        outcome = ConsensusManager().finalize(state)
        self.assertEqual((outcome.status, outcome.final_option), ("majority", "B"))
        self.assertEqual(outcome.reason, "majority outcome with visible support fraction 0.67")

    def test_success_and_unresolved_follow_visible_commitment_counts(self) -> None:
        state, p1, p2, p3 = _three_person_state()
        state.candidate_option = "A"
        for persona in (p1, p2, p3):
            _record_vote(state, persona, "B")
        self.assertEqual(ConsensusManager().detect(state).final_option, "B")

        split, p1, p2, p3 = _three_person_state()
        for persona, option_id in zip((p1, p2, p3), ("A", "B", "C")):
            _record_vote(split, persona, option_id)
        outcome = ConsensusManager().finalize(split)
        self.assertEqual((outcome.status, outcome.final_option), ("unresolved", None))
        self.assertIn("visible-support majority", outcome.reason)

    def test_moderator_separates_holdout_from_missing_commitment(self) -> None:
        state, p1, p2, p3 = _three_person_state()
        _record_vote(state, p1, "B")
        state.turns.append(TurnRecord(
            index=2,
            speaker_id=p2.id,
            speaker_name=p2.name,
            text="Beta Trail is easiest for me.",
            phase=Phase.OPENING,
            act=DialogueAct(p2.id, "", ActType.OPENING, option_refs=["B"]),
            intent=MoveIntent(p2.id, ActType.OPENING, "opening", option_focus=["B"]),
        ))
        _record_vote(state, p3, "A")

        self.assertEqual(visible_candidate_status(state, p2.id, "B"), ("missing", None))
        self.assertEqual(visible_candidate_status(state, p3.id, "B"), ("holdout", "A"))

        missing_prompt = prompts.moderator_holdout_prompt(state, "B", [], [p2.id])
        self.assertIn("Ask only for a direct confirmation", missing_prompt)
        self.assertIn("no visible alternative or objection", missing_prompt)

        mixed_prompt = prompts.moderator_holdout_prompt(state, "B", [p3.id], [p2.id])
        self.assertIn("Cy visibly backed another option", mixed_prompt)
        self.assertIn("Bo has not visibly committed", mixed_prompt)
        self.assertIn("Do not call Bo opposed", mixed_prompt)

    def test_moderator_concentration_ignores_hidden_or_corrupted_controller_leans(self) -> None:
        state, p1, p2, p3 = _three_person_state()
        for runtime in state.runtimes.values():
            runtime.current_preference = "A"
        for persona, option_id in zip((p1, p2, p3), ("A", "B", "C")):
            state.turns.append(TurnRecord(
                index=len(state.turns) + 1,
                speaker_id=persona.id,
                speaker_name=persona.name,
                text=f"{state.scenario.option(option_id).name} fits me.",
                phase=Phase.OPENING,
                act=DialogueAct(persona.id, "", ActType.OPENING, option_refs=[option_id]),
                intent=MoveIntent(persona.id, ActType.OPENING, "opening", option_focus=[option_id]),
            ))
        self.assertAlmostEqual(visible_preference_concentration(state), 1 / 3)
        split = prompts._camp_split(state)
        self.assertIn("1 leaning toward Alpha Trail", split)
        self.assertIn("1 leaning toward Beta Trail", split)
        self.assertIn("1 leaning toward Gamma Trail", split)

    def test_majority_closure_names_visible_non_support(self) -> None:
        state, p1, p2, p3 = _three_person_state()
        _record_vote(state, p1, "B")
        _record_vote(state, p2, "B")
        _record_vote(state, p3, "A")
        outcome = RunOutcome("majority", "B", "visible majority", 3)
        prompt = prompts.moderator_closure_prompt(outcome, state.scenario, state)
        self.assertIn("Ana, Bo visibly supported Beta Trail", prompt)
        self.assertIn("Cy visibly backed Alpha Trail", prompt)
        self.assertIn("agreement wasn't unanimous", prompt)

        farewell = prompts.farewell_line(p3, state.scenario, outcome, state, [p1.name, p2.name], 18)
        self.assertIn("without your visible support", farewell)
        self.assertIn("without implying that you agreed", farewell)

    def test_unresolved_closure_uses_visible_commitments_not_hidden_leans(self) -> None:
        state, p1, p2, p3 = _three_person_state()
        _record_vote(state, p1, "A")
        _record_vote(state, p2, "B")
        outcome = RunOutcome("unresolved", None, "no majority", 2)
        prompt = prompts.moderator_closure_prompt(outcome, state.scenario, state)
        self.assertIn("Ana visibly supported Alpha Trail", prompt)
        self.assertIn("Bo visibly supported Beta Trail", prompt)
        self.assertIn("Cy made no visible commitment", prompt)
        self.assertIn("preferences were expressed but no final commitment", prompt)


class SocialBeatTests(unittest.TestCase):
    def _make_personas(self, extraversions: list[int]) -> list[Persona]:
        return [
            _persona(f"p{i+1}", f"Name{i+1}", TraitProfile(3, 3, e, 3, 2), "A")
            for i, e in enumerate(extraversions)
        ]

    def test_social_speakers_returns_at_most_one(self) -> None:
        personas = self._make_personas([5, 5, 5, 5, 5])
        random.seed(0)
        for _ in range(50):
            result = Orchestrator._social_speakers(personas)
            self.assertLessEqual(len(result), 1)

    def test_social_speakers_picks_most_extraverted(self) -> None:
        personas = self._make_personas([2, 5, 3])
        random.seed(42)
        for _ in range(100):
            result = Orchestrator._social_speakers(personas)
            if result:
                self.assertEqual(result[0].id, "p2")

    def test_social_speakers_can_return_empty(self) -> None:
        personas = self._make_personas([1])
        # extraversion=1, trait_max=5 → 20% draw; seed for at least one miss
        found_empty = False
        random.seed(7)
        for _ in range(30):
            if not Orchestrator._social_speakers(personas):
                found_empty = True
                break
        self.assertTrue(found_empty, "low-extraversion persona should sometimes produce no social beat")

    def test_farewell_prompt_omits_background(self) -> None:
        persona = _persona("p1", "Ana", TraitProfile(3, 3, 3, 3, 2), "A", background="Ana grew up near the coast.")
        outcome = RunOutcome(status="successful", final_option="A", reason="unanimous", turns=10)
        options = [OptionCard("A", "Alpha", "Alpha", {"x": "1"}, "up", "trade", "concern", "best")]
        scenario = Scenario("Pick one", "generic", "What matters?", options, [])
        state = initialise_state(scenario, [persona])
        prompt = prompts.farewell_line(persona, scenario, outcome, state, ["Bo"], 18)
        self.assertNotIn(persona.background, prompt)

    def test_greeting_prompt_avoids_arrival_framing(self) -> None:
        persona = _persona("p1", "Ana", TraitProfile(3, 3, 3, 3, 2), "A")
        prompt = prompts.greeting_line(persona, ["Bo"], 14)
        self.assertNotIn("first text", prompt)
        self.assertNotIn("fires off", prompt)
        self.assertIn("ongoing thread", prompt)


if __name__ == "__main__":
    unittest.main()
