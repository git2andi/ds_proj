"""Run orchestration for one discussion."""

from __future__ import annotations

import datetime as _dt
from typing import Optional

import prompts
from config_loader import cfg
from consensus import ConsensusManager
from controller import DialogueController
from logger import DialogueLogger
from persona import PersonaBuilder
from router import TurnRouter
from scenario_builder import ScenarioBuilder
from schemas import DialogueRunResult, MoveIntent, Phase, RunOutcome
from state import StateTracker, initialise_state
from utterance_generator import UtteranceGenerator
from utils import OptionResolver
from validator import MessageValidator


class Orchestrator:
    def __init__(self, topic: str) -> None:
        self.topic = topic.strip()
        if not self.topic:
            raise ValueError("Topic must not be empty.")
        self.run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.logger = DialogueLogger(self.run_id, self.topic)
        self.scenario_builder = ScenarioBuilder()
        self.controller = DialogueController()
        self.router = TurnRouter()
        self.generator = UtteranceGenerator()
        self.consensus = ConsensusManager()

    def run(self) -> DialogueRunResult:
        scenario = self.scenario_builder.build(self.topic)
        personas = PersonaBuilder(self.topic).build(int(cfg.simulation.num_participants), scenario.options)
        state = initialise_state(scenario, personas)
        tracker = StateTracker(scenario, personas)
        validator = MessageValidator(OptionResolver(scenario.options))

        self._append_moderator_opening(state, tracker)

        while not self.controller.should_stop(state):
            self.controller.update_phase(state)
            detected = self.consensus.detect(state)
            if detected:
                state.outcome = detected
                break
            if state.phase == Phase.CLOSURE:
                break
            intent = self.router.next_move(state)
            text, prompt, tok_in, tok_out, issue_codes = self._generate_validated_turn(state, intent, validator)
            self.logger.write_prompt(prompt, f"{intent.speaker_id}_{intent.act.value}")
            speaker = state.persona_by_id(intent.speaker_id)
            tracker.apply_turn(
                state,
                speaker_id=speaker.id,
                speaker_name=speaker.name,
                text=text,
                phase=state.phase,
                intent=intent,
                tokens_in=tok_in,
                tokens_out=tok_out,
                validation_issues=issue_codes,
            )

        outcome = state.outcome or self.consensus.finalize(state)
        state.outcome = outcome
        self._append_moderator_closure(state, tracker, outcome)
        paths = self.logger.finish(state, outcome)
        transcript = [f"{turn.speaker_name}: {turn.text}" for turn in state.turns]
        return DialogueRunResult(scenario, personas, transcript, outcome, paths)

    def _generate_validated_turn(
        self,
        state,
        intent: MoveIntent,
        validator: MessageValidator,
    ) -> tuple[str, str, int, int, list[str]]:
        text, prompt, tok_in, tok_out = self.generator.generate(state, intent)
        validation = validator.validate(text, state, intent)
        attempts = 0
        while validation.needs_repair and attempts < int(cfg.simulation.max_repairs_per_turn):
            attempts += 1
            repaired, repair_prompt, repair_in, repair_out = self.generator.repair(text, validation, state, intent)
            self.logger.write_prompt(repair_prompt, f"{intent.speaker_id}_{intent.act.value}_repair")
            text = repaired
            tok_in += repair_in
            tok_out += repair_out
            validation = validator.validate(text, state, intent)
            prompt = repair_prompt
        if validation.needs_repair:
            text = self.generator.fallback(state, intent)
            validation = validator.validate(text, state, intent)
        return text, prompt, tok_in, tok_out, validation.codes()

    def _append_moderator_opening(self, state, tracker: StateTracker) -> None:
        text = prompts.moderator_opening(state.scenario)
        tracker.apply_turn(state, "moderator", "Moderator", text.replace("Moderator:", "", 1).strip(), Phase.OPENING)
        state.moderator_interventions += 1

    def _append_moderator_closure(self, state, tracker: StateTracker, outcome: RunOutcome) -> None:
        if outcome.final_option:
            option = state.scenario.option(outcome.final_option)
            text = prompts.moderator_close_consensus(option)
        else:
            text = prompts.moderator_close_unresolved()
        tracker.apply_turn(state, "moderator", "Moderator", text.replace("Moderator:", "", 1).strip(), Phase.CLOSURE)
        state.moderator_interventions += 1


# Backward-compatible alias for older callers.
def run_dialogue(topic: str) -> DialogueRunResult:
    return Orchestrator(topic).run()
