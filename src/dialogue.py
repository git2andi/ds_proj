"""Dialogue runner for option-grounded multi-user simulation.

The runner separates four responsibilities:
1. environment setup (scenario + option board),
2. simulator state (private lean, agenda, tunable behavior),
3. routing policy (who speaks, to whom, with which dialogue act), and
4. visible transcript observation (votes/outcomes from public text only).
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import prompts
from aliases import short_alias_map
from builders import SetupBuilder
from config_loader import cfg
from logger import DialogueLogger, token_summary_for
from llm_client import get_llm_client
from models import (
    ActType,
    AgendaStatus,
    DialogueAct,
    DialogueRunResult,
    DialogueState,
    MoveIntent,
    OpenQuestion,
    OptionCoverage,
    ParticipantRuntime,
    Persona,
    Phase,
    RunOutcome,
    TurnRecord,
)
from parsing import OptionResolver, parse_dialogue_act
from simulator import mark_agenda_done, next_agenda_item
from utils import normalise_lines, normalise_ws, strip_speaker_prefix, weighted_choice

_DECISION_ACTS = {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}
_DISCUSSION_ACTS = {
    ActType.BUILD,
    ActType.AGREE,
    ActType.CHALLENGE,
    ActType.ASK,
    ActType.ANSWER,
    ActType.COMPARE,
    ActType.INVITE,
    ActType.PROPOSE_COMPROMISE,
}


@dataclass(slots=True)
class ValidationReport:
    issues: list[str]
    block_state_mutation: bool = False


class DialogueRunner:
    def __init__(self, topic: str) -> None:
        self.topic = topic.strip()
        if not self.topic:
            raise ValueError("topic must not be empty")
        seed = cfg.simulation.get("random_seed", None)
        if seed is not None:
            random.seed(int(seed))
        self._llm = get_llm_client()
        self._resolver: OptionResolver | None = None
        self._intervention_count = 0
        self._last_intervention_turn = -999

    def run(self) -> DialogueRunResult:
        n = int(cfg.simulation.num_participants)
        self._llm.reset_session()
        scenario, personas = SetupBuilder(self.topic).build(n)
        setup_tokens_in = self._llm.session_tokens_in
        setup_tokens_out = self._llm.session_tokens_out
        self._llm.reset_session()

        state = initialise_state(scenario, personas)
        state.setup_tokens_in = setup_tokens_in
        state.setup_tokens_out = setup_tokens_out
        self._resolver = OptionResolver(scenario.options)
        self._derive_pacing(state)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.logger = DialogueLogger(run_id, self.topic)

        self._print_header(state)
        self._emit(self._append_moderator(state, prompts.moderator_opening(scenario), Phase.OPENING))

        self._opening_round(state)
        self._discussion_loop(state)
        self._decision_loop(state)

        outcome = ConsensusManager.finalize(state)
        state.outcome = outcome
        closing = self._moderator_say(prompts.moderator_closure_prompt(outcome, scenario, state), state)
        self._emit(self._append_moderator(state, closing, Phase.CLOSURE))
        self._mark_phase(state, Phase.CLOSURE, f"closed as {outcome.status}")

        state.dialogue_tokens_in = self._llm.session_tokens_in
        state.dialogue_tokens_out = self._llm.session_tokens_out
        paths = self.logger.finish(state, outcome)
        transcript = [f"{turn.speaker_name}: {turn.text}" for turn in state.turns]
        return DialogueRunResult(scenario, personas, transcript, outcome, paths, token_summary_for(state))

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _opening_round(self, state: DialogueState) -> None:
        state.phase = Phase.OPENING
        for persona in self._opening_order(state.personas):
            intent = MoveIntent(
                speaker_id=persona.id,
                act=ActType.OPENING,
                reason="state the current favorite and one grounded reason without making a final vote",
                option_focus=[persona.preferred_option],
                moves_lean=True,
            )
            self._emit(self._generate_and_append(state, intent))
        self._mark_phase(state, Phase.DISCUSSION, "all participants gave an opening view")

    def _discussion_loop(self, state: DialogueState) -> None:
        state.phase = Phase.DISCUSSION
        while True:
            if self._ready_for_vote(state):
                self._mark_phase(state, Phase.NARROWING, self._vote_reason(state))
                return
            maybe_nudge = self._maybe_moderator_nudge(state)
            if maybe_nudge:
                self._emit(maybe_nudge)
            intent = self._route_discussion_turn(state)
            self._emit(self._generate_and_append(state, intent))

    def _decision_loop(self, state: DialogueState) -> None:
        state.phase = Phase.NARROWING
        for round_index in range(int(cfg.conversation.max_vote_rounds)):
            candidate = self._candidate_for_vote(state)
            state.candidate_option = candidate
            reason = "let's test where everyone stands" if round_index == 0 else "the first vote did not settle it"
            self._emit(self._moderator_vote_nudge(state, candidate, reason))
            for persona in self._vote_order(state, candidate):
                self._emit(self._generate_and_append(state, self._vote_intent(state, persona, candidate)))
            provisional = ConsensusManager.finalize(state)
            if provisional.status in {"successful", "majority"}:
                state.outcome = provisional
                self._mark_phase(state, Phase.CLOSURE, f"{provisional.status} visible after vote round {round_index + 1}")
                return
        self._mark_phase(state, Phase.CLOSURE, "vote rounds exhausted without visible consensus")

    # ------------------------------------------------------------------
    # Routing policy: who / what / whom
    # ------------------------------------------------------------------

    def _route_discussion_turn(self, state: DialogueState) -> MoveIntent:
        open_question = self._next_answerable_question(state)
        if open_question is not None:
            return MoveIntent(
                speaker_id=open_question.target_id,
                act=ActType.ANSWER,
                reason="answer the direct question, then add one implication for the decision",
                addressee_id=open_question.asked_by,
                option_focus=open_question.option_focus or self._focus_from_recent(state),
                respond_to_turn=open_question.turn_id,
            )

        coverage_gap = self._coverage_gap_option(state)
        if coverage_gap is not None:
            speaker = self._speaker_for_option_coverage(state, coverage_gap)
            current = state.runtimes[speaker.id].current_preference or speaker.preferred_option
            focus = [coverage_gap]
            if current in state.scenario.option_ids and current != coverage_gap:
                focus.append(current)
            return MoveIntent(
                speaker_id=speaker.id,
                act=ActType.COMPARE,
                reason="briefly bring in an option that has not yet been socially processed, then compare it with the current lean",
                option_focus=focus,
                moves_lean=False,
            )

        speaker = self._choose_speaker(state)
        agenda_pair = next_agenda_item(speaker)
        if agenda_pair and random.random() < (0.45 + 0.35 * speaker.sim_params.initiative):
            agenda_index, agenda_item = agenda_pair
            act = agenda_item.act
            focus = [agenda_item.option] if agenda_item.option in state.scenario.option_ids else []
        else:
            agenda_index = None
            act = self._choose_discussion_act(state, speaker)
            focus = []

        target_turn = self._choose_target_turn(state, speaker, act)
        target_speaker = target_turn.speaker_id if target_turn and target_turn.speaker_id != "moderator" else None
        addressee = target_speaker if target_speaker and random.random() < float(cfg.routing.direct_address_probability) else None
        if not focus:
            focus = self._focus_options(state, speaker, act, target_turn)
        reason = self._reason_for_act(state, speaker, act, focus, target_turn)
        moves_lean = act in {ActType.AGREE, ActType.PROPOSE_COMPROMISE} and bool(focus)
        return MoveIntent(
            speaker_id=speaker.id,
            act=act,
            reason=reason,
            addressee_id=addressee,
            option_focus=focus,
            respond_to_turn=target_turn.index if target_turn else None,
            agenda_index=agenda_index,
            moves_lean=moves_lean,
        )

    def _choose_speaker(self, state: DialogueState) -> Persona:
        last_speaker = self._last_participant_id(state)
        min_turns = min(rt.turn_count for rt in state.runtimes.values())
        candidates: list[Persona] = []
        weights: list[float] = []
        for persona in state.personas:
            if len(state.personas) > 1 and persona.id == last_speaker:
                continue
            rt = state.runtimes[persona.id]
            p = persona.sim_params
            base = 0.35 + p.engagement + 0.35 * p.initiative
            if rt.turn_count == min_turns:
                base += float(cfg.routing.quiet_speaker_boost)
            elif rt.turn_count > min_turns:
                base *= max(0.25, 1.0 - 0.25 * (rt.turn_count - min_turns))
            candidates.append(persona)
            weights.append(base)
        if not candidates:
            candidates = state.personas[:]
            weights = [1.0 for _ in candidates]
        return weighted_choice(candidates, weights)

    def _choose_discussion_act(self, state: DialogueState, speaker: Persona) -> ActType:
        raw = dict(cfg.routing.move_weights.items())
        p = speaker.sim_params
        raw["ask"] = raw.get("ask", 0.0) * (0.75 + p.initiative)
        raw["challenge"] = raw.get("challenge", 0.0) * (0.70 + p.stubbornness)
        raw["agree"] = raw.get("agree", 0.0) * (0.60 + (1.0 - p.stubbornness))
        raw["invite"] = raw.get("invite", 0.0) * (0.50 + p.responsiveness)
        if self._recent_question_count(state) >= 1:
            raw["ask"] *= 0.25
        if participant_turn_count(state) >= max(state.min_discussion_turns, state.force_narrow_turns - 2):
            raw["ask"] *= 0.25
            raw["invite"] *= 0.5
        if self._latent_leading_count(state) >= max(2, math.ceil(0.5 * len(state.personas))):
            raw["propose_compromise"] = raw.get("propose_compromise", 0.0) + 0.10
        mapping = {
            "build": ActType.BUILD,
            "agree": ActType.AGREE,
            "challenge": ActType.CHALLENGE,
            "ask": ActType.ASK,
            "answer": ActType.ANSWER,
            "compare": ActType.COMPARE,
            "invite": ActType.INVITE,
            "propose_compromise": ActType.PROPOSE_COMPROMISE,
        }
        names = [k for k in raw if k in mapping]
        act = mapping[weighted_choice(names, [float(raw[k]) for k in names])]
        if act == ActType.ANSWER and not self._next_answerable_question(state):
            return ActType.BUILD
        return act

    def _choose_target_turn(self, state: DialogueState, speaker: Persona, act: ActType) -> TurnRecord | None:
        participant_turns = [t for t in state.turns if t.speaker_id not in {"moderator", speaker.id}]
        if not participant_turns:
            return None
        if act in {ActType.AGREE, ActType.CHALLENGE, ActType.ASK, ActType.ANSWER, ActType.INVITE}:
            return participant_turns[-1]
        if random.random() < 0.7:
            return participant_turns[-1]
        return random.choice(participant_turns[-min(4, len(participant_turns)):])

    def _focus_options(self, state: DialogueState, speaker: Persona, act: ActType, target_turn: TurnRecord | None) -> list[str]:
        ids: list[str] = []
        if target_turn:
            ids.extend(target_turn.act.option_refs)
        current = state.runtimes[speaker.id].current_preference or speaker.preferred_option
        if current and current not in ids:
            ids.append(current)
        if act in {ActType.COMPARE, ActType.CHALLENGE, ActType.ASK}:
            rival = self._rival_option(state, speaker, exclude=set(ids))
            if rival:
                ids.append(rival)
        leading = self._latent_leading_option(state)
        if act == ActType.PROPOSE_COMPROMISE and leading and leading not in ids:
            ids.insert(0, leading)
        return [x for x in ids if x in state.scenario.option_ids][:3]

    def _reason_for_act(self, state: DialogueState, speaker: Persona, act: ActType, focus: list[str], target_turn: TurnRecord | None) -> str:
        names = short_alias_map(state.scenario.options)
        focus_names = ", ".join(names[o] for o in focus) if focus else "the options"
        if act == ActType.BUILD:
            return f"add a new grounded reason about {focus_names}, connected to the current discussion"
        if act == ActType.AGREE:
            return f"agree with the recent point, but add a small new angle about {focus_names}"
        if act == ActType.CHALLENGE:
            return f"push back on the recent point or name a real concern about {focus_names}"
        if act == ActType.ASK:
            return f"ask one concrete question that helps compare {focus_names}"
        if act == ActType.ANSWER:
            return f"answer the recent question using only the option facts, then move the decision forward"
        if act == ActType.COMPARE:
            return f"compare {focus_names} with one clear trade-off"
        if act == ActType.INVITE:
            quiet = self._quietest_other(state, speaker.id)
            return f"bring {quiet.name if quiet else 'someone quieter'} into the discussion with a useful prompt"
        if act == ActType.PROPOSE_COMPROMISE:
            return f"test whether {focus_names} could be common ground without claiming it is already decided"
        return "respond naturally and move the decision forward"

    def _vote_intent(self, state: DialogueState, persona: Persona, candidate: str) -> MoveIntent:
        blocked = persona.rejection == candidate or candidate in state.runtimes[persona.id].hard_rejections
        current = state.runtimes[persona.id].current_preference or persona.preferred_option
        if blocked:
            alternative = current if current != candidate and current in state.scenario.option_ids else persona.preferred_option
            return MoveIntent(
                speaker_id=persona.id,
                act=ActType.VOTE,
                reason="cast a clear visible vote for the best acceptable alternative and briefly mention why the candidate is blocked",
                option_focus=[alternative, candidate] if alternative != candidate else [alternative],
                length_hint="short",
                moves_lean=True,
            )
        if self._should_compromise_to_candidate(state, persona, candidate):
            return MoveIntent(
                speaker_id=persona.id,
                act=ActType.ACCEPT if current != candidate else ActType.VOTE,
                reason="make a clear visible commitment to the likely group option",
                option_focus=[candidate],
                length_hint="short",
                moves_lean=True,
            )
        return MoveIntent(
            speaker_id=persona.id,
            act=ActType.VOTE,
            reason="cast a clear visible vote for the option they actually choose now",
            option_focus=[current if current in state.scenario.option_ids else candidate],
            length_hint="short",
            moves_lean=True,
        )

    # ------------------------------------------------------------------
    # Generation, observation, and state mutation
    # ------------------------------------------------------------------

    def _generate_and_append(self, state: DialogueState, intent: MoveIntent) -> TurnRecord:
        persona = state.persona_by_id(intent.speaker_id)
        max_words = self._max_words_for(intent, persona)
        recent_lines = self._recent_lines(state)
        focus_options = [state.scenario.option(i) for i in intent.option_focus if i in state.scenario.option_ids]
        addressee_name = state.name_for(intent.addressee_id) if intent.addressee_id else None
        prompt = prompts.sim_utterance(
            persona=persona,
            state=state,
            intent=intent,
            recent_lines=recent_lines,
            focus_options=focus_options,
            addressee_name=addressee_name,
            max_words=max_words,
        )
        self.logger.write_prompt(prompt, f"{state.turn_index + 1:03d}_{persona.id}_{intent.act.value}")
        text, tokens_in, tokens_out = self._call_participant(prompt, persona.name, max_words)
        act = self._parse_act(state, persona, text, intent)
        report = self._validate_turn_text(text, state, persona, intent, act)
        repaired = False
        trigger_codes = list(report.issues)
        attempts = int(cfg.simulation.max_repairs_per_turn)
        while report.issues and attempts > 0:
            attempts -= 1
            repaired = True
            repair_prompt = prompts.repair_utterance(
                original_text=text,
                issue_codes=report.issues,
                persona=persona,
                state=state,
                recent_lines=recent_lines,
                intent=intent,
                max_words=max_words,
            )
            self.logger.write_prompt(repair_prompt, f"{state.turn_index + 1:03d}_{persona.id}_repair")
            candidate_text, ti, to = self._call_participant(repair_prompt, persona.name, max_words)
            tokens_in += ti
            tokens_out += to
            candidate_act = self._parse_act(state, persona, candidate_text, intent)
            candidate_report = self._validate_turn_text(candidate_text, state, persona, intent, candidate_act)
            if not candidate_report.issues or len(candidate_report.issues) <= len(report.issues):
                text, act, report = candidate_text, candidate_act, candidate_report

        block = report.block_state_mutation or self._semantic_block(persona, intent, act)
        record = self._append_participant(
            state,
            persona,
            text,
            act,
            intent,
            tokens_in,
            tokens_out,
            report.issues,
            repaired,
            trigger_codes,
            block,
        )
        if not block:
            self._apply_semantics(state, record)
            mark_agenda_done(persona, intent.agenda_index)
        else:
            state.no_progress_count += 1
        return record

    def _call_participant(self, prompt: str, speaker_name: str, max_words: int) -> tuple[str, int, int]:
        raw = self._llm.generate(prompt, profile="dialogue")
        text = clean_generated(raw, speaker_name, max_words)
        return text, self._llm.last_tokens_in, self._llm.last_tokens_out

    def _parse_act(self, state: DialogueState, persona: Persona, text: str, intent: MoveIntent) -> DialogueAct:
        assert self._resolver is not None
        previous = self._last_participant_id(state)
        names = {p.id: p.name for p in state.personas}
        return parse_dialogue_act(
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            resolver=self._resolver,
            participant_names=names,
            intent=intent,
            previous_speaker_id=previous,
        )

    def _validate_turn_text(self, text: str, state: DialogueState, persona: Persona, intent: MoveIntent, act: DialogueAct) -> ValidationReport:
        issues: list[str] = []
        block = False
        if not text.strip():
            issues.append("EMPTY")
            block = True
        if "\n" in text.strip():
            issues.append("MULTI_TURN_OUTPUT")
        if re.search(r"\[\s*(?:act|opt|stance)\s*=", text, re.I):
            issues.append("LEAKED_METADATA")
        if self._resolver and self._resolver.invalid_option_refs(text):
            issues.append("INVALID_OPTION_REFERENCE")
            block = True
        if (
            intent.option_focus
            and "not yet been socially processed" in intent.reason
            and intent.option_focus[0] not in act.option_refs
        ):
            issues.append("MISSING_REQUIRED_OPTION_FOCUS")
            block = True
        if intent.act in {ActType.VOTE, ActType.ACCEPT} and not (act.explicit_vote or act.accepts):
            issues.append("UNCLEAR_VISIBLE_COMMITMENT")
            block = True
        if persona.rejection and (act.explicit_vote == persona.rejection or persona.rejection in act.accepts):
            issues.append("HARD_BLOCKER_ACCEPTED_REJECTED_OPTION")
            block = True
        return ValidationReport(list(dict.fromkeys(issues)), block)

    @staticmethod
    def _semantic_block(persona: Persona, intent: MoveIntent, act: DialogueAct) -> bool:
        if persona.rejection and (act.explicit_vote == persona.rejection or persona.rejection in act.accepts):
            return True
        if intent.act in {ActType.VOTE, ActType.ACCEPT} and not (act.explicit_vote or act.accepts):
            return True
        return False

    def _apply_semantics(self, state: DialogueState, record: TurnRecord) -> None:
        rt = state.runtimes[record.speaker_id]
        act = record.act
        before = self._snapshot_progress(state)

        if act.question_target_id:
            self._register_question(state, record)
        if record.intent and record.intent.act == ActType.ANSWER:
            self._close_answered_questions(state, record.speaker_id)

        for option_id in act.option_refs:
            cov = state.coverage[option_id]
            cov.mentions += 1
            if act.act_type in {ActType.BUILD, ActType.AGREE, ActType.COMPARE, ActType.PROPOSE_COMPROMISE, ActType.OPENING}:
                cov.reasons += 1
            if act.act_type in {ActType.CHALLENGE, ActType.REJECT} or option_id in act.soft_rejects:
                cov.objections += 1

        if act.explicit_vote:
            rt.explicit_vote = act.explicit_vote
            rt.current_preference = act.explicit_vote
        for option_id in act.accepts:
            rt.accepted_options.add(option_id)
            rt.explicit_vote = option_id
            rt.current_preference = option_id
            state.coverage[option_id].acceptances += 1
        for option_id, reason in act.soft_rejects.items():
            rt.soft_rejections[option_id] = reason
        for option_id, reason in act.hard_rejects.items():
            rt.hard_rejections[option_id] = reason

        if record.intent and record.intent.act == ActType.OPENING and record.intent.option_focus:
            rt.current_preference = record.intent.option_focus[0]
        elif record.intent and record.intent.moves_lean and record.intent.option_focus:
            candidate = record.intent.option_focus[0]
            if candidate != rt.current_preference and self._can_shift_to(state.persona_by_id(record.speaker_id), candidate):
                rt.current_preference = candidate
        elif act.proposes_option and self._can_shift_to(state.persona_by_id(record.speaker_id), act.proposes_option):
            if random.random() > state.persona_by_id(record.speaker_id).sim_params.compromise_threshold:
                rt.current_preference = act.proposes_option

        after = self._snapshot_progress(state)
        state.no_progress_count = 0 if after != before else state.no_progress_count + 1

    def _append_participant(
        self,
        state: DialogueState,
        persona: Persona,
        text: str,
        act: DialogueAct,
        intent: MoveIntent,
        tokens_in: int,
        tokens_out: int,
        issues: list[str],
        repaired: bool,
        trigger_codes: list[str],
        block: bool,
    ) -> TurnRecord:
        state.turn_index += 1
        rt = state.runtimes[persona.id]
        rt.turn_count += 1
        rt.last_spoke_turn = state.turn_index
        rt.already_said.append(text)
        record = TurnRecord(
            index=state.turn_index,
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            phase=state.phase,
            act=act,
            intent=intent,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            validation_issues=issues,
            repaired=repaired,
            repair_trigger_codes=trigger_codes,
            state_mutation_blocked=block,
        )
        state.turns.append(record)
        return record

    def _append_moderator(self, state: DialogueState, text: str, phase: Phase) -> TurnRecord:
        state.turn_index += 1
        text = normalise_lines(text)
        act = DialogueAct(speaker_id="moderator", text=text, act_type=ActType.REACT)
        record = TurnRecord(index=state.turn_index, speaker_id="moderator", speaker_name="Moderator", text=text, phase=phase, act=act)
        state.turns.append(record)
        return record

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def _ready_for_vote(self, state: DialogueState) -> bool:
        participant_turns = participant_turn_count(state)
        if participant_turns < state.min_discussion_turns:
            return False
        if self._coverage_gap_option(state) is not None and participant_turns < state.hard_max_turns:
            return False
        if participant_turns >= state.hard_max_turns:
            return True
        if participant_turns >= state.force_narrow_turns:
            return True
        early_gate = state.min_discussion_turns + int(cfg.conversation.early_vote_extra_turns)
        if participant_turns >= early_gate and self._latent_concentration(state) >= float(cfg.conversation.concentration_to_vote):
            return True
        return False

    def _vote_reason(self, state: DialogueState) -> str:
        participant_turns = participant_turn_count(state)
        if participant_turns >= state.hard_max_turns:
            return "hard cap reached; forcing a visible vote instead of closing early"
        if participant_turns >= state.force_narrow_turns:
            return "target discussion length reached"
        return "internal lean concentration held after enough back-and-forth"

    def _candidate_for_vote(self, state: DialogueState) -> str:
        latent = self._latent_leading_option(state)
        return latent or state.personas[0].preferred_option

    def _should_compromise_to_candidate(self, state: DialogueState, persona: Persona, candidate: str) -> bool:
        if not self._can_shift_to(persona, candidate):
            return False
        rt = state.runtimes[persona.id]
        if rt.current_preference == candidate:
            return True
        support = self._latent_leading_count(state)
        pressure = support / max(1, len(state.personas) - 1)
        probability = 0.10 + 0.65 * (1.0 - persona.sim_params.compromise_threshold) + 0.25 * pressure
        if candidate in persona.preferred_options:
            probability += 0.15
        return random.random() < min(0.95, probability)

    @staticmethod
    def _can_shift_to(persona: Persona, option_id: str) -> bool:
        if persona.rejection == option_id:
            return False
        if persona.sim_params.stubbornness >= 0.85 and option_id != persona.preferred_option:
            return random.random() < 0.04
        return True

    # ------------------------------------------------------------------
    # Moderator helpers
    # ------------------------------------------------------------------

    def _maybe_moderator_nudge(self, state: DialogueState) -> TurnRecord | None:
        if self._intervention_count >= int(cfg.conversation.moderator_max_interventions):
            return None
        if participant_turn_count(state) < max(state.min_discussion_turns + 2, state.force_narrow_turns - 1):
            return None
        if state.no_progress_count < int(cfg.conversation.moderator_stall_window):
            return None
        if state.turn_index - self._last_intervention_turn < int(cfg.conversation.moderator_cooldown_turns):
            return None
        candidate = self._latent_leading_option(state)
        candidate_name = state.scenario.option(candidate).name if candidate else None
        target_name, requested_action, focus = self._moderator_intervention_details(state, candidate)
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state,
                "discussion seems to be circling",
                candidate_name,
                target_name=target_name,
                requested_action=requested_action,
                focus_options=focus,
            ),
            state,
        )
        self._intervention_count += 1
        self._last_intervention_turn = state.turn_index
        state.no_progress_count = 0
        return self._append_moderator(state, text, Phase.DISCUSSION)

    def _moderator_vote_nudge(self, state: DialogueState, candidate: str, reason: str) -> TurnRecord:
        candidate_name = state.scenario.option(candidate).name
        target_name, requested_action, focus = self._moderator_intervention_details(state, candidate, voting=True)
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state,
                reason,
                candidate_name,
                target_name=target_name,
                requested_action=requested_action,
                focus_options=focus or [candidate],
            ),
            state,
        )
        return self._append_moderator(state, text, Phase.NARROWING)

    def _moderator_say(self, prompt: str, state: DialogueState) -> str:
        raw = self._llm.generate(prompt, profile="dialogue")
        text = clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.moderator))
        if self._resolver and self._resolver.invalid_option_refs(text):
            raw = self._llm.generate(prompt + "\nOnly use the exact option names already listed.", profile="repair")
            text = clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.moderator))
        return text or "Let’s make this concrete: what is the strongest remaining concern before we choose?"

    def _coverage_gap_option(self, state: DialogueState) -> str | None:
        if not bool(cfg.conversation.get("require_option_coverage_before_vote", True)):
            return None
        # Opening turns may naturally leave some non-preferred options untouched.
        # After everyone has spoken once, the router gives each untouched option a
        # light social check before final voting. This is coverage, not forced support.
        if participant_turn_count(state) <= len(state.personas):
            return None
        gaps = [
            (option_id, coverage.mentions, coverage.reasons + coverage.objections)
            for option_id, coverage in state.coverage.items()
            if coverage.mentions == 0
        ]
        if not gaps:
            return None
        gaps.sort(key=lambda item: (item[1], item[2], item[0]))
        return gaps[0][0]

    def _speaker_for_option_coverage(self, state: DialogueState, option_id: str) -> Persona:
        last = self._last_participant_id(state)
        # Prefer a participant who can plausibly discuss the option: secondary
        # preference first, then high openness/initiative, while avoiding the
        # last speaker in ordinary discussion.
        candidates = [p for p in state.personas if p.id != last] or state.personas[:]
        def score(persona: Persona) -> tuple[float, float]:
            p = persona.sim_params
            preference_bonus = 1.0 if option_id in persona.preferred_options else 0.0
            return (preference_bonus + 0.45 * p.initiative + 0.35 * p.engagement + 0.20 * (1.0 - p.stubbornness), -state.runtimes[persona.id].turn_count)
        return max(candidates, key=score)

    def _moderator_intervention_details(
        self,
        state: DialogueState,
        candidate: str | None,
        *,
        voting: bool = False,
    ) -> tuple[str | None, str, list[str]]:
        gap = self._coverage_gap_option(state)
        if gap is not None and not voting:
            return (
                None,
                f"ask the group to briefly compare Option {gap} before narrowing",
                [gap],
            )

        open_question = self._next_answerable_question(state)
        if open_question is not None:
            return (
                state.name_for(open_question.target_id),
                "ask for a direct answer to the pending question",
                open_question.option_focus,
            )

        uncommitted = [p for p in state.personas if not state.runtimes[p.id].explicit_vote]
        if voting and len(uncommitted) == 1:
            target = uncommitted[0]
            return (
                target.name,
                "ask for one unambiguous final vote, not conditional support",
                [candidate] if candidate else [],
            )
        if voting:
            return (
                None,
                "ask everyone for an unambiguous final vote using a clear phrase like 'I vote for ...'",
                [candidate] if candidate else [],
            )

        if candidate:
            dissenters = [p for p in state.personas if state.runtimes[p.id].current_preference != candidate]
            if len(dissenters) == 1:
                return (
                    dissenters[0].name,
                    "ask what remaining concern would need to be resolved to move",
                    [candidate, state.runtimes[dissenters[0].id].current_preference or dissenters[0].preferred_option],
                )
        return (None, "ask for the strongest remaining concern before choosing", [candidate] if candidate else [])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _derive_pacing(self, state: DialogueState) -> None:
        n = len(state.personas)
        prefs = [p.preferred_option for p in state.personas]
        distinct = len(set(prefs))
        avg_compromise = sum(1.0 - p.sim_params.compromise_threshold for p in state.personas) / max(1, n)
        min_turns = math.ceil(float(cfg.conversation.min_discussion_turns_per_participant) * n)
        target = math.ceil(float(cfg.conversation.target_discussion_turns_per_participant) * n)
        hard = math.ceil(float(cfg.conversation.max_discussion_turns_per_participant) * n)
        if distinct > 1:
            target += int(cfg.conversation.contention_extra_turns)
            hard += int(cfg.conversation.contention_extra_turns)
        if avg_compromise < 0.45:
            target += int(cfg.conversation.low_compromise_extra_turns)
            hard += int(cfg.conversation.low_compromise_extra_turns)
        state.min_discussion_turns = max(n, min_turns)
        vote_buffer = max(1, math.ceil(n / 2))
        state.force_narrow_turns = max(state.min_discussion_turns + vote_buffer, target)
        state.hard_max_turns = max(state.force_narrow_turns + vote_buffer, hard)
        state.phase_history.append(
            f"pacing: min={state.min_discussion_turns}, force={state.force_narrow_turns}, hard={state.hard_max_turns}, distinct_initial_prefs={distinct}, avg_compromise={avg_compromise:.2f}"
        )

    def _mark_phase(self, state: DialogueState, phase: Phase, reason: str) -> None:
        state.phase = phase
        state.phase_history.append(f"turn {state.turn_index}: {phase.value} — {reason}")

    @staticmethod
    def _opening_order(personas: list[Persona]) -> list[Persona]:
        return sorted(personas, key=lambda p: (p.sim_params.initiative, random.random()), reverse=True)

    def _vote_order(self, state: DialogueState, candidate: str) -> list[Persona]:
        return sorted(state.personas, key=lambda p: (state.runtimes[p.id].current_preference != candidate, state.runtimes[p.id].turn_count))

    def _recent_lines(self, state: DialogueState) -> list[str]:
        limit = int(cfg.utterances.recent_turns_in_prompt)
        return [f"{turn.speaker_name}: {turn.text}" for turn in state.turns[-limit:]]

    def _recent_question_count(self, state: DialogueState) -> int:
        recent = [t for t in state.turns[-3:] if t.speaker_id != "moderator"]
        return sum(1 for t in recent if "?" in t.text)

    def _next_answerable_question(self, state: DialogueState) -> OpenQuestion | None:
        live = []
        for question in state.open_questions:
            if any(t.speaker_id == question.target_id and t.index > question.turn_id for t in state.turns):
                continue
            if question.target_id in state.runtimes:
                live.append(question)
        state.open_questions = live[-4:]
        return state.open_questions[0] if state.open_questions else None

    def _register_question(self, state: DialogueState, record: TurnRecord) -> None:
        if record.intent and record.intent.act not in {ActType.ASK, ActType.INVITE}:
            return
        target = record.act.question_target_id
        if not target or target == record.speaker_id or target not in state.runtimes:
            return
        state.open_questions.append(OpenQuestion(turn_id=record.index, asked_by=record.speaker_id, target_id=target, text=record.text, option_focus=record.act.option_refs[:2]))
        state.open_questions = state.open_questions[-4:]

    @staticmethod
    def _close_answered_questions(state: DialogueState, speaker_id: str) -> None:
        state.open_questions = [q for q in state.open_questions if q.target_id != speaker_id]

    @staticmethod
    def _focus_from_recent(state: DialogueState) -> list[str]:
        for turn in reversed(state.turns):
            if turn.act.option_refs:
                return turn.act.option_refs[:2]
        return []

    def _rival_option(self, state: DialogueState, speaker: Persona, exclude: set[str]) -> str | None:
        candidates = [o for o in state.scenario.option_ids if o not in exclude and self._can_shift_to(speaker, o)]
        if not candidates:
            return None
        leading = self._latent_leading_option(state)
        if leading in candidates:
            return leading
        return random.choice(candidates)

    @staticmethod
    def _quietest_other(state: DialogueState, speaker_id: str) -> Persona | None:
        others = [p for p in state.personas if p.id != speaker_id]
        return min(others, key=lambda p: state.runtimes[p.id].turn_count) if others else None

    @staticmethod
    def _last_participant_id(state: DialogueState) -> str | None:
        for turn in reversed(state.turns):
            if turn.speaker_id != "moderator":
                return turn.speaker_id
        return None

    @staticmethod
    def _snapshot_progress(state: DialogueState) -> tuple:
        leans = tuple(sorted((pid, rt.current_preference) for pid, rt in state.runtimes.items()))
        votes = tuple(sorted((pid, rt.explicit_vote) for pid, rt in state.runtimes.items() if rt.explicit_vote))
        questions = tuple((q.turn_id, q.target_id) for q in state.open_questions)
        return leans, votes, questions

    @staticmethod
    def _latent_leading_option(state: DialogueState) -> str | None:
        counts = Counter(rt.current_preference for rt in state.runtimes.values() if rt.current_preference)
        return counts.most_common(1)[0][0] if counts else None

    @staticmethod
    def _latent_leading_count(state: DialogueState) -> int:
        counts = Counter(rt.current_preference for rt in state.runtimes.values() if rt.current_preference)
        return counts.most_common(1)[0][1] if counts else 0

    def _latent_concentration(self, state: DialogueState) -> float:
        return self._latent_leading_count(state) / max(1, len(state.personas))

    @staticmethod
    def _max_words_for(intent: MoveIntent, persona: Persona) -> int:
        budgets = cfg.utterances.word_budgets
        if intent.act == ActType.OPENING:
            base = int(budgets.opening)
        elif intent.act == ActType.ASK:
            base = int(budgets.ask)
        elif intent.act == ActType.ANSWER:
            base = int(budgets.answer)
        elif intent.act in _DECISION_ACTS:
            base = int(budgets.vote)
        else:
            base = int(budgets.discussion)
        return max(7, base + round((persona.sim_params.verbosity - 0.5) * 8))

    @staticmethod
    def _print_header(state: DialogueState) -> None:
        print("\n" + "=" * 72)
        print(f"Topic: {state.scenario.topic}")
        print("Participants: " + ", ".join(p.name for p in state.personas))
        print("=" * 72)

    @staticmethod
    def _emit(record: TurnRecord) -> None:
        print(f"{record.speaker_name}: {record.text}")


def initialise_state(scenario, personas: list[Persona]) -> DialogueState:
    state = DialogueState(scenario=scenario, personas=personas)
    state.runtimes = {p.id: ParticipantRuntime(persona_id=p.id, current_preference=p.preferred_option) for p in personas}
    state.coverage = {option.id: OptionCoverage() for option in scenario.options}
    for persona in personas:
        if persona.rejection:
            state.runtimes[persona.id].hard_rejections[persona.rejection] = persona.rejection_reason
    return state


def participant_turn_count(state: DialogueState) -> int:
    return sum(1 for turn in state.turns if turn.speaker_id != "moderator")


class ConsensusManager:
    @staticmethod
    def finalize(state: DialogueState) -> RunOutcome:
        votes = {
            persona.id: state.runtimes[persona.id].explicit_vote
            for persona in state.personas
            if state.runtimes[persona.id].explicit_vote in state.scenario.option_ids
        }
        counts = Counter(votes.values())
        turns = participant_turn_count(state)
        metadata = {
            "visible_votes": votes,
            "latent_preferences": {pid: rt.current_preference for pid, rt in state.runtimes.items()},
            "phase_history": list(state.phase_history),
            "candidate_option": state.candidate_option,
            "min_discussion_turns": state.min_discussion_turns,
            "force_narrow_turns": state.force_narrow_turns,
            "hard_max_turns": state.hard_max_turns,
        }
        if not counts:
            return RunOutcome("unresolved", None, "No visible votes or acceptances were produced.", turns, metadata)
        winner, support = counts.most_common(1)[0]
        if support == len(state.personas):
            return RunOutcome("successful", winner, "All participants visibly committed to the same option.", turns, metadata)
        threshold = math.ceil(float(cfg.consensus.majority_fraction) * len(state.personas))
        if support >= threshold and list(counts.values()).count(support) == 1:
            return RunOutcome("majority", winner, f"{support}/{len(state.personas)} participants visibly committed to the winning option.", turns, metadata)
        return RunOutcome("unresolved", None, "Visible commitments did not produce a unique majority.", turns, metadata)


def clean_generated(text: str, speaker_name: str, max_words: int) -> str:
    text = strip_speaker_prefix(text, speaker_name)
    text = normalise_ws(text.replace("\n", " "))
    text = text.strip('"“”')
    text = re.sub(r"\s*\[\s*(?:act|opt|stance)\s*=.*$", "", text, flags=re.I).strip()
    text = _remove_generic_filler_tail(text)
    words = text.split()
    if len(words) > max_words:
        chopped = " ".join(words[:max_words]).rstrip(" ,;:")
        sentence_end = max(chopped.rfind("."), chopped.rfind("!"), chopped.rfind("?"))
        if sentence_end >= max(10, len(chopped) // 2):
            text = chopped[: sentence_end + 1].strip()
        else:
            text = _remove_dangling_fragment(_remove_generic_filler_tail(chopped))
            if text and text[-1] not in ".!?":
                text += "."
    return text


def _remove_generic_filler_tail(text: str) -> str:
    patterns = [
        r"\s*(?:what do you think|thoughts|any thoughts)\??$",
        r"\s*(?:what about you|does that help|does that work)\??$",
        r"\s*(?:right|yeah)\??$",
    ]
    out = text
    for pattern in patterns:
        out = re.sub(pattern, "", out, flags=re.I).rstrip(" ,;:")
    return out.strip()


def _remove_dangling_fragment(text: str) -> str:
    patterns = [
        r"\s+(?:even if|even though|although|though|because|since|but|while|whereas|if|when|with|without)$",
        r"\s+(?:though|although|but|because|since|if)\s+(?:it|that|this|we|they|there|the)\s+(?:might|could|would|is|are|was|were|has|have)?\s*$",
        r"\s+(?:what|what do|what do you|can|can you|does|does that),?\.?$",
    ]
    out = text
    for pattern in patterns:
        out = re.sub(pattern, "", out, flags=re.I).rstrip(" ,;:")
    return out
