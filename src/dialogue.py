"""Dialogue runner: top-level orchestration and the generation pipeline.

DialogueRunner composes the concern-specific mixins:
FlowMixin (controller/flow.py — phases, narrowing, voting, repair machine),
PolicyMixin (controller/policy.py — route/speaker/act selection),
ObserverMixin (observer.py — post-turn state updates), and
ValidationMixin (validation.py — accept/repair/reject/fallback).

This file itself owns only run orchestration, the generate→parse→validate→
repair→append pipeline, turn/trace appends, and logging/output.
"""

from __future__ import annotations

import random
from datetime import datetime

import prompts
from builders import SetupBuilder, manual_environment
from config_loader import cfg
from consensus import ConsensusManager
from logger import DialogueLogger, token_summary_for
from llm_client import get_llm_client
from models import (
    ActType,
    DialogueAct,
    DialogueRunResult,
    DialogueState,
    MoveIntent,
    OptionCoverage,
    STANCE_ACCEPTABLE,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
    ParticipantRuntime,
    Persona,
    Phase,
    ThreadStatus,
    ThreadType,
    TurnRecord,
)
from parsing import OptionResolver
from style import strip_leading_name
from utils import clean_generated, normalise_lines
from observer import ObserverMixin
from controller.flow import FlowMixin
from controller.policy import PolicyMixin
from validation import ValidationMixin


class DialogueRunner(FlowMixin, PolicyMixin, ObserverMixin, ValidationMixin):
    def __init__(self, topic: str) -> None:
        # A manual environment carries its own topic; the CLI topic is unused then.
        manual_env = manual_environment()
        self.topic = str((manual_env or {}).get("topic") or topic).strip()
        if not self.topic:
            raise ValueError("topic must not be empty")
        seed = cfg.simulation.get("random_seed", None)
        if seed is not None:
            random.seed(int(seed))
        self._llm = get_llm_client()
        self._resolver: OptionResolver | None = None
        self._intervention_count = 0
        self._last_intervention_turn = -999

    def _mod(self, part: str) -> bool:
        """Whether the moderator voice performs a given structural job (issue 7).

        The controller policy is independent of these flags; they only gate the
        moderator's visible turns. An absent `moderator` config section keeps the
        fully-moderated default (every part on)."""
        mod = getattr(cfg, "moderator", None)
        if mod is None:
            return True
        if not bool(mod.get("enabled", True)):
            return False
        return bool(mod.get(part, True))

    def run(self) -> DialogueRunResult:
        n = cfg.participant_count()
        self._llm.reset_session()
        scenario, personas = SetupBuilder(self.topic).build(n)
        setup_tokens_in = self._llm.session_tokens_in
        setup_tokens_out = self._llm.session_tokens_out
        self._llm.reset_session()

        state = initialise_state(scenario, personas)
        state.setup_tokens_in = setup_tokens_in
        state.setup_tokens_out = setup_tokens_out
        self._record_token_usage(state, "setup", setup_tokens_in, setup_tokens_out)
        self._resolver = OptionResolver(scenario.options)
        self._derive_pacing(state)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.logger = DialogueLogger(run_id, self.topic)

        # When the moderator opening is off, the board is still shown to the
        # reader as plain setup scaffolding (header + transcript "## Options"),
        # just not as a moderator turn.
        self._print_header(state, show_board=not self._mod("opening"))
        if self._mod("opening"):
            self._emit(self._append_moderator(state, prompts.moderator_opening(scenario), Phase.OPENING))
            print("=" * 72)

        self._opening_round(state)
        self._discussion_loop(state)
        self._narrowing_phase(state)
        self._decision_loop(state)

        outcome = ConsensusManager.finalize(state)
        state.outcome = outcome
        if outcome.status == "unresolved":
            self._emit_unresolved_acknowledgement(state, outcome)
        if self._mod("closing"):
            closing = self._moderator_say(prompts.moderator_closure_prompt(outcome, scenario, state), state)
            self._emit(self._append_moderator(state, closing, Phase.CLOSING))
        else:
            self._emit_peer_closing(state, outcome)
        self._mark_phase(state, Phase.CLOSING, f"closed as {outcome.status}")

        state.dialogue_tokens_in = self._llm.session_tokens_in
        state.dialogue_tokens_out = self._llm.session_tokens_out
        paths = self.logger.finish(state, outcome)
        transcript = [f"{turn.speaker_name}: {turn.text}" for turn in state.turns]
        return DialogueRunResult(scenario, personas, transcript, outcome, paths, token_summary_for(state))

    # ------------------------------------------------------------------
    # Controller trace
    # ------------------------------------------------------------------

    def _pre_turn_trace(self, state: DialogueState, intent: MoveIntent, persona: Persona) -> dict:
        """Immutable pre-turn snapshot: why this turn was selected (16.1)."""
        answer_thread = self._required_answer_thread(state)
        # The thread that actually routed this turn, carried on the intent.
        routed = state.threads.get(intent.thread_id) if intent.thread_id else None
        return {
            "routed_thread_id": routed.thread_id if routed else None,
            "routed_thread_type": routed.thread_type.value if routed else None,
            "routed_thread_status": routed.status.value if routed else None,
            "hot_thread_count": sum(1 for t in state.threads.values() if t.status is ThreadStatus.HOT),
            "active_repair_reason": state.active_repair.repair_reason if state.active_repair else None,
            "phase": state.phase.value,
            "route_source": intent.route_source,
            "speaker_id": persona.id,
            "speaker_name": persona.name,
            "selected_act": intent.act.value,
            "selected_addressee_id": intent.addressee_id,
            "selected_option_focus": list(intent.option_focus),
            "respond_to_turn": intent.respond_to_turn,
            "coverage_gaps": sorted(
                oid for oid, cov in state.coverage.items() if cov.mentions == 0
            ),
            "candidate_option": state.candidate_option,
            "required_answer_thread_id": answer_thread.thread_id if answer_thread else None,
            "required_respondent_id": answer_thread.required_respondent if answer_thread else None,
            "open_question_count": sum(
                1 for t in state.threads.values()
                if t.thread_type is ThreadType.QUESTION and t.status is ThreadStatus.HOT
            ),
            "unaddressed_concern_count": sum(
                1 for t in state.threads.values()
                if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
                and t.status is ThreadStatus.HOT
            ),
            "no_progress_count": state.no_progress_count,
        }

    @staticmethod
    def _trace_result(
        state: DialogueState,
        pre: dict,
        record: TurnRecord,
        *,
        appended: bool,
        validation_repair_attempts: int,
        trigger_codes: list[str],
        used_fallback: bool,
        intent: MoveIntent,
    ) -> None:
        act = record.act
        coverage_realized = bool(
            intent.route_source == "coverage"
            and intent.option_focus
            and appended
            and not record.state_mutation_blocked
            and intent.option_focus[0] in act.option_refs
        )
        state.controller_trace.append({
            "type": "turn",
            "turn_index": record.index,
            "pre": pre,
            "result": {
                "appended": appended,
                "realized_act": act.act_type.value,
                "act_mismatch": act.act_type != intent.act,
                "validation_issue_codes": list(record.validation_issues),
                "validation_repair_attempts": validation_repair_attempts,
                "validation_repair_trigger_codes": list(trigger_codes),
                "fallback_used": used_fallback,
                "state_mutation_blocked": record.state_mutation_blocked,
                "final_option_refs": list(act.option_refs),
                "final_addressee_id": act.addressee_id,
                "question_target_id": act.question_target_id,
                "formal_vote_realized": act.explicit_vote,
                "accepts_realized": list(act.accepts),
                "coverage_realized": coverage_realized,
                "tokens_in": record.tokens_in,
                "tokens_out": record.tokens_out,
            },
        })

    @staticmethod
    def _post_turn_route_accounting(state: DialogueState, intent: MoveIntent) -> None:
        """Consume bounded-route attempts once the generation pipeline finished.

        Route selection is read-only (contract 4.1): these bounds are charged
        after the turn completed — appended or dropped — so a failed generation
        still consumes its attempt and cannot re-route the same move forever,
        while a mere route selection never mutates state before text exists.
        Realized effects (coverage mentions, addressed concerns, votes) are
        observed separately from the final accepted text.
        """
        source = intent.route_source
        if source == "coverage" and intent.option_focus:
            option_id = intent.option_focus[0]
            if option_id in state.coverage:
                state.coverage[option_id].coverage_attempts += 1
        elif source == "thread_hot" and intent.act == ActType.ASK and intent.thread_id:
            # A blocker-thread probe: one bounded probe per blocker thread,
            # shared with the moderator probe, charged only post-turn.
            thread = state.threads.get(intent.thread_id)
            if thread is not None:
                thread.probe_count += 1
        elif source == "participant_narrowing":
            state.procedural_move_count += 1
            state.no_progress_count = 0

    def _generate_and_append(self, state: DialogueState, intent: MoveIntent) -> TurnRecord:
        self._apply_style_flags(state, intent)
        persona = state.persona_by_id(intent.speaker_id)
        pre_trace = self._pre_turn_trace(state, intent, persona)
        min_words, max_words = self._word_bounds(intent, persona)
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
            min_words=min_words,
        )
        self.logger.write_prompt(prompt, f"{state.turn_index + 1:03d}_{persona.id}_{intent.act.value}")
        text, tokens_in, tokens_out = self._call_participant(prompt, persona.name, max_words)
        self._record_token_usage(state, "utterance", tokens_in, tokens_out)
        if intent.suppress_name_prefix:
            text = strip_leading_name(text, [p.name for p in state.personas])
        act = self._parse_act(state, persona, text, intent)
        report, gti, gto = self._collect_report(text, state, persona, intent, act, focus_options)
        self._record_token_usage(state, "grounding", gti, gto)
        tokens_in += gti
        tokens_out += gto
        repaired = False
        trigger_codes = list(report.issues)
        attempts = int(cfg.simulation.max_repairs_per_turn)
        validation_repair_attempts = 0
        while report.issues and attempts > 0:
            attempts -= 1
            validation_repair_attempts += 1
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
            self._record_token_usage(state, "repair", ti, to)
            tokens_in += ti
            tokens_out += to
            candidate_act = self._parse_act(state, persona, candidate_text, intent)
            candidate_report, gti, gto = self._collect_report(candidate_text, state, persona, intent, candidate_act, focus_options)
            self._record_token_usage(state, "grounding", gti, gto)
            tokens_in += gti
            tokens_out += gto
            if not candidate_report.issues or len(candidate_report.issues) <= len(report.issues):
                text, act, report = candidate_text, candidate_act, candidate_report

        block = report.block_state_mutation
        used_fallback = False
        if block:
            # Blocking issues survived generation + repair: the LLM text must not
            # reach the transcript (issue I1). Replace it with a deterministic
            # fallback for this intent, re-parse, and re-validate before appending.
            text = self._safe_fallback_text(state, persona, intent, report)
            act = self._parse_act(state, persona, text, intent)
            report = self._validate_turn_text(text, state, persona, intent, act)
            used_fallback = True
            state.fallback_turn_count += 1
            block = report.block_state_mutation
            if block:
                # Last-resort protection: an invalid line must not become visible
                # transcript evidence. The preferred fix is upstream (clearer
                # decision prompts + correct parser/validator alignment), but if
                # those still fail, drop the turn instead of printing evidence the
                # state tracker refuses to count.
                state.no_progress_count += 1
                dropped = TurnRecord(
                    index=state.turn_index,
                    speaker_id=persona.id,
                    speaker_name=persona.name,
                    text="",
                    phase=state.phase,
                    act=act,
                    intent=intent,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    validation_issues=report.issues,
                    repaired=repaired,
                    repair_trigger_codes=trigger_codes,
                    state_mutation_blocked=True,
                    used_fallback=used_fallback,
                )
                self._trace_result(
                    state,
                    pre_trace,
                    dropped,
                    appended=False,
                    validation_repair_attempts=validation_repair_attempts,
                    trigger_codes=trigger_codes,
                    used_fallback=used_fallback,
                    intent=intent,
                )
                self._post_turn_route_accounting(state, intent)
                return dropped
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
        record.used_fallback = used_fallback
        self._trace_result(
            state,
            pre_trace,
            record,
            appended=True,
            validation_repair_attempts=validation_repair_attempts,
            trigger_codes=trigger_codes,
            used_fallback=used_fallback,
            intent=intent,
        )
        self._post_turn_route_accounting(state, intent)
        if not block:
            self._apply_semantics(state, record)
        else:
            state.no_progress_count += 1
        return record

    def _call_participant(self, prompt: str, speaker_name: str, max_words: int) -> tuple[str, int, int]:
        raw = self._llm.generate(prompt, profile="dialogue")
        text = clean_generated(raw, speaker_name, max_words)
        return text, self._llm.last_tokens_in, self._llm.last_tokens_out

    @staticmethod
    def _record_token_usage(state: DialogueState, kind: str, tokens_in: int, tokens_out: int) -> None:
        if tokens_in == 0 and tokens_out == 0:
            return
        bucket = state.token_usage_by_call_type.setdefault(kind, {"in": 0, "out": 0, "calls": 0})
        bucket["in"] += int(tokens_in)
        bucket["out"] += int(tokens_out)
        bucket["calls"] += 1

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
        act = DialogueAct(speaker_id="moderator", text=text, act_type=ActType.SUPPORT)
        record = TurnRecord(index=state.turn_index, speaker_id="moderator", speaker_name="Moderator", text=text, phase=phase, act=act)
        state.turns.append(record)
        state.controller_trace.append({
            "type": "moderator_turn",
            "turn_index": record.index,
            "phase": phase.value,
        })
        return record

    def _append_peer_procedure(
        self,
        state: DialogueState,
        persona: Persona,
        text: str,
        act_type: ActType,
        option_focus: list[str],
        *,
        phase: Phase | None = None,
    ) -> TurnRecord:
        """Append deterministic participant-owned procedure text.

        Split summaries are controller facts: vote counts and the candidate being
        tested must not drift in an LLM paraphrase. Keeping this line
        deterministic also removes one utterance+grounding call in no-moderator
        split cases while still making the procedural move visibly owned by a
        participant.
        """
        if phase is None:
            phase = state.phase
        state.turn_index += 1
        text = normalise_lines(text)
        rt = state.runtimes[persona.id]
        rt.turn_count += 1
        rt.last_spoke_turn = state.turn_index
        rt.already_said.append(text)
        act = DialogueAct(
            speaker_id=persona.id,
            text=text,
            act_type=act_type,
            option_refs=[oid for oid in option_focus if oid in state.scenario.option_ids],
        )
        record = TurnRecord(
            index=state.turn_index,
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            phase=phase,
            act=act,
        )
        state.turns.append(record)
        state.controller_trace.append({
            "type": "peer_procedure",
            "turn_index": record.index,
            "phase": phase.value,
            "speaker_id": persona.id,
            "act": act_type.value,
            "option_focus": list(act.option_refs),
        })
        return record

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _moderator_say(self, prompt: str, state: DialogueState) -> str:
        raw = self._llm.generate(prompt, profile="dialogue")
        self._record_token_usage(state, "moderator", self._llm.last_tokens_in, self._llm.last_tokens_out)
        text = clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.moderator))
        if self._resolver and self._resolver.invalid_option_refs(text):
            raw = self._llm.generate(prompt + "\nOnly use the exact option names already listed.", profile="repair")
            self._record_token_usage(state, "moderator_repair", self._llm.last_tokens_in, self._llm.last_tokens_out)
            text = clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.moderator))
        return text or "Let’s make this concrete: what is the strongest remaining concern before we choose?"

    def _recent_lines(self, state: DialogueState) -> list[str]:
        limit = int(cfg.utterances.recent_turns_in_prompt)
        return [f"{turn.speaker_name}: {turn.text}" for turn in state.turns[-limit:]]

    @staticmethod
    def _print_header(state: DialogueState, show_board: bool = False) -> None:
        print("\n" + "=" * 72)
        print(f"Topic: {state.scenario.topic}")
        print("Participants: " + ", ".join(p.name for p in state.personas))
        if show_board:
            # No moderator opening turn: still show the reader the option board
            # as plain setup scaffolding so the console isn't missing the world.
            print(prompts.moderator_opening(state.scenario))
        print("=" * 72)

    @staticmethod
    def _emit(record: TurnRecord) -> None:
        if not record.text.strip():
            return
        print(f"{record.speaker_name}: {record.text}")


def initialise_state(scenario, personas: list[Persona]) -> DialogueState:
    state = DialogueState(scenario=scenario, personas=personas)
    # Per-sim option ranks are the single runtime source of truth for private
    # stance. Derived helpers read from these ranks; no separate preference/rejection
    # containers are maintained.
    state.runtimes = {}
    for p in personas:
        ranks = {option.id: STANCE_NEUTRAL for option in scenario.options}
        reasons_for: dict[str, str] = {}
        reasons_against: dict[str, str] = {}
        for oid, stance in (p.option_stances or {}).items():
            if oid in ranks:
                ranks[oid] = int(stance.rank)
                if stance.reason_for:
                    reasons_for[oid] = stance.reason_for
                if stance.reason_against:
                    reasons_against[oid] = stance.reason_against
        for oid in p.preferred_options:
            if oid in ranks:
                ranks[oid] = max(ranks[oid], STANCE_PREFERRED if oid == p.preferred_option else STANCE_ACCEPTABLE)
        if p.rejection and p.rejection in ranks:
            ranks[p.rejection] = STANCE_REJECTED
            if p.rejection_reason:
                reasons_against[p.rejection] = p.rejection_reason
        rt = ParticipantRuntime(
            persona_id=p.id,
            option_ranks=ranks,
            reasons_for=reasons_for,
            reasons_against=reasons_against,
            commitment_strength=0.45 + 0.40 * p.sim_params.stubbornness,
            commitment_min=0.45 + 0.40 * p.sim_params.stubbornness,
        )
        state.runtimes[p.id] = rt
    state.coverage = {option.id: OptionCoverage() for option in scenario.options}
    return state
