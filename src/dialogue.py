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
from dataclasses import dataclass
from datetime import datetime

import prompts
from builders import SetupBuilder, manual_environment
from config_loader import cfg
from consensus import ConsensusManager
from interpreter import InterpretationResult, TurnInterpreter
from logger import DialogueLogger, token_summary_for
from llm_client import get_llm_client
from models import (
    ActType,
    AssessmentAction,
    TurnAssessment,
    VisibleEvidence,
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
from parsing import (
    OptionResolver,
    commitment_has_reason,
    visible_commitment,
    visible_question,
)
from models import (
    AnswerEvidence,
    BlockerEvidence,
    CommitmentEvidence,
    ComparisonEvidence,
    ConcernEvidence,
    EvidenceSpan,
    QuestionEvidence,
    SwitchEvidence,
)
from style import strip_leading_name
from utils import clean_generated, extract_utterance, normalise_lines
from observer import ObserverMixin
from controller.flow import FlowMixin
from controller.policy import PolicyMixin
from validation import ValidationMixin, assessment_severity


@dataclass(slots=True)
class _Candidate:
    """One processed candidate: text plus its interpretation and assessment."""

    text: str
    evidence: VisibleEvidence | None
    assessment: TurnAssessment
    tokens_in: int = 0
    tokens_out: int = 0
    # Validation-path telemetry (items 8/14): whether the validator LLM ran
    # for this candidate, and why it was skipped when it did not.
    validator_llm_used: bool = False
    fast_path_reason: str = ""
    requested_categories: tuple[str, ...] = ()


class DialogueRunner(FlowMixin, PolicyMixin, ObserverMixin, ValidationMixin):
    def __init__(self, topic: str, *, force_auto_scenario: bool = False) -> None:
        # A manual environment carries its own topic; an explicit CLI/piped
        # topic overrides it and requests automatic scenario generation.
        self._force_auto_scenario = bool(force_auto_scenario)
        manual_env = None if self._force_auto_scenario else manual_environment()
        self.topic = str((manual_env or {}).get("topic") or topic).strip()
        if not self.topic:
            raise ValueError("topic must not be empty")
        seed = cfg.simulation.get("random_seed", None)
        if seed is not None:
            random.seed(int(seed))
        # Generative role for setup/utterances/moderator/repair. The validator
        # role is resolved eagerly so a misconfigured validator provider fails
        # at startup, not at the first structured-interpretation call.
        self._llm = get_llm_client("dialogue")
        self._validator_llm = get_llm_client("validator")
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
        scenario, personas = SetupBuilder(
            self.topic, force_auto_scenario=self._force_auto_scenario
        ).build(n)
        setup_tokens_in = self._llm.session_tokens_in
        setup_tokens_out = self._llm.session_tokens_out
        self._llm.reset_session()

        state = initialise_state(scenario, personas)
        state.setup_tokens_in = setup_tokens_in
        state.setup_tokens_out = setup_tokens_out
        self._record_token_usage(state, "setup", setup_tokens_in, setup_tokens_out)
        self._resolver = OptionResolver(scenario.options)
        validation_mode = str((cfg.get("validation", None) or {}).get("mode", "selective"))
        self._interpreter = TurnInterpreter(
            self._validator_llm, self._resolver, scenario,
            {p.id: p.name for p in personas},
            mode=validation_mode,
        )
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
        candidate: "_Candidate | None" = None,
    ) -> None:
        mentioned = record.mentioned_options()
        coverage_realized = bool(
            intent.route_source == "coverage"
            and intent.option_focus
            and appended
            and not record.state_mutation_blocked
            and intent.option_focus[0] in mentioned
        )
        assessment = record.assessment
        evidence = record.evidence
        evidence_bindings: dict[str, list[str]] = {}
        unsupported_spans: list[dict] = []
        if evidence is not None:
            for kind, options in (
                ("support", [s.option_id for s in evidence.supports]),
                ("concern", [c.option_id for c in evidence.concerns]),
                ("commitment", [c.option_id for c in evidence.commitments]),
                ("comparison", sorted({o for c in evidence.comparisons for o in c.option_ids})),
                ("proposal", [p.option_id for p in evidence.proposals if p.option_id]),
                ("switch", [s.target for s in evidence.switches]),
                ("blocker", [b.option_id for b in evidence.blockers]),
            ):
                if options:
                    evidence_bindings[kind] = options
            unsupported_spans = [
                {"span": c.span.text, "kind": c.kind, "option": c.option_id, "reason": c.reason}
                for c in evidence.claims if c.supported is False
            ]
        state.controller_trace.append({
            "type": "turn",
            "turn_index": record.index,
            "pre": pre,
            "result": {
                "appended": appended,
                "realized_act": record.realized_act().value,
                "act_mismatch": record.realized_act() != intent.act,
                "validation_issue_codes": list(record.validation_issues),
                "validation_repair_attempts": validation_repair_attempts,
                "validation_repair_trigger_codes": list(trigger_codes),
                "fallback_used": used_fallback,
                "state_mutation_blocked": record.state_mutation_blocked,
                "final_option_refs": mentioned,
                "final_addressee_id": record.question_target(),
                "question_target_id": record.question_target(),
                "formal_vote_realized": record.visible_vote(),
                "accepts_realized": record.visible_accepts(),
                "coverage_realized": coverage_realized,
                # Assessment + evidence diagnostics (item 15): every repair,
                # fallback, and state decision is explainable from the trace.
                "assessment_action": assessment.action.value if assessment else None,
                "intended_act_realized": assessment.intended_act_realized if assessment else None,
                "intended_focus_realized": assessment.intended_focus_realized if assessment else None,
                "assessment_notes": assessment.notes if assessment else "",
                "evidence_kinds": sorted(evidence.evidence_kinds()) if evidence else [],
                "evidence_bindings": evidence_bindings,
                "grounding_claims": len(evidence.claims) if evidence else 0,
                "unsupported_claims": unsupported_spans,
                "ambiguous_references": [s.text for s in evidence.ambiguous_references] if evidence else [],
                "fallback_family": record.fallback_family,
                "tokens_in": record.tokens_in,
                "tokens_out": record.tokens_out,
                # Validation-path telemetry for the FINAL candidate (item 8):
                # whether the validator LLM ran, why it was skipped, and which
                # payload categories were requested when it did run.
                "validator_llm_used": candidate.validator_llm_used if candidate else None,
                "validation_fast_path_reason": candidate.fast_path_reason if candidate else "",
                "validator_categories": list(candidate.requested_categories) if candidate else [],
                "validator_tokens_in": candidate.tokens_in if candidate else 0,
                "validator_tokens_out": candidate.tokens_out if candidate else 0,
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

    def _process_candidate(
        self,
        state: DialogueState,
        persona: Persona,
        intent: MoveIntent,
        text: str,
        structure_flags: list[str],
        constructed_evidence: VisibleEvidence | None = None,
    ) -> "_Candidate":
        """The one candidate-processing path: parse references, interpret
        through the validator role, and assess. Used identically for initial
        generation, repaired generation, and deterministic fallback — no
        candidate bypasses semantic or grounding checks.

        ``constructed_evidence`` is the item-8 fast path for deterministic
        fallback lines whose complete evidence is known from construction: it
        replaces the validator call, but assessment still runs every
        deterministic check on it.
        """
        if constructed_evidence is not None:
            interp = InterpretationResult(
                evidence=constructed_evidence, fast_path=True,
                fast_path_reason="deterministic fallback: evidence known from construction",
            )
        else:
            interp = self._interpret_candidate(state, persona, intent, text)
        if interp.tokens_in or interp.tokens_out:
            self._record_token_usage(state, "validator", interp.tokens_in, interp.tokens_out)
        assessment = self._assess_candidate(
            text=text, state=state, persona=persona, intent=intent,
            evidence=interp.evidence,
            verification_issues=interp.verification_issues,
            structure_flags=structure_flags,
            operational_failure=interp.operational_failure,
        )
        return _Candidate(
            text=text, evidence=interp.evidence, assessment=assessment,
            tokens_in=interp.tokens_in, tokens_out=interp.tokens_out,
            validator_llm_used=not interp.fast_path,
            fast_path_reason=interp.fast_path_reason,
            requested_categories=interp.requested_categories,
        )

    def _constructed_fallback_evidence(
        self, state: DialogueState, intent: MoveIntent, text: str, family: str
    ) -> VisibleEvidence | None:
        """Evidence for deterministic fallback lines (item 8): their semantics
        are known from construction, re-derived here from the TEXT itself via
        the deterministic parsers — hidden intent alone never labels anything.
        Returns None when the construction cannot be re-verified, in which
        case the normal interpretation path runs."""
        if self._resolver is None:
            return None
        stripped = text.strip()
        span = EvidenceSpan(text=stripped, start=0)
        mentions = self._resolver.mentions(text)
        ids = [m.option_id for m in mentions]
        evidence = VisibleEvidence(utterance=text, mentions=mentions)
        if family == "vote":
            commit = visible_commitment(
                text, self._resolver, sanctioned_switch=bool(intent.allow_vote_change)
            )
            if not commit or commit[0] not in ("vote", "accept"):
                return None
            kind, target = commit
            evidence.commitments.append(CommitmentEvidence(kind, target, span))
            # The switch source is read from the VISIBLE text (the other named
            # option in a "switching from X to Y" line), never from hidden
            # intent (item 4): a minimal vote names no source and records no
            # switch; a public-switch line names its own bridge.
            source = next((oid for oid in ids if oid != target), None)
            if source is not None:
                evidence.switches.append(SwitchEvidence(
                    target=target, span=span, source=source,
                    reason_span=span if commitment_has_reason(text) else None,
                ))
            evidence.primary_act = ActType.VOTE
            return evidence
        if family == "comparison":
            pair = list(dict.fromkeys(ids))[:2]
            if len(pair) < 2:
                return None
            evidence.comparisons.append(ComparisonEvidence(option_ids=pair, span=span))
            evidence.primary_act = ActType.COMPARE
            return evidence
        if family in ("answer_listed", "answer_unknown"):
            evidence.answers.append(AnswerEvidence(
                completeness="full", span=span, addresses_target=True,
            ))
            evidence.primary_act = ActType.ANSWER
            return evidence
        if family == "coverage_question":
            question = visible_question(
                text, speaker_id=intent.speaker_id,
                participant_names={p.id: p.name for p in state.personas},
            )
            if question is not None:
                evidence.questions.append(QuestionEvidence(
                    scope=question[0], span=span, addressee_id=question[1],
                    option_ids=ids[:2],
                ))
            evidence.primary_act = ActType.ASK if question else ActType.COMMENT
            return evidence
        if family == "blocker_restate":
            target = next((o for o in intent.option_focus if o in ids), None)
            if target is None:
                return None
            evidence.concerns.append(ConcernEvidence(target, "hard", span))
            evidence.blockers.append(BlockerEvidence(target, "raised", span))
            evidence.primary_act = ActType.CONCERN
            return evidence
        return None

    def _interpret_candidate(self, state: DialogueState, persona: Persona, intent: MoveIntent, text: str):
        """Interpret one candidate with PUBLIC context only: the targeted prior
        turn, the routed thread's focus, and the speaker's previous public
        vote. Hidden intent reaches the validator as requested behavior,
        never as evidence."""
        target = None
        if intent.respond_to_turn is not None:
            target = next((t for t in state.turns if t.index == intent.respond_to_turn), None)
        thread = state.threads.get(intent.thread_id) if intent.thread_id else None
        candidates: list[str] = []
        if target is not None and self._resolver is not None:
            candidates += self._resolver.ids_in_text(target.text)
        if thread is not None:
            candidates += list(thread.focus_options)
        thread_summary = None
        if thread is not None:
            about = ", ".join(thread.focus_options) or "the options"
            issue = f" ({thread.issue_key})" if thread.issue_key and not thread.issue_key.startswith("sig:") else ""
            thread_summary = f"{thread.thread_type.value} thread about {about}{issue}"
        return self._interpreter.interpret(
            text=text,
            speaker_id=persona.id,
            intent=intent,
            target_turn_text=target.text if target is not None else None,
            target_turn_speaker=target.speaker_name if target is not None else None,
            thread_summary=thread_summary,
            previous_vote=state.runtimes[persona.id].explicit_vote,
            context_candidates=tuple(dict.fromkeys(candidates)),
            rejected_options=tuple(sorted(state.runtimes[persona.id].rejected_options())),
            previous_speaker_id=self._last_participant_id(state),
        )

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
        text, tokens_in, tokens_out, structure_flags = self._call_participant(prompt, persona.name)
        self._record_token_usage(state, "utterance", tokens_in, tokens_out)
        if intent.suppress_name_prefix:
            text = strip_leading_name(text, [p.name for p in state.personas])
        candidate = self._process_candidate(state, persona, intent, text, structure_flags)
        tokens_in += candidate.tokens_in
        tokens_out += candidate.tokens_out

        repaired = False
        trigger_codes = [issue.code for issue in candidate.assessment.issues]
        attempts = int(cfg.simulation.max_repairs_per_turn)
        validation_repair_attempts = 0
        while candidate.assessment.action is AssessmentAction.REPAIR and attempts > 0:
            attempts -= 1
            validation_repair_attempts += 1
            repaired = True
            repair_prompt = prompts.repair_utterance(
                original_text=candidate.text,
                issues=list(candidate.assessment.issues),
                persona=persona,
                state=state,
                recent_lines=recent_lines,
                intent=intent,
                max_words=max_words,
            )
            self.logger.write_prompt(repair_prompt, f"{state.turn_index + 1:03d}_{persona.id}_repair")
            repair_text, ti, to, repair_flags = self._call_participant(repair_prompt, persona.name)
            self._record_token_usage(state, "repair", ti, to)
            tokens_in += ti
            tokens_out += to
            repaired_candidate = self._process_candidate(state, persona, intent, repair_text, repair_flags)
            tokens_in += repaired_candidate.tokens_in
            tokens_out += repaired_candidate.tokens_out
            # Selection by assessment severity, never by raw issue count (item
            # 9): the repair is kept only when it is strictly safer/better.
            if assessment_severity(repaired_candidate.assessment) < assessment_severity(candidate.assessment):
                candidate = repaired_candidate

        block = any(issue.blocking for issue in candidate.assessment.issues)
        used_fallback = False
        fallback_family = ""
        if block:
            # Blocking issues survived generation + repair: the LLM text must not
            # reach the transcript (issue I1). Replace it with a narrow
            # act-specific deterministic fallback when a truthful one exists and
            # run it through the SAME complete candidate path (resolve,
            # interpret, critical-check, ground, assess); otherwise drop.
            fallback_text, fallback_family = self._fallback_candidate(
                state, persona, intent, [i.code for i in candidate.assessment.issues]
            )
            if fallback_text is not None:
                constructed = self._constructed_fallback_evidence(
                    state, intent, fallback_text, fallback_family
                )
                candidate = self._process_candidate(
                    state, persona, intent, fallback_text, [],
                    constructed_evidence=constructed,
                )
                tokens_in += candidate.tokens_in
                tokens_out += candidate.tokens_out
                used_fallback = True
                state.fallback_turn_count += 1
                block = any(issue.blocking for issue in candidate.assessment.issues)
            if block:
                # Last-resort protection: an invalid line must not become visible
                # transcript evidence. The preferred fix is upstream (clearer
                # decision prompts + correct parser/validator alignment), but if
                # those still fail, drop the turn instead of printing evidence the
                # state tracker refuses to count.
                candidate.assessment.action = AssessmentAction.DROP
                state.no_progress_count += 1
                dropped = TurnRecord(
                    index=state.turn_index,
                    speaker_id=persona.id,
                    speaker_name=persona.name,
                    text="",
                    phase=state.phase,
                    intent=intent,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    validation_issues=[i.code for i in candidate.assessment.issues],
                    repaired=repaired,
                    repair_trigger_codes=trigger_codes,
                    state_mutation_blocked=True,
                    used_fallback=used_fallback,
                    fallback_family=fallback_family if used_fallback else "",
                    evidence=candidate.evidence,
                    assessment=candidate.assessment,
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
                    candidate=candidate,
                )
                self._post_turn_route_accounting(state, intent)
                return dropped
        record = self._append_participant(
            state,
            persona,
            candidate.text,
            intent,
            tokens_in,
            tokens_out,
            [i.code for i in candidate.assessment.issues],
            repaired,
            trigger_codes,
            block,
        )
        record.used_fallback = used_fallback
        record.fallback_family = fallback_family if used_fallback else ""
        record.evidence = candidate.evidence
        record.assessment = candidate.assessment
        self._trace_result(
            state,
            pre_trace,
            record,
            appended=True,
            validation_repair_attempts=validation_repair_attempts,
            trigger_codes=trigger_codes,
            used_fallback=used_fallback,
            intent=intent,
            candidate=candidate,
        )
        turn_entry = state.controller_trace[-1]
        self._post_turn_route_accounting(state, intent)
        if not block:
            before = self._observable_state_snapshot(state, persona.id)
            self._apply_semantics(state, record)
            changes = self._state_changes(before, self._observable_state_snapshot(state, persona.id))
            # Attach exactly which speaker-local/public fields the observer
            # changed to this turn's trace entry (item 14).
            turn_entry["result"]["state_changes"] = changes
            turn_entry["result"]["state_changed"] = bool(changes)
        else:
            state.no_progress_count += 1
        return record

    @staticmethod
    def _observable_state_snapshot(state: DialogueState, speaker_id: str) -> dict:
        """Compact snapshot of the fields the observer may change."""
        rt = state.runtimes[speaker_id]
        return {
            f"rank:{speaker_id}:{oid}": rank for oid, rank in rt.option_ranks.items()
        } | {
            f"vote:{speaker_id}": rt.explicit_vote,
            f"vote_stance:{speaker_id}": rt.vote_stance,
        } | {
            f"thread:{tid}": thread.status.value for tid, thread in state.threads.items()
        } | {
            f"coverage:{oid}": (cov.mentions, cov.reasons, cov.objections, cov.acceptances)
            for oid, cov in state.coverage.items()
        } | {
            f"lean:{pid}": r.top_option() for pid, r in state.runtimes.items()
        }

    @staticmethod
    def _state_changes(before: dict, after: dict) -> list[str]:
        return sorted(
            f"{key}: {before.get(key)!r} -> {value!r}"
            for key, value in after.items()
            if before.get(key) != value
        )

    def _call_participant(self, prompt: str, speaker_name: str) -> tuple[str, int, int, list[str]]:
        """Generate one participant line; returns (text, tokens_in, tokens_out,
        raw-output structure flags from the envelope extraction)."""
        raw = self._llm.generate(prompt, profile="dialogue")
        text, structure_flags = extract_utterance(raw, speaker_name)
        return text, self._llm.last_tokens_in, self._llm.last_tokens_out, structure_flags

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
        record = TurnRecord(index=state.turn_index, speaker_id="moderator", speaker_name="Moderator", text=text, phase=phase)
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
        # Deterministic controller text: its semantics are known by
        # construction, so the evidence carries exactly the option mentions
        # and the procedural label — never fabricated support or commitments.
        record = TurnRecord(
            index=state.turn_index,
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            phase=phase,
            evidence=VisibleEvidence(
                utterance=text,
                mentions=self._resolver.mentions(text) if self._resolver is not None else [],
                primary_act=act_type,
            ),
        )
        state.turns.append(record)
        state.controller_trace.append({
            "type": "peer_procedure",
            "turn_index": record.index,
            "phase": phase.value,
            "speaker_id": persona.id,
            "act": act_type.value,
            "option_focus": [oid for oid in option_focus if oid in state.scenario.option_ids],
        })
        return record

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _moderator_say(self, prompt: str, state: DialogueState) -> str:
        raw = self._llm.generate(prompt, profile="dialogue")
        self._record_token_usage(state, "moderator", self._llm.last_tokens_in, self._llm.last_tokens_out)
        text = clean_generated(raw, "Moderator")
        if self._resolver and self._resolver.invalid_option_refs(text):
            raw = self._llm.generate(prompt + "\nOnly use the exact option names already listed.", profile="repair")
            self._record_token_usage(state, "moderator_repair", self._llm.last_tokens_in, self._llm.last_tokens_out)
            text = clean_generated(raw, "Moderator")
        return text or "Let’s make this concrete: what is the strongest remaining concern before we choose?"

    def _recent_lines(self, state: DialogueState) -> list[str]:
        limit = int(cfg.utterances.recent_turns_in_prompt)
        return [f"{turn.speaker_name}: {turn.text}" for turn in prompts.recent_turn_window(state, limit)]

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
        )
        state.runtimes[p.id] = rt
    state.coverage = {option.id: OptionCoverage() for option in scenario.options}
    return state
