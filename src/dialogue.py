"""Run orchestration, state tracking, readiness control, and consensus."""

from __future__ import annotations

import datetime as dt
import random
import re
from collections import Counter
import prompts
from builders import SetupBuilder
from config_loader import cfg
from llm_client import get_llm_client
from logger import DialogueLogger, token_summary_for
from models import (
    ActType,
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
from parsing import OptionResolver, TurnMove, parse_dialogue_act, parse_trailer
from router import TurnRouter
from scoring import current_lean, leading_option, visible_candidate_status, visible_leading_option, visible_preference_concentration, visible_support_ids
from utils import compact_words, normalise_lines, normalise_ws, strip_speaker_prefix
from validation import MessageValidator, blocks_state_mutation, classify_claim_slots, classify_discourse_frames, fix_collective_voice, fix_stock_phrases, strip_possessive_opener


class Orchestrator:
    def __init__(self, topic: str) -> None:
        self.topic = topic.strip()
        if not self.topic:
            raise ValueError("Topic must not be empty.")
        self.run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._llm = get_llm_client()
        self.router = TurnRouter()
        self.controller = DialogueController()
        self.consensus = ConsensusManager()
        self.logger = DialogueLogger(self.run_id, self.topic)
        self._last_intervention_turn = -999
        self._intervention_count = 0
        self._nudged_holdouts: set[str] = set()

    def run(self) -> DialogueRunResult:
        self._llm.reset_session()
        scenario, personas = SetupBuilder(self.topic).build(int(cfg.simulation.num_participants))
        setup_in = self._llm.session_tokens_in
        setup_out = self._llm.session_tokens_out
        self._llm.reset_session()

        state = initialise_state(scenario, personas)
        state.setup_tokens_in = setup_in
        state.setup_tokens_out = setup_out
        tracker = StateTracker(state)
        validator = MessageValidator(tracker.resolver, tracker.participant_names)

        self._print_header(scenario, personas)
        self._emit(tracker.apply_moderator(state, prompts.moderator_opening(scenario), phase=Phase.OPENING))
        # Options board first, then people say hi, then opinions.
        self._social_round(state, tracker, Phase.OPENING, int(cfg.utterances.word_budgets.greeting),
                           prompts.greeting_line)

        while not self.controller.should_stop(state):
            self.controller.update_phase(state)
            detected = self.consensus.detect(state)
            if detected and state.phase in {Phase.CONFIRMATION, Phase.CLOSURE}:
                state.outcome = detected
                break
            if state.phase == Phase.CLOSURE:
                break
            nudge = self._moderator_intervention(state)
            if nudge is not None:
                self._emit(tracker.apply_moderator(state, nudge, phase=state.phase))
                continue
            intent = self.router.next_intent(state)
            text, move, prompt, tokens_in, tokens_out, issues, repaired, trigger_codes = self._generate_turn(state, intent, validator)
            self.logger.write_prompt(prompt, f"{state.turn_index + 1:03d}_{intent.speaker_id}_{intent.act.value}")
            self._emit(tracker.apply_participant(state, intent, text, move, tokens_in, tokens_out, issues, repaired, trigger_codes))

        outcome = state.outcome or self.consensus.finalize(state)
        state.outcome = outcome
        closing = self._moderator_say(prompts.moderator_closure_prompt(outcome, scenario, state), state)
        self._emit(tracker.apply_moderator(state, closing, phase=Phase.CLOSURE))
        self._social_round(state, tracker, Phase.CLOSURE, int(cfg.utterances.word_budgets.farewell),
                           lambda p, others, b: prompts.farewell_line(p, scenario, outcome, state, others, b),
                           validate_state=state)
        state.dialogue_tokens_in = self._llm.session_tokens_in
        state.dialogue_tokens_out = self._llm.session_tokens_out
        paths = self.logger.finish(state, outcome)
        transcript = [f"{turn.speaker_name}: {turn.text}" for turn in state.turns]
        return DialogueRunResult(scenario, personas, transcript, outcome, paths, token_summary_for(state))

    @staticmethod
    def _print_header(scenario, personas) -> None:
        print("\n" + "=" * 72)
        print(f"Topic: {scenario.topic}")
        print("Participants: " + ", ".join(p.name for p in personas))
        print("=" * 72)

    @staticmethod
    def _emit(record: TurnRecord) -> None:
        print(f"{record.speaker_name}: {record.text}")

    @staticmethod
    def _moderator_line_is_complete(text: str) -> bool:
        """True when the text ends on a sentence boundary rather than trailing off mid-phrase."""
        stripped = text.rstrip()
        return bool(stripped) and stripped[-1] in ".!?"

    def _moderator_say(self, prompt: str, state: DialogueState) -> str:
        """Generate a moderator facilitation line via the LLM and clean it to one line.
        Prior facilitator lines are fed back in so the moderator doesn't recite the same
        stock phrasing twice (e.g. two identical 'anyone object or lock it in?' nudges).
        Two checks fire in order (each can trigger one retry):
          1. Invalid option references — retry with stricter name grounding.
          2. Incomplete sentence — retry with an explicit finish instruction."""
        prior = [t.text for t in state.turns if t.speaker_id == "moderator"][1:]  # skip the fixed option board
        if prior:
            bullets = "; ".join(compact_words(p, 12) for p in prior[-3:])
            prompt += f"\n\nYou already said, earlier: {bullets}. Say this in different words — don't reuse that phrasing."
        resolver = OptionResolver(state.scenario.options)
        for attempt in range(2):
            raw = self._llm.generate(prompt, profile="dialogue")
            text = clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.medium))
            bad_refs = resolver.invalid_option_refs(text)
            incomplete = not self._moderator_line_is_complete(text)
            if not bad_refs and not incomplete:
                return text
            if attempt == 0:
                if bad_refs:
                    prompt += "\n\nOnly use the exact option names listed above — no invented or paraphrased names."
                if incomplete:
                    prompt += "\n\nEnd the sentence — do not stop mid-phrase. End with a period or question mark."
        return text  # accept the imperfect second attempt rather than aborting

    def _social_round(self, state: DialogueState, tracker: "StateTracker", phase: Phase, budget: int, build_prompt, validate_state: DialogueState | None = None) -> None:
        """A quick, optional social beat (greeting at the start, sign-off at the end) from a
        trait-driven subset. `build_prompt(persona, others, budget)` returns the LLM prompt.
        `validate_state` is passed to _social_say for grounding checks (farewell only)."""
        for persona in self._social_speakers(state.personas):
            others = [p.name for p in state.personas if p.id != persona.id]
            line = self._social_say(build_prompt(persona, others, budget), persona, budget, validate_state)
            if line:
                self._emit(tracker.apply_social(state, persona, line, phase, self._llm.last_tokens_in, self._llm.last_tokens_out))

    def _social_say(self, prompt: str, persona: Persona, budget: int, state: DialogueState | None = None) -> str | None:
        # Greetings/goodbyes are cosmetic; a hiccup on one shouldn't sink the whole run, so
        # we skip that persona's line rather than fabricate or abort.
        try:
            raw = self._llm.generate(prompt, profile="dialogue")
        except Exception:
            return None
        text = clean_generated(raw, persona.name, budget)
        # For farewell turns (state present), check that no invalid option names slipped in.
        if text and state is not None:
            resolver = OptionResolver(state.scenario.options)
            if resolver.invalid_option_refs(text):
                try:
                    raw2 = self._llm.generate(
                        prompt + "\n\nOnly use the exact option names listed above — no invented names.",
                        profile="dialogue",
                    )
                    text2 = clean_generated(raw2, persona.name, budget)
                    if text2 and not resolver.invalid_option_refs(text2):
                        text = text2
                except Exception:
                    pass  # fall through with the original text
        return text or None

    @staticmethod
    def _social_speakers(personas: list[Persona]) -> list[Persona]:
        """Return at most one speaker for a social beat — the most extraverted persona,
        with probability = extraversion / trait_max. Returns [] when the draw fails,
        so greetings and farewells are a single optional line rather than a chorus."""
        if not personas:
            return []
        best = max(personas, key=lambda p: p.traits.extraversion)
        trait_max = float(cfg.personas.trait_max)
        if random.random() < best.traits.extraversion / trait_max:
            return [best]
        return []

    def _generate_turn(self, state: DialogueState, intent: MoveIntent, validator: MessageValidator) -> tuple[str, TurnMove, str, int, int, list[str], bool]:
        persona = state.persona_by_id(intent.speaker_id)
        max_words = max_words_for(intent, persona)
        recent_lines = recent_lines_for_prompt(state, intent)
        focus_options = focus_options_for_prompt(state, intent)
        addressee_name = state.name_for(intent.addressee_id) if intent.addressee_id else None
        option_ids = state.scenario.option_ids
        prompt = prompts.sim_utterance(
            persona=persona,
            state=state,
            recent_lines=recent_lines,
            intent=intent,
            focus_options=focus_options,
            addressee_name=addressee_name,
            max_words=max_words,
        )
        message, move = parse_trailer(self._llm.generate(prompt, profile="dialogue"), option_ids)
        text = clean_generated(message, persona.name, max_words)
        text = strip_possessive_opener(text, state.scenario.options)
        total_tokens_in = self._llm.last_tokens_in
        total_tokens_out = self._llm.last_tokens_out

        result = validator.validate(text, state, intent, move)
        issues = result.codes()
        trigger_codes: list[str] = []
        attempts = int(cfg.simulation.max_repairs_per_turn)
        repaired = False
        while self._needs_repair(result) and attempts > 0:
            if not repaired:
                trigger_codes = list(issues)
            attempts -= 1
            repaired = True
            repair_prompt = prompts.repair_utterance(
                original_text=text,
                issue_codes=issues,
                persona=persona,
                state=state,
                recent_lines=recent_lines,
                intent=intent,
                max_words=max_words,
            )
            self.logger.write_prompt(repair_prompt, f"{state.turn_index + 1:03d}_{intent.speaker_id}_repair")
            message, candidate_move = parse_trailer(self._llm.generate(repair_prompt, profile="repair"), option_ids)
            candidate_text = clean_generated(message, persona.name, max_words)
            candidate_text = strip_possessive_opener(candidate_text, state.scenario.options)
            total_tokens_in += self._llm.last_tokens_in
            total_tokens_out += self._llm.last_tokens_out
            candidate_result = validator.validate(candidate_text, state, intent, candidate_move)
            candidate_issues = candidate_result.codes()
            # A style/format repair must not destroy a semantically usable turn. Keep the
            # provider's original line when the rewrite introduces a blocking state defect.
            if _repair_regresses_state(issues, candidate_issues):
                break
            text, move, result, issues = candidate_text, candidate_move, candidate_result, candidate_issues
        # No fabricated fallback: keep the model's real (possibly imperfect) message and
        # record the remaining issues so they are visible in the logs and metrics.
        return text, move, prompt, total_tokens_in, total_tokens_out, issues, repaired, trigger_codes

    @staticmethod
    def _needs_repair(result) -> bool:
        return any(i.severity in {"repair", "fatal"} for i in result.issues) or (
            bool(cfg.validation.repair_on_warning) and any(i.severity == "warn" for i in result.issues)
        )

    def _moderator_intervention(self, state: DialogueState) -> str | None:
        """Return moderator text when the discussion is circling or has a lone holdout,
        else None. Rate-limited by a cooldown and a per-run cap so the mod stays quiet
        most of the time."""
        conv = cfg.conversation
        if self._intervention_count >= int(conv.moderator_max_interventions):
            return None
        if (state.turn_index - self._last_intervention_turn) < int(conv.moderator_cooldown_turns):
            return None
        # A question is owed an answer before anything else (ANALYSIS #5 — a holdout's "what
        # does low-key mean?" was steamrolled by an immediate narrowing). Yield if a direct
        # question is registered, or if the very last participant line was itself a question,
        # so the room gets one turn to respond before the moderator narrows.
        if state.open_questions:
            return None
        last_turn = next((t for t in reversed(state.turns) if t.speaker_id != "moderator"), None)
        if last_turn is not None and "?" in last_turn.text:
            return None

        if state.phase == Phase.DISCUSSION and everyone_spoke_once(state):
            if visible_preference_concentration(state) >= 1.0:
                candidate = visible_leading_option(state)
                if candidate:
                    self._register_intervention(state)
                    state.no_progress_count = 0
                    state.facilitator_force_narrow = True
                    state.narrowing_called = False  # moderator is handling the narrowing
                    return self._moderator_say(prompts.moderator_agreement_prompt(state, candidate), state)
            # Scale stall window with group size: a 5-person group needs more turns
            # before "no progress" actually means the discussion is circling.
            n = len(state.personas)
            stall_window = int(conv.moderator_stall_window)
            min_turns_before_mod = n * 2
            if (state.no_progress_count >= stall_window
                    and participant_turn_count(state) >= min_turns_before_mod):
                self._register_intervention(state)
                state.no_progress_count = 0
                state.facilitator_force_narrow = True
                state.narrowing_called = False  # moderator is handling the narrowing
                return self._moderator_say(prompts.moderator_stall_prompt(state), state)

        if state.phase == Phase.CONFIRMATION and state.candidate_option:
            pending = [h for h in _candidate_holdouts(state, state.candidate_option) if h not in self._nudged_holdouts]
            # Nudge one or two standouts directly; the confirmation router then has each of
            # them answer in turn. With three or more, leave it to normal routing.
            if 1 <= len(pending) <= 2:
                actual = [
                    participant_id
                    for participant_id in pending
                    if visible_candidate_status(state, participant_id, state.candidate_option)[0] == "holdout"
                ]
                missing = [participant_id for participant_id in pending if participant_id not in actual]
                self._nudged_holdouts.update(pending)
                self._register_intervention(state)
                prompt = prompts.moderator_holdout_prompt(
                    state,
                    state.candidate_option,
                    actual,
                    missing,
                )
                return self._moderator_say(prompt, state)
        return None

    def _register_intervention(self, state: DialogueState) -> None:
        self._last_intervention_turn = state.turn_index
        self._intervention_count += 1


# ---------------------------------------------------------------------------
# Initialisation and state tracking
# ---------------------------------------------------------------------------


def initialise_state(scenario, personas) -> DialogueState:
    min_discussion, force_narrow, hard_max = derive_pacing(personas)
    return DialogueState(
        scenario=scenario,
        personas=personas,
        phase=Phase.OPENING,
        runtimes={
            persona.id: ParticipantRuntime(
                persona_id=persona.id,
                current_preference=persona.preferred_options[0],
                hard_rejections={persona.rejection: persona.rejection_reason} if persona.rejection else {},
            )
            for persona in personas
        },
        coverage={option.id: OptionCoverage() for option in scenario.options},
        min_discussion_turns=min_discussion,
        force_narrow_turns=force_narrow,
        hard_max_turns=hard_max,
    )


def derive_pacing(personas) -> tuple[int, int, int]:
    """Derive this run's pacing from group size and composition so length varies and
    scales with the number of participants instead of hitting one fixed floor. A more
    split, more stubborn, more deliberative group talks longer; jitter keeps even
    similar groups from running identical lengths. Returns
    (min discussion turns before narrowing, force-narrow cap, hard stop)."""
    n = len(personas)
    dd = cfg.conversation.discussion_depth
    distinct = len({p.preferred_option for p in personas})
    contention = distinct / max(1, n)
    avg_compromise = sum(p.traits.compromise_willingness for p in personas) / n
    avg_deliberation = sum(
        (p.traits.openness + p.traits.conscientiousness - 2) / 8
        for p in personas
    ) / n
    lo, hi = (float(x) for x in dd.jitter_per_participant)
    per_participant = (
        float(dd.base_per_participant)
        + contention * float(dd.contention_weight)
        + (1.0 - avg_compromise) * float(dd.stubbornness_weight)
        + avg_deliberation * float(dd.deliberation_weight)
        + random.uniform(lo, hi)
    )
    force_narrow = int(cfg.conversation.force_narrow_turns_per_participant) * n
    hard_max = int(cfg.conversation.hard_max_turns_per_participant) * n
    min_discussion = min(round(per_participant * n), force_narrow)
    return min_discussion, force_narrow, hard_max


class StateTracker:
    def __init__(self, state: DialogueState) -> None:
        self.resolver = OptionResolver(state.scenario.options)
        self.participant_names = {p.id: p.name for p in state.personas}
        self._last_progress_snapshot: tuple | None = None

    def apply_moderator(self, state: DialogueState, text: str, phase: Phase) -> TurnRecord:
        return self._apply_turn(state, "moderator", "Moderator", text, phase, intent=None, move=None, tokens_in=0, tokens_out=0, validation_issues=[], repaired=False)

    def apply_social(self, state: DialogueState, persona: Persona, text: str, phase: Phase, tokens_in: int, tokens_out: int) -> TurnRecord:
        """Append a cosmetic greeting/goodbye: shown and logged, but it carries no dialogue
        act and must not touch stance, coverage, convergence, or turn-taking counts."""
        text = normalise_ws(text)
        state.turn_index += 1
        record = TurnRecord(
            index=state.turn_index,
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            phase=phase,
            act=DialogueAct(speaker_id=persona.id, text=text, act_type=ActType.REACT),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            is_social=True,
        )
        state.turns.append(record)
        return record

    def apply_participant(self, state: DialogueState, intent: MoveIntent, text: str, move: TurnMove, tokens_in: int, tokens_out: int, validation_issues: list[str], repaired: bool, trigger_codes: list[str] | None = None) -> TurnRecord:
        persona = state.persona_by_id(intent.speaker_id)
        return self._apply_turn(state, persona.id, persona.name, text, state.phase, intent, move, tokens_in, tokens_out, validation_issues, repaired, trigger_codes or [])

    def _apply_turn(
        self,
        state: DialogueState,
        speaker_id: str,
        speaker_name: str,
        text: str,
        phase: Phase,
        intent: MoveIntent | None,
        move: TurnMove | None,
        tokens_in: int,
        tokens_out: int,
        validation_issues: list[str],
        repaired: bool,
        repair_trigger_codes: list[str] | None = None,
    ) -> TurnRecord:
        # Moderator messages (e.g. the option board) keep their line breaks; participant
        # turns are single-line.
        text = normalise_lines(text) if speaker_id == "moderator" else normalise_ws(text)
        previous_speaker = previous_participant_speaker(state, exclude=speaker_id)
        act = parse_dialogue_act(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=text,
            resolver=self.resolver,
            participant_names=self.participant_names,
            move=move,
            intent=intent,
            previous_speaker_id=previous_speaker,
        )
        state.turn_index += 1
        state_mutation_blocked = speaker_id != "moderator" and blocks_state_mutation(validation_issues)
        record = TurnRecord(
            index=state.turn_index,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=text,
            phase=phase,
            act=act,
            intent=intent,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            validation_issues=validation_issues,
            repaired=repaired,
            repair_trigger_codes=repair_trigger_codes or [],
            state_mutation_blocked=state_mutation_blocked,
        )
        state.turns.append(record)
        if speaker_id != "moderator":
            apply_semantic_state = not record.state_mutation_blocked
            self._update_runtime(state, record, apply_semantic_state)
            if apply_semantic_state:
                self._update_coverage(state, record)
                self._update_questions(state, record)
            self._update_progress(state, record)
        return record

    def _update_runtime(self, state: DialogueState, record: TurnRecord, apply_semantic_state: bool = True) -> None:
        rt = state.runtimes[record.speaker_id]
        rt.turn_count += 1
        rt.last_spoke_turn = record.index
        rt.already_said.append(record.text)
        if len(rt.already_said) > int(cfg.utterances.recent_turns_after_question):
            rt.already_said = rt.already_said[-int(cfg.utterances.recent_turns_after_question):]
        frames = classify_discourse_frames(record.text)
        rt.discourse_frames.extend(frames)
        if len(rt.discourse_frames) > 8:
            rt.discourse_frames = rt.discourse_frames[-8:]
        if not apply_semantic_state:
            return
        act = record.act
        persona = state.persona_by_id(record.speaker_id)
        if act.act_type == ActType.OPENING and not rt.stated_priority:
            rt.stated_priority = record.text
        # Leanings move on genuine commitment (vote / propose / accept), not merely because
        # the router asked the speaker to air an option. A SUPPORT turn shifts the lean ONLY
        # when the router explicitly handed over a change-of-mind (intent.moves_lean, set by
        # the persuasion path). Otherwise supporting a gap/under-discussed option the persona
        # happens to find acceptable would silently drift their stance — the bug that had a
        # Mountain-Retreat-preferring persona "leaning" Road Trip just because the coverage
        # gap routed them to talk about it.
        moves_lean = bool(record.intent and record.intent.moves_lean)
        if act.explicit_vote:
            rt.explicit_vote = act.explicit_vote
            rt.current_preference = act.explicit_vote
        elif act.proposes_option:
            rt.current_preference = act.proposes_option
        elif (moves_lean and act.act_type == ActType.SUPPORT and act.option_refs):
            rt.current_preference = act.option_refs[0]
        for option_id in act.accepts:
            rt.accepted_options.add(option_id)
            rt.soft_rejections.pop(option_id, None)
            rt.current_preference = option_id  # accepting a compromise moves the lean toward it
        for option_id, reason in act.soft_rejects.items():
            rt.soft_rejections[option_id] = reason
        for option_id, reason in act.hard_rejects.items():
            rt.hard_rejections[option_id] = reason
            rt.accepted_options.discard(option_id)
        # If a routed ACCEPT in CONFIRMATION still failed after repair (state mutation
        # blocked), the persona cannot or will not commit. Record a soft decline so
        # _confirmation_intent() skips them instead of re-routing the same prompt.
        if (record.intent and record.intent.act == ActType.ACCEPT
                and state.phase == Phase.CONFIRMATION
                and record.state_mutation_blocked
                and record.repaired
                and record.intent.option_focus):
            candidate = record.intent.option_focus[0]
            if candidate not in rt.hard_rejections and candidate not in rt.accepted_options:
                rt.soft_rejections.setdefault(candidate, "hedged-confirmation")

    def _update_coverage(self, state: DialogueState, record: TurnRecord) -> None:
        act = record.act
        # Only count visibly mentioned options — options inherited from routing intent but
        # never spoken are not evidence that the option was actually examined.
        visible_ids = self.resolver.ids_in_text(record.text)
        slots = classify_claim_slots(record.text)
        has_slots = bool(slots)
        for option_id in visible_ids:
            if option_id in state.coverage:
                cov = state.coverage[option_id]
                cov.mentions += 1
                # A reason requires the option to be named AND a real claim slot attached —
                # prevents any long sentence mentioning the option from inflating reason counts.
                if has_slots:
                    cov.reasons += 1
                cov.covered_slots.update(slots)
        for option_id in set(act.soft_rejects) | set(act.hard_rejects):
            if option_id in state.coverage:
                state.coverage[option_id].objections += 1
        for option_id in act.accepts:
            if option_id in state.coverage:
                state.coverage[option_id].acceptances += 1

    def _update_questions(self, state: DialogueState, record: TurnRecord) -> None:
        if record.intent and record.intent.respond_to_turn is not None:
            target_id = record.intent.respond_to_turn
            if record.intent.act == ActType.ANSWER and _is_hedge_answer(record.text):
                # First hedge: keep the question open for one more routing cycle.
                # Second hedge (hedge_count already > 0): clear it — the group can't answer.
                for q in state.open_questions:
                    if q.turn_id == target_id and q.hedge_count == 0:
                        q.hedge_count += 1
                        return  # leave question open; don't fall through to clear it
            state.open_questions = [q for q in state.open_questions if q.turn_id != target_id]
        if record.act.question_target_id:
            # If the model was routed to ANSWER but echoed the question anyway (repair
            # failed), don't re-register the echo as a new OpenQuestion — that feeds the
            # next ANSWER cycle and produces the 3-turn echo loop (R26 / R13 gap).
            if (record.intent and record.intent.act == ActType.ANSWER
                    and "QUESTION_ECHO" in record.validation_issues):
                return
            state.open_questions.append(OpenQuestion(
                turn_id=record.index,
                asked_by=record.speaker_id,
                target_id=record.act.question_target_id,
                text=record.text,
                option_focus=record.act.option_refs,
            ))

    def _update_progress(self, state: DialogueState, record: TurnRecord) -> None:
        # Progress means *new* information: a changed vote/accept/reject/preference, or an
        # option earning its first reason. Restating a point about an already-covered
        # option does not count, so genuine circling registers as a stall.
        snapshot = self._progress_snapshot(state)
        if snapshot != self._last_progress_snapshot:
            state.no_progress_count = 0
        else:
            state.no_progress_count += 1
        self._last_progress_snapshot = snapshot

    @staticmethod
    def _progress_snapshot(state: DialogueState) -> tuple:
        stances = tuple(
            (
                pid,
                rt.explicit_vote,
                rt.current_preference,
                frozenset(rt.accepted_options),
                frozenset(rt.soft_rejections),
                frozenset(rt.hard_rejections),
            )
            for pid, rt in state.runtimes.items()
        )
        # Include objections and covered_slots so new argument dimensions register as progress,
        # not just stance changes and first-reason counts. Open-question count also matters:
        # answering a question is progress even if stances don't shift.
        coverage = tuple((c.reasons, c.objections, frozenset(c.covered_slots)) for c in state.coverage.values())
        open_q_count = len(state.open_questions)
        return (stances, coverage, open_q_count)


# ---------------------------------------------------------------------------
# Phase/readiness control
# ---------------------------------------------------------------------------


class DialogueController:
    def update_phase(self, state: DialogueState) -> None:
        if state.outcome is not None:
            state.phase = Phase.CLOSURE
            return
        participant_turns = participant_turn_count(state)
        if state.phase == Phase.OPENING and everyone_spoke_once(state):
            state.phase = Phase.DISCUSSION
        state.readiness_score = self.readiness_score(state)
        if state.phase == Phase.DISCUSSION and self._can_start_narrowing(state):
            # On natural convergence (readiness, not moderator/cap), defer one turn so
            # a participant can call for a vote instead of the phase flipping silently.
            natural = not state.facilitator_force_narrow and participant_turn_count(state) < state.force_narrow_turns
            if natural and not state.narrowing_called:
                state.narrowing_called = True
            else:
                state.phase = Phase.NARROWING
                state.narrowing_called = False
        if state.phase == Phase.NARROWING and everyone_voted(state):
            state.candidate_option = ConsensusManager().leading_candidate(state)
            state.phase = Phase.CONFIRMATION
        if state.phase == Phase.CONFIRMATION:
            # Once a candidate is on the table, bound the confirmation churn: after
            # max_confirmation_turns of accept/reject/compromise without everyone agreeing,
            # close out (finalize picks consensus/fallback/unresolved) instead of looping
            # back into fresh compromise proposals that flip the group around.
            if state.confirmation_start_turns is None:
                state.confirmation_start_turns = participant_turns
            elif participant_turns - state.confirmation_start_turns >= int(cfg.conversation.max_confirmation_turns):
                if not state.open_questions:
                    state.phase = Phase.CLOSURE
        if participant_turns >= state.hard_max_turns:
            state.phase = Phase.CLOSURE

    def should_stop(self, state: DialogueState) -> bool:
        if state.outcome is not None:
            return True
        return participant_turn_count(state) >= state.hard_max_turns

    def readiness_score(self, state: DialogueState) -> float:
        # "Readiness" is now how concentrated the group's leanings are: the fraction
        # of participants sharing the single most-supported option. High concentration
        # means the discussion has converged and is ready to narrow to a vote.
        return concentration_score(state)

    def _can_start_narrowing(self, state: DialogueState) -> bool:
        if state.open_questions:
            return False
        if not everyone_spoke_once(state):
            return False
        if min((rt.turn_count for rt in state.runtimes.values()), default=0) < int(cfg.conversation.min_turns_per_participant_before_narrowing):
            return False
        # Full convergence with stalled discussion: skip the min-turns floor and
        # move to a vote instead of circling with agreement restating.
        if concentration_score(state) >= 1.0 and state.no_progress_count >= 2:
            return True
        # Substance-exhausted narrowing: if every staked option has enough covered
        # claim slots and progress has stalled, allow narrowing before the derived
        # floor — continuing only adds restatements, not new grounded substance.
        if state.no_progress_count >= 2:
            staked = {opt for p in state.personas for opt in p.preferred_options}
            slot_threshold = int(cfg.conversation.slot_exhaustion_threshold)
            min_t = int(cfg.conversation.min_turns_per_participant_before_narrowing)
            each_had_min = min((rt.turn_count for rt in state.runtimes.values()), default=0) >= min_t
            all_staked_covered = staked and all(
                len(state.coverage[opt].covered_slots) >= slot_threshold
                for opt in staked if opt in state.coverage
            )
            if each_had_min and all_staked_covered:
                return True
        if participant_turn_count(state) < state.min_discussion_turns:
            return False
        if sum(1 for c in state.coverage.values() if c.mentions > 0) < int(cfg.conversation.min_options_touched_before_narrowing):
            return False
        # Leading option needs at least one substantive reason before natural narrowing —
        # prevents the group from voting immediately after bare first-position statements.
        lead = leading_option(state)
        if lead and lead in state.coverage and state.coverage[lead].reasons == 0:
            return False
        if state.facilitator_force_narrow:
            return True
        if participant_turn_count(state) >= state.force_narrow_turns:
            return True
        return state.readiness_score >= float(cfg.conversation.concentration_to_narrow)


# ---------------------------------------------------------------------------
# Consensus/final decision
# ---------------------------------------------------------------------------


class ConsensusManager:
    def detect(self, state: DialogueState) -> RunOutcome | None:
        for option_id in state.scenario.option_ids:
            if self._all_accepted_or_voted(state, option_id):
                return RunOutcome("successful", option_id, "all participants visibly accepted or voted for the same option", participant_turn_count(state))
        return None

    def finalize(self, state: DialogueState) -> RunOutcome:
        detected = self.detect(state)
        if detected:
            return detected
        support = {
            option_id: len(visible_support_ids(state, option_id))
            for option_id in state.scenario.option_ids
        }
        candidate = max(support, key=support.get) if support else None
        fraction = self.support_fraction(state, candidate) if candidate else 0.0
        if candidate and fraction >= float(cfg.consensus.majority_fallback_fraction):
            return RunOutcome("majority", candidate, f"majority outcome with visible support fraction {fraction:.2f}", participant_turn_count(state))
        return RunOutcome("unresolved", None, "no option reached the configured visible-support majority", participant_turn_count(state))

    def leading_candidate(self, state: DialogueState) -> str | None:
        return leading_option(state)

    def support_fraction(self, state: DialogueState, option_id: str) -> float:
        """Count only visible VOTE and ACCEPT turns. Hidden preferences and routing-
        assigned leans do not count as social proof in the transcript."""
        return len(visible_support_ids(state, option_id)) / max(1, len(state.personas))

    @staticmethod
    def _all_accepted_or_voted(state: DialogueState, option_id: str) -> bool:
        return not _candidate_holdouts(state, option_id)

# ---------------------------------------------------------------------------
# Prompt helpers / generation cleanup
# ---------------------------------------------------------------------------


def recent_lines_for_prompt(state: DialogueState, intent: MoveIntent) -> list[str]:
    if intent.act == ActType.ANSWER:
        n = int(cfg.utterances.recent_turns_after_question)
    elif intent.act in {ActType.ACCEPT, ActType.REJECT, ActType.VOTE}:
        n = 4
    elif intent.act == ActType.REACT:
        n = 4
    else:
        n = int(cfg.utterances.recent_turns_in_prompt)
    turns = state.turns[-n:]
    if intent.respond_to_turn is not None:
        turns = [turn for turn in turns if turn.index != intent.respond_to_turn]
    return [f"{turn.speaker_name}: {turn.text}" for turn in turns]


def focus_options_for_prompt(state: DialogueState, intent: MoveIntent):
    ids = list(dict.fromkeys(intent.option_focus))
    persona = state.persona_by_id(intent.speaker_id)
    for extra in [state.candidate_option, persona.preferred_option]:
        if extra and extra not in ids:
            ids.append(extra)
    ids = [opt for opt in ids if opt in state.scenario.option_ids]
    ids = ids[: int(cfg.utterances.max_focus_options_in_prompt)]
    return [state.scenario.option(opt) for opt in ids]


def max_words_for(intent: MoveIntent, persona) -> int:
    budgets = cfg.utterances.word_budgets
    if intent.act == ActType.OPENING:
        base = int(budgets.opening)
    elif intent.act == ActType.VOTE:
        base = int(budgets.vote)
    elif intent.act in {ActType.ACCEPT, ActType.REJECT}:
        base = int(budgets.confirm)
    elif intent.act == ActType.ANSWER:
        base = int(budgets.answer)
    elif intent.act == ActType.ASK:
        base = int(budgets.ask)
    else:
        base = int(getattr(budgets, intent.length_hint))
    # Traits shift the word budget around the routed length hint, so a terse and a chatty
    # persona read a bit more differently (without overriding the hint).
    mid = (int(cfg.personas.trait_min) + int(cfg.personas.trait_max)) / 2.0
    shift = persona.traits.response_length - mid
    trait_adjusted = max(8, int(base + shift * int(cfg.utterances.length_trait_word_step)))
    return min(trait_adjusted, int(cfg.utterances.max_chat_words))


def _strip_body_semicolons(text: str) -> str:
    """Replace semicolons in the message body with commas. Protects the
    [act=...; opt=...; stance=...] trailer, which legitimately contains semicolons."""
    trailer_idx = text.rfind("[act=")
    body = text[:trailer_idx] if trailer_idx != -1 else text
    tail = text[trailer_idx:] if trailer_idx != -1 else ""
    return body.replace(";", ",") + tail


def clean_generated(text: str, speaker_name: str, max_words: int) -> str:
    text = strip_speaker_prefix(normalise_ws(text), speaker_name).strip().strip('"')
    if "\n" in text:
        text = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
    text = fix_collective_voice(text)
    text = fix_stock_phrases(text)
    text = _strip_considering_opener(text)
    text = _strip_iget_opener(text)
    text = _surface_cleanup(text)
    text = _strip_body_semicolons(text)
    hard_cap = max_words + int(cfg.utterances.hard_cap_extra_words)
    return compact_words(text, hard_cap)


_CONSIDERING_OPENER_STRIP = re.compile(r"^\s*considering\s+[^,.]{1,60}[,.]\s+", re.I)
# "I get that X, but" / "I hear you, but" / "True, but" — banned in prompt but model
# reaches for them anyway. Strip the acknowledgment prefix and keep the actual point.
_IGET_OPENER_STRIP = re.compile(
    r"^\s*(?:i\s+get\s+that\b.{0,90}?,\s*but\s+|"
    r"i\s+hear\s+you\b[^,]{0,30},?\s*but\s+|"
    r"true,\s*but\s+)",
    re.I,
)


def _strip_considering_opener(text: str) -> str:
    """Deterministically remove 'Considering X, ' or 'Considering X. ' as a turn opener.
    The validation layer escalates this to repair, but the model sometimes regenerates
    it anyway. Strip it here so it can never survive into the final transcript."""
    m = _CONSIDERING_OPENER_STRIP.match(text)
    if not m:
        return text
    rest = text[m.end():]
    return rest[0].upper() + rest[1:] if rest else text


def _strip_iget_opener(text: str) -> str:
    """Strip 'I get that X, but', 'I hear you, but', 'True, but' openers.
    These acknowledgment+pivot templates are banned in the prompt but persist;
    stripping leaves only the actual point the speaker is making."""
    m = _IGET_OPENER_STRIP.match(text)
    if not m:
        return text
    rest = text[m.end():]
    return rest[0].upper() + rest[1:] if rest else text


def _surface_cleanup(text: str) -> str:
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    text = re.sub(r'^["\x27]+|["\x27]+$', "", text).strip()
    # Strip accidental option-letter prefix like "D=Some Option name..."
    text = re.sub(r"^[A-Z]=", "", text).strip()
    return text


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def participant_turn_count(state: DialogueState) -> int:
    return sum(rt.turn_count for rt in state.runtimes.values())


def _repair_regresses_state(current_issues: list[str], candidate_issues: list[str]) -> bool:
    return blocks_state_mutation(candidate_issues) and not blocks_state_mutation(current_issues)


def concentration_score(state: DialogueState) -> float:
    leans = [current_lean(state, p) for p in state.personas]
    leans = [l for l in leans if l]
    if not leans:
        return 0.0
    top = Counter(leans).most_common(1)[0][1]
    return round(top / len(state.personas), 3)


def everyone_spoke_once(state: DialogueState) -> bool:
    return all(rt.turn_count > 0 for rt in state.runtimes.values())


def everyone_voted(state: DialogueState) -> bool:
    return all(bool(rt.explicit_vote) for rt in state.runtimes.values())


def _candidate_holdouts(state: DialogueState, candidate_id: str) -> list[str]:
    return [
        pid
        for pid, rt in state.runtimes.items()
        if rt.explicit_vote != candidate_id and candidate_id not in rt.accepted_options
    ]


def previous_participant_speaker(state: DialogueState, exclude: str | None = None) -> str | None:
    for turn in reversed(state.turns):
        if turn.speaker_id != "moderator" and turn.speaker_id != exclude:
            return turn.speaker_id
    return None


_HEDGE_ANSWER_PATTERN = re.compile(
    r"\b(?:not\s+sure|can'?t\s+(?:say|confirm|tell)|"
    r"(?:we|i)'?d?\s+(?:have\s+to|need\s+to)\s+(?:check|look|verify)|"
    r"no\s+idea|hard\s+to\s+say|unknown|"
    r"i'?m\s+not\s+(?:sure|certain)|"
    r"would\s+need\s+to\s+(?:check|look|verify)|"
    r"have\s+to\s+look|can'?t\s+confirm|"
    r"honestly\s+(?:not\s+sure|don'?t\s+know))\b",
    re.I,
)


def _is_hedge_answer(text: str) -> bool:
    return bool(_HEDGE_ANSWER_PATTERN.search(text))
