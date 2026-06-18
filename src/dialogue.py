"""Run orchestration, state tracking, readiness control, and consensus."""

from __future__ import annotations

import datetime as dt
import random
from collections import Counter
from typing import Optional

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
from scoring import leading_option
from utils import compact_words, normalise_lines, normalise_ws, strip_speaker_prefix
from validation import MessageValidator


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
        self._greeting_round(state, tracker)  # options board first, then people say hi, then opinions

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
            text, move, prompt, tokens_in, tokens_out, issues, repaired = self._generate_turn(state, intent, validator)
            self.logger.write_prompt(prompt, f"{state.turn_index + 1:03d}_{intent.speaker_id}_{intent.act.value}")
            self._emit(tracker.apply_participant(state, intent, text, move, tokens_in, tokens_out, issues, repaired))

        outcome = state.outcome or self.consensus.finalize(state)
        state.outcome = outcome
        closing = self._moderator_say(prompts.moderator_closure_prompt(outcome, scenario))
        self._emit(tracker.apply_moderator(state, closing, phase=Phase.CLOSURE))
        self._farewell_round(state, tracker, outcome)
        state.dialogue_tokens_in = self._llm.session_tokens_in
        state.dialogue_tokens_out = self._llm.session_tokens_out
        paths = self.logger.finish(state, outcome)
        transcript = [f"{turn.speaker_name}: {turn.text}" for turn in state.turns]
        return DialogueRunResult(scenario, personas, transcript, outcome, paths, token_summary_for(state))

    @staticmethod
    def _print_header(scenario, personas) -> None:
        print("\n" + "=" * 72)
        print(f"Topic: {scenario.topic}")
        print("Participants: " + ", ".join(f"{p.name} ({p.role})" for p in personas))
        print("=" * 72)

    @staticmethod
    def _emit(record: TurnRecord) -> None:
        print(f"{record.speaker_name}: {record.text}")

    def _moderator_say(self, prompt: str) -> str:
        """Generate a moderator facilitation line via the LLM and clean it to one line."""
        raw = self._llm.generate(prompt, profile="dialogue")
        return clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.medium))

    def _greeting_round(self, state: DialogueState, tracker: "StateTracker") -> None:
        """A quick, optional hello from a trait-driven subset, right after the options board."""
        budget = int(cfg.utterances.word_budgets.greeting)
        said: list[str] = []  # shown to later speakers so greetings don't all read alike
        for persona in self._social_speakers(state.personas):
            others = [p.name for p in state.personas if p.id != persona.id]
            line = self._social_say(prompts.greeting_line(persona, state.scenario.topic, others, budget, said), persona, budget)
            if line:
                said.append(line)
                self._emit(tracker.apply_social(state, persona, line, Phase.OPENING, self._llm.last_tokens_in, self._llm.last_tokens_out))

    def _farewell_round(self, state: DialogueState, tracker: "StateTracker", outcome: RunOutcome) -> None:
        """A quick, optional sign-off from a trait-driven subset once it's decided."""
        budget = int(cfg.utterances.word_budgets.farewell)
        said: list[str] = []  # shown to later speakers so sign-offs don't all read alike
        for persona in self._social_speakers(state.personas):
            others = [p.name for p in state.personas if p.id != persona.id]
            line = self._social_say(prompts.farewell_line(persona, state.scenario, outcome, others, budget, said), persona, budget)
            if line:
                said.append(line)
                self._emit(tracker.apply_social(state, persona, line, Phase.CLOSURE, self._llm.last_tokens_in, self._llm.last_tokens_out))

    def _social_say(self, prompt: str, persona: Persona, budget: int) -> Optional[str]:
        # Greetings/goodbyes are cosmetic; a hiccup on one shouldn't sink the whole run, so
        # we skip that persona's line rather than fabricate or abort.
        try:
            raw = self._llm.generate(prompt, profile="dialogue")
        except Exception:
            return None
        text = clean_generated(raw, persona.name, budget)
        return text or None

    @staticmethod
    def _social_speakers(personas: list[Persona]) -> list[Persona]:
        """Pick who joins a social beat: more extraverted people are likelier to chime in,
        but at least one always does, so greetings/goodbyes feel like 'some, sometimes all'."""
        trait_max = float(cfg.personas.trait_max)
        chosen = [p for p in personas if random.random() < p.traits.extraversion / trait_max]
        if not chosen:
            chosen = [max(personas, key=lambda p: p.traits.extraversion)]
        return chosen

    def _generate_turn(self, state: DialogueState, intent: MoveIntent, validator: MessageValidator) -> tuple[str, TurnMove, str, int, int, list[str], bool]:
        persona = state.persona_by_id(intent.speaker_id)
        max_words = max_words_for(intent, persona)
        recent_lines = recent_lines_for_prompt(state, intent)
        board = public_board(state)
        focus_options = focus_options_for_prompt(state, intent)
        addressee_name = state.name_for(intent.addressee_id) if intent.addressee_id else None
        option_ids = state.scenario.option_ids
        prompt = prompts.sim_utterance(
            persona=persona,
            state=state,
            recent_lines=recent_lines,
            public_board=board,
            intent=intent,
            focus_options=focus_options,
            addressee_name=addressee_name,
            max_words=max_words,
            own_recent=state.runtimes[intent.speaker_id].already_said,
        )
        message, move = parse_trailer(self._llm.generate(prompt, profile="dialogue"), option_ids)
        text = clean_generated(message, persona.name, max_words)

        result = validator.validate(text, state, intent, move)
        issues = result.codes()
        attempts = int(cfg.simulation.max_repairs_per_turn)
        repaired = False
        while self._needs_repair(result) and attempts > 0:
            attempts -= 1
            repaired = True
            repair_prompt = prompts.repair_utterance(
                original_text=text,
                issue_codes=issues,
                persona=persona,
                state=state,
                recent_lines=recent_lines,
                public_board=board,
                intent=intent,
                focus_options=focus_options,
                addressee_name=addressee_name,
                max_words=max_words,
            )
            self.logger.write_prompt(repair_prompt, f"{state.turn_index + 1:03d}_{intent.speaker_id}_repair")
            message, move = parse_trailer(self._llm.generate(repair_prompt, profile="repair"), option_ids)
            text = clean_generated(message, persona.name, max_words)
            result = validator.validate(text, state, intent, move)
            issues = result.codes()
        # No fabricated fallback: keep the model's real (possibly imperfect) message and
        # record the remaining issues so they are visible in the logs and metrics.
        return text, move, prompt, self._llm.last_tokens_in, self._llm.last_tokens_out, issues, repaired

    @staticmethod
    def _needs_repair(result) -> bool:
        return any(i.severity in {"repair", "fatal"} for i in result.issues) or (
            bool(cfg.validation.repair_on_warning) and any(i.severity == "warn" for i in result.issues)
        )

    def _moderator_intervention(self, state: DialogueState) -> Optional[str]:
        """Return moderator text when the discussion is circling or has a lone holdout,
        else None. Rate-limited by a cooldown and a per-run cap so the mod stays quiet
        most of the time."""
        conv = cfg.conversation
        if self._intervention_count >= int(conv.moderator_max_interventions):
            return None
        if (state.turn_index - self._last_intervention_turn) < int(conv.moderator_cooldown_turns):
            return None

        if state.phase == Phase.DISCUSSION and everyone_spoke_once(state):
            # Recognise genuine early agreement (everyone leaning the same way) and move to
            # lock it in, rather than always firing the generic "we're circling" line.
            if concentration_score(state) >= 1.0:
                candidate = leading_option(state)
                if candidate:
                    self._register_intervention(state)
                    state.no_progress_count = 0
                    state.facilitator_force_narrow = True
                    return self._moderator_say(prompts.moderator_agreement_prompt(state, candidate))
            if state.no_progress_count >= int(conv.moderator_stall_window):
                self._register_intervention(state)
                state.no_progress_count = 0
                state.facilitator_force_narrow = True  # mod steps in -> move toward a decision
                return self._moderator_say(prompts.moderator_stall_prompt(state))

        if state.phase == Phase.CONFIRMATION and state.candidate_option:
            holdouts = [h for h in _candidate_holdouts(state, state.candidate_option) if h not in self._nudged_holdouts]
            # Nudge one or two standouts directly; the confirmation router then has each of
            # them answer in turn. With three or more, leave it to normal routing.
            if 1 <= len(holdouts) <= 2:
                self._nudged_holdouts.update(holdouts)
                self._register_intervention(state)
                return self._moderator_say(prompts.moderator_holdout_prompt(state, state.candidate_option, holdouts))
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
                current_preference=persona.preferred_option,
                hard_rejections={opt: "private hard rejection" for opt in persona.hard_rejections},
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
    split, more stubborn, more detail-oriented group talks longer; jitter keeps even
    similar groups from running identical lengths. Returns
    (min discussion turns before narrowing, force-narrow cap, hard stop)."""
    n = len(personas)
    dd = cfg.conversation.discussion_depth
    distinct = len({p.preferred_option for p in personas})
    contention = distinct / max(1, n)
    avg_compromise = sum(p.traits.compromise_willingness for p in personas) / n
    avg_detail = sum(p.traits.detail for p in personas) / n
    lo, hi = (float(x) for x in dd.jitter_per_participant)
    per_participant = (
        float(dd.base_per_participant)
        + contention * float(dd.contention_weight)
        + (1.0 - avg_compromise) * float(dd.stubbornness_weight)
        + avg_detail * float(dd.detail_weight)
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
        self._last_progress_snapshot: Optional[tuple] = None

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
        )
        state.turns.append(record)
        return record

    def apply_participant(self, state: DialogueState, intent: MoveIntent, text: str, move: TurnMove, tokens_in: int, tokens_out: int, validation_issues: list[str], repaired: bool) -> TurnRecord:
        persona = state.persona_by_id(intent.speaker_id)
        return self._apply_turn(state, persona.id, persona.name, text, state.phase, intent, move, tokens_in, tokens_out, validation_issues, repaired)

    def _apply_turn(
        self,
        state: DialogueState,
        speaker_id: str,
        speaker_name: str,
        text: str,
        phase: Phase,
        intent: Optional[MoveIntent],
        move: Optional[TurnMove],
        tokens_in: int,
        tokens_out: int,
        validation_issues: list[str],
        repaired: bool,
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
        )
        state.turns.append(record)
        if speaker_id != "moderator":
            self._update_runtime(state, record)
            self._update_coverage(state, record)
            self._update_questions(state, record)
            self._update_progress(state, record)
        return record

    def _update_runtime(self, state: DialogueState, record: TurnRecord) -> None:
        rt = state.runtimes[record.speaker_id]
        rt.turn_count += 1
        rt.last_spoke_turn = record.index
        rt.already_said.append(record.text)
        if len(rt.already_said) > int(cfg.utterances.recent_turns_after_question):
            rt.already_said = rt.already_said[-int(cfg.utterances.recent_turns_after_question):]
        act = record.act
        persona = state.persona_by_id(record.speaker_id)
        if act.act_type == ActType.OPENING and not rt.stated_priority:
            rt.stated_priority = record.text
        # Leanings move on genuine commitment (vote / propose / support of an option
        # the persona can live with), not merely because the router asked them to
        # discuss a gap option (e.g. a COMPARE turn).
        if act.explicit_vote:
            rt.explicit_vote = act.explicit_vote
            rt.current_preference = act.explicit_vote
        elif act.proposes_option:
            rt.current_preference = act.proposes_option
        elif act.act_type == ActType.SUPPORT and act.option_refs and act.option_refs[0] in set(persona.acceptable_options) | {persona.preferred_option}:
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

    def _update_coverage(self, state: DialogueState, record: TurnRecord) -> None:
        act = record.act
        for option_id in act.option_refs:
            if option_id in state.coverage:
                cov = state.coverage[option_id]
                cov.mentions += 1
                if _looks_like_reason(record.text):
                    cov.reasons += 1
        for option_id in set(act.soft_rejects) | set(act.hard_rejects):
            if option_id in state.coverage:
                state.coverage[option_id].objections += 1
        for option_id in act.accepts:
            if option_id in state.coverage:
                state.coverage[option_id].acceptances += 1

    def _update_questions(self, state: DialogueState, record: TurnRecord) -> None:
        if record.intent and record.intent.respond_to_turn is not None:
            state.open_questions = [q for q in state.open_questions if q.turn_id != record.intent.respond_to_turn]
        if record.act.question_target_id:
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
        reasoned = tuple(1 if c.reasons > 0 else 0 for c in state.coverage.values())
        return (stances, reasoned)


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
            state.phase = Phase.NARROWING
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
        if participant_turn_count(state) < state.min_discussion_turns:
            return False
        if sum(1 for c in state.coverage.values() if c.mentions > 0) < int(cfg.conversation.min_options_touched_before_narrowing):
            return False
        if state.facilitator_force_narrow:  # moderator stepped in on a stall
            return True
        if participant_turn_count(state) >= state.force_narrow_turns:
            return True
        return state.readiness_score >= float(cfg.conversation.concentration_to_narrow)


# ---------------------------------------------------------------------------
# Consensus/final decision
# ---------------------------------------------------------------------------


class ConsensusManager:
    def detect(self, state: DialogueState) -> Optional[RunOutcome]:
        candidate = state.candidate_option or self.leading_candidate(state)
        if not candidate:
            return None
        if self._hard_blockers_for(state, candidate):
            return None
        if self._all_accepted_or_voted(state, candidate):
            return RunOutcome("consensus", candidate, "all participants accepted or voted for the same option", participant_turn_count(state))
        return None

    def finalize(self, state: DialogueState) -> RunOutcome:
        detected = self.detect(state)
        if detected:
            return detected
        candidate = self.leading_candidate(state)
        if candidate:
            fraction = self.support_fraction(state, candidate)
            # A hard rejection always blocks a fallback pick; otherwise a strong majority can carry it.
            if fraction >= float(cfg.consensus.majority_fallback_fraction) and not self._hard_blockers_for(state, candidate):
                return RunOutcome("fallback", candidate, f"majority fallback with support fraction {fraction:.2f}", participant_turn_count(state))
        return RunOutcome("unresolved", None, "no option reached explicit acceptance by all participants", participant_turn_count(state))

    def leading_candidate(self, state: DialogueState) -> Optional[str]:
        return leading_option(state)

    def support_fraction(self, state: DialogueState, option_id: str) -> float:
        supporters = 0
        for persona in state.personas:
            rt = state.runtimes[persona.id]
            if rt.explicit_vote == option_id or option_id in rt.accepted_options or option_id == persona.preferred_option:
                supporters += 1
        return supporters / max(1, len(state.personas))

    @staticmethod
    def _all_accepted_or_voted(state: DialogueState, option_id: str) -> bool:
        return not _candidate_holdouts(state, option_id)

    @staticmethod
    def _hard_blockers_for(state: DialogueState, option_id: str) -> list[str]:
        return [pid for pid, rt in state.runtimes.items() if option_id in rt.hard_rejections]


# ---------------------------------------------------------------------------
# Prompt helpers / generation cleanup
# ---------------------------------------------------------------------------


def public_board(state: DialogueState) -> str:
    lines = [f"phase={state.phase.value}; candidate={state.candidate_option or '-'}; agreement={state.readiness_score:.2f}"]
    for persona in state.personas:
        rt = state.runtimes[persona.id]
        bits = [f"{persona.name}: turns={rt.turn_count}"]
        if rt.stated_priority:
            bits.append(f"priority='{compact_words(rt.stated_priority, 12)}'")
        if rt.current_preference:
            bits.append(f"lean={rt.current_preference}")
        if rt.explicit_vote:
            bits.append(f"vote={rt.explicit_vote}")
        if rt.accepted_options:
            bits.append(f"accepts={sorted(rt.accepted_options)}")
        if rt.soft_rejections:
            bits.append(f"concerns={sorted(rt.soft_rejections)}")
        lines.append("; ".join(bits))
    coverage_bits = []
    for option_id, c in state.coverage.items():
        coverage_bits.append(f"{option_id}:m{c.mentions}/r{c.reasons}/o{c.objections}/a{c.acceptances}")
    lines.append("coverage " + ", ".join(coverage_bits))
    if state.open_questions:
        q = state.open_questions[0]
        lines.append(f"open question: {state.name_for(q.asked_by)} -> {state.name_for(q.target_id)}: {compact_words(q.text, 16)}")
    return "\n".join(lines[: int(cfg.utterances.max_public_board_lines)])


def recent_lines_for_prompt(state: DialogueState, intent: MoveIntent) -> list[str]:
    n = int(cfg.utterances.recent_turns_after_question if intent.act == ActType.ANSWER else cfg.utterances.recent_turns_in_prompt)
    turns = state.turns[-n:]
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
    return max(8, int(base + shift * int(cfg.utterances.length_trait_word_step)))


def clean_generated(text: str, speaker_name: str, max_words: int) -> str:
    text = strip_speaker_prefix(normalise_ws(text), speaker_name).strip().strip('"')
    if "\n" in text:
        text = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
    hard_cap = max_words + int(cfg.utterances.hard_cap_extra_words)
    return compact_words(text, hard_cap)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def participant_turn_count(state: DialogueState) -> int:
    return sum(rt.turn_count for rt in state.runtimes.values())


def current_lean(state: DialogueState, persona) -> Optional[str]:
    rt = state.runtimes[persona.id]
    return rt.explicit_vote or rt.current_preference or persona.preferred_option


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


def previous_participant_speaker(state: DialogueState, exclude: Optional[str] = None) -> Optional[str]:
    for turn in reversed(state.turns):
        if turn.speaker_id != "moderator" and turn.speaker_id != exclude:
            return turn.speaker_id
    return None


def _looks_like_reason(text: str) -> bool:
    # A turn "carries a reason" if it is substantive rather than a bare reaction.
    # Length is a deliberately simple, content-agnostic proxy (no keyword list).
    return len(text.split()) >= 8
