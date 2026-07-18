"""Phase and floor control for the simplified user-simulator runtime."""

from __future__ import annotations

import random
import re
from typing import Any

import prompts
from builders import SetupBuilder
from config_loader import cfg
from consensus import (
    derive_narrowing_options,
    majority_threshold,
    outcome_from_votes,
    public_preference_counts,
    public_support_counts,
)
from llm_client import get_llm_client
from logger import DialogueLogger
from models import (
    ActionType,
    DialogueRunResult,
    DialogueState,
    DiscussionThread,
    GenerationAttempt,
    Persona,
    Phase,
    RunOutcome,
    Scenario,
    StanceUpdateKind,
    ThreadKind,
    TurnRecord,
    UserAction,
    VoteRecord,
    VoteStatus,
)
from simulator import FloorManager, UserSimulator, initial_runtime
from validation import validate_action, validate_realization


class DialogueRunner:
    def __init__(
        self,
        topic: str,
        *,
        force_auto_scenario: bool = False,
        scenario: Scenario | None = None,
        personas: list[Persona] | None = None,
        llm: Any = None,
        logger: DialogueLogger | None = None,
        rng: random.Random | None = None,
        seed: int | None = None,
    ) -> None:
        self.topic = topic.strip()
        configured_seed = cfg.simulation.get("random_seed", None)
        self.seed = (
            int(seed if seed is not None else configured_seed)
            if seed is not None or configured_seed is not None
            else random.SystemRandom().randint(0, 2**31 - 1)
        )
        self.rng = rng or random.Random(self.seed)
        self._llm = llm or get_llm_client()

        if scenario is None or personas is None:
            self._llm.reset_session()
            scenario, personas = SetupBuilder(
                self.topic,
                force_auto_scenario=force_auto_scenario,
                llm=self._llm,
                rng=self.rng,
            ).build(cfg.participant_count())
            setup_in = int(getattr(self._llm, "session_tokens_in", 0))
            setup_out = int(getattr(self._llm, "session_tokens_out", 0))
            setup_calls = int(getattr(self._llm, "session_calls", 0))
        else:
            setup_in = setup_out = setup_calls = 0

        self.state = initialise_state(scenario, personas)
        self._captured_setup_in = setup_in
        self._captured_setup_out = setup_out
        self.state.stats.setup_llm_calls = setup_calls
        self.state.stats.input_tokens = setup_in
        self.state.stats.output_tokens = setup_out
        self.logger = logger or DialogueLogger(self.topic)
        self._simulators = {
            persona.id: UserSimulator(persona, random.Random(self.rng.getrandbits(64)))
            for persona in personas
        }
        self._floor = FloorManager(self.rng)
        self._thread_counter = 0
        self._last_failure_errors: list[str] = []
        self._moderator_enabled = bool(cfg.moderator.enabled)
        self._stall_prompt_used = False

    def run(self) -> DialogueRunResult:
        self._print_header()
        if self._moderator_enabled:
            self._append_moderator(prompts.moderator_opening(self.state.scenario, variant=self._variant()))

        self._run_opening()
        self._transition(Phase.DISCUSSION)
        self._run_discussion()
        self._run_narrowing()
        outcome = self._run_voting()

        self._transition(Phase.CLOSED)
        if self._moderator_enabled:
            self._append_moderator(
                prompts.moderator_closure(outcome, self.state.scenario, variant=self._variant())
            )
        log_paths = self.logger.write_run(self.state, outcome, seed=self.seed, llm=self._llm)
        return DialogueRunResult(
            state=self.state,
            outcome=outcome,
            log_paths=log_paths,
            token_summary={
                "setup_tokens_in": self._captured_setup_in,
                "setup_tokens_out": self._captured_setup_out,
                "dialogue_tokens_in": max(0, self.state.stats.input_tokens - self._captured_setup_in),
                "dialogue_tokens_out": max(0, self.state.stats.output_tokens - self._captured_setup_out),
                "total_tokens_in": self.state.stats.input_tokens,
                "total_tokens_out": self.state.stats.output_tokens,
                "llm_calls": self.state.stats.llm_calls + self.state.stats.setup_llm_calls,
            },
        )

    def _run_opening(self) -> None:
        order = [persona.id for persona in self.state.personas]
        self.rng.shuffle(order)
        for participant_id in order:
            action = self._simulators[participant_id].opening_action(self.state)
            if self._realize_and_commit(action, mandatory=True, voluntary=False) is None:
                raise RuntimeError(
                    f"mandatory opening failed for {participant_id}: {self._last_failure_errors}"
                )

    def _run_discussion(self) -> None:
        minimum, target, maximum = cfg.conversation_turn_budgets(len(self.state.personas))
        empty_limit = int(cfg.conversation.stagnation_no_bid_rounds)
        thread_cap = int(cfg.conversation.thread_turn_cap)

        while (
            self.state.stats.voluntary_turns < maximum
            or self.state.response_obligation is not None
            or bool(self.state.active_thread and self.state.active_thread.required_answer_pending)
        ):
            if self.state.active_thread and self.state.active_thread.turn_count >= thread_cap:
                self._close_thread("thread turn cap reached")

            if self.state.response_obligation is not None:
                participant_id = self.state.response_obligation
                action = self._simulators[participant_id].propose(self.state)
                record = self._realize_and_commit(action, mandatory=True, voluntary=False)
                if record is None:
                    self.state.stats.response_failures += 1
                    self.state.protocol_errors.append(
                        f"required answer failed for {participant_id}"
                    )
                    self.state.response_obligation = None
                    if self.state.active_thread:
                        self.state.active_thread.required_answer_pending = False
                continue

            bids = [simulator.propose(self.state) for simulator in self._simulators.values()]
            selection = self._floor.select(self.state, bids)
            if selection.action is None:
                if self.state.active_thread is not None:
                    # A group question should receive one answer even if every
                    # eligible participant declined the voluntary draw.
                    if self.state.active_thread.required_answer_pending:
                        forced = self._force_thread_response()
                        if forced:
                            continue
                    self._close_thread("no related simulator bid")
                    continue

                self.state.no_bid_rounds += 1
                voluntary = self.state.stats.voluntary_turns
                if voluntary >= target:
                    break
                if voluntary >= minimum and self._publicly_converged():
                    break
                if self.state.no_bid_rounds >= empty_limit:
                    if not self._stall_prompt_used and self._moderator_enabled:
                        self._append_moderator(prompts.moderator_stall_prompt(variant=self._variant()))
                        self._stall_prompt_used = True
                    if not self._force_open_floor_bid():
                        break
                    self.state.no_bid_rounds = 0
                continue

            self.state.no_bid_rounds = 0
            self._realize_and_commit(selection.action, mandatory=False, voluntary=True)

            voluntary = self.state.stats.voluntary_turns
            if voluntary >= minimum and self._publicly_converged():
                break
            if voluntary >= target and not self._any_novel_bid():
                break

        if self.state.active_thread is not None:
            self._close_thread("discussion ended")

    def _force_thread_response(self) -> bool:
        thread = self.state.active_thread
        if thread is None:
            return False
        candidates: list[UserAction] = []
        for participant_id, simulator in self._simulators.items():
            if participant_id == thread.opened_by or participant_id in thread.participants:
                continue
            action = simulator.propose(self.state, force_willing=True)
            if action.wants_to_speak:
                candidates.append(action)
        selection = self._floor.select(self.state, candidates)
        if selection.action is None:
            return False
        self.state.stats.liveness_forced_turns += 1
        return self._realize_and_commit(
            selection.action,
            mandatory=False,
            voluntary=False,
            liveness_forced=True,
        ) is not None

    def _force_open_floor_bid(self) -> bool:
        candidates = [
            simulator.propose(self.state, force_willing=True)
            for simulator in self._simulators.values()
        ]
        selection = self._floor.select(self.state, candidates)
        if selection.action is None:
            return False
        self.state.stats.liveness_forced_turns += 1
        return self._realize_and_commit(
            selection.action,
            mandatory=False,
            voluntary=False,
            liveness_forced=True,
        ) is not None

    def _any_novel_bid(self) -> bool:
        return any(simulator.has_novel_voluntary_bid(self.state) for simulator in self._simulators.values())

    def _run_narrowing(self) -> int:
        self._transition(Phase.NARROWING)
        preference_counts = public_preference_counts(self.state)
        participant_count = len(self.state.personas)
        threshold = majority_threshold(participant_count)
        movement_before = self.state.movement_events

        if not preference_counts:
            self.state.narrowing_options = ()
            return 0
        _, leading_preferences = preference_counts.most_common(1)[0]
        if leading_preferences >= threshold:
            self.state.narrowing_options = ()
            return 0

        self.state.narrowing_options = derive_narrowing_options(
            self.state,
            rng=self.rng,
        )
        if len(self.state.narrowing_options) != 1:
            return 0
        leader = self.state.narrowing_options[0]
        support_counts = public_support_counts(self.state)
        highest_support = max(support_counts.values(), default=0)
        support_leaders = [
            option_id
            for option_id, count in support_counts.items()
            if count == highest_support
        ]
        preference_highest = max(
            (preference_counts[option_id] for option_id in support_leaders),
            default=0,
        )
        preference_leaders = [
            option_id
            for option_id in support_leaders
            if preference_counts[option_id] == preference_highest
        ]
        selected_from_tie = len(preference_leaders) > 1
        holdout_ids = [
            persona.id
            for persona in self.state.personas
            if self.state.runtimes[persona.id].public_preference != leader
        ]
        if not holdout_ids:
            return 0

        # A prior visible acceptance remains the participant's latest public
        # position. It therefore carries into the final vote even if the
        # participant is not selected to speak again during the bounded
        # narrowing exchange.
        for participant_id in holdout_ids:
            runtime = self.state.runtimes[participant_id]
            if (
                leader in runtime.public_acceptances
                and not self.state.persona(participant_id).hard_blocker
            ):
                runtime.narrowing_acceptance = leader

        if self._moderator_enabled and not (
            self.state.turns and self.state.turns[-1].moderator
        ):
            self._append_moderator(
                prompts.moderator_compromise_prompt(
                    self.state.scenario,
                    leader,
                    tuple(self.state.persona(pid).name for pid in holdout_ids),
                    preference_count=preference_counts[leader],
                    participant_count=participant_count,
                    selected_from_tie=selected_from_tie,
                    variant=self._variant(),
                )
            )
        self.state.stats.compromise_attempts += 1

        rounds = int(cfg.conversation.compromise_window_max_turns)
        used_speakers: set[str] = set()
        for _ in range(rounds):
            bids: list[UserAction] = []
            for participant_id in holdout_ids:
                if participant_id in used_speakers:
                    continue
                action = self._simulators[participant_id].compromise_action(
                    self.state, (leader,)
                )
                if action.wants_to_speak:
                    bids.append(action)
            selection = self._floor.select(self.state, bids)
            if selection.action is None:
                continue
            record = self._realize_and_commit(
                selection.action, mandatory=False, voluntary=True
            )
            if record is not None:
                used_speakers.add(selection.action.speaker_id)

        return self.state.movement_events - movement_before

    def _run_voting(self) -> RunOutcome:
        self._transition(Phase.VOTING)
        self.state.vote_round = 1
        self.state.votes = {}
        self.state.vote_records[self.state.vote_round] = {}
        if self._moderator_enabled and not (
            self.state.turns and self.state.turns[-1].moderator
        ):
            self._append_moderator(
                prompts.moderator_vote_request(
                    scenario=self.state.scenario,
                    variant=self._variant(),
                )
            )

        for persona in self.state.personas:
            action = self._simulators[persona.id].decide_vote(self.state)
            self._commit_deterministic_vote(action)
            self.state.vote_records[self.state.vote_round][persona.id] = VoteRecord(
                participant_id=persona.id,
                round=self.state.vote_round,
                status=VoteStatus.VALID,
                option_id=action.vote_option,
                attempts=0,
                errors=[],
            )
        outcome = outcome_from_votes(self.state, self.state.votes, allow_unresolved=True)
        assert outcome is not None
        return outcome

    def _realize_and_commit(
        self,
        action: UserAction,
        *,
        mandatory: bool,
        voluntary: bool,
        liveness_forced: bool = False,
    ) -> TurnRecord | None:
        persona = self.state.persona(action.speaker_id)
        structured_errors = validate_action(self.state, persona, action)
        if structured_errors:
            raise RuntimeError(
                f"invalid authoritative action for {action.speaker_id}: {structured_errors}"
            )

        prompt = prompts.realization_prompt(self.state, action)
        raw = self._clean_text(self._call_llm(prompt, profile="dialogue"))
        errors = validate_realization(self.state, persona, action, raw)
        attempt = GenerationAttempt(
            speaker_id=action.speaker_id,
            phase=self.state.phase,
            action=action.copy(),
            raw_text=raw,
            validation_errors=list(errors),
        )
        self.state.generation_attempts.append(attempt)

        text = raw
        repair_count = 0
        should_repair = mandatory and action.act is ActionType.OPENING
        if errors and should_repair:
            repair_count = 1
            self.state.stats.repair_calls += 1
            repair = self._clean_text(
                self._call_llm(
                    prompts.repair_prompt(self.state, action, raw, errors),
                    profile="repair",
                )
            )
            repair_errors = validate_realization(self.state, persona, action, repair)
            attempt.repair_text = repair
            attempt.repair_errors = list(repair_errors)
            if not repair_errors:
                text = repair
                errors = []

        if errors:
            fallback = None
            if action.act is ActionType.OPENING:
                fallback = self._opening_fallback_text(action)
            if fallback:
                fallback_errors = validate_realization(self.state, persona, action, fallback)
                if not fallback_errors:
                    text = fallback
                    errors = []
                    attempt.fallback_text = fallback
                    attempt.final_status = "fallback"
                    self.state.stats.fallback_turns += 1
            if errors:
                attempt.final_status = "dropped"
                self.state.stats.dropped_turns += 1
                self._last_failure_errors = list(errors)
                for error in errors:
                    self.state.validation_failures[error] = self.state.validation_failures.get(error, 0) + 1
                return None

        attempt.final_status = attempt.final_status if attempt.final_status == "fallback" else "accepted"
        self._last_failure_errors = []
        return self._commit_action(
            action,
            text,
            mandatory=mandatory,
            voluntary=voluntary,
            liveness_forced=liveness_forced,
            repair_count=repair_count,
        )

    def _commit_deterministic_vote(self, action: UserAction) -> TurnRecord:
        persona = self.state.persona(action.speaker_id)
        structured_errors = validate_action(self.state, persona, action)
        if structured_errors:
            raise RuntimeError(
                f"invalid authoritative vote for {action.speaker_id}: {structured_errors}"
            )
        text = prompts.deterministic_vote_text(
            self.state.scenario,
            action.vote_option or action.option_focus[0],
            variant=self._variant(),
        )
        errors = validate_realization(self.state, persona, action, text)
        if errors:
            raise RuntimeError(
                f"invalid deterministic vote for {action.speaker_id}: {errors}"
            )
        self.state.generation_attempts.append(
            GenerationAttempt(
                speaker_id=action.speaker_id,
                phase=self.state.phase,
                action=action.copy(),
                raw_text=text,
                validation_errors=[],
                final_status="deterministic",
            )
        )
        return self._commit_action(
            action,
            text,
            mandatory=True,
            voluntary=False,
            liveness_forced=False,
            repair_count=0,
        )

    def _commit_action(
        self,
        action: UserAction,
        text: str,
        *,
        mandatory: bool,
        voluntary: bool,
        liveness_forced: bool,
        repair_count: int,
    ) -> TurnRecord:
        runtime = self.state.runtimes[action.speaker_id]
        if action.stance_update is not None:
            target = action.stance_update.option_id
            already_accepted = target in runtime.public_acceptances
            runtime.public_acceptances.add(target)
            runtime.acceptance_reasons[target] = action.stance_update.movement_reason
            if (
                self.state.phase is Phase.NARROWING
                and action.stance_update.kind is StanceUpdateKind.MAKE_ACCEPTABLE
            ):
                runtime.narrowing_acceptance = target
            switched = action.stance_update.kind is StanceUpdateKind.SWITCH_PREFERRED
            if switched:
                runtime.preferred_option = target
                runtime.public_preference = target
                runtime.visible_switches += 1
            if switched or not already_accepted:
                self.state.movement_events += 1
                self.state.stats.visible_movements += 1
                if self.state.phase is Phase.NARROWING:
                    self.state.stats.compromise_acceptances += 1

        if action.act is ActionType.OPENING:
            runtime.public_preference = runtime.preferred_option
            runtime.openings += 1
        elif action.act is ActionType.VOTE:
            self.state.votes[action.speaker_id] = action.vote_option

        if action.act is not ActionType.VOTE:
            for point_key in action.point_keys:
                runtime.used_point_keys.add(point_key)
                self.state.public_point_counts[point_key] = (
                    self.state.public_point_counts.get(point_key, 0) + 1
                )
                self.state.recent_point_keys.append(point_key)
            self.state.recent_point_keys[:] = self.state.recent_point_keys[-2:]

        thread_event = self._update_thread_before_append(action, text)
        _, word_max = prompts.word_budget(action.act, self.state.persona(action.speaker_id).sim_params.verbosity)
        record = TurnRecord(
            index=len(self.state.turns),
            phase=self.state.phase,
            speaker_id=action.speaker_id,
            speaker_name=self.state.persona(action.speaker_id).name,
            text=text,
            action=action.copy(),
            moderator=False,
            mandatory=mandatory,
            voluntary=voluntary,
            liveness_forced=liveness_forced,
            priority=action.priority,
            repair_count=repair_count,
            thread_event=thread_event,
            stance_update=action.stance_update,
            vote_option=action.vote_option,
            narrowing_options=self.state.narrowing_options,
            intended_word_max=word_max,
        )
        self.state.turns.append(record)
        if voluntary:
            runtime.voluntary_turns += 1
            self.state.stats.voluntary_turns += 1
        print(f"{record.speaker_name}: {record.text}")
        return record

    def _update_thread_before_append(self, action: UserAction, text: str) -> str | None:
        if action.act is ActionType.ASK:
            self._thread_counter += 1
            kind = ThreadKind.QUESTION
            thread = DiscussionThread(
                id=f"t{self._thread_counter}",
                kind=kind,
                opened_by=action.speaker_id,
                option_focus=action.option_focus,
                point_key=action.point_key,
                source_text=text,
                addressed_to=action.addressee_id,
                participants={action.speaker_id},
                required_answer_pending=True,
            )
            self.state.active_thread = thread
            if action.point_key:
                self.state.runtimes[action.speaker_id].opened_thread_keys.add(action.point_key)
            if action.addressee_id:
                self.state.response_obligation = action.addressee_id
            return "opened_question"

        if (
            action.act is ActionType.OBJECT
            and self.state.active_thread is None
            and action.point_key
            and action.point_key not in self.state.closed_thread_keys
            and self.state.public_point_counts.get(action.point_key, 0) == 1
        ):
            self._thread_counter += 1
            self.state.active_thread = DiscussionThread(
                id=f"t{self._thread_counter}",
                kind=ThreadKind.CONCERN,
                opened_by=action.speaker_id,
                option_focus=action.option_focus,
                point_key=action.point_key,
                source_text=text,
                participants={action.speaker_id},
            )
            self.state.runtimes[action.speaker_id].opened_thread_keys.add(action.point_key)
            return "opened_concern"

        thread = self.state.active_thread
        if thread is not None and action.speaker_id != thread.opened_by:
            thread.turn_count += 1
            thread.participants.add(action.speaker_id)
            if action.act is ActionType.ANSWER:
                thread.required_answer_pending = False
                self.state.response_obligation = None
                return "answered_question"
            return "thread_follow_up"
        return None

    def _close_thread(self, reason: str) -> None:
        thread = self.state.active_thread
        if thread is None:
            return
        if thread.point_key:
            self.state.closed_thread_keys.add(thread.point_key)
        self.state.response_obligation = None
        self.state.active_thread = None
        del reason

    def _opening_fallback_text(self, action: UserAction) -> str:
        option = self.state.scenario.option(action.option_focus[0])
        reference = option.short_name or option.name
        if re.search(r"\d", reference):
            reference = f"Option {option.id}"
        return f"Hi, I prefer {reference}; it fits my priorities best."

    def _call_llm(self, prompt: str, *, profile: str) -> str:
        text = self._llm.generate(prompt, profile=profile)
        self.state.stats.llm_calls += 1
        self.state.stats.input_tokens += int(getattr(self._llm, "last_tokens_in", 0))
        self.state.stats.output_tokens += int(getattr(self._llm, "last_tokens_out", 0))
        return text

    @staticmethod
    def _clean_text(text: str) -> str:
        value = str(text or "").strip().strip('"').strip()
        value = re.sub(r"^(assistant|participant|speaker)\s*:\s*", "", value, flags=re.I)
        return " ".join(value.split())

    def _publicly_converged(self) -> bool:
        preferences = [runtime.public_preference for runtime in self.state.runtimes.values()]
        return bool(preferences) and None not in preferences and len(set(preferences)) == 1

    def _transition(self, phase: Phase) -> None:
        self.state.phase = phase
        if not self.state.phase_history or self.state.phase_history[-1] != phase.value:
            self.state.phase_history.append(phase.value)

    def _append_moderator(self, text: str) -> None:
        if not text.strip():
            return
        if self.state.turns and self.state.turns[-1].moderator:
            self.state.turns[-1].text = f"{self.state.turns[-1].text} {text.strip()}"
            print(f"Moderator: {text.strip()}")
            return
        record = TurnRecord(
            index=len(self.state.turns),
            phase=self.state.phase,
            speaker_id="moderator",
            speaker_name="Moderator",
            text=text.strip(),
            moderator=True,
        )
        self.state.turns.append(record)
        self.state.stats.moderator_turns += 1
        print(f"Moderator: {record.text}")

    def _variant(self) -> int:
        return self.rng.randrange(4)

    def _print_header(self) -> None:
        print("=" * 72)
        print(f"Topic: {self.state.scenario.topic}")
        print("Participants: " + ", ".join(persona.name for persona in self.state.personas))
        print("=" * 72)
        if self.state.scenario.context_text:
            print(self.state.scenario.context_text)
        print("Options:")
        for option in self.state.scenario.options:
            print(option.public_line())


def initialise_state(scenario: Scenario, personas: list[Persona]) -> DialogueState:
    runtimes = {
        persona.id: initial_runtime(persona, scenario.option_ids) for persona in personas
    }
    return DialogueState(scenario=scenario, personas=personas, runtimes=runtimes)
