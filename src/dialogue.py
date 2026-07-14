"""Single autonomous-simulator dialogue runtime."""

from __future__ import annotations

import random
from collections import Counter
import prompts
from builders import SetupBuilder
from config_loader import cfg
from consensus import derive_narrowing_options, outcome_from_votes
from llm_client import get_llm_client
from logger import DialogueLogger
from models import (
    ActionType,
    ActiveIssue,
    DialogueRunResult,
    DialogueState,
    GenerationAttempt,
    GroupStimulus,
    IssueEffect,
    IssueKind,
    IssueResponseKind,
    IssueStatus,
    OptionCoverage,
    Persona,
    Phase,
    RunOutcome,
    Scenario,
    StanceUpdateKind,
    StimulusKind,
    TurnRecord,
    UserAction,
    VoteRecord,
    VoteStatus,
)
from simulator import FloorManager, UserSimulator, action_cooldown_context, action_signature, initial_runtime
from validation import validate_action, validate_realization


class DialogueRunner:
    def __init__(
        self,
        topic: str,
        *,
        force_auto_scenario: bool = False,
        scenario: Scenario | None = None,
        personas: list[Persona] | None = None,
        llm=None,
        logger: DialogueLogger | None = None,
        rng: random.Random | None = None,
        seed: int | None = None,
    ) -> None:
        self.topic = topic.strip()
        configured_seed = cfg.simulation.get("random_seed", None)
        if seed is not None:
            self.seed = int(seed)
        elif configured_seed is not None:
            self.seed = int(configured_seed)
        else:
            self.seed = random.SystemRandom().randint(0, 2**31 - 1)
        self.rng = rng or random.Random(self.seed)
        self._llm = llm or get_llm_client()
        if scenario is None or personas is None:
            self._llm.reset_session()
            builder = SetupBuilder(
                self.topic,
                force_auto_scenario=force_auto_scenario,
                llm=self._llm,
            )
            scenario, personas = builder.build(cfg.participant_count())
            setup_tokens_in = int(getattr(self._llm, "session_tokens_in", 0))
            setup_tokens_out = int(getattr(self._llm, "session_tokens_out", 0))
            setup_calls = int(getattr(self._llm, "session_calls", 0))
        else:
            setup_tokens_in = setup_tokens_out = setup_calls = 0
        self.state = initialise_state(scenario, personas)
        self._captured_setup_in = setup_tokens_in
        self._captured_setup_out = setup_tokens_out
        self.state.stats.setup_llm_calls = setup_calls
        self.state.stats.input_tokens = setup_tokens_in
        self.state.stats.output_tokens = setup_tokens_out
        self.logger = logger or DialogueLogger(self.topic)
        self._simulators = {
            persona.id: UserSimulator(persona, random.Random(self.rng.getrandbits(64)))
            for persona in personas
        }
        self._floor = FloorManager(self.rng)
        self._issue_counter = 0
        self._stimulus_counter = 0
        self._last_failure_errors: list[str] = []
        self._moderator_enabled = bool((cfg.get("moderator", None) or {}).get("enabled", True))

    def run(self) -> DialogueRunResult:
        state = self.state
        self._print_header(state)
        if self._moderator_enabled:
            self._append_moderator(prompts.moderator_opening(state.scenario))
        self._run_opening()
        self._transition(Phase.DISCUSSION)
        self._run_discussion()
        self._run_narrowing(revote=False)
        outcome = self._run_voting(revote=False)
        if outcome is None:
            state.first_round_votes = dict(state.votes)
            self._run_narrowing(revote=True)
            outcome = self._run_voting(revote=True)
            if outcome is None:
                outcome = outcome_from_votes(state, state.votes, allow_unresolved=True)
        assert outcome is not None
        self._transition(Phase.CLOSED)
        if self._moderator_enabled:
            self._append_moderator(prompts.moderator_closure(outcome, state.scenario))
        log_paths = self.logger.write_run(state, outcome, seed=self.seed)
        return DialogueRunResult(
            state=state,
            outcome=outcome,
            log_paths=log_paths,
            token_summary={
                "setup_tokens_in": self._setup_tokens_in(),
                "setup_tokens_out": self._setup_tokens_out(),
                "dialogue_tokens_in": max(0, state.stats.input_tokens - self._setup_tokens_in()),
                "dialogue_tokens_out": max(0, state.stats.output_tokens - self._setup_tokens_out()),
                "total_tokens_in": state.stats.input_tokens,
                "total_tokens_out": state.stats.output_tokens,
                "llm_calls": state.stats.llm_calls + state.stats.setup_llm_calls,
            },
        )

    def _setup_tokens_in(self) -> int:
        # Setup totals are captured before the runtime begins. Fake clients used
        # in tests may not expose per-call history, so infer them from the state
        # snapshot stored at initialization.
        return int(self._captured_setup_in)

    def _setup_tokens_out(self) -> int:
        return int(self._captured_setup_out)

    def _run_opening(self) -> None:
        state = self.state
        order = [persona.id for persona in state.personas]
        self.rng.shuffle(order)
        for participant_id in order:
            action = self._simulators[participant_id].opening_action(state)
            self._realize_and_commit(action, mandatory=True, voluntary=False)

    def _run_discussion(self) -> None:
        state = self.state
        hard_max = int(cfg.conversation.hard_max_voluntary_turns)
        while (
            self._phase_voluntary_count(Phase.DISCUSSION) < hard_max
            or state.response_obligation is not None
        ):
            if state.response_obligation:
                self._drain_response_obligation_before_transition(
                    failure_reason="mandatory answer failed"
                )
                continue

            if self._phase_voluntary_count(Phase.DISCUSSION) >= hard_max:
                break

            bids = [simulator.propose(state) for simulator in self._simulators.values()]
            issue_continued = bool(
                state.active_issue
                and any(bid.wants_to_speak and bid.issue_id == state.active_issue.id for bid in bids)
            )
            if state.active_issue and not issue_continued:
                self._finish_exhausted_issue("nobody continued the issue")

            accepted = self._select_and_realize(bids, phase=Phase.DISCUSSION)
            if accepted:
                state.no_bid_rounds = 0
            else:
                state.no_bid_rounds += 1
                if self._ready_to_narrow():
                    break
                if state.no_bid_rounds == 1:
                    continue

                if state.no_bid_rounds == 2:
                    if self._moderator_enabled and not state.stall_prompt_used:
                        text = prompts.moderator_stall_prompt()
                        self._append_moderator(text)
                        self._set_group_stimulus(StimulusKind.STALL, (), text)
                        state.stall_prompt_used = True
                    continue

                minimum = int(cfg.conversation.min_voluntary_turns)
                if self._phase_voluntary_count(Phase.DISCUSSION) < minimum:
                    if self._force_liveness(Phase.DISCUSSION):
                        state.no_bid_rounds = 0
                        continue
                break

            if self._coverage_prompt_needed():
                option_id = self._uncovered_options()[0]
                text = prompts.moderator_coverage_prompt(state.scenario, option_id)
                self._append_moderator(text)
                self._set_group_stimulus(StimulusKind.COVERAGE, (option_id,), text)
                state.coverage_prompt_used = True
                continue
            if self._ready_to_narrow():
                break

    def _run_narrowing(self, *, revote: bool) -> None:
        state = self.state
        self._drain_response_obligation_before_transition(
            failure_reason="mandatory answer failed before narrowing"
        )
        state.group_stimulus = None
        self._transition(Phase.NARROWING)
        state.narrowing_options = derive_narrowing_options(state)
        if not state.narrowing_options:
            state.narrowing_options = tuple(state.scenario.option_ids[:2])
        if self._moderator_enabled:
            if revote:
                self._append_moderator("No option reached a majority. We will have one short final narrowing before a re-vote.")
            self._append_moderator(prompts.moderator_narrowing(state.scenario, state.narrowing_options))

        budget = int(
            cfg.conversation.revote_narrowing_voluntary_turns
            if revote else cfg.conversation.narrowing_voluntary_turns
        )
        start_count = self._phase_voluntary_count(Phase.NARROWING)
        empty_rounds = 0
        while self._phase_voluntary_count(Phase.NARROWING) - start_count < budget:
            if state.response_obligation:
                participant_id = state.response_obligation
                action = self._simulators[participant_id].propose(state, mandatory_answer=True)
                if self._realize_and_commit(action, mandatory=True, voluntary=False) is None:
                    self._stale_active_issue("mandatory narrowing answer failed")
                continue
            bids = [simulator.propose(state) for simulator in self._simulators.values()]
            if state.active_issue and not any(
                bid.wants_to_speak and bid.issue_id == state.active_issue.id for bid in bids
            ):
                self._finish_exhausted_issue("nobody continued the narrowing issue")
            if not self._select_and_realize(bids, phase=Phase.NARROWING):
                empty_rounds += 1
                # One empty arbitration round is not enough evidence that every
                # autonomous simulator has finished. Retry once, then progress.
                if empty_rounds < 2:
                    continue
                break
            empty_rounds = 0
        if state.response_obligation:
            participant_id = state.response_obligation
            action = self._simulators[participant_id].propose(state, mandatory_answer=True)
            if self._realize_and_commit(action, mandatory=True, voluntary=False) is None:
                self._stale_active_issue("final mandatory answer failed")
        if state.active_issue:
            self._finish_exhausted_issue("narrowing window ended")
        state.narrowing_options = derive_narrowing_options(state) or state.narrowing_options

    def _drain_response_obligation_before_transition(self, *, failure_reason: str) -> None:
        """Complete one pending direct-answer adjacency pair before a phase change.

        The addressed simulator receives the normal bounded realization path:
        one generation plus the existing single repair attempt. A remaining
        failure explicitly stales the issue so the phase transition cannot
        silently strand the obligation.
        """
        participant_id = self.state.response_obligation
        if participant_id is None:
            return
        action = self._simulators[participant_id].propose(
            self.state,
            mandatory_answer=True,
        )
        if self._realize_and_commit(action, mandatory=True, voluntary=False) is None:
            self._stale_active_issue(failure_reason)

    def _run_voting(self, *, revote: bool) -> RunOutcome | None:
        state = self.state
        self._transition(Phase.VOTING)
        state.vote_round = 2 if revote else 1
        state.votes = {}
        state.vote_records[state.vote_round] = {}
        if self._moderator_enabled:
            self._append_moderator(prompts.moderator_vote_request(revote=revote))
        for persona in state.personas:
            action = self._simulators[persona.id].decide_vote(state, revote=revote)
            before_attempts = len(state.generation_attempts)
            record = self._realize_and_commit(action, mandatory=True, voluntary=False)
            attempt_entries = state.generation_attempts[before_attempts:]
            attempts = sum(1 + int(entry.repair_text is not None) for entry in attempt_entries)
            if record is not None:
                vote_record = VoteRecord(
                    participant_id=persona.id,
                    round=state.vote_round,
                    status=VoteStatus.VALID,
                    option_id=action.vote_option,
                    attempts=max(1, attempts),
                )
            else:
                errors = list(self._last_failure_errors)
                unclear = any("vote" in error.casefold() or "ambiguous" in error.casefold() for error in errors)
                vote_record = VoteRecord(
                    participant_id=persona.id,
                    round=state.vote_round,
                    status=VoteStatus.UNCLEAR if unclear else VoteStatus.GENERATION_FAILED,
                    option_id=None,
                    attempts=max(1, attempts),
                    errors=errors,
                )
                state.votes[persona.id] = None
                state.vote_protocol_degraded = True
                state.vote_protocol_errors.append(
                    f"round {state.vote_round} {persona.id}: {vote_record.status.value}"
                )
            state.vote_records[state.vote_round][persona.id] = vote_record
        return outcome_from_votes(state, state.votes, allow_unresolved=revote)

    def _select_and_realize(self, bids: list[UserAction], *, phase: Phase) -> bool:
        remaining = list(bids)
        while True:
            selection = self._floor.select(self.state, remaining)
            if selection is None:
                return False
            action = selection.action
            record = self._realize_and_commit(action, mandatory=False, voluntary=True)
            if record is not None:
                return True
            remaining = [candidate for candidate in remaining if candidate is not action]

    def _force_liveness(self, phase: Phase) -> bool:
        state = self.state
        simulators = sorted(
            self._simulators.values(),
            key=lambda simulator: simulator.persona.sim_params.engagement,
            reverse=True,
        )
        top_level = simulators[0].persona.sim_params.engagement if simulators else 0
        candidates = [simulator for simulator in simulators if simulator.persona.sim_params.engagement == top_level]
        self.rng.shuffle(candidates)
        for simulator in [*candidates, *[s for s in simulators if s not in candidates]]:
            action = simulator.propose(state, liveness_forced=True)
            if not action.wants_to_speak:
                continue
            record = self._realize_and_commit(
                action,
                mandatory=False,
                voluntary=False,
                liveness_forced=True,
            )
            if record:
                state.stats.liveness_forced_turns += 1
                return True
        return False

    def _realize_and_commit(
        self,
        action: UserAction,
        *,
        mandatory: bool,
        voluntary: bool,
        liveness_forced: bool = False,
    ) -> TurnRecord | None:
        state = self.state
        persona = state.persona(action.speaker_id)
        self._last_failure_errors = []
        action_errors = validate_action(state, persona, action)
        if action_errors:
            self._last_failure_errors = list(action_errors)
            for error in action_errors:
                state.validation_failures[self._validation_category(error)] += 1
            raise ValueError(f"invalid structured action from {persona.name}: {'; '.join(action_errors)}")

        target_question = None
        if action.act == ActionType.ANSWER and state.active_issue:
            target_question = state.active_issue.source_text or state.active_issue.summary
        prompt = prompts.realization_prompt(state, persona, action, target_question=target_question)
        self.logger.write_prompt(prompt, "realization")
        raw = self._call_llm(prompt, profile="dialogue")
        result = validate_realization(raw, state, persona, action, target_question=target_question)
        attempt = GenerationAttempt(
            speaker_id=persona.id,
            phase=state.phase,
            action=action.copy(),
            raw_text=raw,
            validation_errors=list(result.errors),
        )
        state.generation_attempts.append(attempt)
        for error in result.errors:
            state.validation_failures[self._validation_category(error)] += 1

        repair_count = 0
        if not result.ok:
            repair_count = 1
            state.stats.repair_calls += 1
            repair_prompt = prompts.repair_prompt(
                state,
                persona,
                action,
                result.text or raw,
                result.errors,
                target_question=target_question,
            )
            self.logger.write_prompt(repair_prompt, "repair")
            repaired = self._call_llm(repair_prompt, profile="repair")
            repaired_result = validate_realization(
                repaired,
                state,
                persona,
                action,
                target_question=target_question,
            )
            attempt.repair_text = repaired
            attempt.repair_errors = list(repaired_result.errors)
            for error in repaired_result.errors:
                state.validation_failures[self._validation_category(error)] += 1
            result = repaired_result
        if not result.ok:
            attempt.final_status = "dropped"
            self._last_failure_errors = list(result.errors)
            state.stats.dropped_turns += 1
            return None
        attempt.final_status = "accepted"
        return self._commit_action(
            action,
            result.text,
            mandatory=mandatory,
            voluntary=voluntary,
            liveness_forced=liveness_forced,
            repair_count=repair_count,
        )

    @staticmethod
    def _validation_category(error: str) -> str:
        lowered = error.casefold()
        categories = (
            ("repetition", "repetition"),
            ("outside voting", "phase_language"),
            ("dialogue-act label", "phase_language"),
            ("vote", "vote"),
            ("ambiguous", "vote"),
            ("invented", "grounding"),
            ("contradict", "grounding"),
            ("unsupported", "grounding"),
            ("option mention", "option_mention"),
            ("unrelated", "answer_relevance"),
            ("speaker", "format"),
            ("metadata", "format"),
            ("empty", "format"),
        )
        for token, category in categories:
            if token in lowered:
                return category
        return "other"

    def _call_llm(self, prompt: str, *, profile: str) -> str:
        text = self._llm.generate(prompt, profile=profile)
        state = self.state
        state.stats.llm_calls += 1
        state.stats.input_tokens += int(getattr(self._llm, "last_tokens_in", max(1, len(prompt.split()))))
        state.stats.output_tokens += int(getattr(self._llm, "last_tokens_out", max(1, len(text.split()))))
        return text

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
        state = self.state
        persona = state.persona(action.speaker_id)
        runtime = state.runtimes[action.speaker_id]
        issue_event = self._apply_issue_before_turn(action, text)
        self._apply_stance_update(action)
        self._apply_public_action(action)

        intended_word_min, intended_word_max = prompts.word_budget(
            action.act, persona.sim_params.verbosity
        )
        record = TurnRecord(
            index=len(state.turns) + 1,
            phase=state.phase,
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            action=action.copy(),
            mandatory=mandatory,
            voluntary=voluntary,
            liveness_forced=liveness_forced,
            urgency=action.urgency,
            repair_count=repair_count,
            issue_event=issue_event,
            stance_update=action.stance_update,
            vote_option=action.vote_option,
            narrowing_options=tuple(state.narrowing_options),
            prompt_tokens=int(getattr(self._llm, "last_tokens_in", 0)),
            output_tokens=int(getattr(self._llm, "last_tokens_out", 0)),
            intended_word_min=intended_word_min,
            intended_word_max=intended_word_max,
        )
        state.turns.append(record)
        runtime.last_action = action.act
        runtime.last_spoken_turn = record.index
        if action.act == ActionType.OPENING:
            runtime.openings += 1
        elif action.act == ActionType.ANSWER and mandatory:
            runtime.mandatory_answers += 1
        elif action.act == ActionType.VOTE:
            runtime.votes_cast += 1
        elif voluntary:
            runtime.voluntary_turns += 1
            state.stats.voluntary_turns += 1
        if action.reason:
            runtime.stated_reason_keys.add(self._reason_key(action))
        if action.question_key:
            runtime.asked_question_keys.add(action.question_key)
        signature = action_signature(action)
        runtime.action_signature_counts[signature] += 1
        runtime.action_signature_contexts[signature] = action_cooldown_context(state, runtime, action)
        if action.stimulus_id is not None:
            runtime.responded_stimuli.add(action.stimulus_id)
            if state.group_stimulus and state.group_stimulus.id == action.stimulus_id:
                state.group_stimulus = None
        self._record_novelty(record)
        self._apply_issue_after_turn(action)
        print(f"{persona.name}: {text}")
        return record

    def _apply_stance_update(self, action: UserAction) -> None:
        if not action.stance_update:
            return
        state = self.state
        runtime = state.runtimes[action.speaker_id]
        update = action.stance_update
        if update.kind == StanceUpdateKind.MAKE_ACCEPTABLE:
            runtime.ranks[update.option_id] = max(4, runtime.ranks.get(update.option_id, 3))
            runtime.acceptable_options.add(update.option_id)
            runtime.disliked_options.discard(update.option_id)
            runtime.public_acceptances.add(update.option_id)
        elif update.kind == StanceUpdateKind.REMOVE_ACCEPTANCE:
            runtime.acceptable_options.discard(update.option_id)
            runtime.public_acceptances.discard(update.option_id)
            runtime.ranks[update.option_id] = min(3, runtime.ranks.get(update.option_id, 3))
        elif update.kind == StanceUpdateKind.SWITCH_PREFERRED:
            previous = runtime.preferred_option
            runtime.ranks[previous] = max(4, min(4, runtime.ranks.get(previous, 4)))
            runtime.acceptable_options.add(previous)
            runtime.public_acceptances.add(previous)
            runtime.preferred_option = update.option_id
            runtime.ranks[update.option_id] = 5
            runtime.acceptable_options.discard(update.option_id)
            runtime.public_preference = update.option_id
            runtime.visible_switches += 1
            runtime.last_switch_turn = len(state.turns) + 1
            runtime.last_switch_target = update.option_id
            runtime.last_switch_external_evidence_turn = self._latest_external_evidence_turn(
                action.speaker_id,
                {previous, update.option_id},
            )
        elif update.kind == StanceUpdateKind.REJECT:
            runtime.ranks[update.option_id] = 1
            runtime.hard_rejected_options.add(update.option_id)
            runtime.acceptable_options.discard(update.option_id)
            runtime.public_acceptances.discard(update.option_id)
            runtime.public_rejections.add(update.option_id)

    def _apply_public_action(self, action: UserAction) -> None:
        state = self.state
        runtime = state.runtimes[action.speaker_id]
        if action.act == ActionType.OPENING:
            runtime.public_preference = runtime.preferred_option
        if action.act == ActionType.SUPPORT:
            for option_id in action.option_focus:
                state.public_supports[option_id] += 1
                state.public_supporters[option_id].add(action.speaker_id)
        elif action.act == ActionType.CONCERN:
            for option_id in action.option_focus:
                state.public_concerns[option_id] += 1
                state.public_concern_raisers[option_id].add(action.speaker_id)
        elif action.act == ActionType.COMPARE and len(action.option_focus) >= 2:
            pair = tuple(sorted(action.option_focus[:2]))
            state.public_comparisons[pair] += 1
            state.public_comparers.setdefault(pair, set()).add(action.speaker_id)
        elif action.act == ActionType.COMPROMISE:
            for option_id in action.option_focus:
                state.public_supports[option_id] += 1
                state.public_supporters[option_id].add(action.speaker_id)
        elif action.act == ActionType.VOTE:
            state.votes[action.speaker_id] = action.vote_option

        substantive = {
            ActionType.SUPPORT,
            ActionType.CONCERN,
            ActionType.ASK,
            ActionType.ANSWER,
            ActionType.COMPARE,
            ActionType.COMPROMISE,
        }
        if action.act in substantive:
            for option_id in action.option_focus:
                state.coverage[option_id].add(action.speaker_id, action.act)
        if action.stance_update and action.stance_update.kind in {
            StanceUpdateKind.MAKE_ACCEPTABLE,
            StanceUpdateKind.SWITCH_PREFERRED,
            StanceUpdateKind.REJECT,
        }:
            state.coverage[action.stance_update.option_id].add(action.speaker_id, action.act)

    def _latest_external_evidence_turn(
        self,
        participant_id: str,
        option_ids: set[str],
    ) -> int:
        return max((
            turn.index
            for turn in self.state.participant_turns
            if turn.speaker_id != participant_id
            and turn.action is not None
            and bool(option_ids & set(turn.action.option_focus))
        ), default=-1)

    def _apply_issue_before_turn(self, action: UserAction, text: str) -> str | None:
        state = self.state
        if action.issue_effect == IssueEffect.OPEN:
            if state.active_issue:
                self._stale_active_issue("a new issue took priority")
            kind = {
                ActionType.ASK: IssueKind.QUESTION,
                ActionType.CONCERN: IssueKind.CONCERN,
                ActionType.COMPARE: IssueKind.COMPARISON,
            }.get(action.act)
            if kind:
                return self._open_issue(kind, action, text)
        if action.act == ActionType.COMPARE and len(action.option_focus) >= 2 and not state.active_issue:
            pair = tuple(sorted(action.option_focus[:2]))
            # The first comparison is ordinary evidence. A later independent
            # development of the same trade-off creates the local issue.
            if state.public_comparisons[pair] >= 1:
                return self._open_issue(IssueKind.COMPARISON, action, text)
        return "continued" if state.active_issue and action.issue_id == state.active_issue.id else None

    def _apply_issue_after_turn(self, action: UserAction) -> None:
        state = self.state
        issue = state.active_issue
        if not issue or action.issue_id != issue.id:
            return
        issue.last_relevant_turn = len(state.turns)
        is_opening_turn = (
            action.issue_effect == IssueEffect.OPEN
            and issue.opened_at_turn == len(state.turns)
            and action.speaker_id == issue.opened_by
        )
        if not is_opening_turn:
            issue.follow_up_count += 1
        if action.act == ActionType.ANSWER and state.response_obligation == action.speaker_id:
            state.response_obligation = None
        if (
            issue.kind is IssueKind.CONCERN
            and action.speaker_id != issue.opened_by
            and action.issue_response_kind is not None
        ):
            issue.relevant_responder_ids.add(action.speaker_id)
            issue.relevant_response_kinds[action.issue_response_kind.value] += 1
            if action.issue_response_kind is IssueResponseKind.MITIGATION:
                issue.same_attribute_mitigation = True
        if action.issue_effect == IssueEffect.ANSWERED:
            issue.answered = True
            issue.outcome = "answered"
        elif action.issue_effect == IssueEffect.PARTIAL:
            issue.owner_last_evaluated_follow_up_count = issue.follow_up_count
            issue.outcome = "partially_addressed"
        elif action.issue_effect == IssueEffect.MAINTAIN:
            issue.owner_last_evaluated_follow_up_count = issue.follow_up_count
            issue.outcome = "maintained"
        elif action.issue_effect == IssueEffect.RESOLVE:
            issue.owner_last_evaluated_follow_up_count = issue.follow_up_count
            issue.outcome = "resolved"
            self._close_active_issue(IssueStatus.RESOLVED, "explicit structured resolution")
            return
        cap = int(cfg.conversation.issue_follow_up_cap)
        if issue.follow_up_count >= cap:
            self._stale_active_issue("hard follow-up cap reached")

    def _open_issue(self, kind: IssueKind, action: UserAction, text: str) -> str:
        self._issue_counter += 1
        issue_id = f"i{self._issue_counter:03d}"
        action.issue_id = issue_id
        issue = ActiveIssue(
            id=issue_id,
            kind=kind,
            option_focus=tuple(action.option_focus),
            opened_by=action.speaker_id,
            addressed_to=action.addressee_id,
            summary=action.reason or text,
            status=IssueStatus.OPEN,
            opened_at_turn=len(self.state.turns) + 1,
            last_relevant_turn=len(self.state.turns) + 1,
            source_text=text,
            reason_source=action.reason_source,
            issue_key=(action.reason_source.option_id, action.reason_source.attribute_name)
            if action.reason_source else None,
        )
        self.state.active_issue = issue
        if kind == IssueKind.QUESTION and action.addressee_id:
            self.state.response_obligation = action.addressee_id
        runtime = self.state.runtimes[action.speaker_id]
        if kind == IssueKind.CONCERN:
            for option_id in action.option_focus:
                runtime.opened_issue_keys.add(f"concern:{option_id}")
        return f"opened:{issue_id}"

    def _close_active_issue(self, status: IssueStatus, reason: str = "") -> None:
        issue = self.state.active_issue
        if not issue:
            return
        issue.status = status
        issue.close_reason = reason
        self.state.issue_history.append(issue)
        self.state.active_issue = None
        self.state.response_obligation = None

    def _stale_active_issue(self, reason: str) -> None:
        issue = self.state.active_issue
        if not issue:
            return
        self._close_active_issue(IssueStatus.STALE, reason)

    def _finish_exhausted_issue(self, reason: str) -> None:
        issue = self.state.active_issue
        if not issue:
            return
        if issue.kind == IssueKind.QUESTION and issue.answered:
            issue.outcome = issue.outcome or "answered"
            self._close_active_issue(IssueStatus.RESOLVED, "question answered; no further follow-up")
            return
        self._stale_active_issue(reason)

    def _set_group_stimulus(
        self,
        kind: StimulusKind,
        option_focus: tuple[str, ...],
        prompt_text: str,
    ) -> None:
        self._stimulus_counter += 1
        self.state.group_stimulus = GroupStimulus(
            id=self._stimulus_counter,
            kind=kind,
            option_focus=option_focus,
            prompt_text=prompt_text,
            created_at_turn=len(self.state.turns),
        )

    def _coverage_prompt_needed(self) -> bool:
        if not self._moderator_enabled or self.state.coverage_prompt_used:
            return False
        if self._phase_voluntary_count(Phase.DISCUSSION) < int(cfg.conversation.min_voluntary_turns):
            return False
        # Coverage is soft. Do not reopen alternatives after all participants
        # already publicly prefer the same option and no issue is active.
        if self._publicly_converged():
            return False
        ready = self._otherwise_ready_to_narrow()
        return bool(self._uncovered_options()) and ready

    def _uncovered_options(self) -> list[str]:
        return [
            option_id for option_id, coverage in self.state.coverage.items()
            if coverage.substantive_count == 0
        ]

    def _ready_to_narrow(self) -> bool:
        state = self.state
        count = self._phase_voluntary_count(Phase.DISCUSSION)
        minimum = int(cfg.conversation.min_voluntary_turns)
        target = int(cfg.conversation.soft_target_voluntary_turns)
        maximum = int(cfg.conversation.hard_max_voluntary_turns)
        if state.response_obligation or state.active_issue:
            return False
        converged = self._publicly_converged()
        if converged and self._convergence_confirmed():
            return True
        if count < minimum:
            return False
        if (
            self._moderator_enabled
            and self._uncovered_options()
            and not state.coverage_prompt_used
            and not converged
        ):
            return False
        substantive = sum(coverage.substantive_count for coverage in state.coverage.values())
        if converged and substantive >= max(3, len(state.personas)):
            return True
        if count >= maximum:
            return True
        recent = state.recent_novelty[-5:]
        low_novelty = len(recent) >= 4 and sum(recent) <= 1
        return count >= target and low_novelty

    def _publicly_converged(self) -> bool:
        preferences = [runtime.public_preference for runtime in self.state.runtimes.values()]
        return bool(preferences and None not in preferences and len(set(preferences)) == 1)

    def _convergence_confirmed(self) -> bool:
        """Require one voluntary discussion contribution after unanimous openings."""
        if not self._publicly_converged():
            return False
        shared = next(iter({
            runtime.public_preference for runtime in self.state.runtimes.values()
            if runtime.public_preference is not None
        }))
        confirming_acts = {
            ActionType.SUPPORT, ActionType.ACKNOWLEDGE, ActionType.COMMENT,
            ActionType.ANSWER, ActionType.COMPARE, ActionType.COMPROMISE,
        }
        return any(
            turn.phase is Phase.DISCUSSION
            and turn.voluntary
            and turn.action is not None
            and turn.action.act in confirming_acts
            and shared in turn.action.option_focus
            for turn in self.state.participant_turns
        )

    def _otherwise_ready_to_narrow(self) -> bool:
        state = self.state
        if state.response_obligation or state.active_issue:
            return False
        count = self._phase_voluntary_count(Phase.DISCUSSION)
        target = int(cfg.conversation.soft_target_voluntary_turns)
        maximum = int(cfg.conversation.hard_max_voluntary_turns)
        recent = state.recent_novelty[-5:]
        return count >= maximum or (count >= target and (not recent or sum(recent) <= 2))

    def _record_novelty(self, record: TurnRecord) -> None:
        action = record.action
        if not action:
            return
        signature = (
            action.act.value,
            tuple(action.option_focus),
            action.reason_source.attribute_name if action.reason_source else action.reason.casefold()[:30],
        )
        prior = []
        for turn in self.state.participant_turns[:-1][-8:]:
            if not turn.action:
                continue
            prior.append((
                turn.action.act.value,
                tuple(turn.action.option_focus),
                turn.action.reason_source.attribute_name if turn.action.reason_source else turn.action.reason.casefold()[:30],
            ))
        self.state.recent_novelty.append(signature not in prior)
        self.state.recent_novelty = self.state.recent_novelty[-12:]

    @staticmethod
    def _reason_key(action: UserAction) -> str:
        source = action.reason_source
        if source:
            return f"{action.act.value}:{source.option_id}:{source.attribute_name}:{source.public_value}"
        return f"{action.act.value}:{action.reason.casefold()}"

    def _phase_voluntary_count(self, phase: Phase) -> int:
        return sum(1 for turn in self.state.turns if turn.phase == phase and turn.voluntary)

    def _transition(self, phase: Phase) -> None:
        self.state.phase = phase
        self.state.phase_history.append(phase.value)

    def _append_moderator(self, text: str) -> None:
        record = TurnRecord(
            index=len(self.state.turns) + 1,
            phase=self.state.phase,
            speaker_id="moderator",
            speaker_name="Moderator",
            text=text,
            moderator=True,
        )
        self.state.turns.append(record)
        self.state.stats.moderator_turns += 1
        print(f"Moderator: {text}")

    @staticmethod
    def _print_header(state: DialogueState) -> None:
        print("=" * 72)
        print(f"Topic: {state.scenario.topic}")
        print("Participants: " + ", ".join(persona.name for persona in state.personas))
        print("=" * 72)
        if not bool((cfg.get("moderator", None) or {}).get("enabled", True)):
            print("Public setup:")
            for fact in state.scenario.shared_context:
                print(f"- {fact}")
            for option in state.scenario.options:
                print(option.public_line())


def initialise_state(scenario: Scenario, personas: list[Persona]) -> DialogueState:
    if not personas:
        raise ValueError("at least one persona is required")
    option_ids = scenario.option_ids
    runtimes = {persona.id: initial_runtime(persona, option_ids) for persona in personas}
    coverage = {option_id: OptionCoverage() for option_id in option_ids}
    return DialogueState(
        scenario=scenario,
        personas=personas,
        runtimes=runtimes,
        coverage=coverage,
        public_supports=Counter(),
        public_concerns=Counter(),
        public_comparisons=Counter(),
        public_supporters={option_id: set() for option_id in option_ids},
        public_concern_raisers={option_id: set() for option_id in option_ids},
        public_comparers={},
    )
