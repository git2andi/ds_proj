"""Dialogue runtime for autonomous structured user simulators."""

from __future__ import annotations

import random
import re
from collections import Counter

import prompts
from builders import SetupBuilder
from config_loader import cfg
from consensus import derive_narrowing_options, majority_threshold, outcome_from_votes
from llm_client import get_llm_client
from logger import DialogueLogger
from models import (
    ActionType,
    ActiveIssue,
    BidPriority,
    DialogueRunResult,
    DialogueState,
    GenerationAttempt,
    GroupStimulus,
    IssueEffect,
    IssueKind,
    IssueRecord,
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
from simulator import FloorManager, UserSimulator, initial_runtime, public_question_key, reason_key
from validation import mentioned_options, validate_action, validate_realization


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
        self.seed = int(seed if seed is not None else configured_seed) if (seed is not None or configured_seed is not None) else random.SystemRandom().randint(0, 2**31 - 1)
        self.rng = rng or random.Random(self.seed)
        self._llm = llm or get_llm_client()

        if scenario is None or personas is None:
            self._llm.reset_session()
            builder = SetupBuilder(self.topic, force_auto_scenario=force_auto_scenario, llm=self._llm)
            scenario, personas = builder.build(cfg.participant_count())
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
        self._issue_counter = 0
        self._stimulus_counter = 0
        self._last_failure_errors: list[str] = []
        self._moderator_enabled = bool(cfg.moderator.enabled)
        self._vote_prompt_already_emitted = False

    def run(self) -> DialogueRunResult:
        self._print_header(self.state)
        if self._moderator_enabled:
            self._append_moderator(prompts.moderator_opening(self.state.scenario))

        self._run_opening()
        self._transition(Phase.DISCUSSION)
        self._run_discussion()
        self._run_narrowing(revote=False)
        outcome = self._run_voting(revote=False)

        if outcome is None:
            self.state.first_round_votes = dict(self.state.votes)
            _, movement_count = self._run_narrowing(revote=True)
            if movement_count == 0:
                self.state.revote_skipped_no_movement = True
                self.state.stats.revote_skipped_no_movement += 1
                outcome = RunOutcome(
                    "unresolved",
                    None,
                    dict(self.state.first_round_votes),
                    "No option reached a majority and no participant changed or broadened their position during the final discussion",
                )
            else:
                outcome = self._run_voting(revote=True)
                if outcome is None:
                    outcome = outcome_from_votes(
                        self.state,
                        self.state.votes,
                        allow_unresolved=True,
                    )

        assert outcome is not None
        self._transition(Phase.CLOSED)
        if self._moderator_enabled:
            self._append_moderator(prompts.moderator_closure(outcome, self.state.scenario))
        log_paths = self.logger.write_run(self.state, outcome, seed=self.seed)
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
            record = self._realize_and_commit(action, mandatory=True, voluntary=False)
            if record is None:
                # Mandatory openings may not silently disappear. Give the endpoint
                # one completely fresh attempt after the ordinary repair path.
                record = self._realize_and_commit(action, mandatory=True, voluntary=False)
            if record is None:
                raise RuntimeError(f"mandatory opening failed for {participant_id}: {self._last_failure_errors}")

    def _run_discussion(self) -> None:
        minimum, target, maximum = cfg.conversation_turn_budgets(len(self.state.personas))
        stagnation_rounds = int(cfg.conversation.stagnation_no_bid_rounds)
        small_group_extra_round_used = False
        while self._phase_voluntary_count(Phase.DISCUSSION) < maximum or self.state.response_obligation:
            if self.state.response_obligation:
                self._drain_response_obligation("mandatory answer failed")
                continue
            if self._phase_voluntary_count(Phase.DISCUSSION) >= maximum:
                break

            bids = [simulator.propose(self.state) for simulator in self._simulators.values()]
            if self.state.active_issue and not any(
                bid.wants_to_speak and bid.issue_id == self.state.active_issue.id for bid in bids
            ):
                self._close_or_stale_inactive_issue("the group moved on")
                bids = [simulator.propose(self.state) for simulator in self._simulators.values()]

            accepted = self._select_and_realize(bids, phase=Phase.DISCUSSION)
            if accepted:
                self.state.no_bid_rounds = 0
            else:
                self.state.no_bid_rounds += 1
                count = self._phase_voluntary_count(Phase.DISCUSSION)
                if self._public_unanimity_ready_for_vote(count):
                    break
                if (
                    count >= minimum
                    and self.state.no_bid_rounds >= stagnation_rounds
                    and not self.state.compromise_prompt_used
                ):
                    if self._run_compromise_window(
                        phase=Phase.DISCUSSION,
                        announce=True,
                    ):
                        self.state.no_bid_rounds = 0
                        continue
                if (
                    len(self.state.personas) <= 4
                    and count < target
                    and not small_group_extra_round_used
                ):
                    # One ordinary retry gives small groups a little more room
                    # without forcing filler or changing simulator authority.
                    small_group_extra_round_used = True
                    continue
                if count >= minimum:
                    break
                if self._moderator_enabled and not self.state.stall_prompt_used:
                    self.state.stall_prompt_used = True
                    if self._run_stall_window():
                        continue
                if not self._force_liveness(Phase.DISCUSSION):
                    break

            if self.state.response_obligation:
                continue
            count = self._phase_voluntary_count(Phase.DISCUSSION)
            if self._public_unanimity_ready_for_vote(count):
                break
            shared_acceptance_minimum = min(
                target,
                minimum + (3 if len(self.state.personas) <= 4 else 0),
            )
            if count >= minimum and not self.state.active_issue:
                if self._publicly_converged():
                    break
                if (
                    count >= shared_acceptance_minimum
                    and self._shared_acceptable_option() is not None
                ):
                    break
            if count >= minimum and self._coverage_prompt_needed():
                self._run_coverage_window(self._uncovered_options()[0])
            if count >= target and not self.state.active_issue:
                if not any(
                    simulator.has_novel_voluntary_bid(self.state)
                    for simulator in self._simulators.values()
                ):
                    if not self.state.compromise_prompt_used:
                        self._run_compromise_window(
                            phase=Phase.DISCUSSION,
                            announce=True,
                        )
                    break
            if count >= minimum and self._publicly_converged() and not self.state.active_issue:
                break

        if self.state.active_issue:
            self._close_or_stale_inactive_issue("discussion phase ended")

    def _run_coverage_window(self, option_id: str) -> bool:
        if not self._moderator_enabled or self.state.coverage_prompt_used:
            return False
        text = prompts.moderator_coverage_prompt(self.state.scenario, option_id)
        self._set_group_stimulus(StimulusKind.COVERAGE, (option_id,), text)
        bids = [simulator.propose(self.state) for simulator in self._simulators.values()]
        relevant = [
            bid for bid in bids
            if bid.wants_to_speak and option_id in bid.option_focus and bid.stimulus_id == self.state.group_stimulus.id
        ]
        if not self._floor.has_selectable_bid(self.state, relevant):
            self.state.coverage_no_interest.add(option_id)
            self.state.group_stimulus = None
            return False
        accepted = self._select_and_realize(
            relevant,
            phase=Phase.DISCUSSION,
            moderator_before=text,
        )
        if accepted:
            self.state.coverage_prompt_used = True
        self.state.group_stimulus = None
        return accepted

    def _run_stall_window(self) -> bool:
        text = prompts.moderator_stall_prompt()
        self._set_group_stimulus(StimulusKind.STALL, (), text)
        bids = [simulator.propose(self.state, liveness_forced=True) for simulator in self._simulators.values()]
        relevant = [bid for bid in bids if bid.wants_to_speak and bid.stimulus_id == self.state.group_stimulus.id]
        if not self._floor.has_selectable_bid(self.state, relevant):
            self.state.group_stimulus = None
            return False
        accepted = self._select_and_realize(
            relevant,
            phase=Phase.DISCUSSION,
            moderator_before=text,
        )
        self.state.group_stimulus = None
        return accepted

    def _run_narrowing(self, *, revote: bool) -> tuple[int, int]:
        """Run adaptive narrowing without generic restatement rounds."""

        self._transition(Phase.NARROWING)
        self.state.narrowing_options = derive_narrowing_options(self.state)
        split_options = (
            self._public_compromise_options()
            if not self.state.narrowing_options
            else ()
        )
        start_movement = self.state.movement_events
        accepted_count = 0
        optional_reaction_windows_used = 0
        optional_reaction_window_cap = (
            2 if len(self.state.personas) >= 5 else len(self.state.personas)
        )

        unanimous = self._publicly_converged()
        unanimous_option = next(
            (
                runtime.public_preference
                for runtime in self.state.runtimes.values()
                if runtime.public_preference is not None
            ),
            None,
        )
        narrowing_prompt = ""
        if self._moderator_enabled:
            if unanimous and unanimous_option:
                # The unanimous status is combined with the final-vote request
                # so the moderator does not speak twice in a row.
                narrowing_prompt = ""
            else:
                if split_options:
                    narrowing_prompt = prompts.moderator_split_compromise_prompt(
                        self.state.scenario,
                        split_options,
                        revote=revote,
                    )
                else:
                    narrowing_prompt = (
                        prompts.moderator_revote_narrowing(
                            self.state.scenario,
                            self.state.narrowing_options,
                        )
                        if revote
                        else prompts.moderator_narrowing(
                            self.state.scenario,
                            self.state.narrowing_options,
                        )
                    )
            # A clear leader or unanimous state has a real scheduled next step.
            # Tie/compromise prompts are emitted later only if a simulator bid exists.
            if not unanimous and len(self.state.narrowing_options) == 1:
                self._append_moderator(narrowing_prompt)

        unanimous_option = next(
            (
                runtime.public_preference
                for runtime in self.state.runtimes.values()
                if runtime.public_preference is not None
            ),
            None,
        )
        if unanimous and not self._unresolved_concern_owners(unanimous_option):
            return 0, 0

        if len(self.state.narrowing_options) == 1:
            leader = self.state.narrowing_options[0]
            initial_support = self._public_position_support_count(leader)
            # Ask at least one relevant dissenter, but stop once one additional
            # acceptance (or the strict majority threshold) has been reached.
            # This avoids coordinated-looking rounds where every dissenter is
            # marched through the same concession.
            target_support = min(
                len(self.state.personas),
                max(majority_threshold(len(self.state.personas)), initial_support + 1),
            )
            participants = self._clear_leader_participants(leader, revote=revote)
            if len(self.state.personas) >= 5:
                participants = participants[: int(
                    cfg.conversation.large_group_narrowing_final_position_cap
                )]
            for participant_id in participants:
                action = self._simulators[participant_id].final_position_action(
                    self.state,
                    revote=revote,
                )
                record = self._realize_and_commit(
                    action,
                    mandatory=True,
                    voluntary=False,
                )
                if record is None:
                    if action.stance_update is not None:
                        self.state.stats.mandatory_movement_failures += 1
                    continue
                accepted_count += 1
                if self.state.active_issue:
                    accepted_count += self._run_active_issue_window(
                        Phase.NARROWING,
                        max_turns=(1 if len(self.state.personas) >= 5 else None),
                    )
                elif optional_reaction_windows_used < optional_reaction_window_cap:
                    reaction_count = self._run_optional_reaction_window()
                    accepted_count += reaction_count
                    if reaction_count:
                        optional_reaction_windows_used += 1
                self.state.narrowing_options = derive_narrowing_options(self.state)
                if self._publicly_converged():
                    break
                if (
                    self._public_position_support_count(leader) >= target_support
                    and self.state.active_issue is None
                ):
                    break
        else:
            announce = self._moderator_enabled
            if not self.state.narrowing_options and narrowing_prompt and self._moderator_enabled:
                # A complete public split should be visible even when nobody
                # volunteers to move. The prompt names only publicly preferred
                # or accepted options, never an untouched coverage option.
                self._append_moderator(narrowing_prompt)
                announce = False
            accepted_count += self._run_compromise_window(
                phase=Phase.NARROWING,
                announce=announce,
                prompt_text=narrowing_prompt,
            )
            if (
                accepted_count == 0
                and not self.state.narrowing_options
                and narrowing_prompt
                and self._moderator_enabled
            ):
                bridge = prompts.moderator_no_movement_bridge(revote=revote)
                self._append_moderator(bridge)
                if not revote:
                    self._vote_prompt_already_emitted = True
            self.state.narrowing_options = derive_narrowing_options(self.state)

        if self.state.active_issue:
            self._stale_active_issue("voting started")
        movement_count = self.state.movement_events - start_movement
        return accepted_count, movement_count

    def _clear_leader_participants(self, leader: str, *, revote: bool) -> list[str]:
        ids = [persona.id for persona in self.state.personas]
        concern_owners = set(self._unresolved_concern_owners(leader))
        if revote and self.state.first_round_votes:
            dissenters = [
                participant_id
                for participant_id in ids
                if self.state.first_round_votes.get(participant_id) != leader
            ]
        else:
            dissenters = [
                participant_id
                for participant_id in ids
                if self.state.runtimes[participant_id].public_preference != leader
            ]
        selected = list(dict.fromkeys([*dissenters, *concern_owners]))
        self.rng.shuffle(selected)
        return selected

    def _unresolved_concern_owners(self, option_id: str | None) -> list[str]:
        if not option_id:
            return []
        owners: list[str] = []
        issues = list(self.state.issue_history)
        if self.state.active_issue is not None:
            issues.append(self.state.active_issue)
        for issue in issues:
            if (
                issue.kind is IssueKind.CONCERN
                and option_id in issue.option_focus
                and issue.status is not IssueStatus.RESOLVED
                and issue.opened_by not in owners
            ):
                owners.append(issue.opened_by)
        return owners

    def _run_compromise_window(
        self,
        *,
        phase: Phase,
        announce: bool,
        prompt_text: str | None = None,
    ) -> int:
        if phase is Phase.DISCUSSION:
            self.state.compromise_prompt_used = True
        self.state.compromise_opportunity = True

        accepted = 0
        cap = int(cfg.conversation.compromise_window_max_turns)
        used_speakers: set[str] = set()
        announced = False
        try:
            for _ in range(cap):
                bids = [
                    simulator.propose(self.state)
                    for participant_id, simulator in self._simulators.items()
                    if participant_id not in used_speakers
                ]
                compromise_bids = [
                    bid
                    for bid in bids
                    if bid.wants_to_speak and bid.act is ActionType.COMPROMISE
                ]
                if not self._floor.has_selectable_bid(self.state, compromise_bids):
                    break
                moderator_before = (
                    prompt_text or prompts.moderator_compromise_prompt()
                    if announce and self._moderator_enabled and not announced
                    else None
                )
                if not self._select_and_realize(
                    compromise_bids,
                    phase=phase,
                    moderator_before=moderator_before,
                ):
                    break
                if moderator_before is not None:
                    announced = True
                proposer = self.state.last_participant_id
                if proposer:
                    used_speakers.add(proposer)
                accepted += 1
                accepted += self._run_optional_reaction_window()
                self.state.narrowing_options = derive_narrowing_options(self.state)
                if self._publicly_converged():
                    break
        finally:
            self.state.compromise_opportunity = False
        return accepted

    def _run_active_issue_window(
        self,
        phase: Phase,
        *,
        max_turns: int | None = None,
    ) -> int:
        accepted = 0
        cap = int(cfg.conversation.narrowing_reaction_turn_cap)
        if max_turns is not None:
            cap = min(cap, max(0, int(max_turns)))
        for _ in range(cap):
            if not self.state.active_issue:
                break
            if self.state.response_obligation:
                before = len(self.state.turns)
                self._drain_response_obligation("mandatory narrowing answer failed")
                accepted += int(len(self.state.turns) > before)
                continue
            bids = [simulator.propose(self.state) for simulator in self._simulators.values()]
            relevant = [
                bid
                for bid in bids
                if bid.wants_to_speak
                and self.state.active_issue is not None
                and bid.issue_id == self.state.active_issue.id
            ]
            if not self._select_and_realize(relevant, phase=phase):
                self._close_or_stale_inactive_issue("no participant continued the narrowing issue")
                break
            accepted += 1
        if self.state.active_issue:
            self._close_or_stale_inactive_issue("narrowing issue window ended")
        return accepted

    def _run_optional_reaction_window(self) -> int:
        if int(cfg.conversation.narrowing_reaction_turn_cap) <= 0:
            return 0
        latest_speaker = self.state.last_participant_id
        bids = [
            simulator.propose_reaction(self.state)
            for participant_id, simulator in self._simulators.items()
            if participant_id != latest_speaker
        ]
        if not self._select_and_realize(bids, phase=Phase.NARROWING):
            return 0
        if self.state.active_issue:
            return 1 + self._run_active_issue_window(Phase.NARROWING)
        return 1

    def _run_voting(self, *, revote: bool) -> RunOutcome | None:
        self._transition(Phase.VOTING)
        self.state.vote_round = 2 if revote else 1
        self.state.votes = {}
        self.state.vote_records[self.state.vote_round] = {}
        if self._moderator_enabled and not self._vote_prompt_already_emitted:
            unanimous_option = None
            if not revote and self._publicly_converged():
                unanimous_option = next(
                    (
                        runtime.public_preference
                        for runtime in self.state.runtimes.values()
                        if runtime.public_preference is not None
                    ),
                    None,
                )
            self._append_moderator(
                prompts.moderator_vote_request(
                    revote=revote,
                    scenario=self.state.scenario,
                    unanimous_option=unanimous_option,
                )
            )
        self._vote_prompt_already_emitted = False

        for persona in self.state.personas:
            action = self._simulators[persona.id].decide_vote(self.state, revote=revote)
            record = self._realize_and_commit(action, mandatory=True, voluntary=False)
            errors = list(self._last_failure_errors) if record is None else []
            if record is None:
                structured_errors = validate_action(self.state, persona, action)
                if structured_errors:
                    raise RuntimeError(
                        f"invalid authoritative vote for {persona.id}: {structured_errors}"
                    )
                fallback_text = self._vote_fallback_text(action)
                fallback_attempt = GenerationAttempt(
                    persona.id,
                    self.state.phase,
                    action.copy(),
                    "",
                    list(errors),
                    final_status="fallback",
                    fallback_text=fallback_text,
                )
                self.state.generation_attempts.append(fallback_attempt)
                self.state.stats.vote_fallbacks += 1
                record = self._commit_action(
                    action,
                    fallback_text,
                    mandatory=True,
                    voluntary=False,
                    liveness_forced=False,
                    repair_count=0,
                )
            vote_record = VoteRecord(
                persona.id,
                self.state.vote_round,
                VoteStatus.VALID,
                action.vote_option,
                1,
                errors,
            )
            self.state.vote_records[self.state.vote_round][persona.id] = vote_record
        return outcome_from_votes(self.state, self.state.votes, allow_unresolved=revote)

    def _fallback_variant(self, action: UserAction, count: int) -> int:
        token = f"{action.speaker_id}:{action.option_focus}:{action.act.value}"
        return sum(ord(char) for char in token) % count

    def _vote_fallback_text(self, action: UserAction) -> str:
        option_id = action.vote_option or (action.option_focus[0] if action.option_focus else "")
        option = self.state.scenario.option(option_id)
        name = option.short_name or option.name
        if action.stance_update is not None:
            reason = action.stance_update.movement_reason.strip()
            variant = self._fallback_variant(action, 3)
            if action.stance_update.reason_already_public:
                options = (
                    f"I’m going with {name} now.",
                    f"{name} is my choice now.",
                    f"I’m on board with {name} for the final vote.",
                )
                return options[variant]
            if reason:
                options = (
                    f"{name} is my choice now. The point that shifted me: {reason}.",
                    f"I’ve settled on {name}. The deciding consideration: {reason}.",
                    f"I’m going with {name} now. What mattered most: {reason}.",
                )
                return options[variant]
            return f"I’m going with {name} now."
        return f"{name} gets my vote."

    def _movement_fallback_text(self, action: UserAction) -> str:
        update = action.stance_update
        if update is None:
            raise ValueError("movement fallback requires a stance update")
        option = self.state.scenario.option(update.option_id)
        name = option.short_name or option.name
        reason = update.movement_reason.strip() or action.decisive_reason.strip() or action.reason.strip()
        variant = self._fallback_variant(action, 3)
        if update.kind is StanceUpdateKind.SWITCH_PREFERRED:
            if update.reason_already_public or not reason:
                options = (
                    f"That changes my view. I’m going with {name} now.",
                    f"I now prefer {name}.",
                    f"I’m on board with {name} now.",
                )
                return options[variant]
            options = (
                f"{name} is my choice now. The point that shifted me: {reason}.",
                f"I’ve settled on {name}. The deciding consideration: {reason}.",
                f"I’m going with {name} now. What mattered most: {reason}.",
            )
            return options[variant]
        if update.kind is StanceUpdateKind.MAKE_ACCEPTABLE:
            if update.reason_already_public or not reason:
                options = (
                    f"{name} would work for me now.",
                    f"I could support {name} now.",
                    f"I’m good with {name} now.",
                )
                return options[variant]
            basis = update.movement_basis
            if basis == "concern_resolved":
                options = (
                    f"That settles my concern, so {name} works for me now. Relevant point: {reason}.",
                    f"{name} is workable for me now. The response that resolved it: {reason}.",
                    f"I can support {name} now; my earlier concern is settled. Relevant point: {reason}.",
                )
            elif basis in {"common_ground", "common_ground_proposal", "stagnation_compromise"}:
                options = (
                    f"I can go along with {name} as common ground. The benefit I’m weighing: {reason}.",
                    f"{name} works for me as a compromise. What carries the trade-off: {reason}.",
                    f"I could support {name} for the group. The point that matters here: {reason}.",
                )
            else:
                options = (
                    f"{name} works for me now. The relevant point: {reason}.",
                    f"I could support {name} now. What makes it workable: {reason}.",
                    f"I’m comfortable with {name} now. The deciding consideration: {reason}.",
                )
            return options[variant]
        return f"My position on {name} has changed."

    def _select_and_realize(
        self,
        bids: list[UserAction],
        *,
        phase: Phase,
        moderator_before: str | None = None,
    ) -> bool:
        remaining = list(bids)
        while remaining:
            selection = self._floor.select(self.state, remaining)
            if selection is None:
                return False
            record = self._realize_and_commit(
                selection.action,
                mandatory=False,
                voluntary=True,
                moderator_before=moderator_before,
            )
            if record is not None:
                return True
            remaining = [bid for bid in remaining if bid is not selection.action]
        return False

    def _force_liveness(self, phase: Phase) -> bool:
        candidates = [
            simulator.propose(self.state, liveness_forced=True)
            for simulator in self._simulators.values()
        ]
        remaining = list(candidates)
        while remaining:
            selection = self._floor.select(self.state, remaining)
            if selection is None:
                return False
            record = self._realize_and_commit(
                selection.action,
                mandatory=False,
                voluntary=False,
                liveness_forced=True,
            )
            if record:
                self.state.stats.liveness_forced_turns += 1
                return True
            remaining = [bid for bid in remaining if bid is not selection.action]
        return False

    def _drain_response_obligation(self, failure_reason: str) -> None:
        participant_id = self.state.response_obligation
        if not participant_id:
            return
        action = self._simulators[participant_id].propose(self.state, liveness_forced=True)
        for _ in range(2):
            if self._realize_and_commit(action, mandatory=True, voluntary=False):
                return
        self.state.vote_protocol_errors.append(failure_reason)
        self.state.response_obligation = None
        if self.state.active_issue:
            self._stale_active_issue(failure_reason)

    def _realize_and_commit(
        self,
        action: UserAction,
        *,
        mandatory: bool,
        voluntary: bool,
        liveness_forced: bool = False,
        moderator_before: str | None = None,
    ) -> TurnRecord | None:
        if action.stance_update is not None:
            self.state.stats.selected_movement_actions += 1
        persona = self.state.persona(action.speaker_id)
        action_errors = validate_action(self.state, persona, action)
        if action_errors:
            self._last_failure_errors = action_errors
            self.state.stats.dropped_turns += 1
            if action.stance_update is not None:
                self.state.stats.movement_realization_failures += 1
            return None

        prompt = prompts.realization_prompt(self.state, persona, action)
        self.logger.write_prompt(prompt, "dialogue")
        raw = self._call_llm(prompt, profile="dialogue")
        errors = validate_realization(self.state, persona, action, raw)
        attempt = GenerationAttempt(action.speaker_id, self.state.phase, action.copy(), raw, list(errors))
        repair_count = 0
        final = raw

        if errors:
            for error in errors:
                self.state.validation_failures[self._validation_category(error)] += 1
            repair_count = 1
            self.state.stats.repair_calls += 1
            repair = prompts.repair_prompt(self.state, persona, action, raw, errors)
            self.logger.write_prompt(repair, "repair")
            fixed = self._call_llm(repair, profile="repair")
            repair_errors = validate_realization(self.state, persona, action, fixed)
            attempt.repair_text = fixed
            attempt.repair_errors = list(repair_errors)
            if repair_errors:
                if action.stance_update is not None:
                    fallback_text = (
                        self._vote_fallback_text(action)
                        if action.act is ActionType.VOTE
                        else self._movement_fallback_text(action)
                    )
                    attempt.final_status = "fallback"
                    attempt.fallback_text = fallback_text
                    self.state.generation_attempts.append(attempt)
                    self.state.stats.movement_realization_failures += 1
                    self.state.stats.movement_fallbacks += 1
                    if action.act is ActionType.VOTE:
                        self.state.stats.vote_fallbacks += 1
                    if mandatory:
                        self.state.stats.mandatory_movement_failures += 1
                    self._last_failure_errors = []
                    if moderator_before and self._moderator_enabled:
                        self._append_moderator(moderator_before)
                    return self._commit_action(
                        action,
                        fallback_text,
                        mandatory=mandatory,
                        voluntary=voluntary,
                        liveness_forced=liveness_forced,
                        repair_count=repair_count,
                    )
                attempt.final_status = "dropped"
                self.state.generation_attempts.append(attempt)
                self.state.stats.dropped_turns += 1
                if action.stance_update is not None:
                    self.state.stats.movement_realization_failures += 1
                self._last_failure_errors = repair_errors
                return None
            final = fixed

        attempt.final_status = "accepted"
        self.state.generation_attempts.append(attempt)
        self._last_failure_errors = []
        if moderator_before and self._moderator_enabled:
            self._append_moderator(moderator_before)
        return self._commit_action(
            action,
            final.strip(),
            mandatory=mandatory,
            voluntary=voluntary,
            liveness_forced=liveness_forced,
            repair_count=repair_count,
        )

    def _call_llm(self, prompt: str, *, profile: str) -> str:
        text = self._llm.generate(prompt, profile=profile)
        self.state.stats.llm_calls += 1
        self.state.stats.input_tokens += int(getattr(self._llm, "last_tokens_in", 0))
        self.state.stats.output_tokens += int(getattr(self._llm, "last_tokens_out", 0))
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
        runtime = self.state.runtimes[action.speaker_id]
        issue_event = self._apply_issue_before_turn(action, text)
        if action.stance_update:
            self._apply_stance_update(action)
            self.state.stats.committed_movement_actions += 1
        self._apply_public_action(action, text)

        _, max_words = prompts.word_budget(action.act, self.state.persona(action.speaker_id).sim_params.verbosity)
        record = TurnRecord(
            index=len(self.state.turns),
            phase=self.state.phase,
            speaker_id=action.speaker_id,
            speaker_name=self.state.persona(action.speaker_id).name,
            text=text,
            action=action.copy(),
            mandatory=mandatory,
            voluntary=voluntary,
            liveness_forced=liveness_forced,
            priority=action.priority,
            repair_count=repair_count,
            issue_event=issue_event,
            stance_update=action.stance_update,
            vote_option=action.vote_option,
            narrowing_options=self.state.narrowing_options,
            prompt_tokens=int(getattr(self._llm, "last_tokens_in", 0)),
            output_tokens=int(getattr(self._llm, "last_tokens_out", 0)),
            intended_word_max=max_words,
        )
        self.state.turns.append(record)
        print(f"{record.speaker_name}: {record.text}")

        runtime.last_action = action.act
        runtime.last_spoken_turn = record.index
        semantic_key = reason_key(action)
        if semantic_key:
            public_seen = any(
                prior.action is not None and reason_key(prior.action) == semantic_key
                for prior in self.state.turns[:-1]
            )
            if (
                public_seen
                and action.stance_update is None
                and action.act not in {ActionType.ANSWER, ActionType.FINAL_POSITION, ActionType.VOTE}
                and action.issue_effect not in {IssueEffect.RESOLVE, IssueEffect.MAINTAIN, IssueEffect.PARTIAL}
            ):
                self.state.stats.semantic_reason_reuse += 1
            runtime.used_reason_keys.add(semantic_key)
        if action.act is ActionType.ACKNOWLEDGE and action.option_focus:
            runtime.acknowledged_options.add(action.option_focus[0])
        if action.act is ActionType.COMPROMISE and action.option_focus:
            runtime.used_compromise_options.add(action.option_focus[0])
            self.state.stats.compromise_proposals += 1
        if voluntary:
            runtime.voluntary_turns += 1
            self.state.stats.voluntary_turns += 1
        if action.act is ActionType.OPENING:
            runtime.openings += 1
        if action.act is ActionType.ANSWER:
            runtime.mandatory_answers += int(mandatory)
        if action.act is ActionType.VOTE:
            runtime.votes_cast += 1
        if action.stimulus_id is not None:
            runtime.responded_stimuli.add(action.stimulus_id)
        if action.issue_id is not None and action.issue_effect is IssueEffect.RESPOND:
            runtime.responded_issue_ids.add(action.issue_id)

        self._apply_issue_after_turn(action)
        return record

    def _apply_stance_update(self, action: UserAction) -> None:
        update = action.stance_update
        if not update:
            return
        runtime = self.state.runtimes[action.speaker_id]
        if update.kind is StanceUpdateKind.MAKE_ACCEPTABLE:
            newly_visible = update.option_id not in runtime.public_acceptances
            runtime.acceptable_options.add(update.option_id)
            runtime.public_acceptances.add(update.option_id)
            movement_reason = update.movement_reason.strip() or action.decisive_reason.strip() or action.reason.strip()
            if movement_reason:
                runtime.acceptance_reasons[update.option_id] = movement_reason
            runtime.ranks[update.option_id] = max(runtime.ranks.get(update.option_id, 3), 4)
            if newly_visible:
                self.state.movement_events += 1
                self.state.stats.compromise_acceptances += 1
                if self.state.phase is Phase.NARROWING:
                    self.state.stats.narrowing_movements += 1
        elif update.kind is StanceUpdateKind.SWITCH_PREFERRED:
            changed = runtime.preferred_option != update.option_id
            movement_reason = update.movement_reason.strip() or action.decisive_reason.strip() or action.reason.strip()
            if movement_reason:
                runtime.acceptance_reasons[update.option_id] = movement_reason
            runtime.preferred_option = update.option_id
            runtime.public_preference = update.option_id
            runtime.acceptable_options.add(update.option_id)
            runtime.public_acceptances.add(update.option_id)
            runtime.ranks[update.option_id] = 5
            if changed:
                runtime.visible_switches += 1
                runtime.last_switch_turn = len(self.state.turns)
                self.state.movement_events += 1
                if self.state.phase is Phase.NARROWING:
                    self.state.stats.narrowing_movements += 1
        elif update.kind is StanceUpdateKind.REJECT:
            runtime.public_rejections.add(update.option_id)
        elif update.kind is StanceUpdateKind.REMOVE_ACCEPTANCE:
            runtime.public_acceptances.discard(update.option_id)

    def _apply_public_action(self, action: UserAction, text: str) -> None:
        runtime = self.state.runtimes[action.speaker_id]
        if action.act is ActionType.OPENING and action.option_focus:
            runtime.public_preference = action.option_focus[0]
        if action.act in {ActionType.OPENING, ActionType.SUPPORT, ActionType.COMPROMISE}:
            for option_id in action.option_focus:
                self.state.public_supports[option_id] += 1
                self.state.public_supporters.setdefault(option_id, set()).add(action.speaker_id)
                self.state.coverage[option_id].add(action.speaker_id, action.act)
        elif action.act is ActionType.CONCERN:
            for option_id in action.option_focus:
                self.state.public_concerns[option_id] += 1
                self.state.public_concern_raisers.setdefault(option_id, set()).add(action.speaker_id)
                self.state.coverage[option_id].add(action.speaker_id, action.act)
        elif action.act is ActionType.COMPARE:
            visible = mentioned_options(text, self.state)
            pair = tuple(sorted(option_id for option_id in action.option_focus if option_id in visible))
            if len(pair) >= 2:
                self.state.public_comparisons[pair] += 1
                for option_id in pair:
                    self.state.coverage[option_id].add(action.speaker_id, action.act)
            else:
                # Accept useful one-sided realization, but never claim that a
                # comparison became public when both options were not visible.
                for option_id in pair:
                    self.state.coverage[option_id].add(action.speaker_id, ActionType.COMMENT)
        if action.act is ActionType.VOTE:
            self.state.votes[action.speaker_id] = action.vote_option

    def _apply_issue_before_turn(self, action: UserAction, text: str) -> str | None:
        if action.issue_effect is IssueEffect.OPEN:
            if action.act is ActionType.ASK:
                return self._open_issue(IssueKind.QUESTION, action, text)
            if action.act is ActionType.CONCERN:
                return self._open_issue(IssueKind.CONCERN, action, text)
        return None

    def _apply_issue_after_turn(self, action: UserAction) -> None:
        issue = self.state.active_issue
        if not issue or action.issue_id != issue.id:
            return
        issue.last_relevant_turn = len(self.state.turns) - 1
        issue.follow_up_count += 1
        if action.speaker_id != issue.opened_by:
            issue.response_count += 1
            issue.responded_by.add(action.speaker_id)
        else:
            issue.owner_reacted = True

        if issue.kind is IssueKind.QUESTION and action.act is ActionType.ANSWER:
            self.state.response_obligation = None
            issue.required_answer_completed = True
            issue.outcome = "answered"
            if int(cfg.conversation.direct_question_optional_follow_up_cap) <= 0:
                self._close_active_issue(IssueStatus.RESOLVED, "direct question answered")
            return

        if issue.kind is IssueKind.QUESTION and issue.required_answer_completed:
            issue.optional_follow_up_count += 1
            issue.outcome = "answered_with_follow_up"
            if issue.optional_follow_up_count >= int(
                cfg.conversation.direct_question_optional_follow_up_cap
            ):
                self._close_active_issue(
                    IssueStatus.RESOLVED,
                    "direct question answered with voluntary follow-up",
                )
            return

        if action.issue_effect is IssueEffect.RESOLVE:
            issue.outcome = "resolved"
            self._close_active_issue(IssueStatus.RESOLVED, "owner visibly accepted the response")
            return
        if action.issue_effect in {IssueEffect.MAINTAIN, IssueEffect.PARTIAL}:
            issue.outcome = "maintained" if action.issue_effect is IssueEffect.MAINTAIN else "partial"
            self._close_active_issue(IssueStatus.STALE, "owner completed the concern reaction")
            return
        if self.state.active_issue and issue.follow_up_count >= int(cfg.conversation.issue_follow_up_cap):
            self._close_or_stale_inactive_issue("issue follow-up cap reached")

    def _open_issue(self, kind: IssueKind, action: UserAction, text: str) -> str:
        if self.state.active_issue:
            self._stale_active_issue("replaced by a new issue")
        self._issue_counter += 1
        issue_id = f"i{self._issue_counter:03d}"
        summary = action.reason.strip() or text.strip()
        option_id = action.option_focus[0] if action.option_focus else "group"
        semantic_issue = (
            action.reason_source.public_value
            if action.reason_source is not None
            else summary
        )
        key = (option_id, self._normalize_issue(semantic_issue))
        issue = ActiveIssue(
            id=issue_id,
            kind=kind,
            option_focus=action.option_focus,
            opened_by=action.speaker_id,
            addressed_to=action.addressee_id,
            summary=summary,
            status=IssueStatus.OPEN,
            opened_at_turn=len(self.state.turns),
            last_relevant_turn=len(self.state.turns),
            source_text=text,
            reason_source=action.reason_source,
            issue_key=key,
            question_mode=action.question_mode,
        )
        self.state.active_issue = issue
        record = self.state.issue_records.get(key)
        if record is None:
            record = IssueRecord(
                key=key,
                kind=kind,
                status=IssueStatus.OPEN,
                last_issue_id=issue_id,
                last_relevant_turn=len(self.state.turns),
            )
            self.state.issue_records[key] = record
        else:
            if kind is IssueKind.CONCERN and record.kind is IssueKind.CONCERN:
                record.reopen_count += 1
            record.kind = kind
            record.status = IssueStatus.OPEN
            record.last_issue_id = issue_id
            record.last_relevant_turn = len(self.state.turns)
        if kind is IssueKind.QUESTION and action.addressee_id:
            self.state.response_obligation = action.addressee_id
        runtime = self.state.runtimes[action.speaker_id]
        if kind is IssueKind.CONCERN:
            runtime.opened_issue_keys.add(f"concern:{reason_key(action)}")
        if kind is IssueKind.QUESTION:
            runtime.asked_question_keys.add(reason_key(action))
            self.state.asked_public_question_keys.add(public_question_key(action))
        return f"opened:{issue_id}"

    def _close_active_issue(self, status: IssueStatus, reason: str) -> None:
        issue = self.state.active_issue
        if not issue:
            return
        issue.status = status
        issue.close_reason = reason
        issue.last_relevant_turn = len(self.state.turns) - 1
        self.state.issue_history.append(issue)
        if issue.issue_key:
            record = self.state.issue_records.setdefault(
                issue.issue_key,
                IssueRecord(issue.issue_key, kind=issue.kind),
            )
            record.kind = issue.kind
            record.status = status
            record.last_issue_id = issue.id
            record.last_relevant_turn = issue.last_relevant_turn
            record.last_closed_turn = issue.last_relevant_turn
            record.outcome = issue.outcome or status.value
        self.state.active_issue = None
        self.state.response_obligation = None

    def _close_or_stale_inactive_issue(self, reason: str) -> None:
        """Finish an answered question; otherwise preserve unresolved issues as stale."""

        issue = self.state.active_issue
        if issue is None:
            return
        if issue.kind is IssueKind.QUESTION and issue.required_answer_completed:
            issue.outcome = issue.outcome or "answered"
            self._close_active_issue(IssueStatus.RESOLVED, reason)
            return
        self._stale_active_issue(reason)

    def _stale_active_issue(self, reason: str) -> None:
        self._close_active_issue(IssueStatus.STALE, reason)

    def _set_group_stimulus(self, kind: StimulusKind, option_focus: tuple[str, ...], text: str) -> None:
        self._stimulus_counter += 1
        self.state.group_stimulus = GroupStimulus(self._stimulus_counter, kind, option_focus, text, len(self.state.turns))

    def _coverage_prompt_needed(self) -> bool:
        return bool(self._moderator_enabled and not self.state.coverage_prompt_used and self._uncovered_options())

    def _uncovered_options(self) -> list[str]:
        return [
            option_id for option_id, coverage in self.state.coverage.items()
            if coverage.substantive_count == 0 and option_id not in self.state.coverage_no_interest
        ]

    def _publicly_converged(self) -> bool:
        preferences = [runtime.public_preference for runtime in self.state.runtimes.values()]
        return bool(preferences and all(preference == preferences[0] and preference is not None for preference in preferences))

    def _public_unanimity_ready_for_vote(self, voluntary_count: int) -> bool:
        """Allow genuine public agreement to close before liveness filler.

        Openings alone are not enough. The group must have completed roughly
        one post-opening contribution round and have no pending local issue.
        """

        return bool(
            voluntary_count >= max(1, len(self.state.personas))
            and self.state.active_issue is None
            and self.state.response_obligation is None
            and self._publicly_converged()
        )

    def _public_compromise_options(self) -> tuple[str, ...]:
        """Options publicly preferred or accepted by at least one participant."""

        visible: set[str] = set()
        for runtime in self.state.runtimes.values():
            if runtime.public_preference in self.state.scenario.option_ids:
                visible.add(runtime.public_preference)
            visible.update(
                option_id
                for option_id in runtime.public_acceptances
                if option_id in self.state.scenario.option_ids
            )
        return tuple(
            option_id
            for option_id in self.state.scenario.option_ids
            if option_id in visible
        )

    def _public_position_support_count(self, option_id: str) -> int:
        return sum(
            runtime.public_preference == option_id
            or option_id in runtime.public_acceptances
            for runtime in self.state.runtimes.values()
        )

    def _shared_acceptable_option(self) -> str | None:
        """Return public common ground without converting supports into a score."""
        for option_id in self.state.scenario.option_ids:
            if all(
                runtime.public_preference == option_id
                or option_id in runtime.public_acceptances
                for runtime in self.state.runtimes.values()
            ):
                return option_id
        return None

    def _phase_voluntary_count(self, phase: Phase) -> int:
        return sum(1 for turn in self.state.turns if turn.phase is phase and turn.voluntary)

    def _transition(self, phase: Phase) -> None:
        self.state.phase = phase
        self.state.phase_history.append(phase.value)

    def _append_moderator(self, text: str) -> None:
        record = TurnRecord(
            index=len(self.state.turns),
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
    def _validation_category(error: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", error.casefold()).strip("_")[:80]

    @staticmethod
    def _normalize_issue(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", text.casefold())
        return " ".join(words[:12])

    @staticmethod
    def _print_header(state: DialogueState) -> None:
        print("=" * 72)
        print(f"Topic: {state.scenario.topic}")
        if state.scenario.context_text:
            print(f"Scenario context: {state.scenario.context_text}")
        print("Options:")
        for option in state.scenario.options:
            print(option.public_line())
        print("Participants: " + ", ".join(persona.name for persona in state.personas))
        print("=" * 72)


def initialise_state(scenario: Scenario, personas: list[Persona]) -> DialogueState:
    runtimes = {persona.id: initial_runtime(persona, scenario.option_ids) for persona in personas}
    return DialogueState(
        scenario=scenario,
        personas=personas,
        runtimes=runtimes,
        coverage={option_id: OptionCoverage() for option_id in scenario.option_ids},
        public_supporters={option_id: set() for option_id in scenario.option_ids},
        public_concern_raisers={option_id: set() for option_id in scenario.option_ids},
    )
