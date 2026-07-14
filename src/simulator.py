"""Participant-local, seeded simulator policy and floor arbitration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from config_loader import cfg
from models import (
    ActionType,
    ActiveIssue,
    DialogueState,
    GroupStimulus,
    IssueEffect,
    IssueKind,
    IssueResponseKind,
    ParticipantRuntime,
    Persona,
    Phase,
    QuestionIntent,
    ReasonSource,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
    StanceUpdate,
    StanceUpdateKind,
    StimulusKind,
    SwitchDecision,
    UserAction,
)


ENGAGEMENT_BID_PROBABILITY = {1: 0.18, 2: 0.31, 3: 0.47, 4: 0.65, 5: 0.82}
SWITCH_MIN_ACCEPTED_TURN_DISTANCE = 3
SWITCH_FIRST_MARGIN = 0.05
SWITCH_REPEAT_MARGIN = 0.10


def initial_runtime(persona: Persona, option_ids: Iterable[str]) -> ParticipantRuntime:
    ranks = {
        option_id: int(persona.option_stances.get(option_id).rank)
        if option_id in persona.option_stances else STANCE_NEUTRAL
        for option_id in option_ids
    }
    preferred = persona.preferred_option
    ranks[preferred] = STANCE_PREFERRED
    if persona.hard_blocker:
        for option_id in option_ids:
            ranks[option_id] = STANCE_PREFERRED if option_id == preferred else STANCE_REJECTED
    return ParticipantRuntime(
        persona_id=persona.id,
        preferred_option=preferred,
        ranks=ranks,
        acceptable_options={oid for oid, rank in ranks.items() if rank == STANCE_ACCEPTABLE},
        disliked_options={oid for oid, rank in ranks.items() if rank == STANCE_DISLIKED},
        hard_rejected_options={oid for oid, rank in ranks.items() if rank == STANCE_REJECTED},
    )


def switch_probability(stubbornness: int, evidence_strength: float, *, hard_blocker: bool = False) -> float:
    """Probability of voluntarily accepting a public alternative.

    Evidence is normalized to 0..1. Stubbornness monotonically reduces the
    result. Hard blockers have exactly zero probability.
    """
    if hard_blocker or stubbornness >= 5:
        return 0.0
    resistance = {1: 0.08, 2: 0.25, 3: 0.48, 4: 0.72}[int(stubbornness)]
    return max(0.01, min(0.88, float(evidence_strength) * (1.0 - resistance)))


def action_signature(action: UserAction) -> str:
    """Normalized semantic signature used for policy-level repetition control."""
    focus = tuple(sorted(action.option_focus)) if action.act is ActionType.COMPARE else tuple(action.option_focus)
    if action.reason_source:
        reason = (
            action.reason_source.option_id,
            action.reason_source.attribute_name,
            action.reason_source.public_value.casefold(),
        )
    else:
        reason = (action.reason.casefold().strip(),)
    stance = (
        action.stance_update.kind.value,
        action.stance_update.option_id,
    ) if action.stance_update else None
    # Issue effect is deliberately omitted: reopening, maintaining, or resolving
    # the same semantic point must not evade the structured cooldown merely by
    # changing its lifecycle label. Stance updates remain distinct.
    return repr((action.act.value, focus, reason, action.issue_id, stance, action.stimulus_id))



def action_cooldown_context(
    state: DialogueState,
    runtime: ParticipantRuntime,
    action: UserAction,
) -> str:
    """Context key that resets a structured-action cooldown.

    A phase change, stance change, active-issue change, or new public concern
    about a focused option makes an otherwise repeated contribution relevant
    again. Ordinary support-count growth alone does not reset the cooldown.
    """
    focus = set(action.option_focus)
    # Only new public evidence from another participant resets the cooldown.
    # The speaker's own repeated support/concern must not reset itself through
    # aggregate counters.
    external_relevant_turn = max((
        turn.index
        for turn in state.participant_turns
        if turn.speaker_id != runtime.persona_id
        and turn.action is not None
        and bool(focus & set(turn.action.option_focus))
    ), default=-1)
    return repr((
        state.phase.value,
        state.active_issue.id if state.active_issue else None,
        runtime.preferred_option,
        tuple(sorted(runtime.acceptable_options)),
        external_relevant_turn,
    ))

def _weighted_choice(rng: random.Random, items: list, weights: list[float]):
    if not items:
        raise ValueError("cannot choose from an empty list")
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return items[0]
    point = rng.random() * total
    cursor = 0.0
    for item, weight in zip(items, weights):
        cursor += max(0.0, weight)
        if point <= cursor:
            return item
    return items[-1]


@dataclass(slots=True)
class FloorSelection:
    action: UserAction
    candidate_count: int


class FloorManager:
    """Selects a complete participant bid without rewriting it."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def select(self, state: DialogueState, bids: list[UserAction]) -> FloorSelection | None:
        eligible = [bid for bid in bids if bid.wants_to_speak]
        if not eligible:
            return None
        max_consecutive = int(cfg.conversation.max_consecutive_turns)
        non_capped = [
            bid for bid in eligible
            if state.consecutive_turns_by(bid.speaker_id) < max_consecutive
        ]
        if non_capped:
            eligible = non_capped
        last = state.last_participant_id
        weights: list[float] = []
        for bid in eligible:
            weight = max(0.01, float(bid.urgency))
            if bid.speaker_id == last:
                weight *= 0.72
            if state.active_issue and bid.issue_id == state.active_issue.id:
                weight *= 1.08
                if (
                    state.active_issue.kind is IssueKind.CONCERN
                    and state.active_issue.follow_up_count > 0
                    and bid.speaker_id == state.active_issue.opened_by
                ):
                    weight *= 1.35
            if state.group_stimulus and bid.stimulus_id == state.group_stimulus.id:
                weight *= 1.06
            weights.append(weight)
        return FloorSelection(
            action=_weighted_choice(self.rng, eligible, weights),
            candidate_count=len(eligible),
        )


class UserSimulator:
    """Seeded Python policy owned by one simulated user."""

    def __init__(self, persona: Persona, rng: random.Random) -> None:
        self.persona = persona
        self.rng = rng

    @property
    def id(self) -> str:
        return self.persona.id

    def opening_action(self, state: DialogueState) -> UserAction:
        runtime = state.runtimes[self.id]
        reason, source = self._positive_reason(state, runtime.preferred_option)
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            urgency=1.0,
            act=ActionType.OPENING,
            option_focus=(runtime.preferred_option,),
            reason=reason,
            reason_source=source,
            personal_context=self._optional_personal_context(0.45),
        )

    def propose(
        self,
        state: DialogueState,
        *,
        mandatory_answer: bool = False,
        liveness_forced: bool = False,
    ) -> UserAction:
        runtime = state.runtimes[self.id]
        if mandatory_answer:
            return self._answer_action(state, runtime, urgency=1.0)

        candidates = self._filter_repeated_candidates(state, runtime, self._candidate_actions(state, runtime))
        if not candidates:
            return self._silence()

        relevant = bool(
            state.active_issue
            and any(candidate.issue_id == state.active_issue.id for candidate in candidates)
        )
        stimulus_relevant = bool(
            state.group_stimulus
            and any(candidate.stimulus_id == state.group_stimulus.id for candidate in candidates)
        )
        concern_owner_followup = bool(
            state.active_issue
            and state.active_issue.kind is IssueKind.CONCERN
            and state.active_issue.opened_by == self.id
            and state.active_issue.follow_up_count > 0
            and any(candidate.issue_id == state.active_issue.id for candidate in candidates)
        )
        bid_probability = ENGAGEMENT_BID_PROBABILITY[self.persona.sim_params.engagement]
        if relevant:
            bid_probability = min(0.95, bid_probability + 0.14)
        if concern_owner_followup:
            bid_probability = max(0.94, bid_probability)
        if stimulus_relevant:
            bid_probability = min(0.93, bid_probability + 0.10)
        if state.phase == Phase.NARROWING:
            bid_probability = min(0.94, bid_probability + 0.08)
        if liveness_forced:
            bid_probability = 1.0
        if self.rng.random() > bid_probability:
            return self._silence()

        weights = [self._candidate_weight(state, runtime, action) for action in candidates]
        selected = _weighted_choice(self.rng, candidates, weights)
        selected.wants_to_speak = True
        selected.urgency = min(
            1.0,
            selected.urgency + 0.05 * (self.persona.sim_params.engagement - 3),
        )
        return selected

    def decide_vote(self, state: DialogueState, *, revote: bool = False) -> UserAction:
        runtime = state.runtimes[self.id]
        if self.persona.hard_blocker:
            return self._vote_action(
                state,
                runtime.preferred_option,
                runtime,
                reason=self.persona.rejection_reason or "This remains my only acceptable option",
            )

        allowed = [
            option_id for option_id in state.scenario.option_ids
            if option_id not in runtime.hard_rejected_options
        ] or [runtime.preferred_option]
        scores = {
            option_id: self._vote_score(state, runtime, option_id, revote=revote)
            for option_id in allowed
        }
        current = runtime.preferred_option
        best = max(scores, key=scores.get)
        choice = current if current in allowed else best
        if best != current and runtime.rank(best) >= STANCE_NEUTRAL:
            advantage = max(0.0, scores[best] - scores.get(current, 0.0))
            evidence = min(
                1.0,
                self._public_evidence_strength(state, best)
                + 0.30 * advantage
                + (0.18 if best in runtime.public_acceptances else 0.0),
            )
            probability = switch_probability(
                self.persona.sim_params.stubbornness,
                evidence,
                hard_blocker=False,
            )
            gate_open = self._record_switch_opportunity(
                state,
                runtime,
                best,
                target_evidence=evidence,
                current_evidence=self._public_evidence_strength(state, current),
                probability=probability,
            )
            if gate_open and self.rng.random() < probability:
                choice = best
        reason, _source = self._positive_reason(state, choice)
        return self._vote_action(state, choice, runtime, reason=reason)

    def _vote_action(
        self,
        state: DialogueState,
        choice: str,
        runtime: ParticipantRuntime,
        *,
        reason: str,
    ) -> UserAction:
        update = None
        if choice != runtime.preferred_option:
            update = StanceUpdate(
                StanceUpdateKind.SWITCH_PREFERRED,
                choice,
                previous_option_id=runtime.preferred_option,
            )
        public_old = runtime.public_preference
        bridge = reason
        if public_old and public_old != choice:
            bridge = f"Public discussion changed the balance from {public_old}: {reason}"
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            urgency=1.0,
            act=ActionType.VOTE,
            option_focus=(choice,),
            reason=bridge,
            stance_update=update,
            vote_option=choice,
        )

    def _candidate_actions(self, state: DialogueState, runtime: ParticipantRuntime) -> list[UserAction]:
        actions: list[UserAction] = []
        if state.active_issue:
            issue_actions = self._issue_actions(state, runtime, state.active_issue)
            if (
                state.active_issue.kind is IssueKind.CONCERN
                and state.active_issue.opened_by == self.id
                and state.active_issue.follow_up_count > 0
            ):
                # Once others have responded, the concern owner evaluates that
                # response before opening an unrelated topic. The simulator still
                # chooses autonomously among maintain/partial/resolve actions.
                return issue_actions
            actions.extend(issue_actions)
        if state.group_stimulus and state.group_stimulus.id not in runtime.responded_stimuli:
            stimulus_action = self._stimulus_action(state, runtime, state.group_stimulus)
            if stimulus_action:
                actions.append(stimulus_action)
        if state.phase == Phase.NARROWING:
            actions.extend(self._narrowing_actions(state, runtime))
            return actions

        public_preferences = [item.public_preference for item in state.runtimes.values()]
        converged = bool(
            public_preferences
            and None not in public_preferences
            and len(set(public_preferences)) == 1
        )
        # After one real discussion contribution has confirmed unanimous
        # openings, the phase coordinator can narrow immediately. Suppress
        # redundant ordinary bids while preserving issue/stimulus responses.
        if converged and self._public_convergence_confirmed(state):
            return actions
        # Before that confirmation, allow one concern or concise confirmation.
        if not converged:
            concern = self._new_concern_action(state, runtime)
            if concern:
                actions.append(concern)
        support = self._support_action(state, runtime)
        if support:
            actions.append(support)
        if not converged:
            compare = self._compare_action(state, runtime)
            if compare:
                actions.append(compare)
            ask = self._ask_action(state, runtime)
            if ask:
                actions.append(ask)
        acknowledge = self._acknowledge_action(state, runtime)
        if acknowledge:
            actions.append(acknowledge)
        return actions

    @staticmethod
    def _public_convergence_confirmed(state: DialogueState) -> bool:
        preferences = [runtime.public_preference for runtime in state.runtimes.values()]
        if not preferences or None in preferences or len(set(preferences)) != 1:
            return False
        shared = preferences[0]
        return any(
            turn.phase is Phase.DISCUSSION
            and turn.voluntary
            and turn.action is not None
            and shared in turn.action.option_focus
            for turn in state.participant_turns
        )

    def _filter_repeated_candidates(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        candidates: list[UserAction],
    ) -> list[UserAction]:
        result: list[UserAction] = []
        for action in candidates:
            signature = action_signature(action)
            count = runtime.action_signature_counts[signature]
            stored_context = runtime.action_signature_contexts.get(signature)
            same_context = stored_context is None or stored_context == action_cooldown_context(
                state, runtime, action
            )
            limit = 2 if action.issue_id or action.stance_update else 1
            if action.act in {ActionType.ACKNOWLEDGE, ActionType.COMPARE, ActionType.SUPPORT, ActionType.ASK}:
                limit = 1
            if not same_context or count < limit:
                result.append(action)
            else:
                state.stats.suppressed_repetitions += 1
        return result

    def _issue_actions(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        issue: ActiveIssue,
    ) -> list[UserAction]:
        if issue.addressed_to == self.id and state.response_obligation == self.id:
            return [self._answer_action(state, runtime, urgency=1.0)]
        if issue.kind is IssueKind.CONCERN and issue.opened_by == self.id:
            if issue.follow_up_count > issue.owner_last_evaluated_follow_up_count:
                return self._concern_owner_reactions(state, runtime, issue)
            return []
        normal_follow_ups = int(cfg.conversation.issue_normal_follow_ups)
        if issue.follow_up_count >= normal_follow_ups and self.rng.random() < 0.65:
            return []
        relevant = any(
            option_id == runtime.preferred_option
            or option_id in runtime.acceptable_options
            or option_id in runtime.disliked_options
            or option_id in runtime.hard_rejected_options
            for option_id in issue.option_focus
        )
        if not relevant:
            return []

        actions: list[UserAction] = []
        if issue.kind is IssueKind.QUESTION:
            if issue.opened_by != self.id and not issue.answered:
                actions.append(self._answer_action(state, runtime, urgency=0.78))
            elif issue.answered and issue.opened_by != self.id and issue.follow_up_count < 2:
                focus = issue.option_focus or (runtime.preferred_option,)
                best = max(focus, key=runtime.rank)
                reason, source = self._positive_reason(state, best)
                actions.append(UserAction(
                    self.id, True, 0.48, ActionType.COMMENT, tuple(focus),
                    reason=reason, reason_source=source, issue_id=issue.id,
                    issue_effect=IssueEffect.CONTINUE,
                ))
        elif issue.kind is IssueKind.CONCERN:
            actions.extend(self._concern_response_actions(state, runtime, issue))
        elif issue.kind is IssueKind.COMPARISON and len(issue.option_focus) >= 2:
            preferred = max(issue.option_focus, key=runtime.rank)
            other = next(option_id for option_id in issue.option_focus if option_id != preferred)
            reason, source = self._positive_reason(state, preferred)
            actions.append(UserAction(
                self.id, True, 0.70, ActionType.COMPARE, (preferred, other),
                reason=f"The trade-off still favors {preferred}: {reason}",
                reason_source=source, issue_id=issue.id, issue_effect=IssueEffect.CONTINUE,
            ))
        return actions

    def _concern_response_actions(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        issue: ActiveIssue,
    ) -> list[UserAction]:
        """Construct only responses that are visibly relevant to a concern."""
        focus = issue.option_focus[0] if issue.option_focus else runtime.preferred_option
        actions: list[UserAction] = []
        rank = runtime.rank(focus)
        if focus == runtime.preferred_option or rank >= STANCE_ACCEPTABLE:
            mitigation = self._matching_issue_reason_source(state, issue, focus)
            if mitigation is not None:
                attribute = mitigation.attribute_name.replace("_", " ")
                actions.append(UserAction(
                    self.id, True, 0.86, ActionType.SUPPORT, (focus,),
                    addressee_id=issue.opened_by,
                    reason=(
                        f"The public {attribute} information directly limits or clarifies "
                        "the concern"
                    ),
                    reason_source=mitigation, issue_id=issue.id,
                    issue_effect=IssueEffect.CONTINUE,
                    issue_response_kind=IssueResponseKind.MITIGATION,
                ))
            reason, source = self._positive_reason(state, focus)
            if not issue.reason_source or source != issue.reason_source:
                actions.append(UserAction(
                    self.id, True, 0.70, ActionType.SUPPORT, (focus,),
                    addressee_id=issue.opened_by,
                    reason=f"The concern is real, but {reason} matters more to me",
                    reason_source=source, issue_id=issue.id,
                    issue_effect=IssueEffect.CONTINUE,
                    issue_response_kind=IssueResponseKind.TRADE_OFF,
                ))
        elif rank <= STANCE_DISLIKED:
            reason, source = self._negative_reason(state, focus)
            actions.append(UserAction(
                self.id, True, 0.68, ActionType.CONCERN, (focus,),
                addressee_id=issue.opened_by,
                reason=f"I agree that the concern remains relevant: {reason}",
                reason_source=source, issue_id=issue.id,
                issue_effect=IssueEffect.CONTINUE,
                issue_response_kind=IssueResponseKind.AGREEMENT,
            ))
        return actions

    def _matching_issue_reason_source(
        self,
        state: DialogueState,
        issue: ActiveIssue,
        option_id: str,
    ) -> ReasonSource | None:
        source = issue.reason_source
        if source is None or source.option_id != option_id:
            return None
        option = state.scenario.option(option_id)
        if source.attribute_name in option.attrs:
            return ReasonSource(option_id, source.attribute_name, option.attrs[source.attribute_name])
        if source.attribute_name in {"upside", "concern"}:
            issue_words = {
                token for token in ''.join(
                    char if char.isalnum() else ' ' for char in source.public_value.casefold()
                ).split() if len(token) >= 4
            }
            for key, value in option.attrs.items():
                haystack = f"{key.replace('_', ' ')} {value}".casefold()
                if any(word in haystack for word in issue_words):
                    return ReasonSource(option_id, key, value)
        return None

    def _concern_owner_reactions(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        issue: ActiveIssue,
    ) -> list[UserAction]:
        focus = issue.option_focus[0] if issue.option_focus else runtime.preferred_option
        if self.persona.hard_blocker or focus in runtime.hard_rejected_options:
            return [UserAction(
                self.id, True, 0.90, ActionType.CONCERN, (focus,),
                reason=self.persona.rejection_reason or "The concern remains non-negotiable",
                issue_id=issue.id, issue_effect=IssueEffect.MAINTAIN,
            )]

        distinct = len(issue.relevant_responder_ids)
        response_strength = min(
            1.0,
            0.28 * distinct
            + (0.32 if issue.same_attribute_mitigation else 0.0)
            + (0.10 if issue.relevant_response_kinds[IssueResponseKind.TRADE_OFF.value] else 0.0),
        )
        stubbornness = self.persona.sim_params.stubbornness
        movement = switch_probability(stubbornness, response_strength)
        actions = [UserAction(
            self.id, True, 0.44 + 0.10 * stubbornness, ActionType.CONCERN,
            (focus,), reason="The concern still matters to my decision",
            issue_id=issue.id, issue_effect=IssueEffect.MAINTAIN,
        )]
        if stubbornness >= 4:
            return actions

        if response_strength >= 0.25:
            actions.append(UserAction(
                self.id, True, 0.66 + 0.24 * response_strength, ActionType.ACKNOWLEDGE,
                (focus,), reason="The relevant response helps, but does not fully solve the concern",
                issue_id=issue.id, issue_effect=IssueEffect.PARTIAL,
            ))
        if response_strength >= 0.55:
            update = None
            if focus != runtime.preferred_option and focus not in runtime.acceptable_options:
                update = StanceUpdate(StanceUpdateKind.MAKE_ACCEPTABLE, focus)
            actions.append(UserAction(
                self.id, True, 0.80 + 0.20 * response_strength, ActionType.COMPROMISE,
                (focus,), reason="The relevant responses address enough of my concern",
                issue_id=issue.id, issue_effect=IssueEffect.RESOLVE,
                stance_update=update,
            ))
            if (
                focus != runtime.preferred_option
                and runtime.rank(focus) >= STANCE_NEUTRAL
                and stubbornness <= 3
                and response_strength >= 0.72
            ):
                evidence = self._public_evidence_strength(state, focus)
                probability = switch_probability(stubbornness, max(evidence, response_strength))
                if self._record_switch_opportunity(
                    state, runtime, focus,
                    target_evidence=max(evidence, response_strength),
                    current_evidence=self._public_evidence_strength(state, runtime.preferred_option),
                    probability=probability,
                ):
                    reason, source = self._positive_reason(state, focus)
                    actions.append(UserAction(
                        self.id, True, 0.58 + 0.68 * movement, ActionType.COMPROMISE,
                        (runtime.preferred_option, focus), reason=reason, reason_source=source,
                        issue_id=issue.id, issue_effect=IssueEffect.RESOLVE,
                        stance_update=StanceUpdate(
                            StanceUpdateKind.SWITCH_PREFERRED,
                            focus,
                            previous_option_id=runtime.preferred_option,
                        ),
                    ))
        if response_strength >= 0.80 and stubbornness <= 2:
            return [
                action for action in actions
                if action.issue_effect is IssueEffect.RESOLVE
            ]
        return actions

    def _answer_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        *,
        urgency: float,
    ) -> UserAction:
        issue = state.active_issue
        focus = issue.option_focus if issue and issue.option_focus else (runtime.preferred_option,)
        best = max(focus, key=runtime.rank)
        if runtime.rank(best) >= STANCE_NEUTRAL:
            reason, source = self._positive_reason(state, best)
        else:
            reason, source = self._negative_reason(state, best)
        return UserAction(
            self.id, True, urgency, ActionType.ANSWER, tuple(focus),
            addressee_id=issue.opened_by if issue else None,
            reason=reason, reason_source=source,
            issue_id=issue.id if issue else None,
            issue_effect=IssueEffect.ANSWERED if issue else None,
        )

    def _stimulus_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        stimulus: GroupStimulus,
    ) -> UserAction | None:
        if stimulus.kind is StimulusKind.COVERAGE and stimulus.option_focus:
            focus = stimulus.option_focus[0]
            rank = runtime.rank(focus)
            if rank >= STANCE_ACCEPTABLE or focus == runtime.preferred_option:
                reason, source = self._positive_reason(state, focus)
                return UserAction(
                    self.id, True, 0.76, ActionType.SUPPORT, (focus,),
                    reason=reason, reason_source=source, stimulus_id=stimulus.id,
                )
            if rank <= STANCE_DISLIKED:
                reason, source = self._negative_reason(state, focus)
                return UserAction(
                    self.id, True, 0.72, ActionType.CONCERN, (focus,),
                    reason=reason, reason_source=source, stimulus_id=stimulus.id,
                )
            preferred = runtime.preferred_option
            reason, source = self._positive_reason(state, preferred)
            return UserAction(
                self.id, True, 0.58, ActionType.COMPARE, (preferred, focus),
                reason=f"Compare whether {focus} can displace {preferred}: {reason}",
                reason_source=source, stimulus_id=stimulus.id,
            )
        if stimulus.kind is StimulusKind.STALL:
            support = self._support_action(state, runtime)
            if support:
                support.stimulus_id = stimulus.id
                support.urgency = max(support.urgency, 0.58)
                return support
        return None

    def _new_concern_action(self, state: DialogueState, runtime: ParticipantRuntime) -> UserAction | None:
        if state.active_issue:
            return None
        candidates = [
            option_id for option_id in state.scenario.option_ids
            if runtime.rank(option_id) <= STANCE_DISLIKED
            and f"concern:{option_id}" not in runtime.opened_issue_keys
        ]
        if not candidates:
            return None
        option_id = min(candidates, key=runtime.rank)
        reason, source = self._negative_reason(state, option_id)
        return UserAction(
            self.id, True, 0.62 + 0.06 * self.persona.sim_params.stubbornness,
            ActionType.CONCERN, (option_id,), reason=reason, reason_source=source,
            personal_context=self._optional_personal_context(0.25),
            issue_effect=IssueEffect.OPEN,
        )

    def _support_action(self, state: DialogueState, runtime: ParticipantRuntime) -> UserAction | None:
        option_id = runtime.preferred_option
        selected = self._unused_reason(
            state, runtime, ActionType.SUPPORT, option_id, positive=True
        )
        challenged = bool(
            state.active_issue
            and state.active_issue.kind is IssueKind.CONCERN
            and option_id in state.active_issue.option_focus
        )
        if selected is None and challenged:
            selected = self._positive_reason(state, option_id)
        if selected is None:
            return None
        reason, source = selected
        return UserAction(
            self.id, True, 0.55 + 0.04 * self.persona.sim_params.stubbornness,
            ActionType.SUPPORT, (option_id,), reason=reason, reason_source=source,
            personal_context=self._optional_personal_context(0.18),
        )

    def _compare_action(self, state: DialogueState, runtime: ParticipantRuntime) -> UserAction | None:
        if state.active_issue:
            return None
        alternatives = [oid for oid in state.scenario.option_ids if oid != runtime.preferred_option]
        if not alternatives:
            return None
        other = max(
            alternatives,
            key=lambda oid: (
                len(state.public_supporters.get(oid, set())),
                runtime.rank(oid),
            ),
        )
        pair = (runtime.preferred_option, other)
        selected = self._unused_reason(
            state, runtime, ActionType.COMPARE, runtime.preferred_option, positive=True
        )
        if runtime.last_action is ActionType.COMPARE or selected is None:
            return None
        reason, source = selected
        return UserAction(
            self.id, True, 0.48, ActionType.COMPARE, pair,
            reason=f"The trade-off favors {runtime.preferred_option}: {reason}",
            reason_source=source,
        )

    def _ask_action(self, state: DialogueState, runtime: ParticipantRuntime) -> UserAction | None:
        if state.active_issue or runtime.last_action is ActionType.ASK:
            return None
        alternatives = [
            option_id for option_id in state.scenario.option_ids
            if option_id != runtime.preferred_option
        ]
        candidates: list[UserAction] = []

        # A recent visible claim can justify a clarification question even when
        # the option has already been discussed. Uniqueness is governed by the
        # structured question key, not by a broad per-option lock.
        for turn in reversed(state.participant_turns):
            if turn.speaker_id == self.id or turn.action is None or not turn.action.option_focus:
                continue
            if turn.action.act in {ActionType.ASK, ActionType.VOTE, ActionType.ACKNOWLEDGE}:
                continue
            focus = tuple(turn.action.option_focus)
            candidates.append(self._question_action(
                QuestionIntent.CLARIFICATION,
                focus,
                turn.speaker_id,
                f"Ask what the visible point about {', '.join(focus)} means for that participant's choice",
            ))
            break
        for focus in alternatives:
            supporters = sorted(
                pid for pid in state.public_supporters.get(focus, set()) if pid != self.id
            )
            preference_holders = sorted(
                pid for pid, other in state.runtimes.items()
                if pid != self.id and other.public_preference == focus
            )
            rationale_targets = supporters or preference_holders
            if rationale_targets:
                addressee = rationale_targets[0]
                candidates.append(self._question_action(
                    QuestionIntent.RATIONALE, (focus,), addressee,
                    f"Ask why {focus} fits that participant's priority",
                ))

            concern_targets = sorted(
                pid for pid in state.public_concern_raisers.get(focus, set()) if pid != self.id
            )
            if concern_targets:
                addressee = concern_targets[0]
                candidates.append(self._question_action(
                    QuestionIntent.IMPACT, (focus,), addressee,
                    f"Ask how the visible concern about {focus} affects that participant's decision",
                ))

            differing = sorted(
                pid for pid, other in state.runtimes.items()
                if pid != self.id
                and other.public_preference not in {None, focus}
            )
            if differing:
                addressee = differing[0]
                candidates.append(self._question_action(
                    QuestionIntent.ACCEPTABILITY, (focus,), addressee,
                    f"Ask whether that participant could accept {focus} under the current trade-off",
                ))

        # A comparison question is useful when no single-option public signal
        # identifies a stronger information need.
        if alternatives:
            other = max(
                alternatives,
                key=lambda oid: (len(state.public_supporters.get(oid, set())), runtime.rank(oid)),
            )
            differing = sorted(
                pid for pid, other_runtime in state.runtimes.items()
                if pid != self.id and other_runtime.public_preference == other
            )
            candidates.append(self._question_action(
                QuestionIntent.COMPARISON, (runtime.preferred_option, other),
                differing[0] if differing else None,
                f"Ask which trade-off between {runtime.preferred_option} and {other} matters more",
            ))

        for action in candidates:
            if action.question_key not in runtime.asked_question_keys:
                return action
        return None

    def _question_action(
        self,
        intent: QuestionIntent,
        option_focus: tuple[str, ...],
        addressee_id: str | None,
        objective: str,
    ) -> UserAction:
        key = "|".join((intent.value, ",".join(option_focus), addressee_id or "group"))
        return UserAction(
            self.id, True, 0.44, ActionType.ASK, option_focus,
            addressee_id=addressee_id, reason=objective,
            issue_effect=IssueEffect.OPEN, question_intent=intent, question_key=key,
        )

    def _acknowledge_action(self, state: DialogueState, runtime: ParticipantRuntime) -> UserAction | None:
        if not state.participant_turns or runtime.last_action is ActionType.ACKNOWLEDGE:
            return None
        latest = state.participant_turns[-1]
        if latest.speaker_id == self.id:
            return None
        return UserAction(
            self.id, True, 0.25, ActionType.ACKNOWLEDGE,
            tuple(latest.action.option_focus if latest.action else ()),
            addressee_id=latest.speaker_id,
            reason=f"Acknowledge the useful point from turn {latest.index} without changing stance",
        )

    def _narrowing_actions(self, state: DialogueState, runtime: ParticipantRuntime) -> list[UserAction]:
        finalists = list(state.narrowing_options)
        if not finalists:
            return []
        actions: list[UserAction] = []
        preferred = runtime.preferred_option

        if preferred in finalists:
            reason, source = self._positive_reason(state, preferred)
            if self._reason_key(ActionType.SUPPORT, source, reason) not in runtime.stated_reason_keys:
                actions.append(UserAction(
                    self.id, True, 0.66, ActionType.SUPPORT, (preferred,),
                    reason=reason, reason_source=source,
                ))

        if self.persona.hard_blocker:
            if preferred not in finalists:
                focus = finalists[0]
                actions.append(UserAction(
                    self.id, True, 0.92, ActionType.CONCERN, (focus,),
                    reason=self.persona.rejection_reason or "The finalists remain unacceptable",
                ))
            return actions

        viable = [
            option_id for option_id in finalists
            if option_id != preferred
            and option_id not in runtime.hard_rejected_options
            and runtime.rank(option_id) >= STANCE_NEUTRAL
        ]
        for alternative in viable:
            evidence = self._public_evidence_strength(state, alternative)
            if alternative in runtime.public_acceptances:
                evidence = min(1.0, evidence + 0.20)
            movement = switch_probability(self.persona.sim_params.stubbornness, min(1.0, 0.30 + evidence))
            reason, source = self._positive_reason(state, alternative)
            if alternative not in runtime.acceptable_options:
                actions.append(UserAction(
                    self.id, True, 0.34 + 0.62 * movement,
                    ActionType.COMPROMISE, (alternative,), reason=reason,
                    reason_source=source,
                    stance_update=StanceUpdate(StanceUpdateKind.MAKE_ACCEPTABLE, alternative),
                ))
            gate_open = self._record_switch_opportunity(
                state,
                runtime,
                alternative,
                target_evidence=evidence,
                current_evidence=self._public_evidence_strength(state, preferred),
                probability=movement,
            )
            if gate_open and evidence >= 0.28 and self.persona.sim_params.stubbornness <= 3:
                actions.append(UserAction(
                    self.id, True, 0.25 + 0.78 * movement,
                    ActionType.COMPROMISE, (preferred, alternative), reason=reason,
                    reason_source=source,
                    stance_update=StanceUpdate(
                        StanceUpdateKind.SWITCH_PREFERRED,
                        alternative,
                        previous_option_id=preferred,
                    ),
                ))

        if preferred not in finalists and not viable:
            focus = finalists[0]
            reason, source = self._positive_reason(state, preferred)
            actions.append(UserAction(
                self.id, True, 0.84, ActionType.CONCERN, (focus,),
                reason=f"I am maintaining {preferred}; this finalist has not displaced it: {reason}",
                reason_source=source,
            ))
        return actions

    def _candidate_weight(self, state: DialogueState, runtime: ParticipantRuntime, action: UserAction) -> float:
        weight = max(0.05, action.urgency)
        if state.active_issue and action.issue_id == state.active_issue.id:
            weight *= 1.28
            if (
                state.active_issue.kind is IssueKind.CONCERN
                and state.active_issue.follow_up_count > 0
                and state.active_issue.opened_by == self.id
            ):
                weight *= 1.30
        if state.group_stimulus and action.stimulus_id == state.group_stimulus.id:
            weight *= 1.18
        if action.act is runtime.last_action:
            weight *= 0.58
        if action.stance_update:
            weight *= 1.18
        signature = action_signature(action)
        repeats = runtime.action_signature_counts[signature]
        if repeats:
            stored_context = runtime.action_signature_contexts.get(signature)
            same_context = stored_context is None or stored_context == action_cooldown_context(
                state, runtime, action
            )
            weight *= (0.15 if same_context else 0.62) ** repeats
        return weight

    def _positive_reason_candidates(
        self, state: DialogueState, option_id: str
    ) -> list[tuple[str, ReasonSource | None]]:
        stance = self.persona.option_stances.get(option_id)
        option = state.scenario.option(option_id)
        candidates: list[tuple[str, ReasonSource | None]] = []
        if stance and stance.reason_for:
            candidates.append((stance.reason_for, self._best_source(option_id, option, stance.reason_for)))
        if option.upside:
            candidates.append((option.upside, ReasonSource(option_id, "upside", option.upside)))
        for key, value in option.attrs.items():
            candidates.append((
                f"the {key.replace('_', ' ')} of {value} fits my priority",
                ReasonSource(option_id, key, value),
            ))
        if self.persona.private_goal:
            candidates.append((f"it fits my goal to {self.persona.private_goal}", None))
        return self._deduplicate_reasons(candidates)

    def _negative_reason_candidates(
        self, state: DialogueState, option_id: str
    ) -> list[tuple[str, ReasonSource | None]]:
        stance = self.persona.option_stances.get(option_id)
        option = state.scenario.option(option_id)
        candidates: list[tuple[str, ReasonSource | None]] = []
        if stance and stance.reason_against:
            candidates.append((stance.reason_against, self._best_source(option_id, option, stance.reason_against)))
        if option.concern:
            candidates.append((option.concern, ReasonSource(option_id, "concern", option.concern)))
        for key, value in option.attrs.items():
            candidates.append((
                f"the {key.replace('_', ' ')} value of {value} does not fit my priority",
                ReasonSource(option_id, key, value),
            ))
        if self.persona.private_goal:
            candidates.append((f"it conflicts with my goal to {self.persona.private_goal}", None))
        return self._deduplicate_reasons(candidates)

    @staticmethod
    def _deduplicate_reasons(
        candidates: list[tuple[str, ReasonSource | None]],
    ) -> list[tuple[str, ReasonSource | None]]:
        result: list[tuple[str, ReasonSource | None]] = []
        seen: set[tuple[str, str, str] | tuple[str]] = set()
        for reason, source in candidates:
            key = (source.option_id, source.attribute_name, str(source.public_value)) if source else (reason.casefold(),)
            if key not in seen:
                seen.add(key)
                result.append((reason, source))
        return result

    def _unused_reason(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        act: ActionType,
        option_id: str,
        *,
        positive: bool,
    ) -> tuple[str, ReasonSource | None] | None:
        candidates = (
            self._positive_reason_candidates(state, option_id)
            if positive else self._negative_reason_candidates(state, option_id)
        )
        for reason, source in candidates:
            if self._reason_key(act, source, reason) not in runtime.stated_reason_keys:
                return reason, source
        return None

    def _positive_reason(self, state: DialogueState, option_id: str) -> tuple[str, ReasonSource | None]:
        candidates = self._positive_reason_candidates(state, option_id)
        return candidates[0] if candidates else ("it fits my current priority", None)

    def _negative_reason(self, state: DialogueState, option_id: str) -> tuple[str, ReasonSource | None]:
        candidates = self._negative_reason_candidates(state, option_id)
        return candidates[0] if candidates else ("it does not fit my current priority", None)

    @staticmethod
    def _best_source(option_id: str, option, reason: str) -> ReasonSource | None:
        normalized = reason.casefold()
        for key, value in option.attrs.items():
            if key.replace("_", " ").casefold() in normalized or str(value).casefold() in normalized:
                return ReasonSource(option_id, key, str(value))
        if option.upside and any(word in normalized for word in option.upside.casefold().split()[:3]):
            return ReasonSource(option_id, "upside", option.upside)
        if option.concern and any(word in normalized for word in option.concern.casefold().split()[:3]):
            return ReasonSource(option_id, "concern", option.concern)
        return None

    def _vote_score(self, state: DialogueState, runtime: ParticipantRuntime, option_id: str, *, revote: bool) -> float:
        rank_score = 0.62 * (runtime.rank(option_id) / 5.0)
        public = self._public_evidence_strength(state, option_id)
        current_bonus = 0.12 if option_id == runtime.preferred_option else 0.0
        acceptance_bonus = 0.18 if option_id in runtime.public_acceptances else 0.0
        finalist_bonus = 0.08 if option_id in state.narrowing_options else 0.0
        # Re-voting retries the protocol; it is not itself persuasive evidence.
        return rank_score + 0.72 * public + current_bonus + acceptance_bonus + finalist_bonus

    @staticmethod
    def _public_evidence_strength(state: DialogueState, option_id: str) -> float:
        preference_count = sum(runtime.public_preference == option_id for runtime in state.runtimes.values())
        acceptance_count = sum(option_id in runtime.public_acceptances for runtime in state.runtimes.values())
        support = len(state.public_supporters.get(option_id, set()))
        concern = len(state.public_concern_raisers.get(option_id, set()))
        n = max(1, len(state.personas))
        raw = (
            1.25 * preference_count
            + 0.85 * acceptance_count
            + 0.42 * support
            - 0.24 * concern
        ) / (1.65 * n)
        return max(0.0, min(1.0, raw))

    def _switch_gate(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        target_option: str,
        *,
        target_evidence: float,
        current_evidence: float,
    ) -> tuple[bool, str]:
        """Apply deterministic hysteresis before stochastic switching."""
        if self.persona.hard_blocker or target_option in runtime.hard_rejected_options:
            return False, "hard_rejected"
        if target_option == runtime.preferred_option:
            return False, "already_preferred"

        margin = float(target_evidence) - float(current_evidence)
        required_margin = SWITCH_REPEAT_MARGIN if runtime.last_switch_turn >= 0 else SWITCH_FIRST_MARGIN
        if margin < required_margin:
            return False, "insufficient_margin"
        if target_evidence < 0.20:
            return False, "insufficient_evidence"
        if runtime.last_switch_turn < 0:
            return True, ""

        next_turn_index = len(state.turns) + 1
        if next_turn_index - runtime.last_switch_turn < SWITCH_MIN_ACCEPTED_TURN_DISTANCE:
            return False, "cooldown"
        latest_external = self._latest_external_evidence_turn(
            state,
            runtime.persona_id,
            {runtime.preferred_option, target_option},
        )
        if latest_external <= max(
            runtime.last_switch_turn,
            runtime.last_switch_external_evidence_turn,
        ):
            return False, "no_new_external_evidence"
        return True, ""

    def _record_switch_opportunity(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        target_option: str,
        *,
        target_evidence: float,
        current_evidence: float,
        probability: float,
    ) -> bool:
        allowed, reason = self._switch_gate(
            state,
            runtime,
            target_option,
            target_evidence=target_evidence,
            current_evidence=current_evidence,
        )
        latest_external = self._latest_external_evidence_turn(
            state,
            runtime.persona_id,
            {runtime.preferred_option, target_option},
        )
        runtime.switch_opportunities += 1
        runtime.last_switch_probability = float(probability)
        runtime.last_switch_rejection_reason = reason
        if reason in {"cooldown", "no_new_external_evidence"}:
            runtime.switch_cooldown_rejections += 1
        state.switch_decisions.append(SwitchDecision(
            participant_id=self.id,
            phase=state.phase,
            turn_index=len(state.turns) + 1,
            current_option=runtime.preferred_option,
            target_option=target_option,
            target_evidence=round(float(target_evidence), 4),
            current_evidence=round(float(current_evidence), 4),
            evidence_margin=round(float(target_evidence) - float(current_evidence), 4),
            probability=round(float(probability), 4),
            latest_external_evidence_turn=latest_external,
            allowed=allowed,
            rejection_reason=reason,
        ))
        return allowed

    @staticmethod
    def _latest_external_evidence_turn(
        state: DialogueState,
        participant_id: str,
        option_ids: set[str],
    ) -> int:
        relevant_acts = {
            ActionType.SUPPORT,
            ActionType.CONCERN,
            ActionType.ANSWER,
            ActionType.COMPARE,
            ActionType.COMPROMISE,
        }
        return max((
            turn.index
            for turn in state.participant_turns
            if turn.speaker_id != participant_id
            and turn.action is not None
            and turn.action.act in relevant_acts
            and bool(option_ids & set(turn.action.option_focus))
        ), default=-1)

    def _optional_personal_context(self, probability: float) -> str | None:
        return self.persona.private_goal if self.rng.random() < probability else None

    @staticmethod
    def _reason_key(act: ActionType, source: ReasonSource | None, reason: str) -> str:
        if source:
            return f"{act.value}:{source.option_id}:{source.attribute_name}:{source.public_value}"
        return f"{act.value}:{reason.casefold()}"

    def _silence(self) -> UserAction:
        return UserAction(self.id, False, 0.0, ActionType.COMMENT)
