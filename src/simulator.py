"""Participant-local policy and categorical floor arbitration.

Each :class:`UserSimulator` creates a complete authoritative
:class:`~models.UserAction`. The floor selects one intact action and never
rewrites its act, option, addressee, reason, or stance effect.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

from config_loader import cfg
from models import (
    ActionType,
    ActiveIssue,
    BidPriority,
    DialogueState,
    IssueEffect,
    IssueKind,
    IssueStatus,
    OpeningMode,
    ParticipantRuntime,
    Phase,
    QuestionMode,
    ReasonSource,
    ResponseMode,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    StanceUpdate,
    StanceUpdateKind,
    UserAction,
)


def bid_probability(engagement: int) -> float:
    return float(
        cfg.level_value(
            "simulator", "bid_probability_by_engagement", engagement, cast=float
        )
    )


def movement_probability(
    stubbornness: int,
    *_ignored: object,
    hard_blocker: bool = False,
) -> float:
    if hard_blocker:
        return 0.0
    return float(
        cfg.level_value(
            "simulator",
            "movement_probability_by_stubbornness",
            stubbornness,
            cast=float,
        )
    )


def initial_runtime(persona, option_ids: Iterable[str]) -> ParticipantRuntime:
    ranks = {
        option_id: (
            persona.option_stances[option_id].rank
            if option_id in persona.option_stances
            else STANCE_NEUTRAL
        )
        for option_id in option_ids
    }
    acceptable = {
        option_id for option_id, rank in ranks.items() if rank >= STANCE_ACCEPTABLE
    }
    disliked = {
        option_id for option_id, rank in ranks.items() if rank <= STANCE_DISLIKED
    }
    rejected = set(disliked) if persona.hard_blocker else set()
    return ParticipantRuntime(
        persona_id=persona.id,
        preferred_option=persona.preferred_option,
        ranks=ranks,
        acceptable_options=acceptable,
        disliked_options=disliked,
        hard_rejected_options=rejected,
    )


def _normalized_reason(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return " ".join(words[:18])


def _reason_identity(action: UserAction) -> str:
    source = action.reason_source
    if source is not None:
        return (
            f"{source.option_id}:{source.attribute_name}:"
            f"{_normalized_reason(source.public_value)}"
        )
    return _normalized_reason(action.reason)




def public_question_key(action: UserAction) -> tuple[str, str, str]:
    """Return the global identity of a direct question.

    The same participant should not be asked the same option concern again just
    because another simulator formulates it differently.
    """

    addressee = action.addressee_id or "group"
    option_id = action.option_focus[0] if action.option_focus else "group"
    concern = (
        action.reason_source.public_value
        if action.reason_source is not None
        else action.reason
    )
    return addressee, option_id, _normalized_reason(concern)
def reason_key(action: UserAction) -> str:
    """Semantic reason key independent of act and addressee."""

    identity = _reason_identity(action)
    if not identity:
        return ""
    return f"{','.join(action.option_focus)}:{identity}"


@dataclass(slots=True)
class FloorSelection:
    action: UserAction
    eligible_count: int
    priority: BidPriority


class FloorManager:
    """Resolve simultaneous claims using categorical conversational priority."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def eligible_bids(
        self,
        state: DialogueState,
        bids: list[UserAction],
    ) -> list[UserAction]:
        valid = [
            bid
            for bid in bids
            if bid.wants_to_speak
            and bid.speaker_id in state.runtimes
            and state.consecutive_turns_by(bid.speaker_id)
            < int(cfg.conversation.max_consecutive_turns)
        ]
        if not valid:
            return []
        highest = max(bid.priority for bid in valid)
        return [bid for bid in valid if bid.priority == highest]

    def has_selectable_bid(self, state: DialogueState, bids: list[UserAction]) -> bool:
        return bool(self.eligible_bids(state, bids))

    def select(
        self,
        state: DialogueState,
        bids: list[UserAction],
    ) -> FloorSelection | None:
        eligible = self.eligible_bids(state, bids)
        if not eligible:
            return None
        last = state.last_participant_id
        alternatives = [bid for bid in eligible if bid.speaker_id != last]
        pool = alternatives or eligible
        action = self.rng.choice(pool)
        return FloorSelection(
            action=action,
            eligible_count=len(eligible),
            priority=action.priority,
        )


class UserSimulator:
    def __init__(self, persona, rng: random.Random) -> None:
        self.persona = persona
        self.rng = rng

    @property
    def id(self) -> str:
        return self.persona.id

    def opening_action(self, state: DialogueState) -> UserAction:
        runtime = state.runtimes[self.id]
        reason, source = self._positive_reason(state, runtime.preferred_option)
        visible_preferences = [
            other.public_preference
            for other in state.runtimes.values()
            if other.public_preference is not None
        ]
        if not visible_preferences:
            opening_mode = OpeningMode.INITIAL
        elif runtime.preferred_option in visible_preferences:
            opening_mode = OpeningMode.ALIGN
        else:
            opening_mode = OpeningMode.CONTRAST
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            priority=BidPriority.REQUIRED,
            act=ActionType.OPENING,
            option_focus=(runtime.preferred_option,),
            reason=reason,
            reason_source=source,
            personal_context=self.persona.private_goal or self.persona.background or None,
            opening_mode=opening_mode,
        )

    def propose(
        self,
        state: DialogueState,
        *,
        liveness_forced: bool = False,
    ) -> UserAction:
        runtime = state.runtimes[self.id]

        if state.response_obligation == self.id:
            return self._answer_action(state, runtime)

        candidates = self._candidate_actions(state, runtime)
        if not candidates:
            return self._silence()

        if (
            not liveness_forced
            and self.rng.random()
            > bid_probability(self.persona.sim_params.engagement)
        ):
            return self._silence()

        pool = [
            action
            for action in candidates
            if self._action_is_novel_or_required(state, runtime, action)
        ]
        if not pool:
            return self._silence()
        return self.rng.choice(pool)

    def propose_reaction(self, state: DialogueState) -> UserAction:
        runtime = state.runtimes[self.id]
        action = self._reaction_action(state, runtime)
        if action is None:
            return self._silence()
        if self.rng.random() > bid_probability(self.persona.sim_params.engagement):
            return self._silence()
        if not self._action_is_novel_or_required(state, runtime, action):
            return self._silence()
        return action

    def has_novel_voluntary_bid(self, state: DialogueState) -> bool:
        runtime = state.runtimes[self.id]
        return any(
            self._action_is_novel_or_required(state, runtime, action)
            for action in self._candidate_actions(state, runtime)
        )

    def _action_is_novel_or_required(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        action: UserAction,
    ) -> bool:
        key = reason_key(action)
        if not key:
            return True
        if action.stance_update is not None:
            return True
        if action.act in {ActionType.ANSWER, ActionType.FINAL_POSITION, ActionType.VOTE}:
            return True
        if action.issue_effect in {
            IssueEffect.RESOLVE,
            IssueEffect.MAINTAIN,
            IssueEffect.PARTIAL,
        }:
            return True
        if (
            action.issue_effect is IssueEffect.RESPOND
            and action.issue_id is not None
            and action.issue_id not in runtime.responded_issue_ids
        ):
            return True
        allow_reuse = bool(cfg.conversation.get("diagnostic_allow_reason_reuse", False))
        if key in runtime.used_reason_keys and not allow_reuse:
            return False
        if action.act in {
            ActionType.SUPPORT,
            ActionType.CONCERN,
            ActionType.COMPARE,
            ActionType.COMMENT,
        } and self._public_reason_already_visible(state, key) and not allow_reuse:
            return False
        return True

    @staticmethod
    def _public_reason_already_visible(state: DialogueState, key: str) -> bool:
        return any(
            turn.action is not None and reason_key(turn.action) == key
            for turn in state.participant_turns
        )

    def final_position_action(
        self,
        state: DialogueState,
        *,
        revote: bool = False,
    ) -> UserAction:
        """Create a final position only for a participant who still matters.

        The environment selects who receives the opportunity. The participant
        decides whether to accept the leader, switch, or maintain a concrete
        objection. No movement is forced.
        """

        runtime = state.runtimes[self.id]
        current = runtime.preferred_option
        leader = state.narrowing_options[0] if len(state.narrowing_options) == 1 else None

        if leader is None:
            return UserAction(
                self.id,
                True,
                BidPriority.REQUIRED,
                ActionType.FINAL_POSITION,
                option_focus=(current,),
            )
        if leader == current:
            unresolved = self._owned_unresolved_concern(state, leader)
            if unresolved is not None and self._concern_can_reopen(state, unresolved):
                return UserAction(
                    self.id,
                    True,
                    BidPriority.REQUIRED,
                    ActionType.CONCERN,
                    option_focus=(leader,),
                    reason=unresolved.summary,
                    reason_source=unresolved.reason_source,
                    issue_effect=IssueEffect.OPEN,
                )
            return UserAction(
                self.id,
                True,
                BidPriority.REQUIRED,
                ActionType.FINAL_POSITION,
                option_focus=(current,),
            )

        if self.persona.hard_blocker:
            return UserAction(
                self.id,
                True,
                BidPriority.REQUIRED,
                ActionType.FINAL_POSITION,
                option_focus=(current,),
            )

        unresolved = self._owned_unresolved_concern(state, leader)
        if unresolved is not None and self._concern_can_reopen(state, unresolved):
            return UserAction(
                self.id,
                True,
                BidPriority.REQUIRED,
                ActionType.CONCERN,
                option_focus=(leader,),
                reason=unresolved.summary,
                reason_source=unresolved.reason_source,
                issue_effect=IssueEffect.OPEN,
            )

        can_move = self._can_consider(state, runtime, leader)
        movement = can_move and self.rng.random() < movement_probability(
            self.persona.sim_params.stubbornness
        )
        if movement:
            already_accepted = leader in runtime.public_acceptances
            update_kind = (
                StanceUpdateKind.SWITCH_PREFERRED
                if already_accepted
                else StanceUpdateKind.MAKE_ACCEPTABLE
            )
            reason, source = self._positive_reason(state, leader)
            if already_accepted:
                reason = runtime.acceptance_reasons.get(leader) or reason
            return UserAction(
                self.id,
                True,
                BidPriority.REQUIRED,
                ActionType.COMPROMISE,
                option_focus=(leader,),
                reason=reason,
                reason_source=source,
                personal_context=self._personal_context(source),
                decisive_reason=reason,
                stance_update=StanceUpdate(
                    update_kind,
                    leader,
                    previous_option_id=current,
                    movement_reason=reason,
                    movement_basis=("previous_acceptance" if already_accepted else "common_ground"),
                    reason_already_public=already_accepted and leader in runtime.acceptance_reasons,
                ),
            )

        reason, source = self._negative_reason(state, leader)
        return UserAction(
            self.id,
            True,
            BidPriority.REQUIRED,
            ActionType.CONCERN,
            option_focus=(leader,),
            reason=reason,
            reason_source=source,
            personal_context=self._personal_context(source),
            issue_effect=IssueEffect.OPEN,
        )

    def decide_vote(self, state: DialogueState, *, revote: bool = False) -> UserAction:
        runtime = state.runtimes[self.id]
        target = runtime.preferred_option
        update: StanceUpdate | None = None
        movement_source: ReasonSource | None = None

        if not self.persona.hard_blocker:
            candidate_pool = list(state.narrowing_options)
            if not candidate_pool:
                candidate_pool = [
                    option_id
                    for option_id in runtime.public_acceptances
                    if self._publicly_preferred_by_other(state, option_id)
                ]
            accepted = sorted(
                option_id
                for option_id in candidate_pool
                if option_id != target and option_id in runtime.public_acceptances
            )
            if accepted and (
                not state.narrowing_options or target not in state.narrowing_options
            ):
                best_key = max(
                    (
                        runtime.rank(option_id),
                        self._publicly_preferred_by_other(state, option_id),
                    )
                    for option_id in accepted
                )
                tied = [
                    option_id
                    for option_id in accepted
                    if (
                        runtime.rank(option_id),
                        self._publicly_preferred_by_other(state, option_id),
                    ) == best_key
                ]
                target = self.rng.choice(sorted(tied))
                fallback_reason, movement_source = self._positive_reason(state, target)
                movement_reason = runtime.acceptance_reasons.get(target) or fallback_reason
                update = StanceUpdate(
                    StanceUpdateKind.SWITCH_PREFERRED,
                    target,
                    previous_option_id=runtime.preferred_option,
                    movement_reason=movement_reason,
                    movement_basis="previous_acceptance",
                    reason_already_public=target in runtime.acceptance_reasons,
                )

        reason = update.movement_reason if update is not None else ""
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            priority=BidPriority.REQUIRED,
            act=ActionType.VOTE,
            option_focus=(target,),
            reason=reason,
            reason_source=movement_source,
            personal_context=self._personal_context(movement_source),
            decisive_reason=reason,
            stance_update=update,
            vote_option=target,
        )

    def _candidate_actions(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> list[UserAction]:
        if state.active_issue:
            actions = self._issue_actions(state, runtime, state.active_issue)
            if actions:
                return actions

        if state.compromise_opportunity:
            action = self._compromise_action(state, runtime)
            return [action] if action is not None else []

        if state.group_stimulus:
            action = self._stimulus_action(state, runtime)
            return [action] if action is not None else []

        reaction = self._reaction_action(state, runtime)
        if reaction is not None and self._action_is_novel_or_required(state, runtime, reaction):
            return [reaction]

        if state.phase is Phase.NARROWING:
            return []

        actions: list[UserAction] = []
        for action in (
            self._support_action(state, runtime),
            self._concern_action(state, runtime),
            self._question_action(state, runtime),
            self._compare_action(state, runtime),
        ):
            if action is not None:
                actions.append(action)
        return actions

    def _issue_actions(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        issue: ActiveIssue,
    ) -> list[UserAction]:
        if issue.status is not IssueStatus.OPEN:
            return []

        if issue.kind is IssueKind.QUESTION:
            if state.response_obligation == self.id:
                return [self._answer_action(state, runtime)]
            if not issue.required_answer_completed:
                return []
            if issue.optional_follow_up_count >= int(
                cfg.conversation.direct_question_optional_follow_up_cap
            ):
                return []
            if self.id in issue.responded_by:
                return []
            return self._question_follow_up(state, runtime, issue)

        if issue.kind is IssueKind.CONCERN:
            if issue.opened_by == self.id:
                if issue.response_count > 0 and not issue.owner_reacted:
                    return self._owner_reaction(state, runtime, issue)
                return []
            if self.id in issue.responded_by:
                return []
            if issue.response_count >= int(cfg.conversation.concern_external_response_cap):
                return []

            option_id = issue.option_focus[0] if issue.option_focus else runtime.preferred_option
            if runtime.rank(option_id) >= STANCE_NEUTRAL:
                basis, source = self._positive_reason(state, option_id)
                response_mode = ResponseMode.ACCEPT_TRADEOFF
            else:
                basis, source = self._negative_reason(state, option_id)
                response_mode = ResponseMode.MAINTAIN_CONCERN
            return [
                UserAction(
                    self.id,
                    True,
                    BidPriority.ISSUE_RESPONSE,
                    ActionType.COMMENT,
                    option_focus=(option_id,),
                    reason=issue.summary,
                    reason_source=source,
                    personal_context=self._personal_context(source),
                    issue_id=issue.id,
                    issue_effect=IssueEffect.RESPOND,
                    response_mode=response_mode,
                    decisive_reason=basis,
                )
            ]

        return []

    def _question_follow_up(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        issue: ActiveIssue,
    ) -> list[UserAction]:
        """Reuse ordinary reaction logic for one optional post-answer turn.

        An answered question does not manufacture a special response for every
        remaining simulator. A follow-up exists only when the latest answer
        naturally triggers a novel reaction that this simulator could already
        make on the open floor.
        """

        reaction = self._reaction_action(state, runtime)
        if reaction is None:
            return []
        reaction.priority = BidPriority.NORMAL
        reaction.issue_id = issue.id
        reaction.issue_effect = IssueEffect.RESPOND
        return [reaction]

    def _owner_reaction(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        issue: ActiveIssue,
    ) -> list[UserAction]:
        option_id = issue.option_focus[0] if issue.option_focus else runtime.preferred_option
        supportive = self._latest_issue_response_supports(state, issue, option_id)
        can_accept = (
            supportive
            and not self.persona.hard_blocker
            and option_id not in runtime.hard_rejected_options
        )
        draw = self.rng.random()
        probability = movement_probability(self.persona.sim_params.stubbornness)

        if can_accept and draw < probability:
            movement_reason = (
                self._latest_issue_response_reason(state, issue)
                or self._positive_reason(state, option_id)[0]
            )
            return [
                UserAction(
                    self.id,
                    True,
                    BidPriority.ISSUE_RESPONSE,
                    ActionType.COMPROMISE,
                    option_focus=(option_id,),
                    reason=issue.summary,
                    issue_id=issue.id,
                    issue_effect=IssueEffect.RESOLVE,
                    response_mode=ResponseMode.ACCEPT_TRADEOFF,
                    decisive_reason=movement_reason,
                    stance_update=StanceUpdate(
                        StanceUpdateKind.MAKE_ACCEPTABLE,
                        option_id,
                        previous_option_id=runtime.preferred_option,
                        movement_reason=movement_reason,
                        movement_basis="concern_resolved",
                        remaining_concern=issue.summary,
                    ),
                )
            ]

        if can_accept:
            return [
                UserAction(
                    self.id,
                    True,
                    BidPriority.ISSUE_RESPONSE,
                    ActionType.COMMENT,
                    option_focus=(option_id,),
                    reason=issue.summary,
                    issue_id=issue.id,
                    issue_effect=IssueEffect.PARTIAL,
                    response_mode=ResponseMode.MAINTAIN_CONCERN,
                    decisive_reason=self._latest_issue_response_reason(state, issue),
                )
            ]

        negative, source = self._negative_reason(state, option_id)
        return [
            UserAction(
                self.id,
                True,
                BidPriority.ISSUE_RESPONSE,
                ActionType.CONCERN,
                option_focus=(option_id,),
                reason=negative or issue.summary,
                reason_source=source,
                personal_context=self._personal_context(source),
                issue_id=issue.id,
                issue_effect=IssueEffect.MAINTAIN,
                response_mode=ResponseMode.MAINTAIN_CONCERN,
            )
        ]

    def _answer_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> UserAction:
        issue = state.active_issue
        focus = issue.option_focus if issue else (runtime.preferred_option,)
        option_id = focus[0] if focus else runtime.preferred_option
        rank = runtime.rank(option_id)
        if issue is not None and issue.question_mode is QuestionMode.CONDITION:
            reason, source, decisive = "", None, ""
            response_mode = ResponseMode.UNKNOWN
        elif rank >= STANCE_ACCEPTABLE:
            decisive, source = self._positive_reason(state, option_id)
            response_mode = ResponseMode.ACCEPT_TRADEOFF
            reason = issue.summary if issue else "the stated drawback"
        elif rank <= STANCE_DISLIKED:
            reason, source = self._negative_reason(state, option_id)
            decisive = ""
            response_mode = ResponseMode.MAINTAIN_CONCERN
        else:
            reason, source, decisive = "", None, ""
            response_mode = ResponseMode.UNKNOWN
        return UserAction(
            self.id,
            True,
            BidPriority.REQUIRED,
            ActionType.ANSWER,
            option_focus=focus,
            addressee_id=issue.opened_by if issue else None,
            reason=reason,
            reason_source=source,
            personal_context=self.persona.private_goal or None,
            issue_id=issue.id if issue else None,
            issue_effect=IssueEffect.RESPOND if issue else None,
            response_mode=response_mode,
            decisive_reason=decisive,
        )

    def _reaction_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> UserAction | None:
        """React to the latest participant statement before changing topic."""

        latest = next(
            (
                turn
                for turn in reversed(state.turns)
                if not turn.moderator
                and turn.speaker_id != self.id
                and turn.action is not None
            ),
            None,
        )
        if latest is None or latest.action is None:
            return None
        if latest.action.act in {ActionType.OPENING, ActionType.VOTE}:
            return None
        if not latest.action.option_focus:
            return None

        option_id = latest.action.option_focus[0]
        if latest.action.act is ActionType.COMPROMISE:
            if option_id == runtime.preferred_option or option_id in runtime.public_acceptances or runtime.rank(option_id) >= STANCE_ACCEPTABLE:
                return UserAction(
                    self.id,
                    True,
                    BidPriority.NORMAL,
                    ActionType.ACKNOWLEDGE,
                    option_focus=(option_id,),
                    addressee_id=latest.speaker_id,
                    reason="",
                )
            if self._can_consider(state, runtime, option_id) and self.rng.random() < movement_probability(self.persona.sim_params.stubbornness):
                reason, source = self._positive_reason(state, option_id)
                return UserAction(
                    self.id,
                    True,
                    BidPriority.NORMAL,
                    ActionType.COMPROMISE,
                    option_focus=(option_id,),
                    addressee_id=latest.speaker_id,
                    reason=reason,
                    reason_source=source,
                    personal_context=self._personal_context(source),
                    decisive_reason=reason,
                    stance_update=StanceUpdate(
                        StanceUpdateKind.MAKE_ACCEPTABLE,
                        option_id,
                        previous_option_id=runtime.preferred_option,
                        movement_reason=reason,
                        movement_basis="common_ground_proposal",
                    ),
                )

        if runtime.rank(option_id) >= STANCE_ACCEPTABLE:
            for reason, source in self._positive_reason_candidates(state, option_id):
                action = UserAction(
                    self.id,
                    True,
                    BidPriority.NORMAL,
                    ActionType.COMMENT,
                    option_focus=(option_id,),
                    addressee_id=latest.speaker_id,
                    reason=reason,
                    reason_source=source,
                    personal_context=self._personal_context(source),
                )
                if (
                    bool(cfg.conversation.get("diagnostic_allow_reason_reuse", False))
                    or reason_key(action) not in runtime.used_reason_keys
                ):
                    return action
            if (
                option_id not in runtime.acknowledged_options
                and latest.action.act in {
                    ActionType.SUPPORT,
                    ActionType.COMMENT,
                    ActionType.ANSWER,
                    ActionType.COMPARE,
                }
            ):
                return UserAction(
                    self.id,
                    True,
                    BidPriority.NORMAL,
                    ActionType.ACKNOWLEDGE,
                    option_focus=(option_id,),
                    addressee_id=latest.speaker_id,
                )
        elif runtime.rank(option_id) <= STANCE_DISLIKED:
            for reason, source in self._negative_reason_candidates(state, option_id):
                if self._concern_was_opened(state, option_id, reason, source):
                    continue
                action = UserAction(
                    self.id,
                    True,
                    BidPriority.NORMAL,
                    ActionType.CONCERN,
                    option_focus=(option_id,),
                    addressee_id=latest.speaker_id,
                    reason=reason,
                    reason_source=source,
                    personal_context=self._personal_context(source),
                    issue_effect=IssueEffect.OPEN,
                )
                if (
                    bool(cfg.conversation.get("diagnostic_allow_reason_reuse", False))
                    or reason_key(action) not in runtime.used_reason_keys
                ):
                    return action
        return None

    def _compromise_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> UserAction | None:
        if self.persona.hard_blocker:
            return None

        if state.narrowing_options:
            pool = list(state.narrowing_options)
        elif state.phase is Phase.NARROWING:
            publicly_considered: set[str] = set()
            for other in state.runtimes.values():
                if other.public_preference in state.scenario.option_ids:
                    publicly_considered.add(other.public_preference)
                publicly_considered.update(
                    option_id
                    for option_id in other.public_acceptances
                    if option_id in state.scenario.option_ids
                )
            pool = [
                option_id
                for option_id in state.scenario.option_ids
                if option_id in publicly_considered
            ]
        else:
            pool = list(state.scenario.option_ids)
        candidates = [
            option_id
            for option_id in sorted(pool)
            if option_id != runtime.preferred_option
            and option_id not in runtime.hard_rejected_options
            and option_id not in runtime.public_rejections
            and option_id not in runtime.used_compromise_options
            and self._can_consider(state, runtime, option_id)
        ]
        if not candidates:
            return None

        best_rank = max(runtime.rank(option_id) for option_id in candidates)
        candidates = [option_id for option_id in candidates if runtime.rank(option_id) == best_rank]
        publicly_supported = [
            option_id
            for option_id in candidates
            if self._publicly_preferred_by_other(state, option_id)
        ]
        target = self.rng.choice(sorted(publicly_supported or candidates))

        if self.rng.random() >= movement_probability(self.persona.sim_params.stubbornness):
            return None

        already_accepted = target in runtime.public_acceptances
        update_kind = (
            StanceUpdateKind.SWITCH_PREFERRED
            if already_accepted
            else StanceUpdateKind.MAKE_ACCEPTABLE
        )
        reason, source = self._positive_reason(state, target)
        return UserAction(
            self.id,
            True,
            BidPriority.NORMAL,
            ActionType.COMPROMISE,
            option_focus=(target,),
            reason=reason,
            reason_source=source,
            personal_context=self._personal_context(source),
            decisive_reason=reason,
            stance_update=StanceUpdate(
                update_kind,
                target,
                previous_option_id=runtime.preferred_option,
                movement_reason=reason,
                movement_basis=("previous_acceptance" if already_accepted else "stagnation_compromise"),
                reason_already_public=already_accepted and target in runtime.acceptance_reasons,
            ),
        )

    def _stimulus_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> UserAction | None:
        stimulus = state.group_stimulus
        if not stimulus or stimulus.id in runtime.responded_stimuli:
            return None
        option_id = stimulus.option_focus[0] if stimulus.option_focus else runtime.preferred_option
        rank = runtime.rank(option_id)
        if stimulus.option_focus and rank <= STANCE_DISLIKED:
            reason, source = self._negative_reason(state, option_id)
            act = ActionType.CONCERN
        elif option_id == runtime.preferred_option or rank >= STANCE_ACCEPTABLE:
            reason, source = self._positive_reason(state, option_id)
            act = ActionType.SUPPORT
        else:
            option_name = state.scenario.option(option_id).short_name or state.scenario.option(option_id).name
            reason, source = (
                f"{option_name} is possible for me, but it is not one of my leading choices",
                None,
            )
            act = ActionType.COMMENT
        return UserAction(
            self.id,
            True,
            BidPriority.ISSUE_RESPONSE,
            act,
            option_focus=(option_id,),
            reason=reason,
            reason_source=source,
            personal_context=self._personal_context(source),
            stimulus_id=stimulus.id,
        )

    def _support_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> UserAction | None:
        option_id = runtime.preferred_option
        for reason, source in self._positive_reason_candidates(state, option_id):
            action = UserAction(
                self.id,
                True,
                BidPriority.NORMAL,
                ActionType.SUPPORT,
                option_focus=(option_id,),
                reason=reason,
                reason_source=source,
                personal_context=self._personal_context(source),
            )
            if (
                bool(cfg.conversation.get("diagnostic_allow_reason_reuse", False))
                or reason_key(action) not in runtime.used_reason_keys
            ):
                return action
        return None

    def _concern_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> UserAction | None:
        concern_count = sum(
            key.startswith("concern:") for key in runtime.opened_issue_keys
        )
        if concern_count >= int(cfg.conversation.max_concerns_per_participant):
            return None
        candidates = sorted(
            (
                option_id
                for option_id in state.scenario.option_ids
                if option_id != runtime.preferred_option
                and runtime.rank(option_id) <= STANCE_DISLIKED
            ),
            key=runtime.rank,
        )
        for option_id in candidates:
            for reason, source in self._negative_reason_candidates(state, option_id):
                semantic = _reason_identity(
                    UserAction(
                        self.id,
                        True,
                        BidPriority.NORMAL,
                        ActionType.CONCERN,
                        option_focus=(option_id,),
                        reason=reason,
                        reason_source=source,
                    )
                )
                key = f"concern:{option_id}:{semantic}"
                if key in runtime.opened_issue_keys:
                    continue
                if self._concern_was_opened(state, option_id, reason, source):
                    continue
                return UserAction(
                    self.id,
                    True,
                    BidPriority.NORMAL,
                    ActionType.CONCERN,
                    option_focus=(option_id,),
                    reason=reason,
                    reason_source=source,
                    personal_context=self._personal_context(source),
                    issue_effect=IssueEffect.OPEN,
                )
        return None

    def _question_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> UserAction | None:
        others = [
            other
            for other in state.personas
            if other.id != self.id
            and state.runtimes[other.id].public_preference is not None
        ]
        self.rng.shuffle(others)
        configured_modes = [QuestionMode(str(value)) for value in cfg.simulator.question_modes]
        for other in others:
            option_id = state.runtimes[other.id].public_preference
            if option_id is None or option_id == runtime.preferred_option:
                continue
            negative_candidates = self._negative_reason_candidates(state, option_id)
            if not negative_candidates:
                continue
            reason, source = negative_candidates[0]
            decisive = self._latest_public_positive_reason(state, other.id, option_id)
            mode_pool = list(configured_modes)
            if not decisive and QuestionMode.TRADEOFF in mode_pool:
                mode_pool.remove(QuestionMode.TRADEOFF)
            mode_pool = [mode for mode in mode_pool if mode is not QuestionMode.CONDITION]
            unknown_probability = float(
                cfg.simulator.unknown_information_question_probability
            )
            question_mode = (
                QuestionMode.CONDITION
                if self.rng.random() < unknown_probability
                else self.rng.choice(mode_pool or [QuestionMode.CHOICE_IMPACT])
            )
            action = UserAction(
                self.id,
                True,
                BidPriority.NORMAL,
                ActionType.ASK,
                option_focus=(option_id,),
                addressee_id=other.id,
                reason=reason,
                reason_source=source,
                personal_context=self._personal_context(source),
                issue_effect=IssueEffect.OPEN,
                question_mode=question_mode,
                decisive_reason=decisive,
            )
            if (
                reason_key(action) in runtime.asked_question_keys
                or public_question_key(action) in state.asked_public_question_keys
            ):
                continue
            return action
        return None

    def _compare_action(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
    ) -> UserAction | None:
        alternatives = [
            option_id
            for option_id in state.scenario.option_ids
            if option_id != runtime.preferred_option
            and self._can_consider(state, runtime, option_id)
        ]
        if not alternatives:
            return None
        other = max(alternatives, key=runtime.rank)
        pair = tuple(sorted((runtime.preferred_option, other)))
        if state.public_comparisons.get(pair, 0) > 0:
            return None
        first, _ = self._positive_reason(state, runtime.preferred_option)
        second, _ = self._positive_reason(state, other)
        if not first or not second:
            return None
        current_name = state.scenario.option(runtime.preferred_option).short_name or state.scenario.option(runtime.preferred_option).name
        other_name = state.scenario.option(other).short_name or state.scenario.option(other).name
        reason = f"{current_name}: {first}; {other_name}: {second}"
        return UserAction(
            self.id,
            True,
            BidPriority.NORMAL,
            ActionType.COMPARE,
            option_focus=(runtime.preferred_option, other),
            reason=reason,
        )

    def _positive_reason_candidates(
        self,
        state: DialogueState,
        option_id: str,
    ) -> list[tuple[str, ReasonSource | None]]:
        candidates: list[tuple[str, ReasonSource | None]] = []
        stance = self.persona.option_stances.get(option_id)
        if stance and stance.reason_for:
            candidates.append(
                (
                    stance.reason_for,
                    self._source_for_reason(state, option_id, stance.reason_for),
                )
            )
        option = state.scenario.option(option_id)
        if option.upside:
            candidates.append(
                (option.upside, ReasonSource(option_id, "upside", option.upside))
            )
        if not candidates:
            candidates.append(
                (
                    self.persona.private_goal or f"{option.name} fits my priorities",
                    None,
                )
            )
        return self._deduplicate_reasons(candidates)

    def _negative_reason_candidates(
        self,
        state: DialogueState,
        option_id: str,
    ) -> list[tuple[str, ReasonSource | None]]:
        candidates: list[tuple[str, ReasonSource | None]] = []
        stance = self.persona.option_stances.get(option_id)
        if stance and stance.reason_against:
            candidates.append(
                (
                    stance.reason_against,
                    self._source_for_reason(state, option_id, stance.reason_against),
                )
            )
        option = state.scenario.option(option_id)
        if option.concern:
            candidates.append(
                (option.concern, ReasonSource(option_id, "concern", option.concern))
            )
        if not candidates:
            candidates.append((f"{option.name} does not fit my priorities", None))
        return self._deduplicate_reasons(candidates)

    @staticmethod
    def _deduplicate_reasons(
        candidates: list[tuple[str, ReasonSource | None]],
    ) -> list[tuple[str, ReasonSource | None]]:
        seen: set[str] = set()
        result: list[tuple[str, ReasonSource | None]] = []
        for reason, source in candidates:
            key = _normalized_reason(reason)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append((reason, source))
        return result

    def _positive_reason(
        self,
        state: DialogueState,
        option_id: str,
    ) -> tuple[str, ReasonSource | None]:
        return self._positive_reason_candidates(state, option_id)[0]

    def _negative_reason(
        self,
        state: DialogueState,
        option_id: str,
    ) -> tuple[str, ReasonSource | None]:
        return self._negative_reason_candidates(state, option_id)[0]

    @staticmethod
    def _source_for_reason(
        state: DialogueState,
        option_id: str,
        reason: str,
    ) -> ReasonSource | None:
        option = state.scenario.option(option_id)
        lower = reason.casefold()
        for key, value in option.attrs.items():
            if key.replace("_", " ").casefold() in lower or str(value).casefold() in lower:
                return ReasonSource(option_id, key, str(value))
        if option.upside and option.upside.casefold() in lower:
            return ReasonSource(option_id, "upside", option.upside)
        if option.concern and option.concern.casefold() in lower:
            return ReasonSource(option_id, "concern", option.concern)
        return None

    def _personal_context(self, source: ReasonSource | None) -> str | None:
        if source is not None:
            return None
        return self.persona.private_goal or self.persona.background or None

    def _can_consider(
        self,
        state: DialogueState,
        runtime: ParticipantRuntime,
        option_id: str,
    ) -> bool:
        if option_id in runtime.hard_rejected_options or option_id in runtime.public_rejections:
            return False
        if option_id in runtime.acceptable_options or option_id in runtime.public_acceptances:
            return True
        rank = runtime.rank(option_id)
        if rank >= STANCE_NEUTRAL:
            return True
        if rank != STANCE_DISLIKED:
            return False
        return self._own_concern_softened(state, option_id)

    def _own_concern_softened(self, state: DialogueState, option_id: str) -> bool:
        """Allow a disliked option only after this simulator's concern moved.

        A rank-2 option is not ordinary compromise material. It becomes
        eligible only when the participant visibly opened the concrete concern
        and the exchange ended with resolution or partial softening.
        """

        issues = list(state.issue_history)
        if state.active_issue is not None:
            issues.append(state.active_issue)
        return any(
            issue.kind is IssueKind.CONCERN
            and issue.opened_by == self.id
            and option_id in issue.option_focus
            and issue.outcome in {"resolved", "partial"}
            for issue in issues
        )

    def _latest_issue_response_supports(
        self,
        state: DialogueState,
        issue: ActiveIssue,
        option_id: str,
    ) -> bool:
        latest = next(
            (
                turn
                for turn in reversed(state.turns)
                if turn.action is not None
                and turn.action.issue_id == issue.id
                and turn.speaker_id != issue.opened_by
            ),
            None,
        )
        if latest is None:
            return False
        return state.runtimes[latest.speaker_id].rank(option_id) >= STANCE_NEUTRAL


    def _latest_issue_response_reason(
        self,
        state: DialogueState,
        issue: ActiveIssue,
    ) -> str:
        latest = next(
            (
                turn.action
                for turn in reversed(state.turns)
                if turn.action is not None
                and turn.action.issue_id == issue.id
                and turn.speaker_id != issue.opened_by
            ),
            None,
        )
        if latest is None:
            return ""
        return latest.decisive_reason or latest.reason

    def _latest_public_positive_reason(
        self,
        state: DialogueState,
        participant_id: str,
        option_id: str,
    ) -> str:
        latest = next(
            (
                turn.action
                for turn in reversed(state.turns)
                if turn.action is not None
                and turn.speaker_id == participant_id
                and option_id in turn.action.option_focus
                and turn.action.act in {
                    ActionType.OPENING,
                    ActionType.SUPPORT,
                    ActionType.COMMENT,
                    ActionType.COMPROMISE,
                }
            ),
            None,
        )
        if latest is not None:
            return latest.decisive_reason or latest.reason
        return state.scenario.option(option_id).upside

    def _owned_unresolved_concern(
        self,
        state: DialogueState,
        option_id: str,
    ) -> ActiveIssue | None:
        issues = list(reversed(state.issue_history))
        if state.active_issue is not None:
            issues.insert(0, state.active_issue)
        return next(
            (
                issue
                for issue in issues
                if issue.kind is IssueKind.CONCERN
                and issue.opened_by == self.id
                and option_id in issue.option_focus
                and issue.status.value != "resolved"
            ),
            None,
        )

    @staticmethod
    def _concern_was_opened(
        state: DialogueState,
        option_id: str,
        reason: str,
        source: ReasonSource | None = None,
    ) -> bool:
        semantic = source.public_value if source is not None else reason
        record = state.issue_records.get((option_id, _normalized_reason(semantic)))
        return record is not None and record.kind is IssueKind.CONCERN

    @staticmethod
    def _concern_can_reopen(state: DialogueState, issue: ActiveIssue) -> bool:
        if state.phase is not Phase.NARROWING or issue.issue_key is None:
            return False
        record = state.issue_records.get(issue.issue_key)
        return bool(
            record is not None
            and record.kind is IssueKind.CONCERN
            and record.reopen_count < int(cfg.conversation.max_concern_reopens)
        )

    def _publicly_preferred_by_other(
        self,
        state: DialogueState,
        option_id: str,
    ) -> bool:
        return any(
            participant_id != self.id and runtime.public_preference == option_id
            for participant_id, runtime in state.runtimes.items()
        )

    def _silence(self) -> UserAction:
        return UserAction(
            self.id,
            False,
            BidPriority.NORMAL,
            ActionType.COMMENT,
        )
