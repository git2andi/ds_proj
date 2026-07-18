"""Autonomous symbolic policy for one simulated participant.

Each simulator independently constructs a complete action and decides whether
it wants the floor. The floor manager only arbitrates between intact bids.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from config_loader import cfg

MAX_PUBLIC_POINT_USES = 2
from models import (
    ActionType,
    BidPriority,
    DialogueState,
    ParticipantRuntime,
    ReasonSource,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    STANCE_REJECTED,
    StanceUpdate,
    StanceUpdateKind,
    ThreadKind,
    UserAction,
)


def re_sub_words(label: str) -> str:
    words = label.split()
    replacements = {"avg": "average", "hrs": "hours", "mins": "minutes"}
    cleaned = [replacements.get(word.lower(), word) for word in words]
    while cleaned and cleaned[-1].lower() in {"usd", "eur", "gbp"}:
        cleaned.pop()
    return " ".join(cleaned)


def bid_probability(engagement: int) -> float:
    return float(
        cfg.level_value(
            "simulator", "bid_probability_by_engagement", engagement, cast=float
        )
    )


def movement_probability(stubbornness: int) -> float:
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
        option_id: persona.option_stances.get(option_id).rank
        if option_id in persona.option_stances
        else STANCE_NEUTRAL
        for option_id in option_ids
    }
    acceptable = {
        option_id for option_id, rank in ranks.items() if rank >= STANCE_ACCEPTABLE
    }
    rejected = {
        option_id for option_id, rank in ranks.items() if rank == STANCE_REJECTED
    }
    return ParticipantRuntime(
        persona_id=persona.id,
        preferred_option=persona.preferred_option,
        ranks=ranks,
        acceptable_options=acceptable,
        hard_rejected_options=rejected,
    )


@dataclass(slots=True)
class FloorSelection:
    action: UserAction | None
    eligible_count: int = 0


class FloorManager:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def eligible_bids(
        self, state: DialogueState, bids: list[UserAction]
    ) -> list[UserAction]:
        eligible = [bid for bid in bids if bid.wants_to_speak]
        if not eligible:
            return []
        required = [bid for bid in eligible if bid.priority is BidPriority.REQUIRED]
        if required:
            return required
        maximum = int(cfg.conversation.max_consecutive_turns)
        restricted = [
            bid
            for bid in eligible
            if state.consecutive_turns_by(bid.speaker_id) < maximum
        ]
        if restricted:
            eligible = restricted
        highest = max(bid.priority for bid in eligible)
        return [bid for bid in eligible if bid.priority == highest]

    def select(self, state: DialogueState, bids: list[UserAction]) -> FloorSelection:
        eligible = self.eligible_bids(state, bids)
        if not eligible:
            return FloorSelection(None, 0)
        counts = {
            persona.id: state.runtimes[persona.id].voluntary_turns
            for persona in state.personas
        }
        last = state.last_participant_id
        previous = None
        participant_ids = [turn.speaker_id for turn in state.participant_turns]
        if len(participant_ids) >= 2:
            previous = participant_ids[-2]
        weights: list[float] = []
        maximum_count = max(counts.values(), default=0)
        for bid in eligible:
            share_deficit = maximum_count - counts[bid.speaker_id]
            weight = 1.0 + 0.25 * share_deficit
            if bid.speaker_id == last:
                weight *= 0.45
            if bid.speaker_id == previous and last != previous:
                weight *= 0.80
            weights.append(max(0.05, weight))
        return FloorSelection(self.rng.choices(eligible, weights=weights, k=1)[0], len(eligible))


class UserSimulator:
    def __init__(self, persona, rng: random.Random) -> None:
        self.persona = persona
        self.rng = rng

    @property
    def id(self) -> str:
        return self.persona.id

    def opening_action(self, state: DialogueState) -> UserAction:
        option_id = state.runtimes[self.id].preferred_option
        source, reason = self._positive_point(state, option_id, allow_used=True)
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            priority=BidPriority.REQUIRED,
            act=ActionType.OPENING,
            option_focus=(option_id,),
            reason=reason,
            reason_source=source,
            personal_context=self.persona.private_goal,
        )

    def propose(self, state: DialogueState, *, force_willing: bool = False) -> UserAction:
        if state.response_obligation == self.id:
            return self._answer_action(state, required=True)
        if state.response_obligation is not None:
            return self._silence()

        if state.active_thread is not None:
            candidates = self._thread_candidates(state)
        else:
            candidates = self._ordinary_candidates(state)
        if not candidates:
            return self._silence()

        action, weight = self._weighted_choice(candidates)
        willingness = bid_probability(self.persona.sim_params.engagement)
        if state.active_thread is not None:
            willingness = min(1.0, willingness + 0.15)
        action.wants_to_speak = force_willing or self.rng.random() < willingness
        if not action.wants_to_speak:
            return self._silence()
        action.priority = (
            BidPriority.THREAD if state.active_thread is not None else BidPriority.NORMAL
        )
        del weight
        return action

    def has_novel_voluntary_bid(self, state: DialogueState) -> bool:
        candidates = (
            self._thread_candidates(state)
            if state.active_thread is not None
            else self._ordinary_candidates(state)
        )
        return bool(candidates)

    def compromise_action(
        self, state: DialogueState, candidates: tuple[str, ...]
    ) -> UserAction:
        runtime = state.runtimes[self.id]
        if self.persona.hard_blocker:
            return self._silence()
        alternatives = [
            option_id
            for option_id in candidates
            if option_id != runtime.preferred_option
            and runtime.rank(option_id) >= STANCE_NEUTRAL
            and option_id not in runtime.hard_rejected_options
        ]
        if not alternatives:
            return self._silence()
        alternatives.sort(key=lambda option_id: runtime.rank(option_id), reverse=True)
        target = alternatives[0]
        if self.rng.random() >= movement_probability(self.persona.sim_params.stubbornness):
            return self._silence()
        source, reason = self._positive_point(state, target, allow_used=True)
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            priority=BidPriority.NORMAL,
            act=ActionType.ACCEPT,
            option_focus=(target,),
            reason=reason,
            reason_source=source,
            personal_context=self.persona.private_goal,
            stance_update=StanceUpdate(
                kind=StanceUpdateKind.SWITCH_PREFERRED,
                option_id=target,
                previous_option_id=runtime.preferred_option,
                movement_reason=reason,
            ),
        )

    def decide_vote(self, state: DialogueState) -> UserAction:
        option_id = state.runtimes[self.id].preferred_option
        source, reason = self._positive_point(state, option_id, allow_used=True)
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            priority=BidPriority.REQUIRED,
            act=ActionType.VOTE,
            option_focus=(option_id,),
            vote_option=option_id,
            reason=reason,
            reason_source=source,
        )

    def _ordinary_candidates(
        self, state: DialogueState
    ) -> list[tuple[UserAction, float]]:
        candidates: list[tuple[UserAction, float]] = []
        runtime = state.runtimes[self.id]

        support = self._support_action(state)
        if support:
            candidates.append((support, 1.15))

        reaction = self._reaction_action(state)
        if reaction:
            candidates.append((reaction, 1.30))

        objection = self._object_action(state)
        if objection:
            candidates.append((objection, 1.05))

        comparison = self._compare_action(state)
        if comparison:
            candidates.append((comparison, 0.20))

        question = self._question_action(state)
        if question:
            candidates.append((question, 0.45))

        # A participant may voluntarily accept an option that is already under
        # visible discussion. This is less likely than ordinary reactions.
        acceptance = self._accept_action(state)
        if acceptance and runtime.voluntary_turns > 0:
            candidates.append((acceptance, 0.35))

        return self._deduplicate(candidates)

    def _thread_candidates(
        self, state: DialogueState
    ) -> list[tuple[UserAction, float]]:
        thread = state.active_thread
        if thread is None or self.id == thread.opened_by and thread.turn_count <= 1:
            return []
        if self.id in thread.participants:
            return []
        if thread.kind is ThreadKind.QUESTION and thread.required_answer_pending:
            answer = self._answer_action(state, required=False)
            return [(answer, 1.25)] if answer.wants_to_speak else []

        target = thread.option_focus[0] if thread.option_focus else None
        if target is None:
            return []
        runtime = state.runtimes[self.id]
        candidates: list[tuple[UserAction, float]] = []
        if runtime.rank(target) >= STANCE_ACCEPTABLE:
            source, reason = self._positive_point(state, target)
            if source is not None:
                candidates.append((self._action(ActionType.REACT, (target,), reason, source), 1.1))
            accept = self._accept_action(state, forced_target=target)
            if accept:
                candidates.append((accept, 0.65))
        else:
            source, reason = self._negative_point(state, target)
            if source is not None:
                candidates.append((self._action(ActionType.OBJECT, (target,), reason, source), 1.1))
        compare = self._compare_action(state, forced_target=target)
        if compare:
            candidates.append((compare, 0.25))
        return self._deduplicate(candidates)

    def _support_action(self, state: DialogueState) -> UserAction | None:
        runtime = state.runtimes[self.id]
        source, reason = self._positive_point(state, runtime.preferred_option)
        if source is None:
            return None
        return self._action(ActionType.SUPPORT, (runtime.preferred_option,), reason, source)

    def _reaction_action(self, state: DialogueState) -> UserAction | None:
        last = next(
            (
                turn
                for turn in reversed(state.turns)
                if not turn.moderator and turn.speaker_id != self.id and turn.action
            ),
            None,
        )
        if last is None or not last.action.option_focus:
            return None
        target = last.action.option_focus[0]
        runtime = state.runtimes[self.id]
        if runtime.rank(target) >= STANCE_ACCEPTABLE:
            source, reason = self._positive_point(state, target)
        else:
            source, reason = self._negative_point(state, target)
        if source is None:
            source, reason = self._neutral_point(state, target)
        if source is None:
            return None
        return self._action(
            ActionType.REACT,
            (target,),
            reason,
            source,
            addressee=last.speaker_id,
        )

    def _object_action(self, state: DialogueState) -> UserAction | None:
        runtime = state.runtimes[self.id]
        visible = [
            other.public_preference
            for pid, other in state.runtimes.items()
            if pid != self.id
            and other.public_preference
            and other.public_preference != runtime.preferred_option
        ]
        targets = [
            option_id
            for option_id in visible
            if option_id is not None and runtime.rank(option_id) <= STANCE_DISLIKED
        ]
        self.rng.shuffle(targets)
        for target in targets:
            source, reason = self._negative_point(state, target)
            if source is not None:
                return self._action(ActionType.OBJECT, (target,), reason, source)
        return None

    def _compare_action(
        self, state: DialogueState, *, forced_target: str | None = None
    ) -> UserAction | None:
        runtime = state.runtimes[self.id]
        recent_acts = [
            turn.action.act
            for turn in state.participant_turns[-2:]
            if turn.action is not None
        ]
        if ActionType.COMPARE in recent_acts:
            return None
        if forced_target is None and runtime.voluntary_turns == 0:
            return None

        targets = [forced_target] if forced_target else [
            other.public_preference
            for pid, other in state.runtimes.items()
            if pid != self.id and other.public_preference != runtime.preferred_option
        ]
        targets = [
            target
            for target in targets
            if target in state.scenario.option_ids
            and target != runtime.preferred_option
        ]
        self.rng.shuffle(targets)

        own_option = state.scenario.option(runtime.preferred_option)
        for target in targets:
            other_option = state.scenario.option(target)
            shared = [
                key
                for key in own_option.attrs
                if key in other_option.attrs
                and str(own_option.attrs[key]).strip()
                and str(other_option.attrs[key]).strip()
            ]
            fresh = [
                key
                for key in shared
                if (runtime.preferred_option, key.strip().lower()) not in state.recent_point_keys
                and (target, key.strip().lower()) not in state.recent_point_keys
            ]
            choices = fresh or shared
            if not choices:
                continue
            attribute = self.rng.choice(choices)
            own_source = ReasonSource(
                runtime.preferred_option, attribute, str(own_option.attrs[attribute])
            )
            other_source = ReasonSource(
                target, attribute, str(other_option.attrs[attribute])
            )
            reason = (
                f"compare {re_sub_words(attribute)}: "
                f"{own_source.public_value} and {other_source.public_value}"
            )
            return UserAction(
                speaker_id=self.id,
                wants_to_speak=True,
                priority=BidPriority.NORMAL,
                act=ActionType.COMPARE,
                option_focus=(runtime.preferred_option, target),
                reason=reason,
                comparison_sources=(own_source, other_source),
                personal_context=self.persona.private_goal,
            )
        return None

    def _question_action(self, state: DialogueState) -> UserAction | None:
        if any(
            turn.action and turn.action.act is ActionType.ASK
            for turn in state.participant_turns[-2:]
        ):
            return None
        runtime = state.runtimes[self.id]
        supporters = [
            (pid, other.public_preference)
            for pid, other in state.runtimes.items()
            if pid != self.id
            and other.public_preference
            and other.public_preference != runtime.preferred_option
        ]
        self.rng.shuffle(supporters)
        for pid, target in supporters:
            if target is None:
                continue
            source, reason = self._negative_point(
                state, target, require_publicly_unseen=True
            )
            if source is None or source.point_key in state.closed_thread_keys:
                continue
            if source.point_key in runtime.opened_thread_keys:
                continue
            # Half the questions are direct; the others invite the whole group.
            addressee = pid if self.rng.random() < 0.5 else None
            return self._action(
                ActionType.ASK,
                (target,),
                reason,
                source,
                addressee=addressee,
            )
        return None

    def _answer_action(self, state: DialogueState, *, required: bool) -> UserAction:
        thread = state.active_thread
        if thread is None or not thread.option_focus:
            return self._silence()
        target = thread.option_focus[0]
        source, reason = self._thread_point(state, target)
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            priority=BidPriority.REQUIRED if required else BidPriority.THREAD,
            act=ActionType.ANSWER,
            option_focus=(target,),
            addressee_id=thread.opened_by,
            reason=reason,
            reason_source=source,
            personal_context=self.persona.private_goal,
        )


    def _thread_point(
        self, state: DialogueState, option_id: str
    ) -> tuple[ReasonSource | None, str]:
        thread = state.active_thread
        option = state.scenario.option(option_id)
        if thread and thread.point_key and thread.point_key[0] == option_id:
            attribute = thread.point_key[1]
            if attribute == "upside":
                value = option.upside
            elif attribute == "concern":
                value = option.concern
            else:
                value = next(
                    (value for key, value in option.attrs.items() if key.strip().lower() == attribute),
                    "",
                )
            if value:
                return ReasonSource(option_id, attribute, value), self._fact_reason(attribute, value)
        return self._positive_point(state, option_id, allow_used=True)

    def _accept_action(
        self, state: DialogueState, *, forced_target: str | None = None
    ) -> UserAction | None:
        runtime = state.runtimes[self.id]
        targets = [forced_target] if forced_target else [
            other.public_preference
            for pid, other in state.runtimes.items()
            if pid != self.id and other.public_preference
        ]
        candidates = [
            target
            for target in targets
            if target in state.scenario.option_ids
            and target != runtime.preferred_option
            and runtime.rank(target) >= STANCE_ACCEPTABLE
            and target not in runtime.public_acceptances
        ]
        if not candidates or self.persona.hard_blocker:
            return None
        target = self.rng.choice(candidates)
        if self.rng.random() >= movement_probability(self.persona.sim_params.stubbornness):
            return None
        source, reason = self._positive_point(state, target, allow_used=True)
        switch = self._should_switch_during_discussion(state, target)
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            priority=BidPriority.NORMAL,
            act=ActionType.ACCEPT,
            option_focus=(target,),
            reason=reason,
            reason_source=source,
            personal_context=self.persona.private_goal,
            stance_update=StanceUpdate(
                kind=(
                    StanceUpdateKind.SWITCH_PREFERRED
                    if switch
                    else StanceUpdateKind.MAKE_ACCEPTABLE
                ),
                option_id=target,
                previous_option_id=runtime.preferred_option,
                movement_reason=reason,
            ),
        )

    def _should_switch_during_discussion(
        self, state: DialogueState, target: str
    ) -> bool:
        """Allow an occasional grounded switch when it can help convergence.

        The target must already be visible in the recent exchange and have at
        least as much public support as the participant's current preference.
        The ordinary stubbornness movement draw has already succeeded before
        this helper is called.
        """
        runtime = state.runtimes[self.id]
        recent_focus = {
            option_id
            for turn in state.participant_turns[-3:]
            if turn.action is not None
            for option_id in turn.action.option_focus
        }
        if target not in recent_focus:
            return False
        counts = Counter(
            other.public_preference
            for other in state.runtimes.values()
            if other.public_preference in state.scenario.option_ids
        )
        return counts[target] > counts[runtime.preferred_option]

    def _positive_point(
        self, state: DialogueState, option_id: str, *, allow_used: bool = False
    ) -> tuple[ReasonSource | None, str]:
        option = state.scenario.option(option_id)
        stance = self.persona.option_stances.get(option_id)
        points = [
            (
                ReasonSource(option_id, "upside", option.upside),
                stance.reason_for if stance and stance.reason_for else option.upside,
            )
        ] if option.upside else []
        return self._choose_point(state, points, allow_used=allow_used)

    def _negative_point(
        self,
        state: DialogueState,
        option_id: str,
        *,
        allow_used: bool = False,
        require_publicly_unseen: bool = False,
    ) -> tuple[ReasonSource | None, str]:
        option = state.scenario.option(option_id)
        stance = self.persona.option_stances.get(option_id)
        points = [
            (
                ReasonSource(option_id, "concern", option.concern),
                stance.reason_against if stance and stance.reason_against else option.concern,
            )
        ] if option.concern else []
        if require_publicly_unseen:
            points = [
                point
                for point in points
                if state.public_point_counts.get(point[0].point_key, 0) == 0
            ]
        return self._choose_point(state, points, allow_used=allow_used)

    def _neutral_point(
        self,
        state: DialogueState,
        option_id: str,
        *,
        allow_used: bool = False,
    ) -> tuple[ReasonSource | None, str]:
        option = state.scenario.option(option_id)
        points = [
            (ReasonSource(option_id, key, value), self._fact_reason(key, value))
            for key, value in option.attrs.items()
        ]
        return self._choose_point(state, points, allow_used=allow_used)

    def _choose_point(
        self,
        state: DialogueState,
        points: list[tuple[ReasonSource, str]],
        *,
        allow_used: bool,
    ) -> tuple[ReasonSource | None, str]:
        runtime = state.runtimes[self.id]
        available = [
            point
            for point in points
            if allow_used or point[0].point_key not in runtime.used_point_keys
        ]
        if not available:
            return None, ""

        fresh = [
            point
            for point in available
            if point[0].point_key not in state.recent_point_keys
            and state.public_point_counts.get(point[0].point_key, 0) < MAX_PUBLIC_POINT_USES
        ]
        if fresh:
            return self.rng.choice(fresh)
        if allow_used:
            least_used = min(
                state.public_point_counts.get(point[0].point_key, 0)
                for point in available
            )
            fallback = [
                point
                for point in available
                if state.public_point_counts.get(point[0].point_key, 0) == least_used
            ]
            return self.rng.choice(fallback)
        return None, ""

    @staticmethod
    def _fact_reason(attribute: str, value: str) -> str:
        label = " ".join(attribute.replace("_", " ").split())
        label = re_sub_words(label)
        return f"the {label} is {value}" if label else str(value)

    def _action(
        self,
        act: ActionType,
        focus: tuple[str, ...],
        reason: str,
        source: ReasonSource | None,
        *,
        addressee: str | None = None,
    ) -> UserAction:
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=True,
            priority=BidPriority.NORMAL,
            act=act,
            option_focus=focus,
            addressee_id=addressee,
            reason=reason,
            reason_source=source,
            personal_context=self.persona.private_goal,
        )

    def _deduplicate(
        self, candidates: list[tuple[UserAction, float]]
    ) -> list[tuple[UserAction, float]]:
        seen: set[tuple[object, ...]] = set()
        result: list[tuple[UserAction, float]] = []
        for action, weight in candidates:
            identity = (action.act, action.option_focus, action.point_keys, action.addressee_id)
            if identity in seen:
                continue
            seen.add(identity)
            result.append((action, weight))
        return result

    def _weighted_choice(
        self, candidates: list[tuple[UserAction, float]]
    ) -> tuple[UserAction, float]:
        weights = [max(0.01, weight) for _, weight in candidates]
        index = self.rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        return candidates[index]

    def _silence(self) -> UserAction:
        return UserAction(
            speaker_id=self.id,
            wants_to_speak=False,
            priority=BidPriority.NORMAL,
            act=ActionType.REACT,
        )
