"""Routing policy: pure route/speaker/act/option/addressee selection.

PolicyMixin selects — who speaks, which dialogue act, which thread drives the
turn, and which option is the vote candidate — and returns MoveIntent objects.
It is read-only over DialogueState (contract 4.1): no orchestration, no I/O,
no state mutation. Phase transitions, narrowing/vote readiness, and the repair
machine live in controller/flow.py.
"""

from __future__ import annotations

import math
import random
from collections import Counter

from aliases import short_alias_map
from config_loader import cfg
from consensus import discussion_positive_mentions, participant_turn_count, public_evidence
from models import (
    ActType,
    DialogueState,
    MoveIntent,
    Persona,
    Phase,
    ThreadStatus,
    ThreadType,
    TurnRecord,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
    _DECISION_ACTS,
    _DISCUSSION_ACTS,
)
from controller import threads as threads_engine
from parsing import round_reason_snippets, used_commitment_phrases
from simulator import expected_turn_share
from style import (
    first_person_opening_fraction,
    name_prefix_fraction,
    option_opening_fraction,
    repeated_opening_token,
    repeated_pattern,
    surface_pattern,
    we_opening_fraction,
)
from utils import preset_dominance_weight, weighted_choice

# The only acts the NORMAL sampler may select (7.1). answer/process/compromise/
# softening are route- or phase-driven and never sampled.
_NORMAL_ACT_MAPPING = {
    "support": ActType.SUPPORT,
    "concern": ActType.CONCERN,
    "ask": ActType.ASK,
    "compare": ActType.COMPARE,
    "comment": ActType.COMMENT,
}


class PolicyMixin:
    def _rank_split_candidates(
        self,
        state: DialogueState,
        votes_by_id: dict[str, str],
        *,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, list[Persona], list[Persona], dict]]:
        """Rank concrete split-vote candidates from formal visible votes.

        Candidate choice is intentionally deterministic. The previous scorer mixed
        visible vote count with stochastic shift checks, which allowed a one-vote
        option to beat a visible plurality in some ``2-1-1`` splits. This ranking
        treats the vote structure as the first-order social signal: a strict
        plurality is tested first unless every relevant dissenter has a hard
        blocker. Ties are then broken by objection load and compromise fit.
        """
        exclude = exclude or set()
        counts = Counter(v for v in votes_by_id.values() if v in state.scenario.option_ids)
        if not counts:
            return []
        positive_mentions = discussion_positive_mentions(state)
        majority_threshold = math.ceil(float(cfg.consensus.majority_fraction) * len(state.personas))

        ranked: list[tuple[tuple[float, float, float, float, float, str], str, list[Persona], list[Persona], dict]] = []
        for candidate, count in counts.items():
            if candidate in exclude:
                continue
            dissenters = [p for p in state.personas if votes_by_id.get(p.id) != candidate]
            hard_blockers = [p for p in dissenters if self._hard_blocks_candidate(state, p, candidate)]
            if dissenters and len(hard_blockers) == len(dissenters):
                # Nobody outside the candidate camp can even conditionally move.
                # Testing this first would be a performative dead end.
                continue
            movable = [p for p in dissenters if p not in hard_blockers and not self._valid_holdout_against(state, p, candidate)]
            required_movers = max(0, majority_threshold - count)
            if required_movers == 0:
                required_movers = 1
            if len(movable) < required_movers:
                continue
            movable.sort(key=lambda p: (self._candidate_resistance(state, p, candidate), p.id))
            selected_movers = movable[:required_movers]
            resistance = [self._candidate_resistance(state, p, candidate) for p in selected_movers]
            avg_resistance = sum(resistance) / max(1, len(resistance))
            compromise_fit = sum(1.0 - p.sim_params.switch_resistance for p in selected_movers) / max(1, len(selected_movers))
            positive_count = int(positive_mentions.get(candidate, 0))
            support_quality = 0.25 * state.coverage[candidate].reasons + 0.10 * state.coverage[candidate].mentions
            meta = {
                "votes": count,
                "positive_mentions": positive_count,
                "required_movers": required_movers,
                "selected_mover_ids": [p.id for p in selected_movers],
                "hard_blockers": len(hard_blockers),
                "avg_resistance": round(avg_resistance, 3),
                "compromise_fit": round(compromise_fit, 3),
            }
            # Formal vote count dominates. When formal counts tie (for example
            # 1-1-1), the option with the most positive visible discussion
            # mentions is tested. Remaining signals only break further ties.
            key = (
                float(count),
                float(positive_count),
                -float(len(hard_blockers)),
                -avg_resistance,
                compromise_fit + support_quality,
                candidate,
            )
            ranked.append((key, candidate, dissenters, selected_movers, meta))

        ranked.sort(
            key=lambda item: (
                -item[0][0], -item[0][1], -item[0][2], -item[0][3], -item[0][4], item[1]
            )
        )
        return [(candidate, dissenters, movers, meta) for _key, candidate, dissenters, movers, meta in ranked]

    @staticmethod
    def _hard_blocks_candidate(state: DialogueState, persona: Persona, candidate: str) -> bool:
        rt = state.runtimes[persona.id]
        return rt.rank(candidate) <= STANCE_REJECTED

    # ------------------------------------------------------------------
    # Concern-thread readers (thread state is the single concern source)
    # ------------------------------------------------------------------

    @staticmethod
    def _live_own_concerns(state: DialogueState, persona_id: str, option_id: str) -> int:
        """Hot/cooling concern or blocker threads this sim raised against the option."""
        return sum(
            1 for t in state.threads.values()
            if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
            and t.status in (ThreadStatus.HOT, ThreadStatus.COOLING)
            and t.started_by == persona_id
            and option_id in t.focus_options
        )

    @staticmethod
    def _unanswered_own_concerns(state: DialogueState, persona_id: str, option_id: str) -> int:
        """Hot (never relevantly answered) concern/blocker threads by this sim."""
        return sum(
            1 for t in state.threads.values()
            if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
            and t.status is ThreadStatus.HOT
            and t.started_by == persona_id
            and option_id in t.focus_options
        )

    @staticmethod
    def _hot_concern_count(state: DialogueState) -> int:
        return sum(
            1 for t in state.threads.values()
            if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
            and t.status is ThreadStatus.HOT
        )


    @staticmethod
    def _valid_holdout_against(state: DialogueState, persona: Persona, candidate: str) -> bool:
        """Whether keeping dissent is more realistic than converting to consensus.

        This protects majority outcomes using only explicit signals: the
        candidate's rank for this sim, the sim's own live concerns against it,
        and switch_resistance. Hard blockers are handled separately by
        ``_hard_blocks_candidate``.
        """
        rt = state.runtimes[persona.id]
        current = rt.explicit_vote or rt.current_acceptance or rt.public_lean or rt.top_option() or persona.preferred_option
        if candidate not in state.scenario.option_ids or current == candidate:
            return False
        if rt.rank(candidate) <= STANCE_REJECTED:
            return True
        unanswered = PolicyMixin._unanswered_own_concerns(state, persona.id, candidate)
        # An ordinary reservation is not a permanent veto: repair explicitly
        # asks a participant to state a concern before deciding. Only strong
        # resistance plus a clearly negative current stance protects the
        # holdout. Hard blockers remain immovable through the rejected rank.
        if (
            persona.sim_params.switch_resistance >= 0.82
            and rt.rank(candidate) < STANCE_ACCEPTABLE
        ):
            return True
        if (
            unanswered >= 2
            and persona.sim_params.switch_resistance >= 0.65
            and rt.rank(candidate) <= STANCE_DISLIKED
        ):
            return True
        return False

    @staticmethod
    def _visible_candidate_openness(
        state: DialogueState, persona_id: str, candidate: str
    ) -> float:
        """Visible pre-vote openness used only to choose plausible movers.

        This never counts as a vote and never changes public candidate scores.
        It simply prevents the repair controller from ignoring that a person
        already warmed to, accepted, supported, or objected to the tested
        candidate in the transcript.
        """
        score = 0.0
        for turn in state.turns:
            if (
                turn.speaker_id != persona_id
                or turn.state_mutation_blocked
                or turn.evidence is None
                or turn.phase in {Phase.VOTING, Phase.COMPROMISE_REPAIR, Phase.CLOSING}
            ):
                continue
            evidence = turn.evidence
            if any(s.option_id == candidate for s in evidence.softenings):
                score += 2.0
            if any(
                c.option_id == candidate and c.kind == "accept"
                for c in evidence.commitments
            ):
                score += 2.0
            if any(s.option_id == candidate for s in evidence.supports):
                score += 1.0
            if any(p.option_id == candidate for p in evidence.proposals):
                score += 1.0
            if any(c.option_id == candidate for c in evidence.concerns):
                score -= 1.0
            if any(
                b.option_id == candidate and b.action == "raised"
                for b in evidence.blockers
            ):
                score -= 3.0
        return max(-4.0, min(4.0, score))

    @staticmethod
    def _candidate_resistance(state: DialogueState, persona: Persona, candidate: str) -> float:
        """Lower means the dissenter is a better candidate for a switch/stay beat.

        This scores final movement (switch/stay/repair beats), so the trait term
        is switch_resistance, not discussion stubbornness.
        """
        rt = state.runtimes[persona.id]
        if rt.rank(candidate) <= STANCE_REJECTED:
            return 99.0
        resistance = 0.70 * persona.sim_params.switch_resistance
        # A switch must be earned (P6): the sim's own visible, still-unanswered
        # objections against the candidate are resistance, not noise. A bridge
        # phrase alone doesn't resolve a concern the transcript left open.
        if rt.rank(candidate) <= STANCE_DISLIKED:
            resistance += 0.25
        resistance += 0.30 * PolicyMixin._unanswered_own_concerns(state, persona.id, candidate)
        openness = PolicyMixin._visible_candidate_openness(state, persona.id, candidate)
        resistance -= 0.10 * openness
        if rt.current_acceptance == candidate or rt.public_lean == candidate:
            resistance -= 0.25
        if candidate in persona.preferred_options:
            resistance -= 0.25
        if rt.rank(candidate) >= STANCE_ACCEPTABLE:
            resistance -= 0.20
        if rt.rank(candidate) >= STANCE_PREFERRED or rt.top_option() == candidate:
            resistance -= 0.35
        return max(0.0, resistance)

    # ------------------------------------------------------------------
    # Routing policy: who / what / whom
    # ------------------------------------------------------------------

    def _route_discussion_turn(self, state: DialogueState) -> MoveIntent:
        """One readable routing function in the Section 8 priority order.

        1. required repair/clarification and required direct answers
        2. hot primary thread
        3. cooling primary thread, probabilistically
        4. option coverage, only when local interaction is quiet
        5. rare same-speaker continuation
        6. normal weighted discussion act

        Phase/progress nudges (slot 8) run in the discussion loop before this
        router; narrowing/voting gates (slot 9) live in _ready_to_narrow.
        """
        answer_thread = self._required_answer_thread(state)
        if answer_thread is not None:
            # A hot question thread's required respondent answers on the next
            # turn; only a stronger validation/safety condition downstream may
            # prevent it. Thread-driven: no separate obligation state.
            return self._answer_intent_for_thread(state, answer_thread)

        candidate = self._public_candidate(state) or self._latent_leading_option(state)
        candidate_pair = [candidate] if candidate else []

        # Hot primary thread drives the next local move deterministically.
        primary = threads_engine.select_primary_thread(state, candidate_options=candidate_pair)
        if primary is not None:
            intent = self._thread_intent(state, primary)
            if intent is not None:
                return intent

        # Cooling primary thread: optional continuation, never scripted.
        cooling = threads_engine.select_primary_thread(
            state, candidate_options=candidate_pair, include_cooling=True
        )
        if cooling is not None and cooling.status is ThreadStatus.COOLING:
            intent = self._maybe_cooling_continuation(state, cooling)
            if intent is not None:
                return intent

        # Option coverage only when local interaction is quiet: a hot thread —
        # even one that could not route a speaker this turn — blocks coverage.
        coverage_gap = self._coverage_gap_option(state)
        if coverage_gap is not None and primary is None:
            # The bounded attempt is charged post-turn (dialogue._post_turn_route_accounting):
            # route selection stays read-only, but a completed generation attempt
            # still consumes the per-option bound so one ignored option cannot
            # be re-routed forever.
            speaker = self._speaker_for_option_coverage(state, coverage_gap)
            current = state.runtimes[speaker.id].top_option() or speaker.preferred_option
            focus = [coverage_gap]
            if current in state.scenario.option_ids and current != coverage_gap:
                focus.append(current)
            gap_name = short_alias_map(state.scenario.options).get(coverage_gap, coverage_gap)
            return MoveIntent(
                speaker_id=speaker.id,
                act=ActType.COMPARE,
                reason=(
                    f"bring {gap_name} into the discussion — it has not really come up yet — "
                    "and compare it with your current lean"
                ),
                route_source="coverage",
                option_focus=focus,
            )

        # Rare same-speaker continuation (issue 6): a short, genuinely additive
        # follow-up to the speaker's own last message. Normal turns still hard-
        # exclude the last speaker in _choose_speaker.
        continuation = self._maybe_continuation_intent(state)
        if continuation is not None:
            return continuation

        return self._normal_intent(state)

    def _normal_intent(self, state: DialogueState) -> MoveIntent:
        """Slot 6: one normal weighted discussion turn (speaker/act/target/focus)."""
        speaker = self._choose_speaker(state)
        act = self._choose_discussion_act(state, speaker)
        target_turn = self._choose_target_turn(state, speaker, act)
        target_speaker = target_turn.speaker_id if target_turn and target_turn.speaker_id != "moderator" else None
        # Direct addressing scales with group size (P3): with two people the
        # addressee is always obvious, with three usually; names stay for the
        # functional cases (questions, invites, obligations set elsewhere).
        address_probability = float(cfg.routing.direct_address_probability)
        if len(state.personas) == 2:
            address_probability *= 0.15
        elif len(state.personas) == 3:
            address_probability *= 0.6
        addressee = target_speaker if target_speaker and random.random() < address_probability else None
        focus = self._focus_options(state, speaker, act, target_turn)
        reason = self._reason_for_act(state, speaker, act, focus, target_turn)
        return MoveIntent(
            speaker_id=speaker.id,
            act=act,
            reason=reason,
            route_source="normal",
            addressee_id=addressee,
            option_focus=focus,
            respond_to_turn=target_turn.index if target_turn else None,
        )

    def _thread_intent(self, state: DialogueState, thread) -> MoveIntent | None:
        """Route the next turn from the hot primary thread (Section 8, slot 3).

        The thread cools only via the observer, after the routed turn's final
        accepted text actually addresses the issue — a routed intention never
        counts as an addressed concern, mitigation, or comparison response.
        Question threads are handled earlier by the required-answer route.
        """
        aliases = short_alias_map(state.scenario.options)
        last = self._last_participant_id(state)
        option_id = thread.focus_options[0] if thread.focus_options else None

        if thread.thread_type is ThreadType.CONCERN and option_id:
            advocates = [
                p for p in state.personas
                if p.id not in {thread.started_by, last}
                and state.runtimes[p.id].top_option() == option_id
            ]
            others = [
                p for p in state.personas if p.id not in {thread.started_by, last}
            ]
            speaker = self._thread_speaker(state, advocates or others, thread)
            if speaker is None:
                return None
            # The act follows the decided objective (todo_prompt item 4):
            # concede -> CONCERN, defend -> SUPPORT, shared doubt -> CONCERN,
            # neutral fact-grounding -> COMMENT. Concede vs defend is decided
            # by discussion-phase stubbornness; direction never left to the LLM.
            speaker_rank = state.runtimes[speaker.id].rank(option_id)
            if advocates and speaker.sim_params.stubbornness <= 0.35:
                act = ActType.CONCERN
                reason = (
                    f"a concern was raised about {aliases[option_id]}, which you back, and it lands — "
                    "concede the point honestly and say what still matters to you in this choice (no final vote)"
                )
            elif advocates:
                act = ActType.SUPPORT
                reason = (
                    f"a concern was raised about {aliases[option_id]}, which you back — "
                    "respond to the issue directly and defend it with one grounded reason"
                )
            elif speaker_rank <= STANCE_DISLIKED:
                act = ActType.CONCERN
                reason = (
                    f"a concern was raised about {aliases[option_id]} — you share the doubt: add one "
                    "grounded reason from the listed facts why that issue is real"
                )
            else:
                act = ActType.COMMENT
                reason = (
                    f"a concern was raised about {aliases[option_id]} — say in one beat what the listed "
                    "facts actually settle about that issue, without taking a side"
                )
            return MoveIntent(
                speaker_id=speaker.id,
                act=act,
                reason=reason,
                route_source="thread_hot",
                thread_id=thread.thread_id,
                option_focus=[option_id],
                respond_to_turn=thread.source_turn_index,
                addressee_id=thread.started_by if random.random() < 0.5 else None,
            )

        if thread.thread_type is ThreadType.BLOCKER and option_id:
            raiser_id = thread.started_by
            if raiser_id not in state.runtimes:
                return None
            others = [p for p in state.personas if p.id not in {raiser_id, last}]
            speaker = self._thread_speaker(state, others, thread)
            if speaker is None:
                return None
            raiser_name = state.name_for(raiser_id)
            if thread.probe_count == 0:
                # One bounded probe per blocker thread (shared with the
                # moderator probe): ask the blocker what would make it workable.
                return MoveIntent(
                    speaker_id=speaker.id,
                    act=ActType.ASK,
                    reason=(
                        f"{raiser_name} clearly can't accept {aliases[option_id]}; ask them one genuine "
                        "question about what would make it workable or what they'd need instead — no pressure"
                    ),
                    route_source="thread_hot",
                thread_id=thread.thread_id,
                    addressee_id=raiser_id,
                    option_focus=[option_id],
                    respond_to_turn=thread.source_turn_index,
                )
            # The act follows the speaker's own stance toward the blocked
            # option (item 4): a backer points to the addressing fact; anyone
            # else acknowledges the blocker's weight — no "or concede" menus.
            if state.runtimes[speaker.id].rank(option_id) >= STANCE_ACCEPTABLE:
                blocker_act = ActType.SUPPORT
                blocker_reason = (
                    f"{raiser_name} rejected {aliases[option_id]} outright — point to the listed fact "
                    "or concrete condition that could still address their issue; no pressure to convert them"
                )
            else:
                blocker_act = ActType.COMMENT
                blocker_reason = (
                    f"{raiser_name} rejected {aliases[option_id]} outright — acknowledge that blocker "
                    "plainly and say what it means for the group's remaining choices; no pressure"
                )
            return MoveIntent(
                speaker_id=speaker.id,
                act=blocker_act,
                reason=blocker_reason,
                route_source="thread_hot",
                thread_id=thread.thread_id,
                option_focus=[option_id],
                respond_to_turn=thread.source_turn_index,
                addressee_id=raiser_id if random.random() < 0.5 else None,
            )

        if thread.thread_type is ThreadType.COMPARISON and len(thread.focus_options) >= 2:
            pair = list(thread.focus_options[:2])
            others = [p for p in state.personas if p.id not in {thread.started_by, last}]
            speaker = self._thread_speaker(state, others, thread)
            if speaker is None:
                return None
            return MoveIntent(
                speaker_id=speaker.id,
                act=ActType.COMPARE,
                reason=(
                    f"someone just weighed {aliases[pair[0]]} against {aliases[pair[1]]} — engage that "
                    "same trade-off with your own view of it; do not open a different topic"
                ),
                route_source="thread_hot",
                thread_id=thread.thread_id,
                option_focus=pair,
                respond_to_turn=thread.source_turn_index,
            )
        return None

    def _maybe_cooling_continuation(self, state: DialogueState, thread) -> MoveIntent | None:
        """Optional continuation of a cooling thread (Section 8, slot 4).

        Probability depends on thread type (blockers most likely to earn one
        more beat, disagreement next, plain questions/comparisons least).
        Bounded: only a freshly cooled thread continues, and never past the
        soft per-thread contribution cap (the hard cap stops even hot routing,
        inside select_primary_thread).
        """
        tcfg = cfg.threads
        if thread.thread_type is ThreadType.BLOCKER:
            probability = float(tcfg.cooling_continue_if_blocker)
        elif thread.thread_type is ThreadType.CONCERN:
            probability = float(tcfg.cooling_continue_if_disagreement)
        else:
            probability = float(tcfg.cooling_continue_probability)
        if threads_engine.participant_turns_since(state, thread.last_touched_turn) > 1:
            return None
        if thread.contribution_count >= int(tcfg.max_thread_turns_soft):
            return None
        if random.random() >= probability:
            return None
        aliases = short_alias_map(state.scenario.options)
        last = self._last_participant_id(state)
        option_id = thread.focus_options[0] if thread.focus_options else None

        if thread.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER) and option_id:
            # The raiser reacts to the response: visibly accept it or push back
            # once. Their acceptance is what resolves the thread (6.2/6.3).
            raiser_id = thread.started_by
            if raiser_id == last or raiser_id not in state.runtimes:
                return None
            # The controller decides accept vs push-back (item 5), weighted by
            # the raiser's stubbornness, and the act matches the objective
            # (item 4): push-back -> CONCERN, acceptance -> SUPPORT.
            raiser_params = state.persona_by_id(raiser_id).sim_params
            if random.random() < 0.25 + 0.6 * raiser_params.stubbornness:
                raiser_act = ActType.CONCERN
                raiser_reason = (
                    f"someone responded to your point about {aliases[option_id]} — it does not fully "
                    "settle it for you: push back once with the one thing that still bothers you; do "
                    "not restate the original wording"
                )
            else:
                raiser_act = ActType.SUPPORT
                raiser_reason = (
                    f"someone responded to your point about {aliases[option_id]} — it lands: say "
                    "plainly that it settles the issue for you and accept it as a workable tradeoff"
                )
            return MoveIntent(
                speaker_id=raiser_id,
                act=raiser_act,
                reason=raiser_reason,
                route_source="thread_cooling",
                thread_id=thread.thread_id,
                option_focus=[option_id],
                respond_to_turn=thread.last_touched_turn,
                length_hint="short",
            )

        if thread.thread_type is ThreadType.QUESTION:
            # The question stays group-relevant while cooling: another
            # participant may react to the answer just given.
            speaker = self._choose_speaker(state)
            answerer = thread.required_respondent
            if speaker.id in {answerer, last}:
                return None
            act = weighted_choice(
                [ActType.SUPPORT, ActType.CONCERN],
                [0.55 + (1.0 - speaker.sim_params.stubbornness) * 0.3,
                 0.3 + speaker.sim_params.stubbornness * 0.4],
            )
            # The sampled act already fixes the direction (item 5): the reason
            # states that one objective instead of offering a menu.
            if act == ActType.SUPPORT:
                reaction = "say what it settles for you"
            else:
                reaction = "push back once on what it does not settle for you"
            return MoveIntent(
                speaker_id=speaker.id,
                act=act,
                reason=(
                    f"react to the answer just given, staying on the same point: {reaction} — "
                    "do not open a new topic or ask a new question"
                ),
                route_source="thread_cooling",
                thread_id=thread.thread_id,
                option_focus=list(thread.focus_options),
                respond_to_turn=thread.last_touched_turn,
            )

        if thread.thread_type is ThreadType.COMPARISON and len(thread.focus_options) >= 2:
            pair = list(thread.focus_options[:2])
            others = [
                p for p in state.personas
                if p.id != last and p.id not in thread.participants_involved
            ]
            speaker = self._thread_speaker(state, others, thread)
            if speaker is None:
                return None
            return MoveIntent(
                speaker_id=speaker.id,
                act=ActType.COMPARE,
                reason=(
                    f"the group has been weighing {aliases[pair[0]]} against {aliases[pair[1]]} — add "
                    "your own take on that trade-off in one beat"
                ),
                route_source="thread_cooling",
                thread_id=thread.thread_id,
                option_focus=pair,
                respond_to_turn=thread.last_touched_turn,
            )
        return None

    def _thread_speaker(self, state: DialogueState, candidates: list[Persona], thread) -> Persona | None:
        """Thread-turn speaker by relevance, not engagement alone (Section 9.2).

        Deterministic: stance/option relevance dominates, turn-share deficit
        corrects imbalance, engagement is only a secondary participation prior,
        and just-spoke/ping-pong penalties keep the floor moving.
        """
        if not candidates:
            return None
        expected = expected_turn_share(state.personas)
        total = sum(rt.turn_count for rt in state.runtimes.values())
        recent = self._recent_participant_ids(state, 2)

        def score(persona: Persona) -> float:
            rt = state.runtimes[persona.id]
            value = 0.35 * persona.sim_params.engagement  # secondary prior
            for oid in thread.focus_options:
                if rt.rank(oid) != STANCE_NEUTRAL:
                    value += 0.45
                if rt.top_option() == oid:
                    value += 0.60
            # Turn-share deficit corrects real imbalance only; before anyone
            # has spoken there is no imbalance (a zero-total "deficit" would
            # just be engagement again, defeating relevance priority).
            if total > 0:
                value += 3.0 * max(0.0, expected[persona.id] - rt.turn_count / total)
            if recent and persona.id == recent[0]:
                value -= 1.5
            if len(recent) > 1 and persona.id == recent[1]:
                value -= 0.4
            return value

        return max(candidates, key=lambda p: (score(p), p.id))

    @staticmethod
    def _silence_streak(state: DialogueState, persona_id: str) -> int:
        """Participant turns since this sim last spoke (all of them if never)."""
        streak = 0
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                continue
            if turn.speaker_id == persona_id:
                break
            streak += 1
        return streak

    def _maybe_continuation_intent(self, state: DialogueState) -> MoveIntent | None:
        """Rare same-speaker follow-up (issue 6): an afterthought, clarification,
        or small self-correction right after the sim's own turn. Only when it is
        a real continuation-type move — never a repeat of the same question or
        point (validated as CONTINUATION_REPEATS) — short, and chain-capped:
        no same-speaker run longer than 3 turns."""
        participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
        if len(participant_turns) < 2:
            return None
        last = participant_turns[-1]
        # Chain length of the last speaker's current run of consecutive turns.
        chain = 0
        for turn in reversed(participant_turns):
            if turn.speaker_id != last.speaker_id:
                break
            chain += 1
        if chain >= 3 or len(state.personas) < 2:
            return None
        persona = state.persona_by_id(last.speaker_id)
        probability = 0.06
        if chain == 2:
            probability *= 0.5
        # P3: a direct question to another sim means the addressee's answer owns
        # the floor. A continuation may still slip in as a short addendum before
        # the answer, but rarely — it must never replace the expected reply.
        pending_direct = any(
            t.thread_type is ThreadType.QUESTION
            and t.status is ThreadStatus.HOT
            and t.source_turn_index == last.index
            and t.required_respondent
            and t.required_respondent != last.speaker_id
            for t in state.threads.values()
        )
        if pending_direct:
            probability *= 0.4
        if random.random() >= probability:
            return None
        asked_question = "?" in last.text
        if asked_question:
            purpose = (
                "add one quick practical addendum to what you just asked — a detail or why it matters, on "
                "the SAME topic; do not repeat, rephrase, or answer the question you just asked"
            )
        else:
            purpose = random.choice([
                "add one quick afterthought to the point you just made — a small practical detail you forgot ('Oh, and…')",
                "clarify one thing from your last message in different words so it can't be misread ('Just to be clear…')",
                "soften or correct one small thing you just said ('Actually, …') without changing your overall point",
            ])
        # A continuation inherits its own previous focus (P3): it deepens the
        # point just made instead of jumping to another option or issue. When
        # the text named no option, the routed intent still knows the topic.
        focus = [oid for oid in last.mentioned_options() if oid in state.scenario.option_ids][:2]
        if not focus and last.intent:
            focus = [oid for oid in last.intent.option_focus if oid in state.scenario.option_ids][:2]
        return MoveIntent(
            speaker_id=persona.id,
            act=ActType.SUPPORT,
            reason=purpose,
            route_source="continuation",
            option_focus=focus,
            length_hint="short",
            continuation=True,
        )

    def _choose_speaker(self, state: DialogueState) -> Persona:
        recent_speakers = self._recent_participant_ids(state, 2)
        last_speaker = recent_speakers[0] if recent_speakers else None
        prior_speaker = recent_speakers[1] if len(recent_speakers) > 1 else None
        preset = getattr(cfg, "corpus_active", None)
        total_turns = sum(rt.turn_count for rt in state.runtimes.values())
        expected = expected_turn_share(state.personas)
        adaptation = float(cfg.routing.get("trait_share_adaptation", 3.5))
        overshoot = float(cfg.routing.get("max_share_overshoot", 0.12))
        silence_cap = int(cfg.routing.get("max_silence_rounds", 2)) * len(state.personas)
        dominant_id = None
        if preset:
            dominant_id = max(
                state.personas,
                key=lambda p: (p.sim_params.engagement, p.id),
            ).id
        candidates: list[Persona] = []
        weights: list[float] = []
        for persona in state.personas:
            if len(state.personas) > 1 and persona.id == last_speaker:
                continue
            rt = state.runtimes[persona.id]
            p = persona.sim_params
            base = 0.35 + p.engagement
            if preset:
                base = preset_dominance_weight(
                    base, persona.id == dominant_id, rt.turn_count,
                    total_turns, len(state.personas), preset,
                    float(cfg.routing.quiet_speaker_boost),
                )
            else:
                # Trait-weighted participation (issue 1): pull each sim's actual
                # turn share toward its trait-derived target instead of strict
                # turn-count equalization. Behind target -> boosted, ahead ->
                # damped, so engagement visibly shapes who talks how much.
                share = rt.turn_count / total_turns if total_turns > 0 else expected[persona.id]
                base *= math.exp(adaptation * (expected[persona.id] - share))
                # Anti-monopoly: clearly past the target share means a hard damp,
                # not a soft one — high engagement may lead, never monologue.
                # 0.30 (not 0.20) so legitimate trait-derived dominance is bent,
                # not erased (P4); the exp() adaptation above already pulls back.
                if total_turns >= len(state.personas) and share - expected[persona.id] > overshoot:
                    base *= 0.30
                # Minimum visibility: nobody disappears. A sim silent for two
                # full rounds gets pushed back in regardless of traits.
                if self._silence_streak(state, persona.id) >= silence_cap:
                    base += 2.0 * float(cfg.routing.quiet_speaker_boost)
            # Discourage two speakers from ping-ponging when others are available.
            if persona.id == prior_speaker and len(state.personas) > 2:
                base *= 0.5
            candidates.append(persona)
            weights.append(base)
        if not candidates:
            candidates = state.personas[:]
            weights = [1.0 for _ in candidates]
        return weighted_choice(candidates, weights)

    def _choose_discussion_act(self, state: DialogueState, speaker: Persona) -> ActType:
        """Sample one NORMAL discussion act (7.1): support/concern/ask/compare/comment.

        answer is route-driven by question threads, process/compromise belong to
        narrowing and repair flows, and softening is an observed stance effect —
        none of them is ever sampled here.
        """
        raw = {
            key: float(value)
            for key, value in cfg.routing.move_weights.items()
            if key in _NORMAL_ACT_MAPPING
        }
        p = speaker.sim_params
        raw["concern"] = raw.get("concern", 0.0) * (0.55 + 0.75 * p.stubbornness + 0.35 * p.directness)
        raw["support"] = raw.get("support", 0.0) * (0.70 + (1.0 - p.stubbornness))
        # A question is worth asking when an unresolved objection is on the table.
        if self._hot_concern_count(state) > 0:
            raw["ask"] = raw.get("ask", 0.0) * 1.4
        if self._recent_question_count(state) >= 1:
            raw["ask"] = raw.get("ask", 0.0) * 0.25
        if participant_turn_count(state) >= max(state.min_discussion_turns, state.force_narrow_turns - 2):
            raw["ask"] = raw.get("ask", 0.0) * 0.25
        # Right after an answer, keep developing the same thread rather than
        # immediately opening a new question.
        participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
        if participant_turns:
            last_turn = participant_turns[-1]
            last_act = last_turn.intent.act if last_turn.intent else last_turn.realized_act()
            if last_act == ActType.ANSWER:
                raw["ask"] = raw.get("ask", 0.0) * 0.3
                raw["support"] = raw.get("support", 0.0) * 1.2
                raw["concern"] = raw.get("concern", 0.0) * 1.15
        # If recent turns use the same surface pattern, prefer comparison or
        # questions over more support/concern phrasing.
        recent_texts = self._recent_participant_texts(state, int(cfg.style.repeated_pattern_window))
        recent_patterns = [surface_pattern(t) for t in recent_texts]
        templated = sum(recent_patterns.count(pat) for pat in ("concede_but", "worry_but", "tradeoff_but"))
        if templated >= 2:
            raw["support"] = raw.get("support", 0.0) * 0.6
            raw["concern"] = raw.get("concern", 0.0) * 0.7
            raw["compare"] = raw.get("compare", 0.0) * 1.3
            raw["ask"] = raw.get("ask", 0.0) * 1.4
        names = list(raw)
        return _NORMAL_ACT_MAPPING[weighted_choice(names, [raw[k] for k in names])]


    @staticmethod
    def _last_targeted_speaker(state: DialogueState) -> str | None:
        """Whom the most recent ACCEPTED turn responded to, from TurnRecords only.

        Routing keeps no memory of unaccepted selections (contract 4.1): the
        anti-pile-on damping below derives from the accepted history, so
        repeated route selection over identical history is side-effect free.
        """
        by_index = {t.index: t for t in state.turns}
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator" or turn.intent is None:
                continue
            target = (
                by_index.get(turn.intent.respond_to_turn)
                if turn.intent.respond_to_turn is not None
                else None
            )
            return target.speaker_id if target and target.speaker_id != "moderator" else None
        return None

    def _choose_target_turn(self, state: DialogueState, speaker: Persona, act: ActType) -> TurnRecord | None:
        """Score a pool of recent threads instead of always taking the last line.

        Open questions, objections/blockers, minority voices, and turns about the
        leading or under-discussed options outrank plain recency, so earlier
        unresolved points get revisited instead of dying after one reply (I8).
        """
        pool = [t for t in state.turns if t.speaker_id not in {"moderator", speaker.id}]
        if not pool:
            return None
        # An answer turn must target the pending question when one exists.
        if act == ActType.ANSWER:
            question_thread = self._required_answer_thread(state)
            if question_thread:
                for turn in pool:
                    if turn.index == question_thread.source_turn_index:
                        return turn
            return pool[-1]
        window = pool[-int(cfg.routing.get("target_window", 6)):]
        leading = self._public_candidate(state) or self._latent_leading_option(state)
        under = self._least_mentioned_option(state)
        open_turn_ids = {
            t.source_turn_index
            for t in state.threads.values()
            if t.thread_type is ThreadType.QUESTION and t.status is ThreadStatus.HOT
        }
        latest_index = window[-1].index
        last_target = self._last_targeted_speaker(state)
        weights: list[float] = []
        for turn in window:
            score = 1.0 / (1.0 + 0.6 * (latest_index - turn.index))
            if turn.index in open_turn_ids:
                score += 2.0
            elif "?" in turn.text:
                score += 0.5
            mentioned = turn.mentioned_options()
            if turn.objected_options():
                score += 1.0
            if leading and leading in mentioned:
                score += 0.6
            if under and under in mentioned:
                score += 0.4
            rt = state.runtimes.get(turn.speaker_id)
            if leading and rt and rt.top_option() and rt.top_option() != leading:
                score += 0.6  # minority/holdout voices stay in play
            if turn.speaker_id == last_target:
                score *= 0.6  # don't keep targeting the same person
            weights.append(score)
        return weighted_choice(window, weights)

    @staticmethod
    def _least_mentioned_option(state: DialogueState) -> str | None:
        pairs = sorted(state.coverage.items(), key=lambda kv: (kv[1].mentions, kv[0]))
        return pairs[0][0] if pairs else None

    def _focus_options(self, state: DialogueState, speaker: Persona, act: ActType, target_turn: TurnRecord | None) -> list[str]:
        current = state.runtimes[speaker.id].top_option() or speaker.preferred_option
        if act is ActType.COMPARE:
            # A natural comparison has exactly two required sides.  Earlier
            # normal routing could collect two options from the target turn,
            # then append the speaker's current lean and ask the LLM to cover
            # all three.  That produced confused prompts and avoidable drops.
            targets = [
                oid for oid in (target_turn.mentioned_options() if target_turn else [])
                if oid in state.scenario.option_ids and oid != current
            ]
            rival = targets[0] if targets else self._rival_option(
                state, speaker, exclude={current} if current else set()
            )
            pair = [oid for oid in (current, rival) if oid in state.scenario.option_ids]
            if len(pair) < 2 and target_turn:
                for oid in target_turn.mentioned_options():
                    if oid in state.scenario.option_ids and oid not in pair:
                        pair.append(oid)
                    if len(pair) == 2:
                        break
            return pair[:2]

        ids: list[str] = []
        if target_turn:
            ids.extend(target_turn.mentioned_options())
        if current and current not in ids:
            ids.append(current)
        if act in {ActType.COMPARE, ActType.CONCERN, ActType.ASK}:
            rival = self._rival_option(state, speaker, exclude=set(ids))
            if rival:
                ids.append(rival)
        return [x for x in ids if x in state.scenario.option_ids][:3]

    def _reason_for_act(self, state: DialogueState, speaker: Persona, act: ActType, focus: list[str], target_turn: TurnRecord | None) -> str:
        """Instruction fragment for a sampled NORMAL act (7.1) — only the five
        sampled acts reach this; contextual acts get their reasons at their
        route/flow sites."""
        names = short_alias_map(state.scenario.options)
        focus_names = ", ".join(names[o] for o in focus) if focus else "the options"
        current = state.runtimes[speaker.id].top_option() or speaker.preferred_option
        current_name = names.get(current, current)
        if act == ActType.SUPPORT:
            # Support targets ONE option — the speaker's current lean (item 5);
            # a comma-joined focus list left the supported option ambiguous.
            return random.choice([
                f"add a new grounded reason for {current_name}, connected to the current discussion",
                f"bring up a practical, everyday consideration that favors {current_name} and has not come up yet",
                f"say plainly what matters most to you personally in this choice and how {current_name} fits it",
            ])
        if act == ActType.CONCERN:
            rivals = [o for o in focus if o != current]
            if rivals:
                return (
                    f"push back on {names[rivals[0]]} with one concrete concern — "
                    f"you currently favor {current_name}, so aim at the rival, not your own pick"
                )
            return f"name a concern others might raise about {current_name}, and say why it still holds up for you"
        if act == ActType.ASK:
            return f"ask one concrete question that helps compare {focus_names}"
        if act == ActType.COMMENT:
            return random.choice([
                "briefly acknowledge or interpret the point just made — one light beat, no new argument needed",
                f"say in one casual line what matters most to you in this choice, without pushing {focus_names}",
                "add one light observation about where the discussion stands — no support, objection, or question required",
            ])
        if act == ActType.COMPARE:
            return f"compare {focus_names} with one clear trade-off"
        return "respond naturally and move the decision forward"


    def _vote_intent(self, state: DialogueState, persona: Persona, candidate: str) -> MoveIntent:
        rt = state.runtimes[persona.id]
        blocked = candidate in rt.rejected_options()
        current = self._stance_consistent_vote_target(state, persona, candidate)
        if blocked:
            # The alternative must actually be acceptable: never the blocked
            # candidate itself (even when the sim's lean is stuck on it) and
            # never another option this sim has vetoed.
            alternative = next(
                (
                    o for o in [current, *persona.preferred_options, *state.scenario.option_ids]
                    if o in state.scenario.option_ids
                    and o != candidate
                    and o not in rt.rejected_options()
                ),
                candidate,
            )
            return MoveIntent(
                speaker_id=persona.id,
                act=ActType.VOTE,
                reason="cast a clear visible vote for the best acceptable alternative and briefly mention why the candidate is blocked",
                route_source="vote",
                option_focus=[alternative, candidate] if alternative != candidate else [alternative],
                length_hint="short",
                allow_vote_change=True,
                required_vote=alternative,
                old_preference=current,
                allowed_reason="the tested candidate is blocked, so this is the best acceptable option",
            )
        return MoveIntent(
            speaker_id=persona.id,
            act=ActType.VOTE,
            reason=(
                "cast a clear visible final vote for the option you actually choose now; "
                "this formal vote may replace an earlier discussion commitment"
            ),
            route_source="vote",
            option_focus=[current if current in state.scenario.option_ids else candidate],
            length_hint="short",
            allow_vote_change=False,
            required_vote=current if current in state.scenario.option_ids else candidate,
            old_preference=None,
            allowed_reason="this remains your most defensible choice from the visible discussion",
        )

    def _stance_consistent_vote_target(self, state: DialogueState, persona: Persona, candidate: str | None = None) -> str:
        """Best final-vote target consistent with this sim's visible stance.

        The final vote should not silently revert to an old latent favorite after
        the same sim visibly raised objections against it. This helper prefers
        the current/accepted/preferred option only when it has not been rejected
        by the speaker; otherwise it falls back to the best acceptable supported
        option.
        """
        rt = state.runtimes[persona.id]
        rejected = set(rt.disliked_options()) | set(rt.rejected_options())
        def acceptable(oid: str | None) -> bool:
            return bool(oid and oid in state.scenario.option_ids and oid not in rejected)
        candidates: list[str | None] = [
            rt.explicit_vote,
            rt.current_acceptance,
            rt.public_lean,
            rt.top_option(),
            *list(rt.acceptable_options()),
            *persona.preferred_options,
            candidate,
        ]
        visible = sorted(
            state.scenario.option_ids,
            key=lambda oid: (-self._visible_support_count(state, oid, exclude=persona.id), oid),
        )
        candidates.extend(visible)
        for oid in candidates:
            if acceptable(oid):
                return str(oid)
        return next((oid for oid in state.scenario.option_ids if oid not in rt.rejected_options()), state.scenario.option_ids[0])

    # ------------------------------------------------------------------
    # Generation, observation, and state mutation
    # ------------------------------------------------------------------

    def _current_top_pair(self, state: DialogueState) -> list[str]:
        """The two visibly best-supported options (empty/shorter when unsupported)."""
        return list(public_evidence(state).top_pair)

    def _public_candidate(self, state: DialogueState) -> str | None:
        """Option with the strongest public backing, or None without any.

        The score comes from the centralized public-evidence calculation;
        the latent lean only breaks ties among equally backed leaders — it
        shapes whom the moderator asks about, never the outcome.
        """
        leaders = public_evidence(state).candidate_leaders
        if not leaders:
            return None
        # Public ties are resolved deterministically; private ranks never break
        # a public-support tie.
        return leaders[0]

    def _candidate_for_vote(self, state: DialogueState) -> str:
        """Vote candidate: the public candidate, falling back to the latent
        leader only when no public evidence exists at all (a first vote round
        after low-commitment discussion)."""
        return (
            self._public_candidate(state)
            or self._latent_leading_option(state)
            or state.personas[0].preferred_option
        )

    @staticmethod
    def _visible_support_count(state: DialogueState, option_id: str, exclude: str | None = None) -> int:
        """Publicly expressed current backers of the option, never private ranks."""
        backers = public_evidence(state).backing.get(option_id, set())
        return len(backers - {exclude} if exclude else backers)

    @staticmethod
    def _visibly_proposed(state: DialogueState, option_id: str) -> bool:
        return option_id in public_evidence(state).proposals

    def _can_shift_to(
        self,
        state: DialogueState,
        persona: Persona,
        option_id: str,
        *,
        final_decision: bool = False,
    ) -> bool:
        """Whether this sim could plausibly move to the option at all.

        Rank-1 options are hard blocked for everyone. The near-hard trait gate
        depends on context: discussion-phase lean movement is stubbornness
        territory, final switch/vote/repair movement is switch_resistance.
        """
        if option_id in state.runtimes[persona.id].rejected_options():
            return False
        gate_trait = (
            persona.sim_params.switch_resistance
            if final_decision
            else persona.sim_params.stubbornness
        )
        if gate_trait >= 0.85 and option_id != persona.preferred_option:
            return random.random() < 0.04
        return True

    # ------------------------------------------------------------------
    # Moderator helpers
    # ------------------------------------------------------------------

    def _coverage_gap_option(self, state: DialogueState) -> str | None:
        if not bool(cfg.conversation.get("require_option_coverage_before_vote", True)):
            return None
        # Opening turns may naturally leave some non-preferred options untouched.
        # After everyone has spoken once, the router gives each untouched option a
        # light social check before final voting. This is coverage, not forced support.
        turns = participant_turn_count(state)
        if turns <= len(state.personas):
            return None
        # Near the hard cap the remaining turns belong to narrowing/voting, not
        # to a late coverage detour (skip_near_hard_cap).
        if state.hard_max_turns and turns >= state.hard_max_turns - max(2, math.ceil(len(state.personas) / 2)):
            return None
        gaps = [
            (option_id, coverage.mentions, coverage.reasons + coverage.objections)
            for option_id, coverage in state.coverage.items()
            if coverage.mentions == 0 and coverage.coverage_attempts == 0
        ]
        if not gaps:
            return None
        gaps.sort(key=lambda item: (item[1], item[2], item[0]))
        return gaps[0][0]

    def _speaker_for_option_coverage(self, state: DialogueState, option_id: str) -> Persona:
        last = self._last_participant_id(state)
        # Prefer a participant who can plausibly discuss the option: secondary
        # preference first, then engagement and flexibility, while avoiding the
        # last speaker in ordinary discussion.
        candidates = [p for p in state.personas if p.id != last] or state.personas[:]
        def score(persona: Persona) -> tuple[float, float]:
            p = persona.sim_params
            preference_bonus = 1.0 if option_id in persona.preferred_options else 0.0
            return (preference_bonus + 0.60 * p.engagement + 0.20 * (1.0 - p.stubbornness), -state.runtimes[persona.id].turn_count)
        return max(candidates, key=score)

    def _vote_order(self, state: DialogueState, candidate: str) -> list[Persona]:
        return sorted(state.personas, key=lambda p: (state.runtimes[p.id].top_option() != candidate, state.runtimes[p.id].turn_count))

    @staticmethod
    def _recent_participant_texts(state: DialogueState, limit: int) -> list[str]:
        texts = [t.text for t in state.turns if t.speaker_id != "moderator"]
        return texts[-limit:]

    @staticmethod
    def _current_round_texts(state: DialogueState) -> list[str]:
        """Participant turns since the last moderator line — i.e. this vote/beat round."""
        texts: list[str] = []
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                break
            texts.append(turn.text)
        return texts

    def _apply_style_flags(self, state: DialogueState, intent: MoveIntent) -> None:
        """Set compact surface-style flags (no LLM call) to keep dialogue varied.

        Item 8: every tripwire is evidence-based (density/repetition in recent
        visible turns — no proactive random damping), and a turn gets at most
        ONE lexical variation note, the most relevant pattern first. The
        tail-question flag is separate conversational-flow control, not a
        variation note.
        """
        names = [p.name for p in state.personas]
        # Leading names must do interactional work (P5): inviting someone in,
        # asking or challenging a specific person, or answering a specific
        # person in a larger group. Ordinary agreement/build/compare turns
        # don't need the addressee's name up front.
        functional_naming = (
            intent.act == ActType.PROCESS
            or (intent.addressee_id is not None and intent.act in {ActType.ASK, ActType.CONCERN})
            or (intent.addressee_id is not None and len(state.personas) >= 4 and intent.act == ActType.ANSWER)
        )
        window = int(cfg.style.name_prefix_window)
        recent = self._recent_participant_texts(state, window)
        # Tail-question churn (P2): when the last few turns already put two or
        # more questions on the table, statement-type acts should not tack on
        # yet another one. Real question acts (ask/invite/probe) are exempt.
        if (
            intent.act in {ActType.SUPPORT, ActType.ANSWER, ActType.COMPARE, ActType.COMMENT,
                           ActType.COMPROMISE, ActType.CONCERN}
            and sum(1 for t in recent if "?" in t) >= 2
        ):
            intent.suppress_tail_question = True
        # One lexical variation note, most relevant repetition pattern first:
        # rhetorical shape > shared opening word > option-name openings >
        # name-prefix openings > 'I'/'We' openings.
        alias_values = list(short_alias_map(state.scenario.options).values())
        pattern = None
        if intent.act in _DISCUSSION_ACTS:
            pattern_window = int(cfg.style.repeated_pattern_window)
            pattern = repeated_pattern(
                self._recent_participant_texts(state, pattern_window), pattern_window
            )
        opening_window = int(cfg.style.repeated_opening_window)
        exempt_from_opening_notes = intent.act in {ActType.VOTE, ActType.CONCERN}
        if pattern:
            intent.avoid_pattern = pattern
        elif repeated_opening_token(self._recent_participant_texts(state, opening_window), opening_window):
            intent.vary_opening = True
        elif recent and option_opening_fraction(recent, alias_values) >= float(cfg.style.option_opening_max_fraction):
            intent.suppress_option_opening = True
        elif (
            not functional_naming
            and recent
            and name_prefix_fraction(recent, names) >= float(cfg.style.name_prefix_max_fraction)
        ):
            intent.suppress_name_prefix = True
        elif (
            not exempt_from_opening_notes
            and recent
            and first_person_opening_fraction(recent) >= float(cfg.style.i_opening_max_fraction)
        ):
            intent.suppress_i_opening = True
        elif (
            not exempt_from_opening_notes
            and recent
            and we_opening_fraction(recent) >= float(cfg.style.we_opening_max_fraction)
        ):
            intent.suppress_we_opening = True
        # Deterministic anti-chorus for decision beats: phrase families already
        # used in this round OR by this speaker in any earlier turn are
        # off-limits, so a re-asked voter never repeats their own line verbatim
        # across vote rounds (issues #25, I12, I19). Reason snippets stay
        # round-scoped: a persona restating their own justification is coherent.
        if intent.act in _DECISION_ACTS:
            round_texts = self._current_round_texts(state)
            own_texts = [t.text for t in state.turns if t.speaker_id == intent.speaker_id]
            if not intent.avoid_phrases:
                intent.avoid_phrases = used_commitment_phrases(round_texts + own_texts)
            if not intent.avoid_reasons:
                intent.avoid_reasons = round_reason_snippets(round_texts)

    def _recent_question_count(self, state: DialogueState) -> int:
        recent = [t for t in state.turns[-3:] if t.speaker_id != "moderator"]
        return sum(1 for t in recent if "?" in t.text)

    def _rival_option(self, state: DialogueState, speaker: Persona, exclude: set[str]) -> str | None:
        candidates = [o for o in state.scenario.option_ids if o not in exclude and self._can_shift_to(state, speaker, o)]
        if not candidates:
            return None
        leading = self._latent_leading_option(state)
        if leading in candidates:
            return leading
        return random.choice(candidates)

    @staticmethod
    def _last_participant_id(state: DialogueState) -> str | None:
        for turn in reversed(state.turns):
            if turn.speaker_id != "moderator":
                return turn.speaker_id
        return None

    @staticmethod
    def _recent_participant_ids(state: DialogueState, limit: int) -> list[str]:
        """Most recent distinct participant speaker ids, newest first."""
        out: list[str] = []
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator" or turn.speaker_id in out:
                continue
            out.append(turn.speaker_id)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _latent_leading_option(state: DialogueState) -> str | None:
        counts = Counter(rt.top_option() for rt in state.runtimes.values() if rt.top_option())
        return counts.most_common(1)[0][0] if counts else None

    @staticmethod
    def _word_bounds(intent: MoveIntent, persona: Persona) -> tuple[int, int]:
        """Verbosity-driven (min, max) word budget; verbosity only ever
        becomes this numeric range, never prompt prose."""
        budgets = cfg.utterances.word_budgets
        if intent.act == ActType.OPENING:
            base = int(budgets.opening)
        elif intent.act == ActType.ASK:
            base = int(budgets.ask)
        elif intent.act == ActType.ANSWER:
            base = int(budgets.answer)
        elif intent.act in _DECISION_ACTS:
            base = int(budgets.vote)
            # A compromise switch needs room for the bridge clause ("water
            # matters to me, but ...") on top of the direct commitment.
            if intent.allow_vote_change:
                base += 6
        else:
            base = int(budgets.discussion)
        # A continuation is a quick add-on, not a second full turn (issue 6).
        if intent.continuation:
            base = max(8, round(base * 0.55))
        # Reactive beats routed with an explicit short hint (reservation answers,
        # deadlock steps, probe replies) actually get the smaller budget (P4).
        elif intent.length_hint == "short":
            base = max(8, round(base * 0.75))
        p = persona.sim_params
        # A real spread, not a +/-4 nudge: terse sims stay short, chatty ones
        # longer. Verbosity is the only persona parameter affecting length.
        # Keep the range wide enough to be visible in transcripts, not merely
        # in diagnostics: a terse participant receives roughly half the base
        # budget, while a highly verbose one receives about one-and-a-half.
        factor = 0.42 + 1.03 * p.verbosity   # 0.42..1.45, monotonic by trait
        # Verbosity is an average, not a per-turn template: every sim sometimes
        # drops a genuinely short beat (quick agreement, one-line answer), with
        # terse sims doing it more often. Openings, split summaries, and
        # bridge-eligible decisions keep full room for their required content.
        short_beat_ok = (
            intent.act not in {ActType.OPENING, ActType.PROCESS}
            and not (intent.act in _DECISION_ACTS and intent.allow_vote_change)
            and not intent.continuation
        )
        short_probability = 0.02 + 0.36 * (1.0 - p.verbosity) ** 2
        if short_beat_ok and random.random() < short_probability:
            factor *= random.uniform(0.45, 0.60)
        else:
            # Per-turn jitter around the persona's average so consecutive turns
            # by the same sim vary naturally instead of one fixed length (I12).
            factor *= random.uniform(0.97, 1.03)
        max_words = max(6, round(base * factor))
        min_words = max(3, round(max_words * (0.38 + 0.32 * p.verbosity)))
        return min_words, max_words
