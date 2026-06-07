"""Move-intent routing for the next dialogue turn."""

from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from config_loader import cfg
from schemas import ActType, DialogueState, MoveIntent, Phase
from utils import sample_length_hint, weighted_choice


class TurnRouter:
    def next_move(self, state: DialogueState) -> MoveIntent:
        if state.phase == Phase.OPENING:
            return self._opening_move(state)
        if state.open_questions:
            q = state.open_questions[0]
            return MoveIntent(
                speaker_id=q.target_id,
                addressee_id=q.asked_by,
                act=ActType.ANSWER,
                option_focus=self._focus_from_candidate_or_preferences(state, q.target_id),
                reason="answer a direct question before the thread moves on",
                length_hint="medium",
            )
        if state.phase == Phase.NARROWING:
            return self._vote_move(state)
        if state.phase == Phase.CONFIRMATION:
            move = self._confirmation_move(state)
            if move:
                return move
            return self._discussion_move(state, forced_act=ActType.PROPOSE_COMPROMISE)
        return self._discussion_move(state)

    def _opening_move(self, state: DialogueState) -> MoveIntent:
        for persona in state.personas:
            if state.runtimes[persona.id].turn_count == 0:
                return MoveIntent(
                    speaker_id=persona.id,
                    act=ActType.OPENING,
                    option_focus=[persona.preferred_option],
                    reason="briefly state an initial priority, not a final vote",
                    length_hint="short",
                )
        return self._discussion_move(state)

    def _vote_move(self, state: DialogueState) -> MoveIntent:
        for persona in state.personas:
            if not state.runtimes[persona.id].explicit_vote:
                return MoveIntent(
                    speaker_id=persona.id,
                    act=ActType.VOTE,
                    option_focus=[persona.preferred_option],
                    reason="the discussion is developed enough to name a current pick",
                    length_hint="medium",
                )
        # All voted: set candidate to leading option and test it.
        candidate = self._leading_vote(state) or state.candidate_option or state.personas[0].preferred_option
        state.candidate_option = candidate
        state.phase = Phase.CONFIRMATION
        return self._confirmation_move(state) or self._discussion_move(state)

    def _confirmation_move(self, state: DialogueState) -> Optional[MoveIntent]:
        candidate = state.candidate_option or self._leading_vote(state)
        if not candidate:
            return None
        state.candidate_option = candidate
        for persona in state.personas:
            rt = state.runtimes[persona.id]
            if candidate in rt.accepted_options or candidate in rt.rejected_options:
                continue
            act = ActType.ACCEPT
            if persona.is_hard_blocker and candidate in persona.hard_rejections:
                act = ActType.REJECT
            return MoveIntent(
                speaker_id=persona.id,
                act=act,
                option_focus=[candidate],
                reason=f"check whether Option {candidate} is acceptable as the current compromise",
                length_hint="short",
            )
        return None

    def _discussion_move(self, state: DialogueState, forced_act: Optional[ActType] = None) -> MoveIntent:
        speaker = self._select_discussion_speaker(state)
        act = forced_act or self._sample_discussion_act(state, speaker)
        focus = self._focus_for_act(state, speaker, act)
        return MoveIntent(
            speaker_id=speaker,
            act=act,
            option_focus=focus,
            reason=self._reason_for_act(act),
            length_hint=sample_length_hint(cfg.routing.length_hint_probabilities),
        )

    def _select_discussion_speaker(self, state: DialogueState) -> str:
        ids = state.participant_ids()
        if not ids:
            raise ValueError("Cannot route without participants.")
        recent_window = int(cfg.routing.recent_speaker_window)
        recent = [turn.speaker_id for turn in state.turns[-recent_window:] if turn.speaker_id != "moderator"]
        recent_counts = Counter(recent)
        last = recent[-1] if recent else None
        min_turns = min(rt.turn_count for rt in state.runtimes.values())
        weights: list[float] = []
        for pid in ids:
            persona = state.persona_by_id(pid)
            rt = state.runtimes[pid]
            score = 1.0
            if rt.turn_count == 0:
                score += float(cfg.routing.unspoken_boost)
            if rt.turn_count == min_turns:
                score += float(cfg.routing.low_turn_count_boost)
            score += persona.traits.initiative * 0.20
            if pid == last:
                score -= float(cfg.routing.last_speaker_penalty)
            score -= recent_counts[pid] * float(cfg.routing.recent_speaker_penalty)
            if rt.rejected_options:
                score += float(cfg.routing.unresolved_objection_boost)
            weights.append(max(0.01, score))
        return weighted_choice(ids, weights)

    def _sample_discussion_act(self, state: DialogueState, speaker_id: str) -> ActType:
        if state.no_progress_count >= int(cfg.conversation.no_progress_window):
            return ActType.PROPOSE_COMPROMISE
        probs = {
            ActType.COMPARE: float(cfg.routing.compare_intent_probability),
            ActType.PUSH_BACK: float(cfg.routing.pushback_intent_probability),
            ActType.ASK: float(cfg.routing.clarify_intent_probability),
            ActType.PROPOSE_COMPROMISE: float(cfg.routing.compromise_intent_probability),
            ActType.REACT: float(cfg.routing.react_intent_probability),
        }
        # More cooperative people move toward compromise slightly more often.
        persona = state.persona_by_id(speaker_id)
        probs[ActType.PROPOSE_COMPROMISE] += max(0.0, persona.traits.compromise_willingness - 0.6) * 0.12
        acts = list(probs.keys())
        return weighted_choice(acts, [probs[a] for a in acts])

    def _focus_for_act(self, state: DialogueState, speaker_id: str, act: ActType) -> list[str]:
        persona = state.persona_by_id(speaker_id)
        if act == ActType.PROPOSE_COMPROMISE:
            candidate = self._best_compromise_option(state, speaker_id)
            return [candidate]
        if act in {ActType.REACT, ActType.PUSH_BACK, ActType.COMPARE, ActType.ASK}:
            recent_refs = [oid for turn in reversed(state.turns) for oid in turn.act.option_refs]
            if recent_refs:
                focus = [recent_refs[0]]
                if act == ActType.COMPARE and persona.preferred_option not in focus:
                    focus.append(persona.preferred_option)
                return focus[:2]
        return [persona.preferred_option]

    def _focus_from_candidate_or_preferences(self, state: DialogueState, speaker_id: str) -> list[str]:
        if state.candidate_option:
            return [state.candidate_option]
        return [state.persona_by_id(speaker_id).preferred_option]

    def _best_compromise_option(self, state: DialogueState, speaker_id: str) -> str:
        persona = state.persona_by_id(speaker_id)
        counts: Counter[str] = Counter()
        for runtime in state.runtimes.values():
            if runtime.explicit_vote:
                counts[runtime.explicit_vote] += 2
            for option_id in runtime.accepted_options:
                counts[option_id] += 1
        for option_id in persona.acceptable_options:
            counts[option_id] += 1
        if counts:
            return counts.most_common(1)[0][0]
        return persona.acceptable_options[0] if persona.acceptable_options else persona.preferred_option

    def _leading_vote(self, state: DialogueState) -> Optional[str]:
        counts = Counter(rt.explicit_vote for rt in state.runtimes.values() if rt.explicit_vote)
        return counts.most_common(1)[0][0] if counts else None

    def _reason_for_act(self, act: ActType) -> str:
        return {
            ActType.REACT: "add a useful reaction without restating old points",
            ActType.COMPARE: "compare the trade-off between relevant options",
            ActType.PUSH_BACK: "raise a concrete concern, but stay compromise-seeking",
            ActType.ASK: "ask one useful clarification to another participant",
            ActType.PROPOSE_COMPROMISE: "suggest a realistic middle ground if possible",
        }.get(act, "move the discussion forward")
