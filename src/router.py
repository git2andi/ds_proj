"""Turn-taking, speaker selection, addressee selection, and local move routing.

The router follows a practical 3W decomposition for multi-party chat: who should
speak, when they should speak, and whom/what they should respond to.  It returns
MoveIntent objects rather than bare speaker IDs.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from config_loader import cfg
from models import ActType, DialogueState, MoveIntent, Phase, Persona, TurnRecord
from scoring import leading_option, option_support
from utils import weighted_choice


class TurnRouter:
    def next_intent(self, state: DialogueState) -> MoveIntent:
        if state.phase == Phase.OPENING:
            return self._opening_intent(state)

        if state.open_questions:
            q = state.open_questions[0]
            return MoveIntent(
                speaker_id=q.target_id,
                addressee_id=q.asked_by,
                act=ActType.ANSWER,
                option_focus=q.option_focus or self._focus_for_person(state, q.target_id),
                reason="answer the direct question before the thread moves on",
                length_hint="medium",
                respond_to_turn=q.turn_id,
            )

        if state.phase == Phase.NARROWING:
            return self._vote_intent(state)

        if state.phase == Phase.CONFIRMATION:
            confirm = self._confirmation_intent(state)
            if confirm:
                return confirm
            return self._discussion_intent(state, forced_act=ActType.PROPOSE_COMPROMISE)

        return self._discussion_intent(state)

    # ------------------------------------------------------------------

    def _opening_intent(self, state: DialogueState) -> MoveIntent:
        for persona in state.personas:
            if state.runtimes[persona.id].turn_count == 0:
                return MoveIntent(
                    speaker_id=persona.id,
                    act=ActType.OPENING,
                    option_focus=[persona.preferred_option],
                    reason="state an initial priority without voting yet",
                    length_hint="short",
                )
        return self._discussion_intent(state)

    def _vote_intent(self, state: DialogueState) -> MoveIntent:
        for persona in self._least_recent_personas(state):
            if not state.runtimes[persona.id].explicit_vote:
                focus = [state.runtimes[persona.id].current_preference or persona.preferred_option]
                return MoveIntent(
                    speaker_id=persona.id,
                    act=ActType.VOTE,
                    option_focus=focus,
                    reason="the group has discussed enough, so name your current pick and why",
                    length_hint="medium",
                )
        state.candidate_option = leading_option(state) or state.candidate_option
        state.phase = Phase.CONFIRMATION
        return self._confirmation_intent(state) or self._discussion_intent(state, forced_act=ActType.PROPOSE_COMPROMISE)

    def _confirmation_intent(self, state: DialogueState) -> Optional[MoveIntent]:
        candidate = state.candidate_option or leading_option(state)
        if not candidate:
            return None
        state.candidate_option = candidate
        for persona in self._least_recent_personas(state):
            rt = state.runtimes[persona.id]
            if rt.explicit_vote == candidate or candidate in rt.accepted_options:
                continue
            if candidate in rt.hard_rejections:
                continue
            if candidate in rt.soft_rejections and candidate not in persona.acceptable_options:
                act = ActType.REJECT
            elif candidate in persona.hard_rejections:
                act = ActType.REJECT
            elif persona.is_hard_blocker and candidate != persona.preferred_option:
                act = ActType.REJECT
            elif candidate in persona.acceptable_options or candidate == persona.preferred_option:
                act = ActType.ACCEPT
            else:
                # Not initially acceptable: accept only if the candidate clears the persona's
                # private acceptance score (utility-based, not a flat willingness threshold).
                act = ActType.ACCEPT if persona.score_for(candidate) >= int(cfg.scenario.acceptance_score) else ActType.REJECT
            return MoveIntent(
                speaker_id=persona.id,
                act=act,
                option_focus=[candidate],
                reason=f"say whether Option {candidate} works as the current compromise",
                length_hint="short",
            )
        return None

    def _discussion_intent(self, state: DialogueState, forced_act: Optional[ActType] = None) -> MoveIntent:
        gap = self._coverage_gap_option(state)
        if gap:
            speaker_id = self._speaker_for_gap(state, gap)
            act = forced_act or self._act_for_gap(state, speaker_id, gap)
            addressee = self._target_for_act(state, speaker_id, act, gap)
            focus = self._focus_for_act(state, speaker_id, act, gap)
            return MoveIntent(
                speaker_id=speaker_id,
                addressee_id=addressee,
                act=act,
                option_focus=focus,
                reason=f"develop under-discussed Option {gap} with a real trade-off",
                length_hint=self._length_hint(),
            )

        speaker_id = self._select_speaker(state)
        act = forced_act or self._sample_discussion_act(state, speaker_id)
        focus = self._focus_for_act(state, speaker_id, act, None)
        addressee = self._target_for_act(state, speaker_id, act, focus[0] if focus else None)
        return MoveIntent(
            speaker_id=speaker_id,
            addressee_id=addressee,
            act=act,
            option_focus=focus,
            reason=self._reason_for_act(act),
            length_hint=self._length_hint(),
        )

    # ------------------------------------------------------------------

    def _select_speaker(self, state: DialogueState) -> str:
        ids = state.participant_ids()
        recent = self._recent_speaker_ids(state)
        last = recent[-1] if recent else None
        recent_counts = Counter(recent)
        min_turns = min((state.runtimes[pid].turn_count for pid in ids), default=0)
        weights: list[float] = []
        for pid in ids:
            persona = state.persona_by_id(pid)
            rt = state.runtimes[pid]
            score = 1.0
            if rt.turn_count == 0:
                score += float(cfg.routing.unspoken_boost)
            if rt.turn_count == min_turns:
                score += float(cfg.routing.low_turn_count_boost)
            score += persona.traits.initiative * float(cfg.routing.initiative_weight)
            score += len(rt.soft_rejections) * float(cfg.routing.unresolved_objection_boost)
            if pid == last:
                score -= float(cfg.routing.last_speaker_penalty)
            score -= recent_counts[pid] * float(cfg.routing.recent_speaker_penalty)
            weights.append(max(0.01, score))
        return weighted_choice(ids, weights)

    def _speaker_for_gap(self, state: DialogueState, option_id: str) -> str:
        ids = state.participant_ids()
        recent = self._recent_speaker_ids(state)
        last = recent[-1] if recent else None
        min_turns = min((state.runtimes[pid].turn_count for pid in ids), default=0)
        weights: list[float] = []
        for pid in ids:
            persona = state.persona_by_id(pid)
            rt = state.runtimes[pid]
            score = 1.0
            if rt.turn_count == min_turns:
                score += float(cfg.routing.low_turn_count_boost)
            if option_id == persona.preferred_option:
                score += float(cfg.routing.preferred_option_gap_boost)
            if option_id in persona.acceptable_options:
                score += float(cfg.routing.acceptable_option_gap_boost)
            if option_id in persona.soft_rejections or option_id in persona.hard_rejections:
                score += float(cfg.routing.unresolved_objection_boost)
            score += persona.traits.initiative * float(cfg.routing.initiative_weight)
            if pid == last:
                score -= float(cfg.routing.last_speaker_penalty)
            weights.append(max(0.01, score))
        return weighted_choice(ids, weights)

    def _coverage_gap_option(self, state: DialogueState) -> Optional[str]:
        # Make sure every option gets at least one mention and one reason, then stop
        # steering: free discussion and convergence take over (no reason-count padding).
        for option_id, coverage in state.coverage.items():
            if coverage.mentions == 0:
                return option_id
        for option_id, coverage in state.coverage.items():
            if coverage.reasons == 0:
                return option_id
        return None

    def _act_for_gap(self, state: DialogueState, speaker_id: str, option_id: str) -> ActType:
        persona = state.persona_by_id(speaker_id)
        if option_id in persona.soft_rejections or option_id in persona.hard_rejections:
            return ActType.OBJECT
        if option_id == persona.preferred_option or option_id in persona.acceptable_options:
            return ActType.SUPPORT
        return ActType.COMPARE

    def _sample_discussion_act(self, state: DialogueState, speaker_id: str) -> ActType:
        if state.no_progress_count >= int(cfg.conversation.no_progress_window_turns):
            return weighted_choice(
                [ActType.PROPOSE_COMPROMISE, ActType.COMPARE],
                [float(cfg.routing.no_progress_compromise_probability), 1.0 - float(cfg.routing.no_progress_compromise_probability)],
            )
        probs = dict(cfg.routing.act_probabilities.items())
        if state.readiness_score >= float(cfg.conversation.concentration_to_narrow):
            probs[ActType.PROPOSE_COMPROMISE.value] = float(probs.get(ActType.PROPOSE_COMPROMISE.value, 0.0)) + float(cfg.routing.late_discussion_compromise_bonus)
        keys = [ActType(key) for key in probs.keys()]
        weights = [float(probs[key.value]) for key in keys]
        return weighted_choice(keys, weights)

    def _target_for_act(self, state: DialogueState, speaker_id: str, act: ActType, option_id: Optional[str]) -> Optional[str]:
        if not state.turns:
            return None
        last_participant = self._last_participant_turn(state)
        if act in {ActType.ANSWER, ActType.REACT, ActType.PUSH_BACK} and last_participant and last_participant.speaker_id != speaker_id:
            return last_participant.speaker_id
        if act == ActType.ASK and last_participant and last_participant.speaker_id != speaker_id:
            return last_participant.speaker_id
        if act in {ActType.OBJECT, ActType.PUSH_BACK, ActType.COMPARE} and option_id:
            target = self._conflicting_person_for_option(state, speaker_id, option_id)
            if target:
                return target
        if float(cfg.routing.direct_reply_probability) > 0 and last_participant and last_participant.speaker_id != speaker_id:
            return last_participant.speaker_id
        return None

    def _conflicting_person_for_option(self, state: DialogueState, speaker_id: str, option_id: str) -> Optional[str]:
        candidates: list[str] = []
        speaker = state.persona_by_id(speaker_id)
        speaker_likes = option_id == speaker.preferred_option or option_id in speaker.acceptable_options
        for persona in state.personas:
            if persona.id == speaker_id:
                continue
            other_likes = option_id == persona.preferred_option or option_id in persona.acceptable_options
            if other_likes != speaker_likes:
                candidates.append(persona.id)
        if not candidates:
            return None
        return weighted_choice(candidates, [1.0] * len(candidates))

    def _focus_for_act(self, state: DialogueState, speaker_id: str, act: ActType, option_id: Optional[str]) -> list[str]:
        persona = state.persona_by_id(speaker_id)
        if option_id:
            focus = [option_id]
        elif state.candidate_option and act in {ActType.PROPOSE_COMPROMISE, ActType.ACCEPT, ActType.REJECT}:
            focus = [state.candidate_option]
        else:
            focus = [state.runtimes[speaker_id].current_preference or persona.preferred_option]
        if act in {ActType.COMPARE, ActType.PUSH_BACK}:
            pref = state.runtimes[speaker_id].current_preference or persona.preferred_option
            if pref not in focus:
                focus.append(pref)
        if act == ActType.PROPOSE_COMPROMISE:
            candidate = self._best_compromise_focus(state, speaker_id)
            if candidate and candidate not in focus:
                focus.insert(0, candidate)
        return focus[: int(cfg.utterances.max_focus_options_in_prompt)]

    def _focus_for_person(self, state: DialogueState, speaker_id: str) -> list[str]:
        rt = state.runtimes[speaker_id]
        persona = state.persona_by_id(speaker_id)
        return [rt.current_preference or persona.preferred_option]

    def _best_compromise_focus(self, state: DialogueState, speaker_id: str) -> Optional[str]:
        persona = state.persona_by_id(speaker_id)
        candidates = list(dict.fromkeys(persona.acceptable_options + ([state.candidate_option] if state.candidate_option else [])))
        if not candidates:
            return persona.preferred_option
        return max(candidates, key=lambda opt: option_support(state, opt))

    def _least_recent_personas(self, state: DialogueState) -> list[Persona]:
        return sorted(state.personas, key=lambda p: (state.runtimes[p.id].last_spoke_turn is not None, state.runtimes[p.id].last_spoke_turn or -1))

    def _recent_speaker_ids(self, state: DialogueState) -> list[str]:
        window = int(cfg.routing.recent_speaker_window)
        ids = [turn.speaker_id for turn in state.turns if turn.speaker_id != "moderator"]
        return ids[-window:]

    @staticmethod
    def _last_participant_turn(state: DialogueState) -> Optional[TurnRecord]:
        for turn in reversed(state.turns):
            if turn.speaker_id != "moderator":
                return turn
        return None

    def _length_hint(self) -> str:
        probs = cfg.routing.length_hint_probabilities
        keys = ["short", "medium", "long"]
        return weighted_choice(keys, [float(probs.get(k, 0.0)) for k in keys])

    @staticmethod
    def _reason_for_act(act: ActType) -> str:
        return {
            ActType.REACT: "react naturally to the last useful point",
            ActType.ASK: "ask one targeted question that helps the decision",
            ActType.COMPARE: "compare real trade-offs between options",
            ActType.SUPPORT: "add a concrete reason for an option",
            ActType.OBJECT: "raise a soft objection without derailing the group",
            ActType.PUSH_BACK: "push back on a point while staying cooperative",
            ActType.PROPOSE_COMPROMISE: "suggest a workable compromise using an existing option",
        }.get(act, "make a useful local contribution")
