"""Adaptive dialogue phase/readiness control."""

from __future__ import annotations

from config_loader import cfg
from schemas import DialogueState, Phase


class DialogueController:
    def update_phase(self, state: DialogueState) -> None:
        if state.outcome is not None:
            state.phase = Phase.CLOSURE
            return
        participant_turns = sum(1 for t in state.turns if t.speaker_id != "moderator")
        if state.phase == Phase.OPENING and self._everyone_has_spoken_once(state):
            state.phase = Phase.DISCUSSION
        state.readiness_score = self.readiness_score(state)
        if state.phase in {Phase.OPENING, Phase.DISCUSSION} and self._can_start_narrowing(state):
            state.phase = Phase.NARROWING
        if state.phase == Phase.NARROWING and self._has_candidate_to_confirm(state):
            state.phase = Phase.CONFIRMATION
        if participant_turns >= int(cfg.conversation.hard_max_total_turns):
            state.phase = Phase.CLOSURE

    def should_stop(self, state: DialogueState) -> bool:
        if state.outcome is not None:
            return True
        participant_turns = sum(1 for t in state.turns if t.speaker_id != "moderator")
        return participant_turns >= int(cfg.conversation.hard_max_total_turns)

    def readiness_score(self, state: DialogueState) -> float:
        n = max(1, len(state.personas))
        participant_turns = [rt.turn_count for rt in state.runtimes.values()]
        total_participant_turns = sum(participant_turns)
        min_turns = int(cfg.conversation.min_turns_per_participant_before_close)
        soft_min = max(int(cfg.conversation.soft_min_total_turns), n * min_turns)

        everyone_spoke = sum(1 for x in participant_turns if x > 0) / n
        enough_turns = min(1.0, total_participant_turns / max(1, soft_min))
        option_reason_count = sum(1 for c in state.coverage.values() if c.reasons > 0)
        option_coverage = min(1.0, option_reason_count / max(1, int(cfg.conversation.min_options_with_reason_before_narrowing)))
        total_reasons = sum(c.reasons for c in state.coverage.values())
        reason_depth = min(1.0, total_reasons / max(1, int(cfg.conversation.min_total_option_reasons_before_narrowing)))
        question_penalty = 0.0 if not state.open_questions else 0.22
        stall_bonus = min(0.12, state.no_progress_count / max(1, int(cfg.conversation.no_progress_window)) * 0.12)
        score = 0.25 * everyone_spoke + 0.25 * enough_turns + 0.25 * option_coverage + 0.25 * reason_depth
        score = max(0.0, min(1.0, score - question_penalty + stall_bonus))
        return round(score, 3)

    def _everyone_has_spoken_once(self, state: DialogueState) -> bool:
        return all(rt.turn_count > 0 for rt in state.runtimes.values())

    def _can_start_narrowing(self, state: DialogueState) -> bool:
        if state.open_questions:
            return False
        if not self._everyone_has_spoken_once(state):
            return False
        if state.readiness_score < float(cfg.conversation.readiness_threshold):
            return False
        distinct = sum(1 for rt in state.runtimes.values() if rt.turn_count > 0)
        if distinct < int(cfg.conversation.min_distinct_participants_before_narrowing):
            return False
        return True

    def _has_candidate_to_confirm(self, state: DialogueState) -> bool:
        if state.candidate_option:
            return True
        votes = [rt.explicit_vote for rt in state.runtimes.values() if rt.explicit_vote]
        return len(votes) == len(state.runtimes) and bool(votes)
