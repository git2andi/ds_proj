"""Adaptive dialogue phase/readiness control.

The controller deliberately separates *readiness score* from *hard gates*.
A high score is not enough to vote if basic discussion requirements are unmet.
This prevents the failure mode where one opening round immediately turns into
voting because all participants have spoken once.
"""

from __future__ import annotations

from config_loader import cfg
from schemas import ActType, DialogueState, Phase


class DialogueController:
    def update_phase(self, state: DialogueState) -> None:
        if state.outcome is not None:
            state.phase = Phase.CLOSURE
            return

        participant_turns = self._participant_turn_count(state)
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
        return self._participant_turn_count(state) >= int(cfg.conversation.hard_max_total_turns)

    def readiness_score(self, state: DialogueState) -> float:
        n = max(1, len(state.personas))
        participant_turns = [rt.turn_count for rt in state.runtimes.values()]
        total_participant_turns = sum(participant_turns)
        min_turns = int(cfg.conversation.min_turns_per_participant_before_narrowing)
        soft_min = max(int(cfg.conversation.soft_min_total_turns), n * min_turns)

        everyone_spoke = sum(1 for x in participant_turns if x > 0) / n
        enough_turns = min(1.0, total_participant_turns / max(1, soft_min))
        balanced_depth = min(1.0, min(participant_turns or [0]) / max(1, min_turns))

        options_touched = sum(1 for c in state.coverage.values() if c.mentions > 0)
        touched_score = min(1.0, options_touched / max(1, int(cfg.conversation.min_options_touched_before_narrowing)))

        options_with_reasons = sum(1 for c in state.coverage.values() if c.reasons > 0)
        reason_coverage = min(1.0, options_with_reasons / max(1, int(cfg.conversation.min_options_with_reason_before_narrowing)))

        total_reasons = sum(c.reasons for c in state.coverage.values())
        reason_depth = min(1.0, total_reasons / max(1, int(cfg.conversation.min_total_option_reasons_before_narrowing)))

        tradeoff_turns = self._tradeoff_turns(state)
        tradeoff_score = min(1.0, tradeoff_turns / max(1, int(cfg.conversation.min_tradeoff_or_objection_turns_before_narrowing)))

        question_penalty = 0.0 if not state.open_questions else 0.18
        stall_bonus = min(0.10, state.no_progress_count / max(1, int(cfg.conversation.no_progress_window)) * 0.10)

        score = (
            0.15 * everyone_spoke
            + 0.18 * enough_turns
            + 0.14 * balanced_depth
            + 0.18 * touched_score
            + 0.15 * reason_coverage
            + 0.12 * reason_depth
            + 0.08 * tradeoff_score
        )
        return round(max(0.0, min(1.0, score - question_penalty + stall_bonus)), 3)

    def _can_start_narrowing(self, state: DialogueState) -> bool:
        if state.open_questions:
            return False
        if not self._everyone_has_spoken_once(state):
            return False
        if state.readiness_score < float(cfg.conversation.readiness_threshold):
            return False
        if self._participant_turn_count(state) < int(cfg.conversation.soft_min_total_turns):
            return False
        min_each = int(cfg.conversation.min_turns_per_participant_before_narrowing)
        if any(rt.turn_count < min_each for rt in state.runtimes.values()):
            return False
        if self._options_touched(state) < int(cfg.conversation.min_options_touched_before_narrowing):
            return False
        if self._options_with_reasons(state) < int(cfg.conversation.min_options_with_reason_before_narrowing):
            return False
        if sum(c.reasons for c in state.coverage.values()) < int(cfg.conversation.min_total_option_reasons_before_narrowing):
            return False
        if self._tradeoff_turns(state) < int(cfg.conversation.min_tradeoff_or_objection_turns_before_narrowing):
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

    def _everyone_has_spoken_once(self, state: DialogueState) -> bool:
        return all(rt.turn_count > 0 for rt in state.runtimes.values())

    def _participant_turn_count(self, state: DialogueState) -> int:
        return sum(1 for t in state.turns if t.speaker_id != "moderator")

    def _options_touched(self, state: DialogueState) -> int:
        return sum(1 for c in state.coverage.values() if c.mentions > 0)

    def _options_with_reasons(self, state: DialogueState) -> int:
        return sum(1 for c in state.coverage.values() if c.reasons > 0)

    def _tradeoff_turns(self, state: DialogueState) -> int:
        count = 0
        for turn in state.turns:
            if turn.speaker_id == "moderator":
                continue
            lower = turn.text.lower()
            if turn.act.act_type in {ActType.PUSH_BACK, ActType.COMPARE, ActType.REJECT}:
                count += 1
            elif any(cue in lower for cue in ["but", "though", "however", "worry", "concern", "trade-off", "tradeoff", "downside", "price", "cost", "risk"]):
                count += 1
        return count
