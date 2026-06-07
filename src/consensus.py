"""Consensus and final-decision logic."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from config_loader import cfg
from schemas import DialogueState, RunOutcome


class ConsensusManager:
    def detect(self, state: DialogueState) -> Optional[RunOutcome]:
        candidate = state.candidate_option or self.leading_vote(state)
        if not candidate:
            return None
        blockers = self.blockers_for(state, candidate)
        if not blockers and self._all_have_accepted_or_voted(state, candidate):
            return RunOutcome("consensus", candidate, "all participants accepted or voted for the same compromise", state.turn_index)
        return None

    def finalize(self, state: DialogueState) -> RunOutcome:
        detected = self.detect(state)
        if detected:
            return detected
        candidate = state.candidate_option or self.leading_vote(state)
        if candidate and bool(cfg.consensus.allow_majority_fallback_after_max_turns):
            support = self.support_fraction(state, candidate)
            if support >= float(cfg.consensus.majority_fallback_fraction) and not self._has_hard_blocker_against(state, candidate):
                return RunOutcome("fallback", candidate, f"majority fallback with support fraction {support:.2f}", state.turn_index)
        return RunOutcome("unresolved", None, "no option reached explicit acceptance by all participants", state.turn_index)

    def leading_vote(self, state: DialogueState) -> Optional[str]:
        counts = Counter(rt.explicit_vote for rt in state.runtimes.values() if rt.explicit_vote)
        return counts.most_common(1)[0][0] if counts else None

    def support_fraction(self, state: DialogueState, option_id: str) -> float:
        if not state.runtimes:
            return 0.0
        supporters = 0
        for runtime in state.runtimes.values():
            if runtime.explicit_vote == option_id or option_id in runtime.accepted_options:
                supporters += 1
        return supporters / len(state.runtimes)

    def blockers_for(self, state: DialogueState, option_id: str) -> list[str]:
        # Any explicit current rejection blocks consensus. Hard-blocker status only
        # matters for fallback decisions, not for consensus validity.
        return [pid for pid, runtime in state.runtimes.items() if option_id in runtime.rejected_options]

    def _all_have_accepted_or_voted(self, state: DialogueState, option_id: str) -> bool:
        for runtime in state.runtimes.values():
            if runtime.explicit_vote == option_id:
                continue
            if option_id in runtime.accepted_options:
                continue
            return False
        return True

    def _has_hard_blocker_against(self, state: DialogueState, option_id: str) -> bool:
        for pid, runtime in state.runtimes.items():
            persona = state.persona_by_id(pid)
            if persona.is_hard_blocker and option_id in runtime.rejected_options:
                return True
        return False
