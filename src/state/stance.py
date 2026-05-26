"""
state/stance.py
---------------
StanceTable and OptionState — per-(speaker × option) public stance tracking.

Single source of truth for what each participant has publicly said about each option.
Used by the consensus engine (Stage 9) and act planner (Stage 7).
Fisher ratios computed from StanceTable history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state.turn import StanceUpdate


@dataclass
class OptionState:
    """Aggregate stance counts for a single option."""
    option_id: str                                          # "A".."D"
    text: str                                               # canonical option text

    supporters: set[str]                       = field(default_factory=set)
    opponents: set[str]                        = field(default_factory=set)
    ambiguous: set[str]                        = field(default_factory=set)
    conditional_supporters: dict[str, str]     = field(default_factory=dict)  # name → condition
    hard_blockers: dict[str, str]              = field(default_factory=dict)  # name → reason

    support_score: float                       = 0.0
    opposition_score: float                    = 0.0
    last_mentioned_turn: Optional[int]         = None

    def net_score(self) -> float:
        return self.support_score - self.opposition_score

    def apply_update(self, update: "StanceUpdate") -> None:
        """Update aggregate sets from a StanceUpdate."""
        name = update.speaker
        stance = update.stance

        # Clear previous category memberships for this speaker
        self.supporters.discard(name)
        self.opponents.discard(name)
        self.ambiguous.discard(name)
        self.conditional_supporters.pop(name, None)
        self.hard_blockers.pop(name, None)

        if stance == "support":
            self.supporters.add(name)
            self.support_score += update.confidence
        elif stance == "oppose":
            self.opponents.add(name)
            self.opposition_score += update.confidence
        elif stance == "blocker":
            self.opponents.add(name)
            self.hard_blockers[name] = update.condition or "unspecified"
            self.opposition_score += update.confidence * 1.5
        elif stance == "conditional_support":
            self.conditional_supporters[name] = update.condition or ""
            self.support_score += update.confidence * 0.7
        elif stance == "ambiguous":
            self.ambiguous.add(name)
        # "neutral" has no effect on scores


class StanceTable:
    """
    Per-(speaker × option) current stance + history.
    Provides Fisher-style ratios over a rolling window.
    """

    def __init__(self) -> None:
        # (speaker, option) → most recent StanceUpdate
        self._current: dict[tuple[str, str], "StanceUpdate"] = {}
        self._history: list["StanceUpdate"] = []

    def current(self, speaker: str, option: str) -> Optional["StanceUpdate"]:
        return self._current.get((speaker, option))

    def current_stance_label(self, speaker: str, option: str) -> str:
        su = self.current(speaker, option)
        return su.stance if su else "neutral"

    def apply(self, update: "StanceUpdate", option_state: Optional[OptionState] = None) -> None:
        """Record an update and optionally push it to an OptionState aggregate."""
        self._current[(update.speaker, update.option)] = update
        self._history.append(update)
        if option_state is not None:
            option_state.apply_update(update)

    def fisher_ratios(self, window: int = 8) -> dict[str, float]:
        """
        Favorable/unfavorable/ambiguous/conditional ratios over the last `window` updates.
        Returns keys: favor, disfavor, ambiguous, conditional.
        """
        recent = self._history[-window:]
        if not recent:
            return {"favor": 0.0, "disfavor": 0.0, "ambiguous": 0.5, "conditional": 0.0}

        counts: dict[str, int] = {
            "support": 0, "oppose": 0, "blocker": 0,
            "ambiguous": 0, "conditional_support": 0, "neutral": 0,
        }
        for su in recent:
            counts[su.stance] = counts.get(su.stance, 0) + 1

        total = len(recent)
        return {
            "favor":       counts["support"] / total,
            "disfavor":    (counts["oppose"] + counts["blocker"]) / total,
            "ambiguous":   (counts["ambiguous"] + counts["neutral"]) / total,
            "conditional": counts["conditional_support"] / total,
        }

    def majority_favor(self, all_speakers: list[str], min_supporters: int) -> Optional[str]:
        """Return the option letter with at least `min_supporters` supporting speakers."""
        from collections import Counter
        support_counts: Counter = Counter()
        for (speaker, option), su in self._current.items():
            if su.stance in ("support", "conditional_support"):
                support_counts[option] += 1
        for option, count in support_counts.most_common():
            if count >= min_supporters:
                return option
        return None

    def holdouts(self, option: str, all_speakers: list[str]) -> list[str]:
        """Speakers not yet supporting this option."""
        return [
            s for s in all_speakers
            if self.current_stance_label(s, option)
            not in ("support", "conditional_support")
        ]

    def unresolved_blockers(self, option: str) -> dict[str, str]:
        """Active hard blockers for this option: name → reason."""
        return {
            speaker: su.condition or "unspecified"
            for (speaker, opt), su in self._current.items()
            if opt == option and su.stance == "blocker"
        }

    def all_public_votes(self) -> dict[str, str]:
        """Most recent committed vote per speaker (COMMIT_VOTE level confidence only)."""
        votes: dict[str, str] = {}
        for su in reversed(self._history):
            if su.speaker not in votes and su.stance == "support" and su.confidence >= 1.0:
                votes[su.speaker] = su.option
        return votes
