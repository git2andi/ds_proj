"""
state/participant.py
--------------------
ParticipantState — observed behavioral state for one participant across a dialogue.

Tracks public preference, unresolved conditions, cooldowns, and participation debt.
Used by the act planner (Stage 7) and turn policy (Stage 6) for
personality-biased self-selection and MUCA-style strategy cooldowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state.turn import DialogueAct
    from persona import Persona


@dataclass
class ParticipantState:
    """Mutable behavioral state for one participant, updated by StateTracker each turn."""
    name: str

    # Public stance (derived from transcript, not private beliefs)
    public_preference: Optional[str]          = None   # most recent committed vote letter
    public_stances: dict[str, str]            = field(default_factory=dict)  # option → stance label
    unresolved_conditions: dict[str, str]     = field(default_factory=dict)  # option → condition text
    hard_blockers: dict[str, str]             = field(default_factory=dict)  # option → blocker reason

    # Discourse obligations
    open_questions_to_answer: list[int]       = field(default_factory=list)  # turn_ids of unanswered Qs

    # Participation metrics
    turn_count: int                           = 0
    last_spoke_turn: Optional[int]            = None
    participation_debt: float                 = 0.0    # negative = spoke a lot; positive = spoke too little

    # Act history (for cooldowns)
    recent_dialogue_acts: list["DialogueAct"] = field(default_factory=list)
    strategy_cooldowns: dict[str, int]        = field(default_factory=dict)  # act.value → turns_remaining

    # Hard-blocker flag (set by stubbornness sampler in Stage 8)
    is_true_hard_blocker: bool                = False

    # Optional link back to Persona object (attached after init)
    persona_ref: Optional["Persona"]          = field(default=None, repr=False, compare=False)

    def decrement_cooldowns(self) -> None:
        """Call once per turn to age down all active cooldowns."""
        self.strategy_cooldowns = {
            act: max(0, remaining - 1)
            for act, remaining in self.strategy_cooldowns.items()
            if remaining > 1
        }

    def on_cooldown(self, act: "DialogueAct") -> bool:
        return self.strategy_cooldowns.get(act.value, 0) > 0

    def set_cooldown(self, act: "DialogueAct", turns: int) -> None:
        self.strategy_cooldowns[act.value] = turns

    def record_act(self, act: "DialogueAct", max_history: int = 10) -> None:
        self.recent_dialogue_acts.append(act)
        if len(self.recent_dialogue_acts) > max_history:
            self.recent_dialogue_acts = self.recent_dialogue_acts[-max_history:]
