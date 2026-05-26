"""
policy/stubbornness.py
-----------------------
Hard-blocker sampling — Stage 8 (MASTERPLAN §Stage 8C).

At dialogue start, decides whether this run contains a true hard blocker.
Target: hard_blocker_dialogue_rate ≈ cfg.stubbornness.hard_blocker_dialogue_probability.

A true hard blocker:
  - Has is_true_hard_blocker=True on their ParticipantState.
  - Their act_planner steers toward REJECT_WITH_REASON when the candidate
    option is in their beliefs.rejected list.
  - They CAN soften if allow_softening_if_condition_met AND their concession
    condition is publicly addressed during emergence.

Non-blocker low-A personas are blunt and argumentative but not permanently
obstructive — they argue, then concede when their concern is genuinely met.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.participant import ParticipantState


def sample_hard_blockers(
    participant_states: list["ParticipantState"],
    cfg,
) -> list[str]:
    """
    Decide at dialogue start whether this dialogue contains a hard blocker.
    Mutates is_true_hard_blocker on selected ParticipantState(s).
    Returns list of chosen blocker names (empty = no hard blocker this run).

    Called from orchestrator.run_simulation() after attach_personas().
    Requires persona_ref to be set on each ParticipantState.
    """
    prob = cfg.stubbornness.hard_blocker_dialogue_probability
    max_blockers = cfg.stubbornness.max_hard_blockers_per_dialogue

    if random.random() >= prob:
        return []

    # Prefer participants who have explicit rejected options (a genuine reason to block).
    candidates_with_rejection = [
        ps for ps in participant_states
        if ps.persona_ref and ps.persona_ref.beliefs
        and ps.persona_ref.beliefs.rejected
    ]
    candidates = candidates_with_rejection or list(participant_states)

    chosen = random.sample(candidates, min(max_blockers, len(candidates)))
    for ps in chosen:
        ps.is_true_hard_blocker = True
    return [ps.name for ps in chosen]
