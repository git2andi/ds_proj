"""
state/dialogue_state.py
------------------------
StructuredState — the full parallel dialogue state maintained by StateTracker (Stage 5+).

Distinct from orchestrator.DialogueState (the legacy string-based state).
Both coexist during the transition; decisions switch to StructuredState in Stages 6-9.

PhaseEvidence stores Fisher-style evidence ratios and phase confidence,
respecting Fisher's caveat that phases are gradual, not hard switches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

from state.discourse import DiscourseGraph
from state.stance import StanceTable, OptionState
from state.participant import ParticipantState


@dataclass
class PhaseEvidence:
    """
    Fisher-aligned phase representation: confidence + ratio snapshot.
    Phase label is a soft assignment, not a hard switch.
    """
    phase: Literal[
        "orientation", "conflict", "emergence", "reinforcement", "closure"
    ] = "orientation"
    confidence: float     = 0.5      # 0.0..1.0; 1.0 = unambiguous phase signal
    favor_rate: float     = 0.0
    disfavor_rate: float  = 0.0
    ambiguous_rate: float = 0.5
    conditional_rate: float = 0.0
    rounds_in_phase: int  = 0
    entered_at_turn: int  = 0


@dataclass
class StructuredState:
    """
    Full structured dialogue state — the single source of truth once Stage 9 is live.
    In Stage 5 this is populated in parallel with orchestrator.DialogueState.
    """
    # Turn records (full history)
    turns: list = field(default_factory=list)          # list[TurnRecord]
    turn_id_counter: int = 0

    # Participants: name → ParticipantState
    participants: dict[str, ParticipantState] = field(default_factory=dict)

    # Options: letter → OptionState
    options: dict[str, OptionState] = field(default_factory=dict)

    # Stance tracking
    stance_table: StanceTable = field(default_factory=StanceTable)

    # Discourse (pending Qs, reply edges)
    discourse: DiscourseGraph = field(default_factory=DiscourseGraph)

    # Phase evidence (Fisher-aligned)
    phase_evidence: PhaseEvidence = field(default_factory=PhaseEvidence)

    # Consensus state labels
    consensus_state: Literal[
        "none", "candidate_emerging", "majority_candidate",
        "conditional_consensus", "full_consensus", "blocked", "failed"
    ] = "none"
    candidate_option: Optional[str] = None

    # Dialogue outcome
    outcome: Literal["pending", "success", "force_close", "failed"] = "pending"
