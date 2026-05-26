"""
reasoning/phase_detector.py
----------------------------
PhaseDetector — Fisher (1970) evidence ratios → PhaseEvidence + confidence.

Fisher's four decision-emergence phases:
  orientation   → greeting / opening / early negotiation (opinions forming)
  conflict      → sustained disagreement; disfavor dominates
  emergence     → opposition declining, ambiguity / conditional support rising
  reinforcement → favor dominates; group converging; blockers resolved

Key design properties (MASTERPLAN §Stage 9):
  - Phases are GRADUAL, not hard switches (Fisher's own caveat).
  - Stores confidence ∈ [0, 1]; allows short or skipped phases.
  - Runs in PARALLEL with legacy _update_phase() behind cfg.consensus.use_structured.
  - Never called when tracker is None or structured state is unavailable.

Called from orchestrator._store_line() after each StateTracker update.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from config_loader import cfg
from state.dialogue_state import PhaseEvidence

if TYPE_CHECKING:
    from state.dialogue_state import StructuredState


class PhaseDetector:
    """
    Computes updated PhaseEvidence from a StructuredState's StanceTable.
    Instantiated once per dialogue in run_simulation().
    """

    def update(
        self,
        structured: "StructuredState",
        participant_turn_count: int,
    ) -> PhaseEvidence:
        """
        Recompute PhaseEvidence from the current StanceTable.
        Returns the new PhaseEvidence (caller stores it back into structured).
        """
        pc = cfg.phase_policy
        ratios = structured.stance_table.fisher_ratios(window=pc.window_size)
        current = structured.phase_evidence

        new_phase, confidence = self._transition(
            ratios=ratios,
            current=current,
            participant_turn_count=participant_turn_count,
        )

        rounds_in  = current.rounds_in_phase + 1 if new_phase == current.phase else 1
        entered_at = (current.entered_at_turn
                      if new_phase == current.phase else participant_turn_count)

        return PhaseEvidence(
            phase          = new_phase,
            confidence     = round(min(0.99, max(0.0, confidence)), 3),
            favor_rate     = round(ratios["favor"], 3),
            disfavor_rate  = round(ratios["disfavor"], 3),
            ambiguous_rate = round(ratios["ambiguous"], 3),
            conditional_rate = round(ratios["conditional"], 3),
            rounds_in_phase  = rounds_in,
            entered_at_turn  = entered_at,
        )

    # ------------------------------------------------------------------
    # Phase transition logic
    # ------------------------------------------------------------------

    def _transition(
        self,
        ratios: dict[str, float],
        current: PhaseEvidence,
        participant_turn_count: int,
    ) -> tuple[str, float]:
        """
        Determine (new_phase, confidence) given current ratios and phase.
        Transition thresholds come from cfg.phase_policy.
        """
        fav  = ratios["favor"]
        dis  = ratios["disfavor"]
        amb  = ratios["ambiguous"]
        cnd  = ratios["conditional"]
        pc   = cfg.phase_policy
        phase = current.phase

        if phase == "orientation":
            if participant_turn_count >= pc.min_turns_before_conflict and dis > amb:
                conf = min(0.92, 0.55 + (dis - amb) * 0.8)
                return "conflict", conf
            return "orientation", max(0.40, 0.70 - dis * 0.5)

        if phase == "conflict":
            emer      = pc.emergence
            dis_thr   = emer.opposition_decline_threshold
            amb_thr   = emer.ambiguity_rise_threshold
            if dis <= dis_thr and (amb + cnd) >= amb_thr:
                conf = min(0.90, 0.50 + (amb_thr - dis) * 0.4 + cnd * 0.3)
                return "emergence", conf
            return "conflict", max(0.40, 0.60 - fav * 0.2)

        if phase == "emergence":
            reinf   = pc.reinforcement
            sup_thr = reinf.support_rate_threshold
            if fav >= sup_thr:
                conf = min(0.94, 0.50 + (fav - sup_thr) * 1.5)
                return "reinforcement", conf
            return "emergence", max(0.40, 0.50 + cnd * 0.4)

        if phase == "reinforcement":
            return "reinforcement", min(0.95, 0.70 + fav * 0.3)

        # Unknown phase — hold steady
        return phase, current.confidence
