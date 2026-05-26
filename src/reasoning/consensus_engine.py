"""
reasoning/consensus_engine.py
------------------------------
ConsensusEngine — StanceTable-based consensus state computation (Stage 9).

Replaces regex-based logic in consensus.py when cfg.consensus.use_structured is true.

Consensus state ladder (MASTERPLAN §Stage 9):
  none               — no option has meaningful support
  candidate_emerging — one option has plurality but ≥1 participant still pending/ambiguous
  majority_candidate — plurality with no active opposition (conditional/ambiguous only)
  conditional_consensus — all participants support or conditionally support
  full_consensus     — all participants explicitly support; no unresolved conditions
  blocked            — a hard blocker actively opposes the leading candidate
  failed             — hard ceiling reached without progress (set externally)

Key design:
  - Uses PUBLIC stances only (StanceTable). Private beliefs never enter scoring.
  - best_available_decision() uses cfg.consensus.stance_weights for scoring.
  - Confirmation rejection maps to stance="blocker" via StateTracker, eliminating
    the confirmation re-test loop (MASTERPLAN §Stage 9, confirmation rejection).
  - A/B safety: both old and new paths run; cfg.consensus.use_structured controls
    which drives actual orchestrator behavior.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from state.dialogue_state import StructuredState
    from state.stance import StanceTable, OptionState

ConsensusState = str  # one of the literals listed in the docstring


class ConsensusEngine:
    """
    Derive consensus state from public StanceTable.
    Instantiated once per dialogue in orchestrator.run_simulation().
    """

    def compute_state(
        self,
        structured: "StructuredState",
        participant_names: list[str],
    ) -> ConsensusState:
        """
        Determine current consensus state from public stances.
        Never reads private beliefs.
        """
        if not participant_names:
            return "none"

        stance  = structured.stance_table
        options = list(structured.options.keys())
        n       = len(participant_names)

        # Check for hard blockers on the current candidate first
        candidate = structured.candidate_option
        if candidate:
            if stance.unresolved_blockers(candidate):
                return "blocked"

        best_opt = _leading_option(stance, options)
        if best_opt is None:
            return "none"

        supporters:   set[str] = set()
        conditional:  set[str] = set()
        ambiguous_s:  set[str] = set()
        opposing:     set[str] = set()

        for name in participant_names:
            lbl = stance.current_stance_label(name, best_opt)
            if lbl == "support":
                supporters.add(name)
            elif lbl == "conditional_support":
                conditional.add(name)
            elif lbl == "ambiguous":
                ambiguous_s.add(name)
            elif lbl in ("oppose", "blocker"):
                opposing.add(name)

        covered = supporters | conditional | ambiguous_s | opposing

        if len(supporters) == n:
            return "full_consensus"

        if len(supporters) + len(conditional) == n:
            return "conditional_consensus"

        if not opposing and len(supporters) + len(conditional) + len(ambiguous_s) == n:
            return "majority_candidate"

        if len(supporters) + len(conditional) > n // 2:
            return "candidate_emerging"

        if supporters or conditional:
            return "candidate_emerging"

        return "none"

    def leading_candidate(
        self,
        structured: "StructuredState",
        participant_names: list[str],
    ) -> Optional[str]:
        """Return the option letter with the highest public support score, or None."""
        return _leading_option(
            structured.stance_table,
            list(structured.options.keys()),
        )

    def best_available_decision(
        self,
        structured: "StructuredState",
        participant_names: list[str],
    ) -> Optional[str]:
        """
        Select the best option from public stances only (no private beliefs).
        Uses cfg.consensus.stance_weights to score each option per participant.
        Restricted to options with ≥1 support or conditional_support first;
        falls back to all mentioned options, then all options (always logged).

        Replaces orchestrator._force_conclusion() private-belief scoring when
        cfg.consensus.use_structured is true.
        """
        weights = cfg.consensus.stance_weights
        stance  = structured.stance_table
        options = list(structured.options.keys())
        if not options:
            return None

        def _score(opt: str) -> float:
            return sum(
                weights.get(stance.current_stance_label(name, opt), 0.0)
                for name in participant_names
            )

        scores = {opt: _score(opt) for opt in options}

        # Prefer options with at least one positive stance update
        voted = [
            opt for opt in options
            if any(
                stance.current_stance_label(name, opt)
                in ("support", "conditional_support")
                for name in participant_names
            )
        ]
        pool = voted if voted else options
        return max(pool, key=lambda o: scores[o])


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _leading_option(stance: "StanceTable", options: list[str]) -> Optional[str]:
    """Return the option with the highest combined support + conditional score."""
    best_opt:   Optional[str] = None
    best_score: float         = -1.0

    for opt in options:
        score = sum(
            (1.0 if su.stance == "support" else
             0.7 if su.stance == "conditional_support" else 0.0)
            for (_, o), su in stance._current.items()
            if o == opt
        )
        if score > best_score:
            best_score = score
            best_opt   = opt

    return best_opt if best_score > 0.0 else None
