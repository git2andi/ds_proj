"""
reasoning.py
------------
Consensus + Fisher phase detection + lightweight grounding fact-check.
Merged from the former reasoning/ and realize/grounding modules.

ConsensusEngine  -- uses public StanceTable; no private beliefs.
PhaseDetector    -- Fisher (1970) ratio-driven phase confidence.
fact_check()     -- deterministic invented-attribute detector for turns.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

from config_loader import cfg
from state import PhaseEvidence

if TYPE_CHECKING:
    from state import StanceTable, StructuredState


# =============================================================================
# ConsensusEngine
# =============================================================================

ConsensusState = str   # one of: none | candidate_emerging | majority_candidate |
                       #          conditional_consensus | full_consensus | blocked | failed


class ConsensusEngine:
    """Derive consensus state and best-available decision from public StanceTable."""

    def compute_state(
        self, structured: "StructuredState", participant_names: list[str]
    ) -> ConsensusState:
        if not participant_names:
            return "none"

        stance  = structured.stance_table
        options = list(structured.options.keys())
        n       = len(participant_names)

        candidate = structured.candidate_option
        if candidate and stance.unresolved_blockers(candidate):
            return "blocked"

        best = _leading_option(stance, options)
        if best is None:
            return "none"

        supporters: set[str]  = set()
        conditional: set[str] = set()
        ambiguous: set[str]   = set()
        opposing: set[str]    = set()

        for name in participant_names:
            lbl = stance.current_stance_label(name, best)
            if lbl == "support":
                supporters.add(name)
            elif lbl == "conditional_support":
                conditional.add(name)
            elif lbl == "ambiguous":
                ambiguous.add(name)
            elif lbl in ("oppose", "blocker"):
                opposing.add(name)

        if len(supporters) == n:
            return "full_consensus"
        if len(supporters) + len(conditional) == n:
            return "conditional_consensus"
        if not opposing and len(supporters) + len(conditional) + len(ambiguous) == n:
            return "majority_candidate"
        if len(supporters) + len(conditional) > n // 2 or (supporters or conditional):
            return "candidate_emerging"
        return "none"

    def leading_candidate(
        self, structured: "StructuredState", participant_names: list[str]
    ) -> Optional[str]:
        return _leading_option(structured.stance_table, list(structured.options.keys()))

    def best_available_decision(
        self, structured: "StructuredState", participant_names: list[str]
    ) -> Optional[str]:
        weights = cfg.consensus.stance_weights
        stance  = structured.stance_table
        options = list(structured.options.keys())
        if not options:
            return None

        def score(opt: str) -> float:
            return sum(
                weights.get(stance.current_stance_label(name, opt), 0.0)
                for name in participant_names
            )

        scores = {opt: score(opt) for opt in options}
        voted = [
            opt for opt in options
            if any(stance.current_stance_label(name, opt) in ("support", "conditional_support")
                   for name in participant_names)
        ]
        pool = voted if voted else options
        return max(pool, key=lambda o: scores[o])


def _leading_option(stance: "StanceTable", options: list[str]) -> Optional[str]:
    best_opt: Optional[str] = None
    best_score: float = -1.0
    for opt in options:
        score = sum(
            (1.0 if su.stance == "support" else
             0.7 if su.stance == "conditional_support" else 0.0)
            for (_, o), su in stance._current.items()
            if o == opt
        )
        if score > best_score:
            best_score = score
            best_opt = opt
    return best_opt if best_score > 0.0 else None


# =============================================================================
# PhaseDetector -- Fisher (1970)
# =============================================================================

class PhaseDetector:
    """Fisher ratio -> phase + confidence. Phases are gradual, not hard switches."""

    def update(
        self, structured: "StructuredState", participant_turn_count: int
    ) -> PhaseEvidence:
        pc = cfg.phase_policy
        ratios = structured.stance_table.fisher_ratios(window=pc.window_size)
        current = structured.phase_evidence

        new_phase, conf = self._transition(ratios, current, participant_turn_count)

        rounds_in  = current.rounds_in_phase + 1 if new_phase == current.phase else 1
        entered_at = current.entered_at_turn if new_phase == current.phase else participant_turn_count

        return PhaseEvidence(
            phase=new_phase,
            confidence=round(min(0.99, max(0.0, conf)), 3),
            favor_rate=round(ratios["favor"], 3),
            disfavor_rate=round(ratios["disfavor"], 3),
            ambiguous_rate=round(ratios["ambiguous"], 3),
            conditional_rate=round(ratios["conditional"], 3),
            rounds_in_phase=rounds_in,
            entered_at_turn=entered_at,
        )

    def _transition(self, ratios: dict[str, float], current: PhaseEvidence,
                    participant_turn_count: int) -> tuple[str, float]:
        fav, dis, amb, cnd = ratios["favor"], ratios["disfavor"], ratios["ambiguous"], ratios["conditional"]
        pc = cfg.phase_policy
        phase = current.phase

        if phase == "orientation":
            if participant_turn_count >= pc.min_turns_before_conflict and dis > amb:
                return "conflict", min(0.92, 0.55 + (dis - amb) * 0.8)
            return "orientation", max(0.40, 0.70 - dis * 0.5)

        if phase == "conflict":
            emer = pc.emergence
            if dis <= emer.opposition_decline_threshold and (amb + cnd) >= emer.ambiguity_rise_threshold:
                return "emergence", min(0.90, 0.50 + (emer.ambiguity_rise_threshold - dis) * 0.4 + cnd * 0.3)
            return "conflict", max(0.40, 0.60 - fav * 0.2)

        if phase == "emergence":
            if fav >= pc.reinforcement.support_rate_threshold:
                return "reinforcement", min(0.94, 0.50 + (fav - pc.reinforcement.support_rate_threshold) * 1.5)
            return "emergence", max(0.40, 0.50 + cnd * 0.4)

        if phase == "reinforcement":
            return "reinforcement", min(0.95, 0.70 + fav * 0.3)

        return phase, current.confidence


# =============================================================================
# Grounding -- deterministic invented-fact detection
# =============================================================================

def fact_check(turn_text: str, option_texts: list[str], topic: str) -> list[str]:
    """Return suspicious phrases (numbers, quotes, parenthesised asides) not in source."""
    source = " ".join([topic.lower(), *[o.lower() for o in option_texts]])
    suspicious: list[str] = []

    for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?%?\b", turn_text):
        token = m.group(0)
        if token not in source:
            suspicious.append(token)

    for m in re.finditer(r'"([^"]{3,80})"', turn_text):
        phrase = m.group(1).strip()
        if phrase.lower() not in source:
            suspicious.append(f'"{phrase}"')

    for m in re.finditer(r"\(([^)]{4,60})\)", turn_text):
        phrase = m.group(1).strip()
        if re.search(r"\d", phrase) and phrase.lower() not in source:
            suspicious.append(f"({phrase})")

    return suspicious


def repair_directive() -> str:
    return (
        "IMPORTANT: Use only attributes explicitly listed in the options. "
        "Do not invent numbers, prices, dates, names, or unstated details."
    )
