"""
reasoning.py
------------
Consensus + Fisher ratios (logging only) + lightweight grounding fact-check.

ConsensusEngine  — uses public StanceTable; no private beliefs.
fisher_ratios()  — Fisher (1970) favorable / unfavorable / ambiguous ratios.
                   Logged per dialogue for analysis; does not drive control flow.
fact_check()     — deterministic invented-attribute detector for turns.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from state import StanceTable, StructuredState


# =============================================================================
# ConsensusEngine
# =============================================================================

ConsensusState = str   # none | candidate_emerging | majority_candidate |
                       # conditional_consensus | full_consensus | blocked | failed


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
        if supporters or conditional:
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
            for (_, o), su in stance.current_items()
            if o == opt
        )
        if score > best_score:
            best_score = score
            best_opt = opt
    return best_opt if best_score > 0.0 else None


# =============================================================================
# Fisher (1970) ratios — logged per dialogue, not used for control
# =============================================================================

def fisher_ratios(structured: "StructuredState", window: Optional[int] = None) -> dict[str, float]:
    """Favorable / unfavorable / ambiguous / conditional ratios over recent stances."""
    w = window if window is not None else cfg.fisher.window_size
    recent = structured.stance_table.history()[-w:]
    if not recent:
        return {"favor": 0.0, "disfavor": 0.0, "ambiguous": 0.0, "conditional": 0.0}
    counts: Counter = Counter(su.stance for su in recent)
    total = len(recent)
    return {
        "favor":       round(counts.get("support", 0) / total, 3),
        "disfavor":    round((counts.get("oppose", 0) + counts.get("blocker", 0)) / total, 3),
        "ambiguous":   round((counts.get("ambiguous", 0) + counts.get("neutral", 0)) / total, 3),
        "conditional": round(counts.get("conditional_support", 0) / total, 3),
    }


# =============================================================================
# Grounding — deterministic invented-fact detection
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
