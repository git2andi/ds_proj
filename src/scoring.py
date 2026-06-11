"""Shared option-support scoring used by both routing and consensus.

Keeping this in one place avoids two slightly different copies of "how much does
the group back option X" drifting apart.
"""

from __future__ import annotations

from typing import Optional

from config_loader import cfg
from models import DialogueState


def option_support(state: DialogueState, option_id: str) -> float:
    """How strongly the group currently backs an option: explicit votes and accepts
    weigh most, then current leanings and initial preference, plus the hidden private
    utility, minus rejections."""
    acceptance = int(cfg.scenario.acceptance_score)
    score = 0.0
    for persona in state.personas:
        rt = state.runtimes[persona.id]
        if rt.explicit_vote == option_id:
            score += 4.0
        if option_id in rt.accepted_options:
            score += 3.0
        if rt.current_preference == option_id:
            score += 2.0
        if persona.preferred_option == option_id:
            score += 1.0
        score += (persona.score_for(option_id) - acceptance) * 0.6
        if option_id in rt.hard_rejections:
            score -= 5.0
        elif option_id in rt.soft_rejections:
            score -= 0.7
    return score


def leading_option(state: DialogueState) -> Optional[str]:
    if not state.scenario.option_ids:
        return None
    return max(state.scenario.option_ids, key=lambda opt: option_support(state, opt))
