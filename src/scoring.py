"""Shared option-support scoring used by both routing and consensus.

Keeping this in one place avoids two slightly different copies of "how much does
the group back option X" drifting apart.
"""

from __future__ import annotations

from typing import Optional

from config_loader import cfg
from models import DialogueState, Persona


def current_lean(state: DialogueState, persona: Persona) -> Optional[str]:
    """The option a persona effectively backs right now: an explicit vote wins, then their
    moved leaning, falling back to their original preferred option. One definition shared by
    routing, consensus, concentration, and the moderator's standings."""
    rt = state.runtimes[persona.id]
    return rt.explicit_vote or rt.current_preference or persona.preferred_option


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
        # Latent acceptability is weighted lightly so an option only a few people *could*
        # live with doesn't out-rank options people are actively backing — otherwise the
        # group drifts onto a bland common compromise nobody actually argued for.
        score += (persona.score_for(option_id) - acceptance) * 0.4
        if option_id in rt.hard_rejections:
            score -= 5.0
        elif option_id in rt.soft_rejections:
            score -= 0.7
    return score


def leading_option(state: DialogueState) -> Optional[str]:
    if not state.scenario.option_ids:
        return None
    return max(state.scenario.option_ids, key=lambda opt: option_support(state, opt))
