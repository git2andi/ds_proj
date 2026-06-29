"""Shared option-support scoring used by both routing and consensus.

Keeping this in one place avoids two slightly different copies of "how much does
the group back option X" drifting apart.
"""

from __future__ import annotations

from models import DialogueState, Persona


def current_lean(state: DialogueState, persona: Persona) -> str | None:
    """The option a persona effectively backs right now: an explicit vote wins, then their
    moved leaning, falling back to their original preferred option. One definition shared by
    routing, consensus, concentration, and the moderator's standings."""
    rt = state.runtimes[persona.id]
    return rt.explicit_vote or rt.current_preference or persona.preferred_option


def option_support(state: DialogueState, option_id: str) -> float:
    """How strongly the group currently backs an option: explicit votes and accepts
    weigh most, then current leanings and initial preference, minus rejections."""
    score = 0.0
    for persona in state.personas:
        rt = state.runtimes[persona.id]
        if rt.explicit_vote == option_id:
            score += 4.0
        if option_id in rt.accepted_options:
            score += 3.0
        if rt.current_preference == option_id:
            score += 2.0
        if option_id in persona.preferred_options:
            score += 1.0
        if option_id in rt.hard_rejections:
            score -= 5.0
        elif option_id in rt.soft_rejections:
            score -= 0.7
    return score


def leading_option(state: DialogueState) -> str | None:
    if not state.scenario.option_ids:
        return None
    return max(state.scenario.option_ids, key=lambda opt: option_support(state, opt))


def best_overlap_option(state: DialogueState) -> str | None:
    """The option that best bridges the room: one that people currently in *different*
    lean-camps can all live with (it's their lean, an accepted option, or in their
    acceptable set) and that nobody hard-rejects. This is the common-ground candidate a
    facilitator should surface when the group is split — e.g. a Mountain Retreat both the
    beach person and the road-trip person can accept. Returns None if the group has already
    converged (one camp) or no single option bridges two camps."""
    leans = {p.id: current_lean(state, p) for p in state.personas}
    if len({l for l in leans.values() if l}) <= 1:
        return None
    best: str | None = None
    best_key: tuple[int, int] = (1, 0)  # require strictly more than one bridged camp
    for opt in state.scenario.option_ids:
        backers = 0
        camps: set[str] = set()
        for persona in state.personas:
            rt = state.runtimes[persona.id]
            if opt in rt.hard_rejections:
                continue
            if leans[persona.id] == opt or opt in rt.accepted_options or opt in persona.preferred_options:
                backers += 1
                if leans[persona.id]:
                    camps.add(leans[persona.id])
        key = (len(camps), backers)
        if len(camps) >= 2 and key > best_key:
            best, best_key = opt, key
    return best
