"""Shared option-support scoring used by both routing and consensus.

Keeping this in one place avoids two slightly different copies of "how much does
the group back option X" drifting apart.
"""

from __future__ import annotations

from collections import Counter

from models import ActType, DialogueState, Persona
from parsing import OptionResolver


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


def visible_support_ids(state: DialogueState, option_id: str) -> list[str]:
    """Participants with a binding visible vote or acceptance for ``option_id``."""
    return [
        persona.id
        for persona in state.personas
        if state.runtimes[persona.id].explicit_vote == option_id
        or option_id in state.runtimes[persona.id].accepted_options
    ]


def visible_preference_option(state: DialogueState, persona_id: str) -> str | None:
    """Latest option the participant visibly chose or leaned toward in the transcript."""
    resolver = OptionResolver(state.scenario.options)
    for turn in reversed(state.turns):
        if turn.speaker_id != persona_id or turn.is_social:
            continue
        if turn.act.explicit_vote:
            return turn.act.explicit_vote
        if turn.act.accepts:
            return turn.act.accepts[0]
        visible_options = resolver.ids_in_text(turn.text)
        if len(visible_options) != 1:
            continue
        if turn.intent and turn.intent.act in {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}:
            return visible_options[0]
        if turn.act.act_type == ActType.OPENING or (turn.intent and turn.intent.moves_lean):
            return visible_options[0]
    return None


def visible_leading_option(state: DialogueState) -> str | None:
    preferences = [visible_preference_option(state, persona.id) for persona in state.personas]
    counts = Counter(option_id for option_id in preferences if option_id)
    return counts.most_common(1)[0][0] if counts else None


def visible_preference_concentration(state: DialogueState) -> float:
    preferences = [visible_preference_option(state, persona.id) for persona in state.personas]
    counts = Counter(option_id for option_id in preferences if option_id)
    return (counts.most_common(1)[0][1] / len(state.personas)) if counts and state.personas else 0.0


def visible_candidate_status(
    state: DialogueState,
    persona_id: str,
    candidate_id: str,
) -> tuple[str, str | None]:
    """Return ``(supporter|holdout|missing, visible_alternative)``.

    Hidden persona preferences and initialized routing leans are excluded. A failed
    decision turn naming the candidate is aligned but still ``missing`` because its
    visible text did not establish a binding commitment.
    """
    if persona_id in visible_support_ids(state, candidate_id):
        return "supporter", None
    for turn in reversed(state.turns):
        if turn.speaker_id != persona_id or turn.is_social:
            continue
        if candidate_id in turn.act.hard_rejects or candidate_id in turn.act.soft_rejects:
            return "holdout", None
        if turn.act.explicit_vote and turn.act.explicit_vote != candidate_id:
            return "holdout", turn.act.explicit_vote
        alternatives = [option_id for option_id in turn.act.accepts if option_id != candidate_id]
        if alternatives:
            return "holdout", alternatives[0]
    visible_preference = visible_preference_option(state, persona_id)
    if visible_preference:
        return ("missing", None) if visible_preference == candidate_id else ("holdout", visible_preference)
    return "missing", None


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
