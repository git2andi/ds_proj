"""Simple public narrowing and clear visible vote counting."""

from __future__ import annotations

import math
import random
from collections import Counter

from config_loader import cfg
from models import DialogueState, RunOutcome


def public_preference_counts(state: DialogueState) -> Counter[str]:
    return Counter(
        runtime.public_preference
        for runtime in state.runtimes.values()
        if runtime.public_preference in state.scenario.option_ids
    )


def public_support_counts(state: DialogueState) -> Counter[str]:
    """Count each participant once per publicly supported option."""

    counts: Counter[str] = Counter()
    for runtime in state.runtimes.values():
        supported = set(runtime.public_acceptances)
        if runtime.public_preference in state.scenario.option_ids:
            supported.add(runtime.public_preference)
        for option_id in supported:
            if option_id in state.scenario.option_ids:
                counts[option_id] += 1
    return counts


def derive_narrowing_options(
    state: DialogueState,
    *,
    rng: random.Random | None = None,
) -> tuple[str, ...]:
    """Return one public leader from visible preferences and acceptances.

    The leader is chosen from current public preferences and visible
    acceptances. Current preference counts first break a support tie. If
    several top options remain tied, a supplied seeded RNG selects one as the
    bounded compromise target. The moderator therefore always gets one target
    for a no-majority tie without forcing any participant to accept it.
    """

    support = public_support_counts(state)
    if not support:
        return ()
    highest = max(support.values())
    leaders = [option_id for option_id, count in support.items() if count == highest]
    if len(leaders) == 1:
        return (leaders[0],)

    preferences = public_preference_counts(state)
    preference_highest = max((preferences[option_id] for option_id in leaders), default=0)
    preference_leaders = sorted(
        option_id for option_id in leaders if preferences[option_id] == preference_highest
    )
    if len(preference_leaders) == 1:
        return (preference_leaders[0],)
    if rng is not None and preference_leaders:
        return (rng.choice(preference_leaders),)
    return ()


def majority_threshold(participant_count: int, fraction: float | None = None) -> int:
    configured = float(cfg.consensus.majority_fraction if fraction is None else fraction)
    return max(participant_count // 2 + 1, math.ceil(configured * participant_count))


def vote_counts(votes: dict[str, str | None]) -> Counter[str]:
    return Counter(option_id for option_id in votes.values() if option_id)


def outcome_from_votes(
    state: DialogueState,
    votes: dict[str, str | None],
    *,
    allow_unresolved: bool,
) -> RunOutcome | None:
    counts = vote_counts(votes)
    n = len(state.personas)
    if not counts:
        return RunOutcome("unresolved", None, dict(votes), "No valid votes were recorded") if allow_unresolved else None
    option_id, count = counts.most_common(1)[0]
    if count == n:
        return RunOutcome("successful", option_id, dict(votes), "All participants voted for the same option")
    if count >= majority_threshold(n):
        return RunOutcome("majority", option_id, dict(votes), f"{count} of {n} participants selected the option")
    if allow_unresolved:
        return RunOutcome("unresolved", None, dict(votes), "No option reached a majority in the final vote")
    return None
