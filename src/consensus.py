"""Simple public narrowing and clear visible vote counting."""

from __future__ import annotations

import math
from collections import Counter

from config_loader import cfg
from models import DialogueState, RunOutcome


def public_preference_counts(state: DialogueState) -> Counter[str]:
    return Counter(
        runtime.public_preference
        for runtime in state.runtimes.values()
        if runtime.public_preference in state.scenario.option_ids
    )


def public_acceptance_counts(state: DialogueState) -> Counter[str]:
    counts: Counter[str] = Counter()
    for runtime in state.runtimes.values():
        for option_id in runtime.public_acceptances:
            if option_id in state.scenario.option_ids:
                counts[option_id] += 1
    return counts


def derive_narrowing_options(state: DialogueState) -> tuple[str, ...]:
    """Derive a leader/top pair only from latest visible public positions.

    A complete tie does not fabricate finalists. Public acceptability is used
    only as a common-ground fallback when one option is acceptable to a
    majority of the group.
    """

    preferences = public_preference_counts(state)
    if preferences:
        highest = max(preferences.values())
        leaders = sorted(option_id for option_id, count in preferences.items() if count == highest)
        if len(leaders) == 1:
            return (leaders[0],)
        if len(leaders) == 2:
            return tuple(leaders)

    threshold = majority_threshold(len(state.personas))
    acceptances = public_acceptance_counts(state)
    common = sorted(option_id for option_id, count in acceptances.items() if count >= threshold)
    if len(common) == 1:
        return (common[0],)
    if len(common) == 2:
        return tuple(common)
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
        return RunOutcome("unresolved", None, dict(votes), "No option reached a majority after the bounded re-vote")
    return None
