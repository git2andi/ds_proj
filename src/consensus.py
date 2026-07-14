"""Public candidate derivation and formal vote counting."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from config_loader import cfg
from models import DialogueState, RunOutcome


@dataclass(slots=True, frozen=True)
class CandidateStanding:
    option_id: str
    score: float
    public_preferences: int
    public_acceptances: int
    supports: int
    concerns: int


def candidate_standings(state: DialogueState) -> list[CandidateStanding]:
    standings: list[CandidateStanding] = []
    for option_id in state.scenario.option_ids:
        preferences = sum(runtime.public_preference == option_id for runtime in state.runtimes.values())
        acceptances = sum(option_id in runtime.public_acceptances for runtime in state.runtimes.values())
        supports = len(state.public_supporters.get(option_id, set()))
        concerns = len(state.public_concern_raisers.get(option_id, set()))
        comparison_mentions = sum(
            len(state.public_comparers.get(pair, set()))
            for pair in state.public_comparisons
            if option_id in pair
        )
        score = 3.0 * preferences + 1.4 * acceptances + 0.8 * supports - 0.65 * concerns + 0.15 * comparison_mentions
        standings.append(CandidateStanding(
            option_id=option_id,
            score=score,
            public_preferences=preferences,
            public_acceptances=acceptances,
            supports=supports,
            concerns=concerns,
        ))
    return sorted(standings, key=lambda row: (row.score, row.public_preferences, row.supports, row.option_id), reverse=True)


def derive_narrowing_options(state: DialogueState) -> tuple[str, ...]:
    standings = candidate_standings(state)
    if not standings:
        return ()
    n = len(state.personas)
    leader = standings[0]
    if leader.public_preferences == n:
        return (leader.option_id,)
    if len(standings) == 1:
        return (leader.option_id,)
    runner_up = standings[1]
    # A clearly dominant public leader can be announced alone; otherwise retain
    # the top pair. Hidden private ranks never enter this calculation.
    if leader.public_preferences >= max(2, math.floor(n / 2) + 1) and leader.score - runner_up.score >= 3.0:
        return (leader.option_id,)
    return (leader.option_id, runner_up.option_id)


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
    valid_votes = sum(counts.values())
    if valid_votes == n and count == n:
        return RunOutcome("successful", option_id, dict(votes), "All participants voted for the same option")
    if count >= majority_threshold(n):
        return RunOutcome("majority", option_id, dict(votes), f"{count} of {n} participants selected the option")
    if allow_unresolved:
        return RunOutcome("unresolved", None, dict(votes), "No option reached a majority after the bounded re-vote")
    return None
