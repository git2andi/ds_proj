"""Outcome logic: turn counting and visible-vote consensus.

Kept separate from the controller (`dialogue.py`): outcomes are computed from
visible transcript evidence only, so this module depends on nothing in the
controller and can be tested in isolation.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from config_loader import cfg
from models import DialogueState, Phase, RunOutcome

# Only these phases produce outcome-relevant commitments (13.1): formal votes
# and repair-phase final acceptances/switches. Opening leans and discussion
# support update public stance state but never silently become final votes.
_COMMITMENT_PHASES = {Phase.VOTING, Phase.COMPROMISE_REPAIR}


def participant_turn_count(state: DialogueState) -> int:
    return sum(1 for turn in state.turns if turn.speaker_id != "moderator")


def visible_votes_from_transcript(state: DialogueState) -> dict[str, str]:
    """Last formal visible commitment per participant, read from transcript turns.

    Runtime votes are useful for routing, but final outcomes must be grounded in
    what the transcript visibly says during the formal commitment phases. A
    later repair-phase concession replaces an earlier formal vote. Scanning
    committed turns in order keeps outcome metadata and transcript evidence
    synchronized.
    """
    option_ids = set(state.scenario.option_ids)
    votes: dict[str, str] = {}
    for turn in state.turns:
        if turn.state_mutation_blocked or not turn.is_formal_commitment_turn():
            continue
        if turn.speaker_id not in state.runtimes:
            continue
        vote = turn.visible_vote()
        if vote not in option_ids:
            accepted = [oid for oid in turn.visible_accepts() if oid in option_ids]
            if len(accepted) == 1:
                vote = accepted[0]
        if vote in option_ids:
            votes[turn.speaker_id] = vote
    return votes


def public_support(
    state: DialogueState,
    *,
    phase: Phase | None = None,
    include_support_acts: bool = False,
) -> dict[str, set[str]]:
    """Current visible backing only; superseded positions never remain active."""
    support: dict[str, set[str]] = {oid: set() for oid in state.scenario.option_ids}
    option_ids = set(state.scenario.option_ids)
    if phase is not None:
        # Phase-scoped diagnostics use the last visible option-bound signal in
        # that phase per participant.
        current: dict[str, str] = {}
        for turn in state.turns:
            if turn.phase is not phase or turn.speaker_id not in state.runtimes or turn.state_mutation_blocked:
                continue
            if turn.evidence is None:
                continue
            commitment = turn.evidence.sole_commitment()
            if commitment and commitment.option_id in option_ids:
                current[turn.speaker_id] = commitment.option_id
            elif include_support_acts and turn.evidence.supports:
                oid = turn.evidence.supports[-1].option_id
                if oid in option_ids:
                    current[turn.speaker_id] = oid
        for pid, oid in current.items():
            if oid not in state.runtimes[pid].rejected_options():
                support[oid].add(pid)
        return support

    current: dict[str, str] = {}
    for turn in state.turns:
        if turn.speaker_id not in state.runtimes or turn.state_mutation_blocked or turn.evidence is None:
            continue
        commitment = turn.evidence.sole_commitment()
        if commitment and commitment.option_id in option_ids:
            current[turn.speaker_id] = commitment.option_id
            continue
        softened = [x.option_id for x in turn.evidence.softenings if x.option_id in option_ids]
        if softened:
            current[turn.speaker_id] = softened[-1]
            continue
        if include_support_acts and turn.evidence.supports:
            oid = turn.evidence.supports[-1].option_id
            if oid in option_ids:
                current[turn.speaker_id] = oid
    # Runtime values cover synthetic/current states whose trace was not built by
    # the normal observer, but transcript-visible evidence wins when present.
    for pid, rt in state.runtimes.items():
        current.setdefault(pid, rt.explicit_vote or rt.current_acceptance or rt.public_lean)
    for pid, oid in current.items():
        if oid in option_ids and oid not in state.runtimes[pid].rejected_options():
            support[oid].add(pid)
    return support


def discussion_positive_mentions(state: DialogueState) -> Counter:
    """Count visible positive option references before formal voting.

    Each accepted participant turn contributes at most once per option. Pure
    mentions and concerns do not count. The signal is historical by design: it
    is used only to choose which equally voted option should be tested as the
    single compromise candidate after a tie.
    """
    option_ids = set(state.scenario.option_ids)
    counts: Counter = Counter()
    for turn in state.turns:
        if (
            turn.speaker_id == "moderator"
            or turn.state_mutation_blocked
            or turn.phase in _COMMITMENT_PHASES
            or turn.evidence is None
        ):
            continue
        positive: set[str] = set()
        positive.update(s.option_id for s in turn.evidence.supports if s.option_id in option_ids)
        positive.update(
            c.favored for c in turn.evidence.comparisons
            if c.favored in option_ids
        )
        positive.update(
            s.option_id for s in turn.evidence.softenings
            if s.option_id in option_ids
        )
        positive.update(
            c.option_id for c in turn.evidence.commitments
            if c.option_id in option_ids
        )
        positive.update(
            p.option_id for p in turn.evidence.proposals
            if p.option_id in option_ids
        )
        for option_id in positive:
            counts[option_id] += 1
    return counts


@dataclass(frozen=True)
class PublicEvidence:
    """The single public-evidence snapshot consumed by policy and flow.

    Everything here derives from accepted visible turns (plus the runtime's
    current public commitment, which is itself set only from parsed visible
    commitments). Private rank tables never contribute.
    """

    backing: dict[str, set[str]]        # option -> visible backers (vote/acceptance, any phase)
    formal_votes: dict[str, str]        # participant -> last formal vote (voting/repair phases)
    formal_counts: Counter              # option -> formal vote count
    proposals: set[str]                 # options visibly offered as common ground
    candidate_scores: dict[str, int]    # weighted public backing per option
    candidate_leaders: tuple[str, ...]  # top-scoring options; empty without public evidence
    top_pair: tuple[str, ...]           # up to two best publicly backed options, sorted


def public_evidence(state: DialogueState) -> PublicEvidence:
    """One centralized public-evidence calculation (discussion support, formal
    votes/counts, public candidate scores, public top pair).

    Candidate scoring: a participant's current visible commitment weighs
    double, their other visible backing once, and each visible compromise
    proposal once — the same weighting the old vote-candidate helper used,
    minus the private acceptable-rank leak.
    """
    option_ids = set(state.scenario.option_ids)
    backing = public_support(state)
    formal = visible_votes_from_transcript(state)
    commitments = Counter(
        rt.explicit_vote for rt in state.runtimes.values() if rt.explicit_vote in option_ids
    )
    scores: dict[str, int] = {}
    for oid in state.scenario.option_ids:
        current_backing = len(backing[oid])
        score = current_backing + commitments.get(oid, 0)
        if score:
            scores[oid] = score
    best = max(scores.values(), default=0)
    leaders = tuple(sorted(oid for oid, s in scores.items() if s == best)) if best else ()
    ranked = sorted(
        ((len(pids), oid) for oid, pids in backing.items() if pids),
        key=lambda item: (-item[0], item[1]),
    )
    top_pair = tuple(sorted(oid for _count, oid in ranked[:2]))
    return PublicEvidence(
        backing=backing,
        formal_votes=formal,
        formal_counts=Counter(formal.values()),
        proposals=set(),
        candidate_scores=scores,
        candidate_leaders=leaders,
        top_pair=top_pair,
    )


class ConsensusManager:
    @staticmethod
    def finalize(state: DialogueState) -> RunOutcome:
        votes = visible_votes_from_transcript(state)
        counts = Counter(votes.values())
        turns = participant_turn_count(state)
        metadata = {
            "visible_votes": votes,
            "latent_preferences": {pid: rt.top_option() for pid, rt in state.runtimes.items()},
            "stance_ranks": {pid: dict(rt.option_ranks) for pid, rt in state.runtimes.items()},
            "phase_history": list(state.phase_history),
            "candidate_option": state.candidate_option,
            "min_discussion_turns": state.min_discussion_turns,
            "force_narrow_turns": state.force_narrow_turns,
            "hard_max_turns": state.hard_max_turns,
        }
        if not counts:
            return RunOutcome("unresolved", None, "No visible votes or acceptances were produced.", turns, metadata)
        winner, support = counts.most_common(1)[0]
        if support == len(state.personas):
            active_blockers = [pid for pid, rt in state.runtimes.items() if winner in rt.rejected_options()]
            if active_blockers:
                metadata["active_blockers_on_winner"] = active_blockers
                return RunOutcome("unresolved", None, "A unanimous tally conflicts with an active blocker.", turns, metadata)
            return RunOutcome("successful", winner, "All participants visibly committed to the same option.", turns, metadata)
        threshold = math.ceil(float(cfg.consensus.majority_fraction) * len(state.personas))
        if support >= threshold and list(counts.values()).count(support) == 1:
            return RunOutcome("majority", winner, f"{support}/{len(state.personas)} participants visibly committed to the winning option.", turns, metadata)
        return RunOutcome("unresolved", None, "Visible commitments did not produce a unique majority.", turns, metadata)
