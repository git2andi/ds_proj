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
        if turn.speaker_id == "moderator" or turn.state_mutation_blocked:
            continue
        if turn.speaker_id not in state.runtimes:
            continue
        if turn.phase not in _COMMITMENT_PHASES:
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
    """Who visibly backed each option, from accepted evidence only.

    A validated vote or acceptance commitment always counts; with
    ``include_support_acts`` validated option-bound support evidence counts
    too (used by the narrowing gate). Private ranks never count — this is the
    public layer. A participant's later visible hard rejection withdraws
    their backing.
    """
    option_ids = set(state.scenario.option_ids)
    support: dict[str, set[str]] = {oid: set() for oid in state.scenario.option_ids}
    for turn in state.turns:
        if turn.speaker_id == "moderator" or turn.state_mutation_blocked:
            continue
        if turn.speaker_id not in state.runtimes:
            continue
        if phase is not None and turn.phase is not phase:
            continue
        evidence = turn.evidence
        if evidence is None:
            continue  # no accepted evidence = no public semantic signal
        backed = {c.option_id for c in evidence.commitments if c.option_id in option_ids}
        if include_support_acts:
            backed.update(s.option_id for s in evidence.supports if s.option_id in option_ids)
        for oid in backed:
            support[oid].add(turn.speaker_id)
    for oid in support:
        support[oid] = {
            pid for pid in support[oid] if oid not in state.runtimes[pid].rejected_options()
        }
    return support


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
    proposal_counts: Counter = Counter()
    for turn in state.turns:
        if turn.speaker_id == "moderator" or turn.state_mutation_blocked:
            continue
        if turn.evidence is None:
            continue
        for proposal in turn.evidence.proposals:
            if proposal.option_id in option_ids:
                proposal_counts[proposal.option_id] += 1
    commitments = Counter(
        rt.explicit_vote for rt in state.runtimes.values() if rt.explicit_vote in option_ids
    )
    scores: dict[str, int] = {}
    for oid in state.scenario.option_ids:
        other_backing = sum(
            1 for pid in backing[oid] if state.runtimes[pid].explicit_vote != oid
        )
        score = 2 * commitments.get(oid, 0) + other_backing + proposal_counts.get(oid, 0)
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
        proposals=set(proposal_counts),
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
            return RunOutcome("successful", winner, "All participants visibly committed to the same option.", turns, metadata)
        threshold = math.ceil(float(cfg.consensus.majority_fraction) * len(state.personas))
        if support >= threshold and list(counts.values()).count(support) == 1:
            return RunOutcome("majority", winner, f"{support}/{len(state.personas)} participants visibly committed to the winning option.", turns, metadata)
        return RunOutcome("unresolved", None, "Visible commitments did not produce a unique majority.", turns, metadata)
