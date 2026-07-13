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
from models import ActType, DialogueState, Phase, PublicParticipantState, RunOutcome, ThreadStatus, ThreadType

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
    """Current visible backing derived only from accepted transcript evidence.

    A participant's latest visible option-bound stance supersedes an earlier one.
    Hidden runtime ranks/preferences never fill missing public evidence.
    """
    support: dict[str, set[str]] = {oid: set() for oid in state.scenario.option_ids}
    option_ids = set(state.scenario.option_ids)
    current: dict[str, str] = {}
    for turn in state.turns:
        if turn.speaker_id not in state.runtimes or turn.state_mutation_blocked:
            continue
        if phase is not None and turn.phase is not phase:
            continue
        if turn.evidence is None:
            continue
        raised_blockers = {
            blocker.option_id for blocker in turn.evidence.blockers
            if blocker.action == "raised" and blocker.option_id in option_ids
        }
        if current.get(turn.speaker_id) in raised_blockers:
            current.pop(turn.speaker_id, None)
        commitment = turn.evidence.sole_commitment()
        if commitment and commitment.option_id in option_ids:
            current[turn.speaker_id] = commitment.option_id
            continue
        softened = [x.option_id for x in turn.evidence.softenings if x.option_id in option_ids]
        if softened:
            current[turn.speaker_id] = softened[-1]
            continue
        if include_support_acts and turn.evidence.supports:
            supported = [x.option_id for x in turn.evidence.supports if x.option_id in option_ids]
            if supported:
                current[turn.speaker_id] = supported[-1]
    for pid, oid in current.items():
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



def discussion_objection_mentions(state: DialogueState) -> Counter:
    """Count accepted visible objections per option before formal voting."""
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
        objected = {
            concern.option_id for concern in turn.evidence.concerns
            if concern.option_id in option_ids
        }
        objected.update(
            blocker.option_id for blocker in turn.evidence.blockers
            if blocker.action == "raised" and blocker.option_id in option_ids
        )
        for option_id in objected:
            counts[option_id] += 1
    return counts

def visible_proposals(state: DialogueState) -> set[str]:
    """Existing options visibly proposed as common ground in accepted turns."""
    option_ids = set(state.scenario.option_ids)
    return {
        proposal.option_id
        for turn in state.turns
        if turn.speaker_id in state.runtimes
        and not turn.state_mutation_blocked
        and turn.evidence is not None
        for proposal in turn.evidence.proposals
        if proposal.option_id in option_ids
    }


def public_participant_ledger(state: DialogueState) -> dict[str, PublicParticipantState]:
    """Derive one compact social ledger from accepted visible evidence only."""
    option_ids = set(state.scenario.option_ids)
    latest_position: dict[str, str | None] = {p.id: None for p in state.personas}
    supported: dict[str, set[str]] = {p.id: set() for p in state.personas}
    concerned: dict[str, set[str]] = {p.id: set() for p in state.personas}
    last_act: dict[str, ActType | None] = {p.id: None for p in state.personas}
    last_turn: dict[str, int | None] = {p.id: None for p in state.personas}
    last_focus: dict[str, tuple[str, ...]] = {p.id: () for p in state.personas}
    pending_from: dict[str, str | None] = {p.id: None for p in state.personas}
    pending_to: dict[str, str | None] = {p.id: None for p in state.personas}
    issue_keys: dict[str, set[str]] = {p.id: set() for p in state.personas}

    for turn in state.turns:
        pid = turn.speaker_id
        if pid not in latest_position or turn.state_mutation_blocked or turn.evidence is None:
            continue
        ev = turn.evidence
        last_act[pid] = turn.realized_act()
        last_turn[pid] = turn.index
        focus = tuple(dict.fromkeys(
            oid for oid in (turn.intent.option_focus if turn.intent else turn.mentioned_options())
            if oid in option_ids
        ))
        last_focus[pid] = focus
        for entry in ev.supports:
            if entry.option_id in option_ids:
                supported[pid].add(entry.option_id)
                latest_position[pid] = entry.option_id
        for entry in ev.softenings:
            if entry.option_id in option_ids:
                supported[pid].add(entry.option_id)
                latest_position[pid] = entry.option_id
        for entry in ev.commitments:
            if entry.option_id in option_ids:
                supported[pid].add(entry.option_id)
                latest_position[pid] = entry.option_id
        for entry in ev.proposals:
            if entry.option_id in option_ids:
                supported[pid].add(entry.option_id)
        for entry in ev.concerns:
            if entry.option_id in option_ids:
                concerned[pid].add(entry.option_id)
        for entry in ev.blockers:
            if entry.option_id not in option_ids:
                continue
            if entry.action == "raised":
                concerned[pid].add(entry.option_id)
            elif entry.action == "resolved":
                concerned[pid].discard(entry.option_id)

    for thread in state.threads.values():
        if thread.status not in (ThreadStatus.HOT, ThreadStatus.COOLING):
            continue
        owner = thread.started_by
        if owner in issue_keys and thread.issue_key:
            issue_keys[owner].add(thread.issue_key)
        if thread.thread_type is ThreadType.QUESTION:
            if thread.required_respondent in pending_from:
                pending_from[str(thread.required_respondent)] = owner
                if owner in pending_to:
                    pending_to[owner] = str(thread.required_respondent)
        elif thread.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER) and owner in concerned:
            concerned[owner].update(oid for oid in thread.focus_options if oid in option_ids)

    return {
        persona.id: PublicParticipantState(
            participant_id=persona.id,
            name=persona.name,
            public_position=latest_position[persona.id],
            supported_options=tuple(oid for oid in state.scenario.option_ids if oid in supported[persona.id]),
            concerned_options=tuple(oid for oid in state.scenario.option_ids if oid in concerned[persona.id]),
            last_act=last_act[persona.id],
            last_turn_index=last_turn[persona.id],
            last_focus_options=last_focus[persona.id],
            pending_question_from=pending_from[persona.id],
            pending_question_to=pending_to[persona.id],
            active_issue_keys=tuple(sorted(issue_keys[persona.id])),
        )
        for persona in state.personas
    }


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
    objection_counts: Counter           # accepted public objections per option
    candidate_scores: dict[str, int]    # weighted public backing less objection load
    candidate_leaders: tuple[str, ...]  # top-scoring options; empty without public evidence
    top_pair: tuple[str, ...]           # up to two best publicly backed options, sorted


def public_evidence(state: DialogueState) -> PublicEvidence:
    """Centralized public-evidence snapshot used by policy and flow.

    Candidate scores are derived solely from accepted visible turns: current
    backing, formal votes, positive discussion evidence, and visible proposals.
    Stable scenario option order breaks exact ties.
    """
    backing = public_support(state, include_support_acts=True)
    formal = visible_votes_from_transcript(state)
    formal_counts = Counter(formal.values())
    positive = discussion_positive_mentions(state)
    proposals = visible_proposals(state)
    objections = discussion_objection_mentions(state)
    scores: dict[str, int] = {}
    for oid in state.scenario.option_ids:
        score = (
            2 * len(backing[oid])
            + 2 * int(formal_counts.get(oid, 0))
            + int(positive.get(oid, 0))
            + (2 if oid in proposals else 0)
            - int(objections.get(oid, 0))
        )
        if score > 0:
            scores[oid] = score
    best = max(scores.values(), default=0)
    leaders = tuple(oid for oid in state.scenario.option_ids if scores.get(oid) == best) if best else ()
    ranked = sorted(
        (oid for oid in state.scenario.option_ids if oid in scores),
        key=lambda oid: (-scores[oid], state.scenario.option_ids.index(oid)),
    )
    return PublicEvidence(
        backing=backing,
        formal_votes=formal,
        formal_counts=formal_counts,
        proposals=proposals,
        objection_counts=objections,
        candidate_scores=scores,
        candidate_leaders=leaders,
        top_pair=tuple(ranked[:2]),
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
