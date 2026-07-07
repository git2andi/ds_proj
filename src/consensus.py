"""Outcome logic: turn counting and visible-vote consensus.

Kept separate from the controller (`dialogue.py`): outcomes are computed from
visible transcript evidence only, so this module depends on nothing in the
controller and can be tested in isolation.
"""

from __future__ import annotations

import math
from collections import Counter

from config_loader import cfg
from models import DialogueState, RunOutcome


def participant_turn_count(state: DialogueState) -> int:
    return sum(1 for turn in state.turns if turn.speaker_id != "moderator")


def visible_votes_from_transcript(state: DialogueState) -> dict[str, str]:
    """Last visible public commitment per participant, read from transcript turns.

    Runtime votes are useful for routing, but final outcomes must be grounded in
    what the transcript visibly says. Earlier discussion commitments can become
    stale when a participant changes their formal final vote without saying the
    word "switch". Scanning committed turns in order keeps outcome metadata and
    transcript evidence synchronized.
    """
    option_ids = set(state.scenario.option_ids)
    votes: dict[str, str] = {}
    for turn in state.turns:
        if turn.speaker_id == "moderator" or turn.state_mutation_blocked:
            continue
        if turn.speaker_id not in state.runtimes:
            continue
        vote = turn.act.explicit_vote
        if vote not in option_ids and turn.act.accepts:
            accepted = [oid for oid in turn.act.accepts if oid in option_ids]
            if len(accepted) == 1:
                vote = accepted[0]
        if vote in option_ids:
            votes[turn.speaker_id] = vote
    return votes


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
