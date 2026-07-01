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


class ConsensusManager:
    @staticmethod
    def finalize(state: DialogueState) -> RunOutcome:
        votes = {
            persona.id: state.runtimes[persona.id].explicit_vote
            for persona in state.personas
            if state.runtimes[persona.id].explicit_vote in state.scenario.option_ids
        }
        counts = Counter(votes.values())
        turns = participant_turn_count(state)
        metadata = {
            "visible_votes": votes,
            "latent_preferences": {pid: rt.current_preference for pid, rt in state.runtimes.items()},
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
