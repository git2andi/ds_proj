"""
state/turn.py
--------------
Core per-turn data types for the structured state layer (Stage 5+).

DialogueAct  — enumerated move types derived from Fisher, SSJ, MUCA
StanceUpdate — a single speaker×option stance observation
TurnRecord   — full structured record for one turn, supersedes TurnTrace
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DialogueAct(Enum):
    GREET              = "GREET"
    OPEN_PRIORITY      = "OPEN_PRIORITY"
    ASSERT_SUPPORT     = "ASSERT_SUPPORT"       # Fisher: favorable
    ASSERT_OPPOSITION  = "ASSERT_OPPOSITION"    # Fisher: unfavorable
    ASSERT_AMBIGUOUS   = "ASSERT_AMBIGUOUS"     # Fisher: ambiguous
    ASK_CLARIFICATION  = "ASK_CLARIFICATION"
    ASK_PREFERENCE     = "ASK_PREFERENCE"
    ANSWER             = "ANSWER"
    CHALLENGE          = "CHALLENGE"
    CONCEDE            = "CONCEDE"
    CONDITIONAL_ACCEPT = "CONDITIONAL_ACCEPT"
    PROPOSE_COMPROMISE = "PROPOSE_COMPROMISE"
    COMMIT_VOTE        = "COMMIT_VOTE"
    CONFIRM            = "CONFIRM"
    REJECT_WITH_REASON = "REJECT_WITH_REASON"
    SUMMARIZE          = "SUMMARIZE"
    GOODBYE            = "GOODBYE"
    SILENCE            = "SILENCE"
    MODERATOR          = "MODERATOR"


@dataclass
class StanceUpdate:
    """One speaker's publicly-observable stance toward one option at one moment."""
    speaker: str
    option: str                         # "A".."D"
    stance: str                         # "support"|"oppose"|"ambiguous"|
                                        # "conditional_support"|"blocker"|"neutral"
    confidence: float = 1.0            # 0.0..1.0 — deterministic rules use fixed values
    condition: Optional[str] = None    # for conditional_support / blocker


@dataclass
class TurnRecord:
    """
    Full structured per-turn record.
    Replaces the lightweight TurnTrace from Stage 1 when Stage 5 is wired in.
    Written to {dialogue_id}_state.jsonl.
    """
    turn_id: int
    speaker: str
    text: str
    phase: str
    is_moderator: bool

    # Discourse
    addressees: list[str]              = field(default_factory=list)
    reply_to: Optional[int]            = None   # turn_id of the question being answered
    is_question: bool                  = False
    answers_question_id: Optional[int] = None   # turn_id of the question this answers

    # Content
    dialogue_act: DialogueAct          = DialogueAct.ASSERT_AMBIGUOUS
    mentioned_options: list[str]       = field(default_factory=list)
    stance_updates: list[StanceUpdate] = field(default_factory=list)

    # Selection metadata (filled by orchestrator)
    selected_reason: str = ""
    tokens_in: int       = 0
    tokens_out: int      = 0
    prompt_type: str     = "sim_turn"

    def to_dict(self) -> dict:
        """Serialisable dict for JSONL logging."""
        d = dataclasses.asdict(self)
        d["dialogue_act"] = self.dialogue_act.value
        d["stance_updates"] = [
            {k: v for k, v in su.items()} for su in d["stance_updates"]
        ]
        return d

    @staticmethod
    def from_dict(d: dict) -> "TurnRecord":
        stance_updates = [
            StanceUpdate(**su) for su in d.pop("stance_updates", [])
        ]
        act_raw = d.pop("dialogue_act", "ASSERT_AMBIGUOUS")
        try:
            act = DialogueAct(act_raw)
        except ValueError:
            act = DialogueAct.ASSERT_AMBIGUOUS
        return TurnRecord(**d, dialogue_act=act, stance_updates=stance_updates)
