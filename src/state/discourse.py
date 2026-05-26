"""
state/discourse.py
------------------
DiscourseGraph — tracks pending questions and reply structure (Ouchi & Tsuboi 2016).

Pending questions persist across multiple turns until answered.
This is the key data structure that enables the SSJ rule 1a obligation
(addressed party must respond) in Stage 6's turn policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiscourseGraph:
    """
    First-class question/answer adjacency-pair tracking.

    pending_questions  : turn_id → list of addressed participant names
    reply_edges        : answer_turn_id → question_turn_id
    open_invitations   : turn_ids of group-directed (no specific addressee) questions
    last_addressed     : most recently name-mentioned participant (may or may not be a question)
    """
    pending_questions: dict[int, list[str]] = field(default_factory=dict)
    reply_edges: dict[int, int]             = field(default_factory=dict)
    open_invitations: list[int]             = field(default_factory=list)
    last_addressed: Optional[str]           = None

    def has_obligation_for(self, name: str) -> bool:
        """True if `name` appears as an addressee in any pending question."""
        return any(name in addressees for addressees in self.pending_questions.values())

    def resolve_question(self, answering_turn_id: int, speaker: str) -> Optional[int]:
        """
        If `speaker` is in any pending question's addressee list, mark it answered.
        Returns the resolved question's turn_id, or None if nothing was resolved.
        """
        for q_id, addressees in list(self.pending_questions.items()):
            if speaker in addressees:
                self.reply_edges[answering_turn_id] = q_id
                del self.pending_questions[q_id]
                return q_id
        return None

    def add_question(self, turn_id: int, addressees: list[str]) -> None:
        """Register a new question. Empty addressees → open invitation."""
        if addressees:
            self.pending_questions[turn_id] = list(addressees)
        else:
            self.open_invitations.append(turn_id)

    def oldest_pending_addressees(self) -> list[str]:
        """Addressees of the oldest unanswered question (FIFO priority)."""
        if not self.pending_questions:
            return []
        oldest_id = min(self.pending_questions)
        return self.pending_questions[oldest_id]
