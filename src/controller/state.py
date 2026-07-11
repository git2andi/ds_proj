"""Controller runtime state: phases, threads, and the repair objective.

These are the controller's own runtime dataclasses (todo 5). Stable domain
objects (scenario, personas, dialogue acts, turns, the shared DialogueState
container) stay in models.py, which re-exports these types for a single stable
import surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Phase(str, Enum):
    """Explicit controller phases (todo 5.1). Allowed transitions:

    opening -> discussion -> narrowing -> voting -> closing
    narrowing -> discussion        (at most once, on candidate collapse)
    voting -> compromise_repair -> voting | closing
    """

    OPENING = "opening"
    DISCUSSION = "discussion"
    NARROWING = "narrowing"
    VOTING = "voting"
    COMPROMISE_REPAIR = "compromise_repair"
    CLOSING = "closing"


class ThreadType(str, Enum):
    """Persistent local-interaction thread types (todo 5.2)."""

    QUESTION = "question"
    CONCERN = "concern"
    BLOCKER = "blocker"
    COMPARISON = "comparison"


class ThreadStatus(str, Enum):
    """Thread lifecycle statuses (todo 5.3).

    hot      — should influence the next route; may block coverage/narrowing/voting
    cooling  — received one relevant response; may continue but does not force it
    resolved — no further routing attention needed
    stale    — no longer blocks progress; kept for repetition avoidance/reactivation
    """

    HOT = "hot"
    COOLING = "cooling"
    RESOLVED = "resolved"
    STALE = "stale"


class BlockingStrength(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


QuestionScope = Literal["direct", "group"]


@dataclass(slots=True)
class ThreadState:
    """Minimal persistent thread record (todo 5.4/5.5).

    Identity normally follows (thread_type, focus_options, issue_key); the
    source text is retrievable via source_turn_index — no copied turn text, no
    LLM summaries, no derived age counters. Staleness removes routing priority
    only; it never erases a participant's rejection/concern from stance state.
    """

    thread_id: str
    thread_type: ThreadType
    status: ThreadStatus = ThreadStatus.HOT
    focus_options: list[str] = field(default_factory=list)
    issue_key: str = ""
    started_by: str = ""
    required_respondent: str | None = None
    participants_involved: list[str] = field(default_factory=list)
    # Accepted turns folded into this thread (open/restate/response/reaction);
    # the soft/hard per-thread routing caps count these, not unique speakers.
    contribution_count: int = 0
    # Bounded blocker probes charged post-turn (peer or moderator, whichever
    # fires first): each blocker THREAD gets at most one natural probe, so one
    # sim's probe never suppresses another sim's blocker on the same option.
    probe_count: int = 0
    source_turn_index: int = 0
    created_turn: int = 0
    last_touched_turn: int = 0
    blocking_strength: BlockingStrength = BlockingStrength.NONE
    resolution_reason: str | None = None
    # Question semantics (5.5); None for non-question threads.
    question_scope: QuestionScope | None = None


@dataclass(slots=True)
class RepairState:
    """The single active repair objective (todo 13.7).

    repair_reason: unclear_vote | hard_blocker | majority_holdout | split_vote |
    two_person_deadlock. Only one repair objective may be active at once.
    """

    repair_reason: str
    candidate_or_pair: list[str] = field(default_factory=list)
    participants_involved: list[str] = field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 1
    status: str = "active"  # active | resolved | exhausted
