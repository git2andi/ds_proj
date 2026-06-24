"""Core typed objects for the simulator.

The LLM may generate text, but routing, consensus, logging, and validation operate
on these objects only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


class Phase(str, Enum):
    OPENING = "opening"
    DISCUSSION = "discussion"
    NARROWING = "narrowing"
    CONFIRMATION = "confirmation"
    CLOSURE = "closure"


class ActType(str, Enum):
    OPENING = "opening"
    ANSWER = "answer"
    REACT = "react"
    ASK = "ask"
    SUPPORT = "support"
    OBJECT = "object"
    COMPARE = "compare"
    PUSH_BACK = "push_back"
    PROPOSE_COMPROMISE = "propose_compromise"
    VOTE = "vote"
    ACCEPT = "accept"
    REJECT = "reject"


LengthHint = Literal["short", "medium", "long"]


@dataclass(slots=True)
class OptionCard:
    id: str
    name: str
    attrs: dict[str, str] = field(default_factory=dict)
    upside: str = ""
    tradeoff: str = ""
    concern: str = ""
    best_for: str = ""

    def public_line(self) -> str:
        attrs = ", ".join(f"{k}: {v}" for k, v in self.attrs.items())
        parts = [f"{self.id}) {self.name}"]
        if attrs:
            parts.append(attrs)
        if self.upside:
            parts.append(f"plus: {self.upside}")
        if self.tradeoff:
            parts.append(f"trade-off: {self.tradeoff}")
        return "; ".join(parts)

    def prompt_card(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in self.attrs.items())
        pieces = [
            f"Option {self.id}: {self.name}",
            f"attrs: {attrs}" if attrs else "attrs: none",
            f"upside: {self.upside}" if self.upside else "",
            f"tradeoff: {self.tradeoff}" if self.tradeoff else "",
            f"concern: {self.concern}" if self.concern else "",
            f"best_for: {self.best_for}" if self.best_for else "",
        ]
        return "; ".join(p for p in pieces if p)


@dataclass(slots=True)
class Scenario:
    topic: str
    decision_kind: str
    opening_question: str
    options: list[OptionCard]

    @property
    def option_ids(self) -> list[str]:
        return [o.id for o in self.options]

    def option(self, option_id: str) -> OptionCard:
        for option in self.options:
            if option.id == option_id:
                return option
        raise KeyError(option_id)


@dataclass(slots=True)
class TraitProfile:
    openness: int
    conscientiousness: int
    extraversion: int
    agreeableness: int
    neuroticism: int
    response_length: int
    compromise_willingness: float
    initiative: float
    directness: float
    detail: float


@dataclass(slots=True)
class Persona:
    id: str
    name: str
    role: str
    traits: TraitProfile
    speech_style: str
    private_goal: str
    backstory: str
    main_concern: str
    preferred_option: str
    acceptable_options: list[str]
    soft_rejections: list[str]
    hard_rejections: list[str]
    reasons: dict[str, list[str]]
    reservation: str
    reconsider_if: str
    option_scores: dict[str, int] = field(default_factory=dict)  # hidden 1..5 utility per option
    is_hard_blocker: bool = False

    def score_for(self, option_id: str) -> int:
        return self.option_scores.get(option_id, 3)


@dataclass(slots=True)
class MoveIntent:
    speaker_id: str
    act: ActType
    reason: str
    addressee_id: Optional[str] = None
    option_focus: list[str] = field(default_factory=list)
    length_hint: LengthHint = "medium"
    respond_to_turn: Optional[int] = None
    # True only when the router is deliberately handing the speaker a change-of-mind
    # (the persuasion "won over" turn). A SUPPORT move only moves the persona's lean
    # when this is set, so ordinary discussion/coverage-gap support of an acceptable
    # option no longer silently drifts their stance.
    moves_lean: bool = False


@dataclass(slots=True)
class DialogueAct:
    speaker_id: str
    text: str
    act_type: ActType
    option_refs: list[str] = field(default_factory=list)
    addressee_id: Optional[str] = None
    question_target_id: Optional[str] = None
    explicit_vote: Optional[str] = None
    accepts: list[str] = field(default_factory=list)
    soft_rejects: dict[str, str] = field(default_factory=dict)
    hard_rejects: dict[str, str] = field(default_factory=dict)
    proposes_option: Optional[str] = None


@dataclass(slots=True)
class OpenQuestion:
    turn_id: int
    asked_by: str
    target_id: str
    text: str
    option_focus: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OptionCoverage:
    mentions: int = 0
    reasons: int = 0
    objections: int = 0
    acceptances: int = 0
    covered_slots: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ParticipantRuntime:
    persona_id: str
    turn_count: int = 0
    last_spoke_turn: Optional[int] = None
    stated_priority: str = ""
    current_preference: Optional[str] = None
    explicit_vote: Optional[str] = None
    accepted_options: set[str] = field(default_factory=set)
    soft_rejections: dict[str, str] = field(default_factory=dict)
    hard_rejections: dict[str, str] = field(default_factory=dict)
    already_said: list[str] = field(default_factory=list)
    discourse_frames: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TurnRecord:
    index: int
    speaker_id: str
    speaker_name: str
    text: str
    phase: Phase
    act: DialogueAct
    intent: Optional[MoveIntent] = None
    tokens_in: int = 0
    tokens_out: int = 0
    validation_issues: list[str] = field(default_factory=list)  # issues remaining after any repair
    repaired: bool = False  # whether a repair attempt actually ran for this turn


@dataclass(slots=True)
class RunOutcome:
    status: str
    final_option: Optional[str]
    reason: str
    turns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DialogueState:
    scenario: Scenario
    personas: list[Persona]
    phase: Phase = Phase.OPENING
    turns: list[TurnRecord] = field(default_factory=list)
    runtimes: dict[str, ParticipantRuntime] = field(default_factory=dict)
    coverage: dict[str, OptionCoverage] = field(default_factory=dict)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    candidate_option: Optional[str] = None
    outcome: Optional[RunOutcome] = None
    turn_index: int = 0
    readiness_score: float = 0.0
    no_progress_count: int = 0
    facilitator_force_narrow: bool = False
    confirmation_start_turns: Optional[int] = None  # participant-turn count when confirmation began
    # Pacing targets derived once per run (group size + composition), so length varies
    # and scales with the number of participants instead of using one fixed floor.
    min_discussion_turns: int = 0
    force_narrow_turns: int = 0
    hard_max_turns: int = 0
    setup_tokens_in: int = 0
    setup_tokens_out: int = 0
    dialogue_tokens_in: int = 0
    dialogue_tokens_out: int = 0

    def participant_ids(self) -> list[str]:
        return [p.id for p in self.personas]

    def persona_by_id(self, persona_id: str) -> Persona:
        for persona in self.personas:
            if persona.id == persona_id:
                return persona
        raise KeyError(persona_id)

    def name_for(self, persona_id: Optional[str]) -> str:
        if persona_id is None:
            return ""
        if persona_id == "moderator":
            return "Moderator"
        return self.persona_by_id(persona_id).name


@dataclass(slots=True)
class DialogueRunResult:
    scenario: Scenario
    personas: list[Persona]
    transcript: list[str]
    outcome: RunOutcome
    log_paths: dict[str, str]
    token_summary: dict[str, int]


@dataclass(slots=True)
class ValidationIssue:
    code: str
    severity: str
    message: str


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def codes(self) -> list[str]:
        return [issue.code for issue in self.issues]
