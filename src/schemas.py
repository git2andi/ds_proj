"""Core dataclasses for the discussion simulator.

The purpose of this module is to make the simulation state explicit.  LLM calls
may produce text, but all control decisions operate over these typed objects.
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
    COMPARE = "compare"
    PUSH_BACK = "push_back"
    CONCEDE = "concede"
    PROPOSE_COMPROMISE = "propose_compromise"
    VOTE = "vote"
    ACCEPT = "accept"
    REJECT = "reject"
    CLOSE = "close"
    OTHER = "other"


LengthHint = Literal["short", "medium", "long"]


@dataclass(slots=True)
class OptionCard:
    id: str
    name: str
    attrs: dict[str, str] = field(default_factory=dict)
    upside: str = ""
    tradeoff: str = ""
    concern: str = ""
    fit: str = ""
    risk: str = ""
    best_for: str = ""

    def source_text(self) -> str:
        attr_text = ", ".join(f"{k}={v}" for k, v in self.attrs.items())
        pieces = [
            f"Option {self.id} - {self.name}",
            f"attrs: {attr_text}" if attr_text else "attrs: none",
            f"upside: {self.upside}",
            f"tradeoff: {self.tradeoff}",
            f"concern: {self.concern}",
            f"fit: {self.fit}",
            f"risk: {self.risk}",
            f"best for: {self.best_for}",
        ]
        return "; ".join(p for p in pieces if p)

    def short_text(self) -> str:
        return f"Option {self.id}: {self.name} — {self.upside or self.fit or self.tradeoff}".strip()


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
    patience: float
    initiative: float
    conflict_directness: float
    detail_level: float


@dataclass(slots=True)
class Persona:
    id: str
    name: str
    role: str
    traits: TraitProfile
    speech_style: str
    private_goal: str
    preferred_option: str
    acceptable_options: list[str]
    soft_rejections: list[str]
    hard_rejections: list[str]
    reasons: dict[str, list[str]]
    reservation: str
    reconsider_if: str
    is_hard_blocker: bool = False

    def can_accept_initially(self, option_id: str) -> bool:
        return option_id in self.acceptable_options or option_id == self.preferred_option


@dataclass(slots=True)
class ParticipantRuntime:
    persona_id: str
    turn_count: int = 0
    last_spoke_turn: Optional[int] = None
    explicit_vote: Optional[str] = None
    accepted_options: set[str] = field(default_factory=set)
    rejected_options: set[str] = field(default_factory=set)
    stated_reasons: list[str] = field(default_factory=list)
    already_said: list[str] = field(default_factory=list)
    current_preference: Optional[str] = None


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
    rejects: list[str] = field(default_factory=list)
    proposes_option: Optional[str] = None


@dataclass(slots=True)
class TurnRecord:
    index: int
    speaker_id: str
    speaker_name: str
    text: str
    phase: Phase
    act: DialogueAct
    intent: Optional["MoveIntent"] = None
    tokens_in: int = 0
    tokens_out: int = 0
    validation_issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OpenQuestion:
    asked_by: str
    target_id: str
    text: str
    turn_index: int
    answered: bool = False


@dataclass(slots=True)
class OptionCoverage:
    mentions: int = 0
    reasons: int = 0
    objections: int = 0
    acceptances: int = 0


@dataclass(slots=True)
class CompromiseProposal:
    proposed_by: str
    option_id: str
    text: str
    turn_index: int


@dataclass(slots=True)
class MoveIntent:
    speaker_id: str
    act: ActType
    reason: str
    addressee_id: Optional[str] = None
    option_focus: list[str] = field(default_factory=list)
    length_hint: LengthHint = "medium"
    is_moderator: bool = False


@dataclass(slots=True)
class DialogueState:
    scenario: Scenario
    personas: list[Persona]
    phase: Phase = Phase.OPENING
    turn_index: int = 0
    turns: list[TurnRecord] = field(default_factory=list)
    runtimes: dict[str, ParticipantRuntime] = field(default_factory=dict)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    coverage: dict[str, OptionCoverage] = field(default_factory=dict)
    candidate_option: Optional[str] = None
    compromise_proposals: list[CompromiseProposal] = field(default_factory=list)
    moderator_interventions: int = 0
    no_progress_count: int = 0
    readiness_score: float = 0.0
    outcome: Optional["RunOutcome"] = None

    def persona_by_id(self, persona_id: str) -> Persona:
        for persona in self.personas:
            if persona.id == persona_id:
                return persona
        raise KeyError(persona_id)

    def name_for(self, persona_id: str) -> str:
        return self.persona_by_id(persona_id).name

    def participant_ids(self) -> list[str]:
        return [p.id for p in self.personas]


@dataclass(slots=True)
class ValidationIssue:
    code: str
    severity: Literal["warn", "repair", "fatal"]
    message: str


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    repaired_text: Optional[str] = None

    @property
    def needs_repair(self) -> bool:
        return any(i.severity in {"repair", "fatal"} for i in self.issues)

    def codes(self) -> list[str]:
        return [i.code for i in self.issues]


@dataclass(slots=True)
class RunOutcome:
    status: Literal["consensus", "fallback", "unresolved"]
    final_option: Optional[str]
    reason: str
    turns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DialogueRunResult:
    scenario: Scenario
    personas: list[Persona]
    transcript: list[str]
    outcome: RunOutcome
    log_paths: dict[str, str] = field(default_factory=dict)
