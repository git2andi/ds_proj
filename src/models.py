"""Typed objects for the option-grounded multi-user simulator.

The project simulates several user simulators in a shared decision environment.
LLMs render utterances, but the environment owns the option board, simulator
parameters, agenda items, routing state, visible commitments, and outcome logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Phase(str, Enum):
    OPENING = "opening"
    DISCUSSION = "discussion"
    NARROWING = "narrowing"
    CLOSURE = "closure"


class ActType(str, Enum):
    OPENING = "opening"
    BUILD = "build"
    AGREE = "agree"
    CHALLENGE = "challenge"
    ASK = "ask"
    ANSWER = "answer"
    COMPARE = "compare"
    INVITE = "invite"
    PROPOSE_COMPROMISE = "propose_compromise"
    VOTE = "vote"
    ACCEPT = "accept"
    REJECT = "reject"
    REACT = "react"


class AgendaStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    OBSOLETE = "obsolete"


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
    short_name: str = ""

    def public_line(self, max_attrs: int = 3, note_words: int = 9) -> str:
        attr_bits = [
            f"{key.replace('_', ' ')}: {value}"
            for key, value in ordered_attrs(self.attrs)[: max(0, max_attrs)]
        ]
        details = "; ".join(attr_bits)
        suffix_bits: list[str] = []
        if self.upside:
            suffix_bits.append(f"+ {_clip_words(self.upside, note_words)}")
        if self.tradeoff:
            suffix_bits.append(f"− {_clip_words(self.tradeoff, note_words)}")
        suffix = f" ({'; '.join(suffix_bits)})" if suffix_bits else ""
        return f"{self.id}) {self.name}" + (f" — {details}" if details else "") + suffix

    def prompt_card(self) -> str:
        attrs = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in ordered_attrs(self.attrs))
        pieces = [
            f"{self.id}) {self.name}",
            f"attrs: {attrs}" if attrs else "attrs: none",
            f"upside: {self.upside}" if self.upside else "",
            f"tradeoff: {self.tradeoff}" if self.tradeoff else "",
            f"concern: {self.concern}" if self.concern else "",
            f"best_for: {self.best_for}" if self.best_for else "",
        ]
        return "; ".join(p for p in pieces if p)


def _clip_words(text: str, limit: int) -> str:
    words = str(text).split()
    if len(words) <= limit:
        return str(text).strip().rstrip(" .;:")
    return " ".join(words[:limit]).rstrip(" ,.;:") + "…"


def ordered_attrs(attrs: dict[str, str]) -> list[tuple[str, str]]:
    priority = [
        "cost", "price", "budget", "wait", "duration", "time", "length", "pages",
        "genre", "rating", "release", "difficulty", "availability", "flight",
        "departure", "arrival", "baggage", "distance", "drive", "capacity",
        "temperature", "ambiance", "setting", "activity", "effort", "battery",
        "comfort", "privacy", "accuracy", "safety", "space", "yield", "maintenance",
    ]

    def rank_for(key: str) -> tuple[int, str]:
        normalised = key.lower().replace("_", " ")
        for index, needle in enumerate(priority):
            if needle in normalised:
                return index, normalised
        return 999, normalised

    return sorted(attrs.items(), key=lambda item: rank_for(item[0]))


@dataclass(slots=True)
class Scenario:
    topic: str
    decision_kind: str
    opening_question: str
    options: list[OptionCard]
    shared_context: list[str] = field(default_factory=list)
    environment_type: str = "option_grounded_group_decision"

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

    @property
    def compromise_willingness(self) -> float:
        agree = (self.agreeableness - 1) / 4
        openness = (self.openness - 1) / 4
        calm = (5 - self.neuroticism) / 4
        value = 0.65 * agree + 0.20 * openness + 0.15 * calm
        floor = 0.05 if self.agreeableness == 1 else 0.35
        return max(floor, min(0.95, value))


@dataclass(slots=True)
class SimulatorParameters:
    """Operational controls for a simulated user.

    OCEAN traits are retained as a compact personality source, but routing and
    prompts use these explicit parameters because they are easier to tune and
    evaluate.
    """

    engagement: float
    verbosity: float
    initiative: float
    responsiveness: float
    stubbornness: float
    directness: float
    compromise_threshold: float

    def clipped(self) -> "SimulatorParameters":
        return SimulatorParameters(**{name: _clip01(getattr(self, name)) for name in self.__dataclass_fields__})


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class AgendaItem:
    act: ActType
    option: str | None = None
    reason: str = ""
    priority: float = 1.0
    status: AgendaStatus = AgendaStatus.PENDING


@dataclass(slots=True)
class Persona:
    id: str
    name: str
    traits: TraitProfile
    sim_params: SimulatorParameters
    background: str
    private_goal: str
    preferred_options: list[str]
    rejection: str | None = None
    rejection_reason: str = ""
    agenda: list[AgendaItem] = field(default_factory=list)

    @property
    def preferred_option(self) -> str:
        return self.preferred_options[0]


@dataclass(slots=True)
class MoveIntent:
    speaker_id: str
    act: ActType
    reason: str
    addressee_id: str | None = None
    option_focus: list[str] = field(default_factory=list)
    length_hint: LengthHint = "medium"
    respond_to_turn: int | None = None
    agenda_index: int | None = None
    suppress_name_prefix: bool = False
    suppress_option_opening: bool = False
    suppress_i_opening: bool = False
    suppress_we_opening: bool = False
    vary_opening: bool = False
    avoid_pattern: str | None = None
    avoid_phrases: list[str] = field(default_factory=list)
    allow_vote_change: bool = False


@dataclass(slots=True)
class DialogueAct:
    speaker_id: str
    text: str
    act_type: ActType
    option_refs: list[str] = field(default_factory=list)
    addressee_id: str | None = None
    question_target_id: str | None = None
    explicit_vote: str | None = None
    accepts: list[str] = field(default_factory=list)
    soft_rejects: dict[str, str] = field(default_factory=dict)
    hard_rejects: dict[str, str] = field(default_factory=dict)
    proposes_option: str | None = None
    resolves_blocker: str | None = None    # option whose earlier blocker this line resolves
    conditional_support: str | None = None  # option supported only conditionally
    offers_compromise: str | None = None    # option visibly proposed as common ground


@dataclass(slots=True)
class OpenQuestion:
    turn_id: int
    asked_by: str
    target_id: str
    text: str
    option_focus: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResponseObligation:
    """A pending duty for one participant to respond to a direct address.

    Created when the moderator or another participant directs a question at a
    named participant. The router consumes it before normal speaker selection so
    the addressed participant answers within the next turn or two.
    """

    target_id: str
    source_id: str            # "moderator" or a persona id
    question_text: str
    expected_act: ActType
    created_turn: int
    expires_after: int        # turn_index after which the obligation lapses
    option_focus: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OptionCoverage:
    mentions: int = 0
    reasons: int = 0
    objections: int = 0
    acceptances: int = 0
    coverage_attempts: int = 0  # times the controller routed a turn to cover this option


@dataclass(slots=True)
class ParticipantRuntime:
    persona_id: str
    turn_count: int = 0
    last_spoke_turn: int | None = None
    current_preference: str | None = None  # latent simulator preference, not public evidence
    explicit_vote: str | None = None       # observed public commitment from visible text
    vote_stance: str | None = None         # how the vote was stated: "vote" (direct) or "accept"
    accepted_options: set[str] = field(default_factory=set)
    soft_rejections: dict[str, str] = field(default_factory=dict)
    hard_rejections: dict[str, str] = field(default_factory=dict)
    already_said: list[str] = field(default_factory=list)
    # Visible vote movements: {"from": old-or-initial, "to": new, "has_reason": bool}.
    switch_events: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class TurnRecord:
    index: int
    speaker_id: str
    speaker_name: str
    text: str
    phase: Phase
    act: DialogueAct
    intent: MoveIntent | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    validation_issues: list[str] = field(default_factory=list)
    repaired: bool = False
    repair_trigger_codes: list[str] = field(default_factory=list)
    state_mutation_blocked: bool = False
    used_fallback: bool = False


@dataclass(slots=True)
class RunOutcome:
    status: str
    final_option: str | None
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
    response_obligation: "ResponseObligation | None" = None
    unanswered_obligations: int = 0
    candidate_option: str | None = None
    compromise_attempted: bool = False
    minority_check_attempted: bool = False
    outcome: RunOutcome | None = None
    turn_index: int = 0
    no_progress_count: int = 0
    fallback_turn_count: int = 0
    invalid_printed_turn_count: int = 0
    phase_history: list[str] = field(default_factory=list)
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

    def name_for(self, persona_id: str | None) -> str:
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
