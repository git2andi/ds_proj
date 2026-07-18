"""Core data models for the option-grounded user-simulator runtime.

The simulator owns structured actions. The language model only realizes the
selected action as text. Public preferences and thread state are updated only
from accepted, visibly grounded turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class Phase(str, Enum):
    OPENING = "OPENING"
    DISCUSSION = "DISCUSSION"
    NARROWING = "NARROWING"
    VOTING = "VOTING"
    CLOSED = "CLOSED"


class ActionType(str, Enum):
    OPENING = "opening"
    REACT = "react"
    SUPPORT = "support"
    OBJECT = "object"
    COMPARE = "compare"
    ASK = "ask"
    ANSWER = "answer"
    ACCEPT = "accept"
    VOTE = "vote"


class BidPriority(IntEnum):
    NORMAL = 1
    THREAD = 2
    REQUIRED = 3


class ThreadKind(str, Enum):
    QUESTION = "question"
    CONCERN = "concern"


class VoteStatus(str, Enum):
    VALID = "valid"
    ABSTAINED = "abstained"
    UNCLEAR = "unclear"
    GENERATION_FAILED = "generation_failed"


class StanceUpdateKind(str, Enum):
    MAKE_ACCEPTABLE = "make_acceptable"
    SWITCH_PREFERRED = "switch_preferred"


@dataclass(slots=True)
class OptionCard:
    id: str
    name: str
    attrs: dict[str, str] = field(default_factory=dict)
    upside: str = ""
    concern: str = ""
    short_name: str = ""
    aliases: tuple[str, ...] = ()

    def public_line(self) -> str:
        attrs = "; ".join(
            f"{key.replace('_', ' ')}: {value}" for key, value in self.attrs.items()
        )
        extras: list[str] = []
        if self.upside:
            extras.append(f"+ {self.upside.strip().rstrip(' .;:')}")
        if self.concern:
            extras.append(f"− {self.concern.strip().rstrip(' .;:')}")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        return f"{self.id}) {self.name}" + (f" — {attrs}" if attrs else "") + suffix

    def public_values(self) -> list[str]:
        return [*self.attrs.values(), self.upside, self.concern]


@dataclass(slots=True)
class Scenario:
    topic: str
    options: list[OptionCard]
    shared_context: list[str] = field(default_factory=list)
    setup_notes: list[str] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return " ".join(part.strip() for part in self.shared_context if part.strip())

    @property
    def option_ids(self) -> list[str]:
        return [option.id for option in self.options]

    def option(self, option_id: str) -> OptionCard:
        for option in self.options:
            if option.id == option_id:
                return option
        raise KeyError(option_id)


@dataclass(slots=True)
class SimulatorParameters:
    engagement: int
    verbosity: int
    directness: int
    stubbornness: int

    def validated(self, *, hard_blocker: bool = False) -> "SimulatorParameters":
        for name in ("engagement", "verbosity", "directness"):
            value = int(getattr(self, name))
            if not 1 <= value <= 5:
                raise ValueError(f"{name} must be in [1, 5], got {value}")
            setattr(self, name, value)
        self.stubbornness = int(self.stubbornness)
        if hard_blocker:
            self.stubbornness = 5
        elif not 1 <= self.stubbornness <= 4:
            raise ValueError(
                f"normal stubbornness must be in [1, 4], got {self.stubbornness}"
            )
        return self


STANCE_REJECTED = 1
STANCE_DISLIKED = 2
STANCE_NEUTRAL = 3
STANCE_ACCEPTABLE = 4
STANCE_PREFERRED = 5


@dataclass(slots=True)
class OptionStance:
    option_id: str
    rank: int = STANCE_NEUTRAL
    reason_for: str = ""
    reason_against: str = ""

    def clipped(self) -> "OptionStance":
        return OptionStance(
            option_id=self.option_id,
            rank=max(STANCE_REJECTED, min(STANCE_PREFERRED, int(self.rank))),
            reason_for=self.reason_for.strip(),
            reason_against=self.reason_against.strip(),
        )


@dataclass(slots=True)
class Persona:
    id: str
    name: str
    sim_params: SimulatorParameters
    background: str
    private_goal: str
    preferred_options: list[str]
    age: int
    speech_style: str
    style_tendencies: tuple[str, ...] = ()
    rejection: str | None = None
    rejection_reason: str = ""
    option_stances: dict[str, OptionStance] = field(default_factory=dict)
    hard_blocker: bool = False

    @property
    def preferred_option(self) -> str:
        return self.preferred_options[0]


@dataclass(slots=True, frozen=True)
class ReasonSource:
    option_id: str
    attribute_name: str
    public_value: str

    @property
    def point_key(self) -> tuple[str, str]:
        return (self.option_id, self.attribute_name.strip().lower())


@dataclass(slots=True, frozen=True)
class StanceUpdate:
    kind: StanceUpdateKind
    option_id: str
    previous_option_id: str | None = None
    movement_reason: str = ""


@dataclass(slots=True)
class UserAction:
    speaker_id: str
    wants_to_speak: bool
    priority: BidPriority
    act: ActionType
    option_focus: tuple[str, ...] = ()
    addressee_id: str | None = None
    reason: str = ""
    reason_source: ReasonSource | None = None
    personal_context: str | None = None
    stance_update: StanceUpdate | None = None
    vote_option: str | None = None

    @property
    def point_key(self) -> tuple[str, str] | None:
        return self.reason_source.point_key if self.reason_source else None

    def copy(self) -> "UserAction":
        return UserAction(
            speaker_id=self.speaker_id,
            wants_to_speak=self.wants_to_speak,
            priority=self.priority,
            act=self.act,
            option_focus=tuple(self.option_focus),
            addressee_id=self.addressee_id,
            reason=self.reason,
            reason_source=self.reason_source,
            personal_context=self.personal_context,
            stance_update=self.stance_update,
            vote_option=self.vote_option,
        )


@dataclass(slots=True)
class DiscussionThread:
    id: str
    kind: ThreadKind
    opened_by: str
    option_focus: tuple[str, ...]
    point_key: tuple[str, str] | None
    source_text: str
    addressed_to: str | None = None
    turn_count: int = 1
    participants: set[str] = field(default_factory=set)
    required_answer_pending: bool = False


@dataclass(slots=True)
class VoteRecord:
    participant_id: str
    round: int
    status: VoteStatus
    option_id: str | None = None
    attempts: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GenerationAttempt:
    speaker_id: str
    phase: Phase
    action: UserAction
    raw_text: str
    validation_errors: list[str]
    repair_text: str | None = None
    repair_errors: list[str] = field(default_factory=list)
    final_status: str = "pending"
    fallback_text: str | None = None


@dataclass(slots=True)
class ParticipantRuntime:
    persona_id: str
    preferred_option: str
    ranks: dict[str, int]
    acceptable_options: set[str] = field(default_factory=set)
    hard_rejected_options: set[str] = field(default_factory=set)
    public_preference: str | None = None
    public_acceptances: set[str] = field(default_factory=set)
    acceptance_reasons: dict[str, str] = field(default_factory=dict)
    used_point_keys: set[tuple[str, str]] = field(default_factory=set)
    opened_thread_keys: set[tuple[str, str]] = field(default_factory=set)
    voluntary_turns: int = 0
    openings: int = 0
    visible_switches: int = 0

    def rank(self, option_id: str) -> int:
        return int(self.ranks.get(option_id, STANCE_NEUTRAL))


@dataclass(slots=True)
class TurnRecord:
    index: int
    phase: Phase
    speaker_id: str
    speaker_name: str
    text: str
    action: UserAction | None = None
    moderator: bool = False
    mandatory: bool = False
    voluntary: bool = False
    liveness_forced: bool = False
    priority: BidPriority = BidPriority.NORMAL
    repair_count: int = 0
    thread_event: str | None = None
    stance_update: StanceUpdate | None = None
    vote_option: str | None = None
    narrowing_options: tuple[str, ...] = ()
    prompt_tokens: int = 0
    output_tokens: int = 0
    intended_word_max: int = 0

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(slots=True)
class RuntimeStats:
    llm_calls: int = 0
    setup_llm_calls: int = 0
    repair_calls: int = 0
    dropped_turns: int = 0
    fallback_turns: int = 0
    liveness_forced_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    moderator_turns: int = 0
    voluntary_turns: int = 0
    visible_movements: int = 0
    response_failures: int = 0
    compromise_attempts: int = 0
    compromise_acceptances: int = 0


@dataclass(slots=True)
class DialogueState:
    scenario: Scenario
    personas: list[Persona]
    runtimes: dict[str, ParticipantRuntime]
    phase: Phase = Phase.OPENING
    turns: list[TurnRecord] = field(default_factory=list)
    active_thread: DiscussionThread | None = None
    closed_thread_keys: set[tuple[str, str]] = field(default_factory=set)
    public_point_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    recent_point_keys: list[tuple[str, str]] = field(default_factory=list)
    response_obligation: str | None = None
    votes: dict[str, str | None] = field(default_factory=dict)
    vote_records: dict[int, dict[str, VoteRecord]] = field(default_factory=dict)
    vote_round: int = 0
    protocol_errors: list[str] = field(default_factory=list)
    narrowing_options: tuple[str, ...] = ()
    phase_history: list[str] = field(default_factory=lambda: [Phase.OPENING.value])
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    no_bid_rounds: int = 0
    movement_events: int = 0
    generation_attempts: list[GenerationAttempt] = field(default_factory=list)
    validation_failures: dict[str, int] = field(default_factory=dict)

    @property
    def participant_turns(self) -> list[TurnRecord]:
        return [turn for turn in self.turns if not turn.moderator]

    @property
    def last_participant_id(self) -> str | None:
        for turn in reversed(self.turns):
            if not turn.moderator:
                return turn.speaker_id
        return None

    def consecutive_turns_by(self, participant_id: str) -> int:
        count = 0
        for turn in reversed(self.turns):
            if turn.moderator:
                continue
            if turn.speaker_id != participant_id:
                break
            count += 1
        return count

    def persona(self, participant_id: str) -> Persona:
        for persona in self.personas:
            if persona.id == participant_id:
                return persona
        raise KeyError(participant_id)


@dataclass(slots=True)
class RunOutcome:
    status: str
    final_option: str | None
    votes: dict[str, str | None]
    reason: str


@dataclass(slots=True)
class DialogueRunResult:
    state: DialogueState
    outcome: RunOutcome
    log_paths: dict[str, str]
    token_summary: dict[str, int]
