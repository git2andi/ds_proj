"""Core data models for the autonomous option-grounded simulator.

Structured :class:`UserAction` objects are authoritative. Natural-language
utterances render those actions; only state-changing semantics are checked for
visible realization.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Phase(str, Enum):
    OPENING = "OPENING"
    DISCUSSION = "DISCUSSION"
    NARROWING = "NARROWING"
    VOTING = "VOTING"
    CLOSED = "CLOSED"


class ActionType(str, Enum):
    OPENING = "opening"
    SUPPORT = "support"
    CONCERN = "concern"
    ASK = "ask"
    ANSWER = "answer"
    COMPARE = "compare"
    ACKNOWLEDGE = "acknowledge"
    COMMENT = "comment"
    COMPROMISE = "compromise"
    FINAL_POSITION = "final_position"
    VOTE = "vote"


class BidPriority(IntEnum):
    """Categorical floor priority; no arbitrary urgency weights are used."""

    NORMAL = 1
    ISSUE_RESPONSE = 2
    ISSUE_OWNER_REACTION = 3
    REQUIRED = 4


class IssueKind(str, Enum):
    QUESTION = "question"
    CONCERN = "concern"
    COMPARISON = "comparison"


class IssueStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    STALE = "stale"


class OpeningMode(str, Enum):
    INITIAL = "initial"
    ALIGN = "align"
    CONTRAST = "contrast"


class QuestionMode(str, Enum):
    CHOICE_IMPACT = "choice_impact"
    TRADEOFF = "tradeoff"
    CONDITION = "condition"


class ResponseMode(str, Enum):
    KNOWN_MITIGATION = "known_mitigation"
    ACCEPT_TRADEOFF = "accept_tradeoff"
    MAINTAIN_CONCERN = "maintain_concern"
    UNKNOWN = "unknown"


class IssueEffect(str, Enum):
    OPEN = "open"
    RESPOND = "respond"
    PARTIAL = "partial"
    RESOLVE = "resolve"
    MAINTAIN = "maintain"


class StimulusKind(str, Enum):
    COVERAGE = "coverage"
    STALL = "stall"
    COMPROMISE = "compromise"


class VoteStatus(str, Enum):
    VALID = "valid"
    ABSTAINED = "abstained"
    UNCLEAR = "unclear"
    GENERATION_FAILED = "generation_failed"


class StanceUpdateKind(str, Enum):
    MAKE_ACCEPTABLE = "make_acceptable"
    REMOVE_ACCEPTANCE = "remove_acceptance"
    SWITCH_PREFERRED = "switch_preferred"
    REJECT = "reject"


@dataclass(slots=True)
class OptionCard:
    id: str
    name: str
    attrs: dict[str, str] = field(default_factory=dict)
    upside: str = ""
    concern: str = ""
    short_name: str = ""

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

    def prompt_card(self, *, include_attrs: set[str] | None = None) -> str:
        attrs = self.attrs
        if include_attrs is not None:
            attrs = {key: value for key, value in attrs.items() if key in include_attrs}
        pieces = [f"{self.id}) {self.name}"]
        if attrs:
            pieces.append(", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in attrs.items()))
        if self.upside:
            pieces.append(f"upside: {self.upside}")
        if self.concern:
            pieces.append(f"concern: {self.concern}")
        return "; ".join(pieces)

    def public_values(self) -> list[str]:
        return [*self.attrs.values(), self.upside, self.concern]


@dataclass(slots=True)
class Scenario:
    topic: str
    options: list[OptionCard]
    shared_context: list[str] = field(default_factory=list)
    environment_type: str = "option_grounded_group_decision"
    setup_notes: list[str] = field(default_factory=list)

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
            raise ValueError(f"normal stubbornness must be in [1, 4], got {self.stubbornness}")
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


@dataclass(slots=True, frozen=True)
class StanceUpdate:
    kind: StanceUpdateKind
    option_id: str
    previous_option_id: str | None = None


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
    issue_id: str | None = None
    issue_effect: IssueEffect | None = None
    stance_update: StanceUpdate | None = None
    vote_option: str | None = None
    stimulus_id: int | None = None
    opening_mode: OpeningMode | None = None
    question_mode: QuestionMode | None = None
    response_mode: ResponseMode | None = None
    decisive_reason: str = ""
    condition: str = ""

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
            issue_id=self.issue_id,
            issue_effect=self.issue_effect,
            stance_update=self.stance_update,
            vote_option=self.vote_option,
            stimulus_id=self.stimulus_id,
            opening_mode=self.opening_mode,
            question_mode=self.question_mode,
            response_mode=self.response_mode,
            decisive_reason=self.decisive_reason,
            condition=self.condition,
        )


@dataclass(slots=True)
class IssueRecord:
    key: tuple[str, str]
    status: IssueStatus = IssueStatus.OPEN
    last_issue_id: str | None = None
    last_relevant_turn: int = -1
    last_closed_turn: int = -1
    outcome: str | None = None


@dataclass(slots=True)
class ActiveIssue:
    id: str
    kind: IssueKind
    option_focus: tuple[str, ...]
    opened_by: str
    addressed_to: str | None
    summary: str
    status: IssueStatus
    opened_at_turn: int
    last_relevant_turn: int
    follow_up_count: int = 0
    source_text: str = ""
    outcome: str | None = None
    close_reason: str = ""
    reason_source: ReasonSource | None = None
    issue_key: tuple[str, str] | None = None
    response_count: int = 0
    owner_reacted: bool = False
    question_mode: QuestionMode | None = None


@dataclass(slots=True)
class GroupStimulus:
    id: int
    kind: StimulusKind
    option_focus: tuple[str, ...]
    prompt_text: str
    created_at_turn: int


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
    disliked_options: set[str] = field(default_factory=set)
    hard_rejected_options: set[str] = field(default_factory=set)
    public_preference: str | None = None
    public_acceptances: set[str] = field(default_factory=set)
    public_rejections: set[str] = field(default_factory=set)
    used_reason_keys: set[str] = field(default_factory=set)
    opened_issue_keys: set[str] = field(default_factory=set)
    asked_question_keys: set[str] = field(default_factory=set)
    responded_stimuli: set[int] = field(default_factory=set)
    responded_issue_ids: set[str] = field(default_factory=set)
    used_compromise_options: set[str] = field(default_factory=set)
    voluntary_turns: int = 0
    mandatory_answers: int = 0
    openings: int = 0
    votes_cast: int = 0
    last_action: ActionType | None = None
    last_spoken_turn: int = -1
    visible_switches: int = 0
    last_switch_turn: int = -1

    def rank(self, option_id: str) -> int:
        return int(self.ranks.get(option_id, STANCE_NEUTRAL))


@dataclass(slots=True)
class OptionCoverage:
    substantive_count: int = 0
    participant_ids: set[str] = field(default_factory=set)
    action_types: Counter[str] = field(default_factory=Counter)

    def add(self, speaker_id: str, act: ActionType) -> None:
        self.substantive_count += 1
        self.participant_ids.add(speaker_id)
        self.action_types[act.value] += 1


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
    issue_event: str | None = None
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
    liveness_forced_turns: int = 0
    suppressed_repetitions: int = 0
    suppressed_duplicate_candidates: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    moderator_turns: int = 0
    voluntary_turns: int = 0
    compromise_proposals: int = 0
    compromise_acceptances: int = 0
    narrowing_movements: int = 0
    revote_skipped_no_movement: int = 0
    semantic_reason_reuse: int = 0
    vote_fallbacks: int = 0
    mandatory_movement_failures: int = 0


@dataclass(slots=True)
class DialogueState:
    scenario: Scenario
    personas: list[Persona]
    runtimes: dict[str, ParticipantRuntime]
    phase: Phase = Phase.OPENING
    turns: list[TurnRecord] = field(default_factory=list)
    active_issue: ActiveIssue | None = None
    issue_history: list[ActiveIssue] = field(default_factory=list)
    issue_records: dict[tuple[str, str], IssueRecord] = field(default_factory=dict)
    response_obligation: str | None = None
    group_stimulus: GroupStimulus | None = None
    coverage: dict[str, OptionCoverage] = field(default_factory=dict)
    public_supports: Counter[str] = field(default_factory=Counter)
    public_concerns: Counter[str] = field(default_factory=Counter)
    public_comparisons: Counter[tuple[str, ...]] = field(default_factory=Counter)
    public_supporters: dict[str, set[str]] = field(default_factory=dict)
    public_concern_raisers: dict[str, set[str]] = field(default_factory=dict)
    votes: dict[str, str | None] = field(default_factory=dict)
    first_round_votes: dict[str, str | None] = field(default_factory=dict)
    vote_records: dict[int, dict[str, VoteRecord]] = field(default_factory=dict)
    vote_round: int = 0
    vote_protocol_degraded: bool = False
    vote_protocol_errors: list[str] = field(default_factory=list)
    narrowing_options: tuple[str, ...] = ()
    phase_history: list[str] = field(default_factory=lambda: [Phase.OPENING.value])
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    stall_prompt_used: bool = False
    coverage_prompt_used: bool = False
    coverage_no_interest: set[str] = field(default_factory=set)
    no_bid_rounds: int = 0
    compromise_opportunity: bool = False
    compromise_prompt_used: bool = False
    movement_events: int = 0
    revote_skipped_no_movement: bool = False
    generation_attempts: list[GenerationAttempt] = field(default_factory=list)
    validation_failures: Counter[str] = field(default_factory=Counter)

    @property
    def participant_turns(self) -> list[TurnRecord]:
        return [turn for turn in self.turns if not turn.moderator]

    @property
    def voluntary_turn_count(self) -> int:
        return sum(1 for turn in self.turns if turn.voluntary)

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

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "preferences": {
                pid: runtime.public_preference
                for pid, runtime in self.runtimes.items()
                if runtime.public_preference is not None
            },
            "acceptances": {
                pid: sorted(runtime.public_acceptances)
                for pid, runtime in self.runtimes.items()
                if runtime.public_acceptances
            },
            "supports": dict(self.public_supports),
            "concerns": dict(self.public_concerns),
            "active_issue": self.active_issue,
            "narrowing_options": self.narrowing_options,
            "votes": dict(self.votes),
        }


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
