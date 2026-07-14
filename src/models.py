"""Domain models for the autonomous user-simulator runtime.

The structured :class:`UserAction` is the semantic authority.  Accepted natural
language is a rendering of that action; dialogue state is never reconstructed by
parsing the utterance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
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
    VOTE = "vote"


class IssueKind(str, Enum):
    QUESTION = "question"
    CONCERN = "concern"
    COMPARISON = "comparison"


class IssueStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    STALE = "stale"


class IssueEffect(str, Enum):
    OPEN = "open"
    CONTINUE = "continue"
    ANSWERED = "answered"
    PARTIAL = "partial"
    RESOLVE = "resolve"
    MAINTAIN = "maintain"


class QuestionIntent(str, Enum):
    RATIONALE = "rationale"
    IMPACT = "impact"
    ACCEPTABILITY = "acceptability"
    COMPARISON = "comparison"
    CLARIFICATION = "clarification"


class IssueResponseKind(str, Enum):
    MITIGATION = "mitigation"
    TRADE_OFF = "trade_off"
    AGREEMENT = "agreement"


class StimulusKind(str, Enum):
    COVERAGE = "coverage"
    STALL = "stall"


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

    def prompt_card(self) -> str:
        attrs = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in self.attrs.items())
        pieces = [f"{self.id}) {self.name}", f"attrs: {attrs}" if attrs else "attrs: none"]
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
    """Direct, user-facing simulator traits.

    The scale is integer 1..5 (low..high).  Normal stubbornness is restricted to
    1..4; value 5 is reserved for explicit hard blockers.
    """

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
        expected_high = 5 if hard_blocker else 4
        if hard_blocker:
            self.stubbornness = 5
        elif not 1 <= self.stubbornness <= expected_high:
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
    urgency: float
    act: ActionType
    option_focus: tuple[str, ...] = ()
    addressee_id: str | None = None
    reason: str = ""
    reason_source: ReasonSource | None = None
    personal_context: str | None = None
    issue_id: str | None = None
    issue_effect: IssueEffect | None = None
    issue_response_kind: IssueResponseKind | None = None
    question_intent: QuestionIntent | None = None
    question_key: str | None = None
    stance_update: StanceUpdate | None = None
    vote_option: str | None = None
    stimulus_id: int | None = None

    def copy(self) -> "UserAction":
        """An explicit value copy useful in tests proving floor non-rewriting."""
        return UserAction(
            speaker_id=self.speaker_id,
            wants_to_speak=self.wants_to_speak,
            urgency=self.urgency,
            act=self.act,
            option_focus=tuple(self.option_focus),
            addressee_id=self.addressee_id,
            reason=self.reason,
            reason_source=self.reason_source,
            personal_context=self.personal_context,
            issue_id=self.issue_id,
            issue_effect=self.issue_effect,
            issue_response_kind=self.issue_response_kind,
            question_intent=self.question_intent,
            question_key=self.question_key,
            stance_update=self.stance_update,
            vote_option=self.vote_option,
            stimulus_id=self.stimulus_id,
        )


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
    answered: bool = False
    outcome: str | None = None
    close_reason: str = ""
    reason_source: ReasonSource | None = None
    issue_key: tuple[str, str] | None = None
    relevant_responder_ids: set[str] = field(default_factory=set)
    relevant_response_kinds: Counter[str] = field(default_factory=Counter)
    same_attribute_mitigation: bool = False
    owner_last_evaluated_follow_up_count: int = 0


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


@dataclass(slots=True)
class SwitchDecision:
    participant_id: str
    phase: Phase
    turn_index: int
    current_option: str
    target_option: str
    target_evidence: float
    current_evidence: float
    evidence_margin: float
    probability: float
    latest_external_evidence_turn: int
    allowed: bool
    rejection_reason: str = ""


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
    stated_reason_keys: set[str] = field(default_factory=set)
    opened_issue_keys: set[str] = field(default_factory=set)
    asked_question_keys: set[str] = field(default_factory=set)
    action_signature_counts: Counter[str] = field(default_factory=Counter)
    action_signature_contexts: dict[str, str] = field(default_factory=dict)
    responded_stimuli: set[int] = field(default_factory=set)
    voluntary_turns: int = 0
    mandatory_answers: int = 0
    openings: int = 0
    votes_cast: int = 0
    last_action: ActionType | None = None
    last_spoken_turn: int = -1
    visible_switches: int = 0
    last_switch_turn: int = -1
    last_switch_target: str | None = None
    last_switch_external_evidence_turn: int = -1
    switch_opportunities: int = 0
    switch_cooldown_rejections: int = 0
    last_switch_probability: float = 0.0
    last_switch_rejection_reason: str = ""

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
    urgency: float = 0.0
    repair_count: int = 0
    issue_event: str | None = None
    stance_update: StanceUpdate | None = None
    vote_option: str | None = None
    narrowing_options: tuple[str, ...] = ()
    prompt_tokens: int = 0
    output_tokens: int = 0
    intended_word_min: int = 0
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
    input_tokens: int = 0
    output_tokens: int = 0
    moderator_turns: int = 0
    voluntary_turns: int = 0


@dataclass(slots=True)
class DialogueState:
    scenario: Scenario
    personas: list[Persona]
    runtimes: dict[str, ParticipantRuntime]
    phase: Phase = Phase.OPENING
    turns: list[TurnRecord] = field(default_factory=list)
    active_issue: ActiveIssue | None = None
    issue_history: list[ActiveIssue] = field(default_factory=list)
    response_obligation: str | None = None
    group_stimulus: GroupStimulus | None = None
    coverage: dict[str, OptionCoverage] = field(default_factory=dict)
    public_supports: Counter[str] = field(default_factory=Counter)
    public_concerns: Counter[str] = field(default_factory=Counter)
    public_comparisons: Counter[tuple[str, ...]] = field(default_factory=Counter)
    public_supporters: dict[str, set[str]] = field(default_factory=dict)
    public_concern_raisers: dict[str, set[str]] = field(default_factory=dict)
    public_comparers: dict[tuple[str, ...], set[str]] = field(default_factory=dict)
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
    no_bid_rounds: int = 0
    recent_novelty: list[bool] = field(default_factory=list)
    generation_attempts: list[GenerationAttempt] = field(default_factory=list)
    validation_failures: Counter[str] = field(default_factory=Counter)
    switch_decisions: list[SwitchDecision] = field(default_factory=list)

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
        """Public state only; deliberately excludes all private persona fields."""
        return {
            "phase": self.phase.value,
            "preferences": {
                pid: runtime.public_preference for pid, runtime in self.runtimes.items()
                if runtime.public_preference is not None
            },
            "acceptances": {
                pid: sorted(runtime.public_acceptances) for pid, runtime in self.runtimes.items()
                if runtime.public_acceptances
            },
            "supports": dict(self.public_supports),
            "concerns": dict(self.public_concerns),
            "distinct_supporters": {
                option_id: sorted(participant_ids)
                for option_id, participant_ids in self.public_supporters.items()
                if participant_ids
            },
            "distinct_concern_raisers": {
                option_id: sorted(participant_ids)
                for option_id, participant_ids in self.public_concern_raisers.items()
                if participant_ids
            },
            "active_issue": self.active_issue,
            "group_stimulus": self.group_stimulus,
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
