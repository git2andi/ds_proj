"""Typed objects for the option-grounded multi-user simulator.

The project simulates several user simulators in a shared decision environment.
LLMs render utterances, but the environment owns the option board, simulator
parameters, global agenda items, routing state, visible commitments, and outcome logic.
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
    """Compact dialogue-act vocabulary used by routing, prompting, and logs."""

    OPENING = "opening"
    SUPPORT = "support"
    CONCERN = "concern"
    ASK = "ask"
    ANSWER = "answer"
    COMPARE = "compare"
    SOFTEN_TOWARD = "soften_toward"
    COMPROMISE = "compromise"
    PROCESS = "process"
    VOTE = "vote"
    CLOSING = "closing"


# Act groupings shared across the routing/observation/validation modules.
_DECISION_ACTS = {ActType.VOTE}
_DISCUSSION_ACTS = {
    ActType.SUPPORT,
    ActType.CONCERN,
    ActType.ASK,
    ActType.ANSWER,
    ActType.COMPARE,
    ActType.SOFTEN_TOWARD,
    ActType.COMPROMISE,
    ActType.PROCESS,
}


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
    concern: str = ""
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
        if self.concern:
            suffix_bits.append(f"− {_clip_words(self.concern, note_words)}")
        suffix = f" ({'; '.join(suffix_bits)})" if suffix_bits else ""
        return f"{self.id}) {self.name}" + (f" — {details}" if details else "") + suffix

    def prompt_card(self) -> str:
        attrs = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in ordered_attrs(self.attrs))
        pieces = [
            f"{self.id}) {self.name}",
            f"attrs: {attrs}" if attrs else "attrs: none",
            f"upside: {self.upside}" if self.upside else "",
            f"concern: {self.concern}" if self.concern else "",
        ]
        return "; ".join(p for p in pieces if p)


def _clip_words(text: str, limit: int) -> str:
    words = str(text).split()
    if len(words) <= limit:
        return str(text).strip().rstrip(" .;:")
    return " ".join(words[:limit]).rstrip(" ,.;:") + "…"


def ordered_attrs(attrs: dict[str, str]) -> list[tuple[str, str]]:
    """Attributes in the order the setup provided them.

    No hard-coded dimension priority: the setup chooses attributes that are
    natural for the topic, and its own ordering is kept for display.
    """
    return list(attrs.items())


@dataclass(slots=True)
class Scenario:
    topic: str
    options: list[OptionCard]
    shared_context: list[str] = field(default_factory=list)
    environment_type: str = "option_grounded_group_decision"
    setup_notes: list[str] = field(default_factory=list)  # deterministic setup repairs (e.g. cap clamps)

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


@dataclass(slots=True)
class SimulatorParameters:
    """Operational controls for a simulated user.

    OCEAN traits are hidden setup traits: they only derive these four explicit
    parameters (and plausible persona content). Routing and prompts read the
    parameters, never the traits.

    engagement    -> expected speaker frequency / turn share
    verbosity     -> average utterance length via numeric word budgets
    directness    -> blunt vs soft wording
    stubbornness  -> resistance to changing stance, strength of stance defense
    """

    engagement: float
    verbosity: float
    directness: float
    stubbornness: float

    def clipped(self) -> "SimulatorParameters":
        return SimulatorParameters(**{name: _clip01(getattr(self, name)) for name in self.__dataclass_fields__})


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class DiscussionAgendaItem:
    """One chat-level item the controller should cover before narrowing.

    This is a global discussion checklist, not a per-simulator script. It keeps
    required decision work explicit while persona-specific reasons stay inside
    OptionStance.reason_for / reason_against.
    """

    key: str
    act: ActType
    reason: str
    option: str | None = None
    required: bool = True
    status: AgendaStatus = AgendaStatus.PENDING


STANCE_REJECTED = 1
STANCE_DISLIKED = 2
STANCE_NEUTRAL = 3
STANCE_ACCEPTABLE = 4
STANCE_PREFERRED = 5


@dataclass(slots=True)
class OptionStance:
    """One simulator's stance toward one option.

    Rank is the single source of truth for preference/rejection buckets:
    5 preferred, 4 acceptable, 3 neutral, 2 disliked, 1 rejected/hard-blocked.
    Reasons are deliberately short setup hints; they are not hidden transcript
    facts, only controller guidance for plausible moves.
    """

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
    traits: TraitProfile
    sim_params: SimulatorParameters
    background: str
    private_goal: str
    preferred_options: list[str]
    age: int
    speech_style: str
    rejection: str | None = None
    rejection_reason: str = ""
    option_stances: dict[str, OptionStance] = field(default_factory=dict)

    @property
    def preferred_option(self) -> str:
        return self.preferred_options[0]

    @property
    def agenda(self) -> list[object]:
        """Compatibility shim for older run/eval code paths.

        Per-person scripted agendas were removed in favor of the chat-level
        DialogueState.discussion_agenda checklist. Returning an empty list keeps
        stale readers from crashing while ensuring no per-sim agenda can steer
        the dialogue.
        """
        return []


@dataclass(slots=True)
class MoveIntent:
    speaker_id: str
    act: ActType
    reason: str
    addressee_id: str | None = None
    option_focus: list[str] = field(default_factory=list)
    length_hint: LengthHint = "medium"
    respond_to_turn: int | None = None
    agenda_key: str | None = None
    suppress_name_prefix: bool = False
    suppress_option_opening: bool = False
    suppress_i_opening: bool = False
    suppress_we_opening: bool = False
    suppress_tail_question: bool = False  # enough questions are open; end on a statement (P2)
    vary_opening: bool = False
    avoid_pattern: str | None = None
    avoid_phrases: list[str] = field(default_factory=list)
    avoid_reasons: list[str] = field(default_factory=list)  # justification snippets already used this round
    allow_vote_change: bool = False
    required_vote: str | None = None   # controller-selected decision target; validation blocks drift
    old_preference: str | None = None  # controller-visible previous pick for sanctioned switches
    allowed_reason: str | None = None  # grounded reason fragment the LLM may use for a vote/switch
    soften_toward: str | None = None  # routed softening beat's attractor (issue 3)
    continuation: bool = False        # same-speaker follow-up turn (issue 6): short addendum/clarification


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
    softens_toward: str | None = None       # option the line visibly warms to without committing (issue 3)


@dataclass(slots=True)
class Concern:
    """A visible objection about an option, kept alive as a short thread (issue 2).

    The router tries to get a reaction from an advocate of the option within a
    turn or two, so the discussion does not jump away from a raised concern.
    A thread expires unaddressed after a few participant turns.
    """

    turn_id: int
    raised_by: str
    option_id: str
    text: str
    addressed_by: str | None = None
    age: int = 0                # participant turns since the concern was raised


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
    # Stance ranks are the runtime source of truth.
    # 5 preferred, 4 acceptable, 3 neutral, 2 disliked, 1 rejected.
    option_ranks: dict[str, int] = field(default_factory=dict)
    reasons_for: dict[str, str] = field(default_factory=dict)
    reasons_against: dict[str, str] = field(default_factory=dict)
    commitment_strength: float = 0.6
    commitment_min: float = 1.0
    challenges_received: int = 0
    concessions_made: int = 0
    explicit_vote: str | None = None
    vote_stance: str | None = None
    already_said: list[str] = field(default_factory=list)
    switch_events: list[dict] = field(default_factory=list)

    def rank(self, option_id: str | None) -> int:
        if not option_id:
            return STANCE_NEUTRAL
        return int(self.option_ranks.get(option_id, STANCE_NEUTRAL))

    def reason_for(self, option_id: str | None) -> str:
        return self.reasons_for.get(option_id or "", "")

    def reason_against(self, option_id: str | None) -> str:
        return self.reasons_against.get(option_id or "", "")

    def set_rank(self, option_id: str, rank: int, *, reason_for: str = "", reason_against: str = "") -> None:
        self.option_ranks[option_id] = max(STANCE_REJECTED, min(STANCE_PREFERRED, int(rank)))
        if reason_for:
            self.reasons_for[option_id] = reason_for.strip()
        if reason_against:
            self.reasons_against[option_id] = reason_against.strip()

    def adjust_rank(self, option_id: str, delta: int, *, reason_for: str = "", reason_against: str = "") -> None:
        self.set_rank(option_id, self.rank(option_id) + int(delta), reason_for=reason_for, reason_against=reason_against)

    def top_option(self, *, fallback: str | None = None) -> str | None:
        if not self.option_ranks:
            return fallback
        best_rank = max(self.option_ranks.values())
        candidates = [oid for oid, rank in self.option_ranks.items() if rank == best_rank]
        if fallback in candidates:
            return fallback
        return sorted(candidates)[0] if candidates else fallback

    def options_at_rank(self, rank: int) -> set[str]:
        return {oid for oid, value in self.option_ranks.items() if value == rank}

    def acceptable_options(self) -> set[str]:
        top = self.top_option()
        return {
            oid for oid, value in self.option_ranks.items()
            if value >= STANCE_ACCEPTABLE and oid != top
        }

    def liked_options(self) -> set[str]:
        return {oid for oid, value in self.option_ranks.items() if value >= STANCE_ACCEPTABLE}

    def disliked_options(self) -> set[str]:
        return {oid for oid, value in self.option_ranks.items() if value == STANCE_DISLIKED}

    def rejected_options(self) -> set[str]:
        return {oid for oid, value in self.option_ranks.items() if value == STANCE_REJECTED}

    def is_acceptable(self, option_id: str | None) -> bool:
        return self.rank(option_id) >= STANCE_ACCEPTABLE

    def is_disliked(self, option_id: str | None) -> bool:
        return self.rank(option_id) == STANCE_DISLIKED

    def is_rejected(self, option_id: str | None) -> bool:
        return self.rank(option_id) == STANCE_REJECTED

    def promote_to_preferred(self, option_id: str, *, reason_for: str = "") -> None:
        old = self.top_option()
        if old and old != option_id and self.rank(old) >= STANCE_PREFERRED:
            self.option_ranks[old] = STANCE_ACCEPTABLE
        self.set_rank(option_id, STANCE_PREFERRED, reason_for=reason_for)

    def mark_acceptable(self, option_id: str, *, reason_for: str = "") -> None:
        if self.rank(option_id) > STANCE_REJECTED:
            self.set_rank(option_id, max(self.rank(option_id), STANCE_ACCEPTABLE), reason_for=reason_for)

    def mark_disliked(self, option_id: str, *, reason_against: str = "") -> None:
        if self.rank(option_id) > STANCE_REJECTED and self.rank(option_id) < STANCE_PREFERRED:
            self.set_rank(option_id, STANCE_DISLIKED, reason_against=reason_against)

    def mark_rejected(self, option_id: str, *, reason_against: str = "") -> None:
        self.set_rank(option_id, STANCE_REJECTED, reason_against=reason_against)

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
    discussion_agenda: list[DiscussionAgendaItem] = field(default_factory=list)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    open_concerns: list[Concern] = field(default_factory=list)
    concerns_raised_total: int = 0
    concerns_addressed_total: int = 0
    response_obligation: "ResponseObligation | None" = None
    obligations_created: int = 0
    unanswered_obligations: int = 0
    candidate_option: str | None = None
    compromise_attempted: bool = False
    two_person_deadlock_attempted: bool = False
    minority_check_attempted: bool = False
    reservation_exchange_done: bool = False  # the bounded holdout/supporter exchange ran (issue 4)
    split_reservation_exchanges: int = 0     # reservation/supporter pairs during split/tie narrowing
    procedural_move_count: int = 0           # participant-owned structure beats taken (split summaries/probes)
    outcome: RunOutcome | None = None
    turn_index: int = 0
    no_progress_count: int = 0
    fallback_turn_count: int = 0
    invalid_printed_turn_count: int = 0
    blocker_probes: set[str] = field(default_factory=set)  # options whose blocker was already probed
    # Lightweight issue ledger (P7): practical unknowns (parking, booking, …)
    # that were already raised as not-decidable-from-the-board, so the
    # discussion stops reopening them. issue -> {"mentions": int, "options": [ids]}
    issue_ledger: dict[str, dict] = field(default_factory=dict)
    stagnation_break_done: bool = False  # the one bounded circling-rescue beat was used (I20)
    softened_sims: set[str] = field(default_factory=set)  # sims already routed to a visible softening beat (issue 3)
    discussion_lean_shifts: int = 0      # latent-lean movements during the discussion phase (issue 3)
    phase_history: list[str] = field(default_factory=list)
    min_discussion_turns: int = 0
    force_narrow_turns: int = 0
    hard_max_turns: int = 0
    setup_tokens_in: int = 0
    setup_tokens_out: int = 0
    dialogue_tokens_in: int = 0
    dialogue_tokens_out: int = 0
    token_usage_by_call_type: dict[str, dict[str, int]] = field(default_factory=dict)

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
