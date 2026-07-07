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
    SOFTEN = "soften"
    CALL_VOTE = "call_vote"
    SUMMARIZE_SPLIT = "summarize_split"
    PROBE_HOLDOUT = "probe_holdout"
    SUGGEST_NARROWING = "suggest_narrowing"
    POST_RESERVATION_DECISION = "post_reservation_decision"
    VOTE = "vote"
    ACCEPT = "accept"
    REJECT = "reject"
    REACT = "react"


# Act groupings shared across the routing/observation/validation modules
# (defined here to avoid a shared-constant import cycle between them, issue 8).
_DECISION_ACTS = {ActType.VOTE, ActType.ACCEPT, ActType.REJECT, ActType.POST_RESERVATION_DECISION}
_DISCUSSION_ACTS = {
    ActType.BUILD,
    ActType.AGREE,
    ActType.CHALLENGE,
    ActType.ASK,
    ActType.ANSWER,
    ActType.COMPARE,
    ActType.INVITE,
    ActType.PROPOSE_COMPROMISE,
    ActType.SOFTEN,
    ActType.CALL_VOTE,
    ActType.SUMMARIZE_SPLIT,
    ActType.PROBE_HOLDOUT,
    ActType.SUGGEST_NARROWING,
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
    # Derived social tone (P8): high = warm/encouraging, low = dry/blunt-toned.
    # Never hostility — low friendliness stays cooperative, just unsoftened.
    friendliness: float = 0.5

    def clipped(self) -> "SimulatorParameters":
        return SimulatorParameters(**{name: _clip01(getattr(self, name)) for name in self.__dataclass_fields__})


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class AgendaItem:
    """One pending private communicative goal.

    Part of a weak hint list consulted only in quiet moments — not an
    agenda-based simulation mechanism (see simulator.build_initial_agenda).
    """

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
    # 1-2 compact personal anchors (P7): small trait-consistent reasons a person
    # gives for preferences ("budget-sensitive", "prefers calm settings"). Used
    # at most about once per discussion; never scenario facts.
    anchors: list[str] = field(default_factory=list)

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
    suppress_tail_question: bool = False  # enough questions are open; end on a statement (P2)
    vary_opening: bool = False
    avoid_pattern: str | None = None
    avoid_phrases: list[str] = field(default_factory=list)
    avoid_reasons: list[str] = field(default_factory=list)  # justification snippets already used this round
    allow_vote_change: bool = False
    soften_toward: str | None = None  # routed softening beat's attractor (issue 3)
    continuation: bool = False        # same-speaker follow-up turn (issue 6): short addendum/clarification
    # Compact trait-derived delivery label (P2): rendered as one short prompt
    # line so traits shape phrasing mid-discussion, not only at vote time.
    # One of: challenge_directly | soften_and_bridge | restate_concern | bridge_condition.
    trait_color: str | None = None
    # Personal anchor offered to this turn's prompt (P7); at most once per sim per run.
    anchor: str | None = None


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
    deferred: bool = False    # a low-responsiveness sim already sat out one beat


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
    # Stateful stance tracking (issue 2): how firmly the sim holds its current
    # favorite. Eroded by challenges against it and by visible group support for
    # rivals (scaled by stubbornness); rebuilt a little by defending it. Low
    # commitment makes softening/switching easier, during discussion and at votes.
    commitment_strength: float = 0.6
    commitment_min: float = 1.0            # lowest commitment reached during the run (tuning telemetry)
    challenges_received: int = 0           # visible challenges landed on this sim's favorite
    concessions_made: int = 0              # times this sim visibly conceded a point/switched
    explicit_vote: str | None = None       # observed public commitment from visible text
    vote_stance: str | None = None         # how the vote was stated: "vote" (direct) or "accept"
    accepted_options: set[str] = field(default_factory=set)
    # Options this sim visibly raised a concern/challenge about (persistent for
    # the whole run, unlike the short-lived open_concerns threads). Feeds the
    # stubborn restate-concern route (P2).
    concerns_raised: dict[str, str] = field(default_factory=dict)
    soft_rejections: dict[str, str] = field(default_factory=dict)
    hard_rejections: dict[str, str] = field(default_factory=dict)
    already_said: list[str] = field(default_factory=list)
    # Visible vote movements:
    # {"from": old-or-initial, "to": new, "has_reason": bool, "has_bridge": bool}.
    # has_bridge is the issue-5 signal: the switch line links the old stance to
    # the new pick with a reason, not just any loose reason clause.
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
    procedural_move_count: int = 0           # participant-owned structure beats taken (issue 5)
    peer_vote_call_done: bool = False        # a participant already called for final picks (issue 5)
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
    restated_concerns: set[str] = field(default_factory=set)  # sims that already got their stubborn restate beat (P2)
    micro_reaction_count: int = 0        # deterministic tiny reaction beats emitted (P4)
    anchors_used: set[str] = field(default_factory=set)  # sims whose personal anchor was already offered to a prompt (P7)
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
