"""Typed objects for the option-grounded multi-user simulator.

The project simulates several user simulators in a shared decision environment.
LLMs render utterances, but the environment owns the option board, simulator
parameters, thread state, routing state, visible commitments, and outcome logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# Controller runtime state lives in controller/state.py; re-exported here so
# every module keeps one stable import surface for the typed objects.
from controller.state import (
    BlockingStrength,
    Phase,
    QuestionScope,
    RepairState,
    ThreadState,
    ThreadStatus,
    ThreadType,
)

# Intentional re-exports (stable import surface): the controller enums below
# are unused inside this module but imported from `models` across src/tests.
# Declared so static tools count them as used; nothing star-imports models.
__all__ = ["BlockingStrength", "ThreadStatus", "ThreadType"]


class ActType(str, Enum):
    """Compact dialogue-act vocabulary used by routing, prompting, and logs."""

    OPENING = "opening"
    SUPPORT = "support"
    CONCERN = "concern"
    ASK = "ask"
    ANSWER = "answer"      # route-driven by an active question thread, never sampled
    COMPARE = "compare"
    COMMENT = "comment"    # light contribution: acknowledge/interpret/state a priority
    COMPROMISE = "compromise"  # narrowing/repair prompts only, not sampled
    PROCESS = "process"        # participant-led narrowing/procedure only, not sampled
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
    ActType.COMMENT,
    ActType.COMPROMISE,
    ActType.PROCESS,
}


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

    engagement        -> expected speaker frequency / turn share
    verbosity         -> average utterance length via numeric word budgets
    directness        -> blunt vs soft wording
    stubbornness      -> strength of stance defense during discussion
    switch_resistance -> resistance to final switching, compromise acceptance,
                         holdout concession, and vote/repair movement
    """

    engagement: float
    verbosity: float
    directness: float
    stubbornness: float
    switch_resistance: float = 0.5

    def clipped(self) -> "SimulatorParameters":
        return SimulatorParameters(**{name: _clip01(getattr(self, name)) for name in self.__dataclass_fields__})


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
    # Sampled exclusive hard blocker (setup metadata): exactly one preferred
    # option, every other option hard-rejected (rank 1). Multi-rejection is
    # represented by the rank table itself; the singular `rejection` field
    # remains the manual single-rejection input.
    hard_blocker: bool = False

    @property
    def preferred_option(self) -> str:
        return self.preferred_options[0]


@dataclass(slots=True)
class MoveIntent:
    speaker_id: str
    act: ActType
    reason: str
    # Why the controller selected this turn (trace metadata, no routing effect).
    # Values: opening, answer_required, thread_hot, thread_cooling, coverage,
    # continuation, normal, participant_narrowing, vote, vote_clarification,
    # majority_holdout_repair, split_vote_repair, two_person_deadlock_repair.
    route_source: str = "normal"
    addressee_id: str | None = None
    option_focus: list[str] = field(default_factory=list)
    length_hint: LengthHint = "medium"
    respond_to_turn: int | None = None
    thread_id: str | None = None       # thread that routed this turn (trace metadata)
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
    continuation: bool = False        # same-speaker follow-up turn (issue 6): short addendum/clarification


# ---------------------------------------------------------------------------
# Visible-evidence contract (todo_validation item 3)
#
# The typed multi-label representation of what ONE final visible utterance
# publicly says. Populated by the interpretation pipeline (deterministic
# critical parser + validator LLM), verified against the raw text, and — once
# migration completes — the only semantic evidence allowed to update dialogue
# state. Controller intent is never encoded here: this is visible text only.
# ---------------------------------------------------------------------------

# Controlled vocabularies shared with the validator schema and test fixtures.
SUPPORT_STRENGTHS = ("weak", "conditional", "firm")
CONCERN_SEVERITIES = ("ordinary", "hard")
QUESTION_SCOPES = ("direct", "group", "rhetorical")
ANSWER_COMPLETENESS = ("full", "partial", "evasive", "unrelated")
COMMITMENT_KINDS = ("vote", "accept")
BLOCKER_ACTIONS = ("raised", "resolved")
CLAIM_KINDS = (
    "listed_fact",            # exact option-attribute-value or shared-context fact
    "arithmetic",             # simple reproducible arithmetic over listed values
    "opinion",                # subjective judgment, never a scenario fact
    "inference",              # qualified conclusion drawn from listed facts
    "uncertainty",            # explicit uncertainty or a question about missing info
    "invented_detail",        # concrete detail not present in the scenario
    "cross_option_transfer",  # another option's value applied to this option
    "ungrounded_inference",   # concrete conclusion asserted with no listed support
    "contradiction",          # directly conflicts with a listed option fact
)


@dataclass(slots=True)
class EvidenceSpan:
    """Exact substring of the visible utterance backing one piece of evidence.

    ``start`` is the character offset in the final utterance; -1 means the
    span text was reported without a located offset (verification then only
    checks substring membership).
    """

    text: str
    start: int = -1


@dataclass(slots=True)
class OptionMention:
    option_id: str
    span: EvidenceSpan
    order: int = 0                  # textual order among mentions, 0-based
    alias_form: str = ""            # surface form used ("Museum", "Option B", ...)
    resolution: str = "explicit"    # explicit | context (pronoun/thread resolved)


@dataclass(slots=True)
class SupportEvidence:
    option_id: str
    strength: str                   # SUPPORT_STRENGTHS
    span: EvidenceSpan


@dataclass(slots=True)
class ConcernEvidence:
    option_id: str
    severity: str                   # CONCERN_SEVERITIES ("hard" = rejection-strength)
    span: EvidenceSpan


@dataclass(slots=True)
class ComparisonEvidence:
    option_ids: list[str]
    span: EvidenceSpan
    favored: str | None = None      # only when the text visibly favors one side
    dimension: str | None = None    # only when visible in the text


@dataclass(slots=True)
class QuestionEvidence:
    """Visible question, detected deterministically (scope + addressee)."""

    scope: str                      # QUESTION_SCOPES
    span: EvidenceSpan
    addressee_id: str | None = None  # participant id for direct questions
    option_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnswerEvidence:
    completeness: str               # ANSWER_COMPLETENESS
    span: EvidenceSpan
    addresses_target: bool = True


@dataclass(slots=True)
class SofteningEvidence:
    """Visible stance movement short of a commitment: warming toward an
    option and/or explicitly conceding a point / moving off an earlier stance."""

    span: EvidenceSpan
    option_id: str | None = None    # option warmed toward, when visible
    concession: bool = False        # explicitly yields a prior position


@dataclass(slots=True)
class ProposalEvidence:
    """Visible compromise/common-ground proposal, including question forms."""

    option_id: str | None
    span: EvidenceSpan


@dataclass(slots=True)
class CommitmentEvidence:
    kind: str                       # COMMITMENT_KINDS: vote | accept
    option_id: str
    span: EvidenceSpan


@dataclass(slots=True)
class SwitchEvidence:
    target: str
    span: EvidenceSpan
    source: str | None = None       # None when the old pick is not visible
    reason_span: EvidenceSpan | None = None  # visible reason, when present


@dataclass(slots=True)
class BlockerEvidence:
    option_id: str
    action: str                     # BLOCKER_ACTIONS: raised | resolved
    span: EvidenceSpan


@dataclass(slots=True)
class GroundingClaim:
    """One atomic checkable statement extracted from the utterance."""

    span: EvidenceSpan
    kind: str                        # CLAIM_KINDS
    option_id: str | None = None     # subject option when applicable
    attribute: str | None = None     # attempted attribute for factual claims
    value: str | None = None         # attempted value for factual claims
    source_facts: list[str] = field(default_factory=list)  # e.g. "A.cost=24 euros", "context:0"
    supported: bool | None = None    # deterministic verification outcome (None = not yet checked)
    reason: str = ""                 # why unsupported, when supported is False


@dataclass(slots=True)
class VisibleEvidence:
    """Multi-label visible semantics of one final utterance.

    One utterance may carry several evidence types simultaneously (e.g. a
    concern, a switch commitment with a reason, and a group question); earlier
    evidence never erases later evidence. Every entry binds to its specific
    option(s) — nothing applies globally to all mentioned options.
    """

    utterance: str = ""
    mentions: list[OptionMention] = field(default_factory=list)
    supports: list[SupportEvidence] = field(default_factory=list)
    concerns: list[ConcernEvidence] = field(default_factory=list)
    comparisons: list[ComparisonEvidence] = field(default_factory=list)
    questions: list[QuestionEvidence] = field(default_factory=list)
    answers: list[AnswerEvidence] = field(default_factory=list)
    softenings: list[SofteningEvidence] = field(default_factory=list)
    proposals: list[ProposalEvidence] = field(default_factory=list)
    commitments: list[CommitmentEvidence] = field(default_factory=list)
    switches: list[SwitchEvidence] = field(default_factory=list)
    blockers: list[BlockerEvidence] = field(default_factory=list)
    claims: list[GroundingClaim] = field(default_factory=list)
    # References that could not be resolved to exactly one public referent.
    ambiguous_references: list[EvidenceSpan] = field(default_factory=list)
    # Whether the utterance addressed the routed public thread (None = no thread).
    thread_relevant: bool | None = None
    # Derived primary label for compatibility, logs, and summary metrics only —
    # never the sole semantic authority over the multi-label evidence above.
    primary_act: ActType | None = None

    def sole_commitment(self) -> CommitmentEvidence | None:
        """The single visible commitment, or None when absent or conflicting."""
        targets = {c.option_id for c in self.commitments}
        if len(self.commitments) >= 1 and len(targets) == 1:
            return self.commitments[0]
        return None

    def evidence_kinds(self) -> set[str]:
        kinds: set[str] = set()
        for name, entries in (
            ("support", self.supports),
            ("concern", self.concerns),
            ("comparison", self.comparisons),
            ("question", self.questions),
            ("answer", self.answers),
            ("proposal", self.proposals),
            ("commitment", self.commitments),
            ("switch", self.switches),
        ):
            if entries:
                kinds.add(name)
        # A softening warms toward a specific option; a pure concession (no
        # target option) is movement off a stance without a visible direction.
        if any(s.option_id for s in self.softenings):
            kinds.add("softening")
        if any(b.action == "raised" for b in self.blockers):
            kinds.add("blocker")
        if any(b.action == "resolved" for b in self.blockers):
            kinds.add("blocker_resolution")
        if any(s.concession for s in self.softenings):
            kinds.add("concession")
        return kinds


class AssessmentAction(str, Enum):
    """Explicit action decided by validation (replaces issue-count logic)."""

    ACCEPT = "accept"
    ACCEPT_WITH_METRIC = "accept_with_metric"  # safe deviation, telemetry only
    REPAIR = "repair"
    FALLBACK = "fallback"
    DROP = "drop"


@dataclass(slots=True)
class ValidationIssue:
    code: str
    explanation: str = ""
    span: str = ""                  # offending exact span when applicable
    option_id: str | None = None
    # Blocking issues must never reach the transcript as state evidence; a
    # candidate that keeps one after repair is replaced or dropped. Non-blocking
    # issues are worth one repair but may print if repair cannot improve them.
    blocking: bool = False


@dataclass(slots=True)
class TurnAssessment:
    """Validation outcome for one candidate utterance: what to do and why."""

    action: AssessmentAction = AssessmentAction.ACCEPT
    issues: list[ValidationIssue] = field(default_factory=list)
    intended_act_realized: bool | None = None    # requested function visibly realized
    intended_focus_realized: bool | None = None  # requested option focus visibly present
    notes: str = ""


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

    def top_option(self, *, fallback: str | None = None) -> str | None:
        if not self.option_ranks:
            return fallback
        best_rank = max(self.option_ranks.values())
        candidates = [oid for oid, rank in self.option_ranks.items() if rank == best_rank]
        if fallback in candidates:
            return fallback
        return sorted(candidates)[0] if candidates else fallback

    def acceptable_options(self) -> set[str]:
        top = self.top_option()
        return {
            oid for oid, value in self.option_ranks.items()
            if value >= STANCE_ACCEPTABLE and oid != top
        }

    def disliked_options(self) -> set[str]:
        return {oid for oid, value in self.option_ranks.items() if value == STANCE_DISLIKED}

    def rejected_options(self) -> set[str]:
        return {oid for oid, value in self.option_ranks.items() if value == STANCE_REJECTED}

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
    intent: MoveIntent | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    validation_issues: list[str] = field(default_factory=list)
    repaired: bool = False
    repair_trigger_codes: list[str] = field(default_factory=list)
    state_mutation_blocked: bool = False
    used_fallback: bool = False
    fallback_family: str = ""  # which act-specific fallback produced the text
    # Final accepted visible evidence and its validation assessment: the ONE
    # semantic authority for this turn. Every appended participant turn
    # carries evidence (pipeline, peer procedure, and test fixtures alike);
    # a record without evidence (moderator lines) carries no public semantic
    # signal at all.
    evidence: "VisibleEvidence | None" = None
    assessment: "TurnAssessment | None" = None

    def mentioned_options(self) -> list[str]:
        if self.evidence is None:
            return []
        seen: list[str] = []
        for mention in self.evidence.mentions:
            if mention.option_id not in seen:
                seen.append(mention.option_id)
        return seen

    def visible_vote(self) -> str | None:
        if self.evidence is None:
            return None
        commitment = self.evidence.sole_commitment()
        return commitment.option_id if commitment else None

    def visible_accepts(self) -> list[str]:
        if self.evidence is None:
            return []
        return [c.option_id for c in self.evidence.commitments if c.kind == "accept"]

    def objected_options(self) -> set[str]:
        if self.evidence is None:
            return set()
        return {c.option_id for c in self.evidence.concerns} | {
            b.option_id for b in self.evidence.blockers if b.action == "raised"
        }

    def question_target(self) -> str | None:
        if self.evidence is None:
            return None
        return next(
            (q.addressee_id for q in self.evidence.questions if q.scope == "direct"), None
        )

    def realized_act(self) -> ActType:
        """Display/trace label only — never a state authority. Derived from the
        accepted visible evidence; falls back to the routed intent act for
        records without evidence (e.g. moderator lines)."""
        if self.evidence is not None and self.evidence.primary_act is not None:
            return self.evidence.primary_act
        return self.intent.act if self.intent is not None else ActType.COMMENT


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
    # Thread-based local interaction state: questions, concerns, blockers, and
    # comparisons are thread-driven; there is no content agenda.
    threads: dict[str, ThreadState] = field(default_factory=dict)
    thread_counter: int = 0
    active_repair: "RepairState | None" = None
    candidate_option: str | None = None
    narrowing_returned: bool = False       # the one allowed narrowing -> discussion fallback was used
    top_pair_history: list[tuple] = field(default_factory=list)  # visible top pair per turn (stability trigger)
    # One bounded repair objective at a time (13.7); completed objectives are
    # appended to repair_history (each reason runs at most once per run).
    repair_history: list["RepairState"] = field(default_factory=list)
    reservation_exchanges: int = 0           # holdout/supporter reservation pairs run inside repairs
    procedural_move_count: int = 0           # participant-owned structure beats taken (split summaries/probes)
    outcome: RunOutcome | None = None
    turn_index: int = 0
    no_progress_count: int = 0
    fallback_turn_count: int = 0
    invalid_printed_turn_count: int = 0
    discussion_lean_shifts: int = 0      # latent-lean movements during the discussion phase (issue 3)
    discussion_lean_shift_turns: list[int] = field(default_factory=list)  # source turn per shift (item 14)
    phase_history: list[str] = field(default_factory=list)
    # Immutable per-turn controller trace: why each turn was selected (pre) and
    # what the final accepted text actually realized (result). Entries are plain
    # dicts with copied values; nothing routes from this list.
    controller_trace: list[dict] = field(default_factory=list)
    min_discussion_turns: int = 0
    force_narrow_turns: int = 0
    hard_max_turns: int = 0
    setup_tokens_in: int = 0
    setup_tokens_out: int = 0
    dialogue_tokens_in: int = 0
    dialogue_tokens_out: int = 0
    token_usage_by_call_type: dict[str, dict[str, int]] = field(default_factory=dict)

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
