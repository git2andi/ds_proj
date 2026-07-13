"""Simulator policy: participant-owned behavioral decisions.

This module is the single owner of *what a simulated user wants to do*. It
translates hidden OCEAN traits into the five explicit simulator parameters
(`derive_simulator_parameters`) and — the core of the authority-split design —
lets each simulator decide, from its own persona, private stance, and the
PUBLIC dialogue state, whether it wants to claim an open floor and, if so, which
communicative act/target/option-focus/reason it intends.

Authority split:
- the floor manager (controller/floor.py) arbitrates access to the floor and
  imposes protocol obligations, but never rewrites a simulator's chosen act,
  target, focus, reason, vote, or compromise;
- this module reads only the deciding simulator's own private state plus the
  shared scenario and accepted public conversation state — never another
  simulator's private goal, hidden ranks, or hidden reasons (private-info
  boundary, todo 9/17);
- the LLM later realizes the selected intent as one natural utterance.

All stochastic choices go through the process-global seeded ``random`` (seeded
from ``simulation.random_seed`` at run start), so bids, winners, and the turn
sequence reproduce under a fixed seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from aliases import short_alias_map
from config_loader import cfg
from consensus import public_evidence, public_participant_ledger
from models import (
    ActType,
    DialogueState,
    DiscussionStimulus,
    MoveIntent,
    OptionCoverage,
    ParticipantRuntime,
    Persona,
    Phase,
    PublicParticipantState,
    Scenario,
    SimulatorBid,
    SimulatorParameters,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
    ThreadStatus,
    ThreadType,
    TraitProfile,
    TurnObligation,
    TurnRecord,
)
from utils import weighted_choice


def derive_simulator_parameters(traits: TraitProfile) -> SimulatorParameters:
    open01 = (traits.openness - 1) / 4
    consc01 = (traits.conscientiousness - 1) / 4
    extra01 = (traits.extraversion - 1) / 4
    agree01 = (traits.agreeableness - 1) / 4
    neuro01 = (traits.neuroticism - 1) / 4

    return SimulatorParameters(
        engagement=0.25 + 0.60 * extra01 + 0.15 * consc01,
        verbosity=0.20 + 0.55 * extra01 + 0.25 * open01,
        directness=0.25 + 0.35 * consc01 + 0.25 * extra01 + 0.15 * (1.0 - agree01),
        stubbornness=(
            0.45 * (1.0 - agree01)
            + 0.25 * neuro01
            + 0.20 * (1.0 - open01)
            + 0.10 * consc01
        ),
        # Final-decision movement resistance: how hard the sim is to move during
        # narrowing/voting/repair (candidate switches, compromise acceptance,
        # holdout concession). Distinct from stubbornness, which only governs
        # discussion-phase stance defense. Low agreeableness dominates; high
        # conscientiousness adds commitment to the announced pick; low openness
        # and high neuroticism add a smaller reluctance to change course.
        switch_resistance=(
            0.40 * (1.0 - agree01)
            + 0.25 * consc01
            + 0.20 * (1.0 - open01)
            + 0.15 * neuro01
        ),
    ).clipped()


def expected_turn_share(personas: list[Persona]) -> dict[str, float]:
    """Engagement-derived target participation share per sim.

    Engagement is the only participation-share parameter; the constant floor
    keeps even a fully disengaged sim at a visible minimum share. This is the
    single contract used by both the willingness baseline and the evaluation's
    engagement-realization metrics.
    """
    raw = {p.id: 0.30 + p.sim_params.engagement for p in personas}
    total = sum(raw.values()) or 1.0
    return {pid: value / total for pid, value in raw.items()}


# ---------------------------------------------------------------------------
# Policy constants (documented behavioral weights, kept central per todo 4/25)
# ---------------------------------------------------------------------------

# Willingness factors — additive terms combined then clipped to [0, 1].
_W_ENGAGEMENT = 0.42          # baseline floor-claim tendency (engagement only)
_W_CHALLENGED = 0.34         # own current/preferred option was visibly challenged
_W_RIVAL_SUPPORT = 0.24      # a disliked/rejected option gained visible support
_W_RELEVANT_LAST = 0.14      # the last accepted turn touches this sim's option/concern
_W_OWN_CONCERN = 0.14        # this sim's own concern was answered/engaged
_W_ANSWERABLE_Q = 0.26       # a relevant open group question this sim can answer
_W_UNUSED_REASON = 0.10      # an unused grounded reason/comparison is available
_W_SILENCE = 0.30            # silent relative to engagement-derived expected share
_W_UNDER_OPTION = 0.10       # a relevant under-discussed option matches own stance
_W_NARROWING_STAKE = 0.16    # narrowing around an option this sim strongly (dis)likes

_W_SPOKE_LAST = 0.55         # spoke on the immediately preceding turn
_W_OVERSHOOT = 0.40          # recent over-participation vs expected share
_W_REPEAT = 0.30             # the intended point repeats a recent own contribution

# Below this willingness a claim is only sampled, never forced; a bid whose best
# available act scores below _MIN_ACT_SCORE never claims the floor at all.
_MIN_WILLINGNESS = 0.05
_MIN_ACT_SCORE = 0.30

# Normal open-floor acts the policy may sample.
_OPEN_FLOOR_ACTS = (
    ActType.ANSWER,
    ActType.SUPPORT,
    ActType.CONCERN,
    ActType.ASK,
    ActType.COMPARE,
    ActType.COMMENT,
    ActType.COMPROMISE,
)


# ---------------------------------------------------------------------------
# Movement helpers (read only the deciding sim's own private ranks)
# ---------------------------------------------------------------------------

def hard_blocks(rt: ParticipantRuntime, candidate: str | None) -> bool:
    """Whether this sim hard-rejects the candidate (rank 1)."""
    return bool(candidate) and rt.rank(candidate) <= STANCE_REJECTED


def can_move_to(persona: Persona, rt: ParticipantRuntime, option_id: str, *, final: bool) -> bool:
    """Whether this sim could plausibly move to the option at all.

    Rank-1 options are hard blocked for everyone. The near-hard trait gate is
    stubbornness in discussion-phase lean movement, switch_resistance for final
    switch/vote/repair movement.
    """
    if option_id in rt.rejected_options():
        return False
    gate = persona.sim_params.switch_resistance if final else persona.sim_params.stubbornness
    if gate >= 0.85 and option_id != persona.preferred_option:
        return random.random() < 0.04
    return True


def _visible_candidate_openness(state: DialogueState, persona_id: str, candidate: str) -> float:
    """Own visible pre-vote openness toward the candidate, from this sim's own
    accepted discussion turns only (never a vote, never a public score)."""
    score = 0.0
    for turn in state.turns:
        if (
            turn.speaker_id != persona_id
            or turn.state_mutation_blocked
            or turn.evidence is None
            or turn.phase in {Phase.VOTING, Phase.COMPROMISE_REPAIR, Phase.CLOSING}
        ):
            continue
        ev = turn.evidence
        if any(s.option_id == candidate for s in ev.softenings):
            score += 2.0
        if any(c.option_id == candidate and c.kind == "accept" for c in ev.commitments):
            score += 2.0
        if any(s.option_id == candidate for s in ev.supports):
            score += 1.0
        if any(p.option_id == candidate for p in ev.proposals):
            score += 1.0
        if any(c.option_id == candidate for c in ev.concerns):
            score -= 1.0
        if any(b.option_id == candidate and b.action == "raised" for b in ev.blockers):
            score -= 3.0
    return max(-4.0, min(4.0, score))


def _own_unanswered_concerns(state: DialogueState, persona_id: str, option_id: str) -> int:
    return sum(
        1 for t in state.threads.values()
        if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
        and t.status is ThreadStatus.HOT
        and t.started_by == persona_id
        and option_id in t.focus_options
    )


def candidate_resistance(state: DialogueState, persona: Persona, candidate: str) -> float:
    """Lower means the sim is more willing to move to the candidate (final
    movement); reads only its own ranks, own concerns, and its own visible
    openness."""
    rt = state.runtimes[persona.id]
    if rt.rank(candidate) <= STANCE_REJECTED:
        return 99.0
    resistance = 0.70 * persona.sim_params.switch_resistance
    if rt.rank(candidate) <= STANCE_DISLIKED:
        resistance += 0.25
    resistance += 0.30 * _own_unanswered_concerns(state, persona.id, candidate)
    resistance -= 0.10 * _visible_candidate_openness(state, persona.id, candidate)
    if rt.current_acceptance == candidate or rt.public_lean == candidate:
        resistance -= 0.25
    if candidate in persona.preferred_options:
        resistance -= 0.25
    if rt.rank(candidate) >= STANCE_ACCEPTABLE:
        resistance -= 0.20
    if rt.rank(candidate) >= STANCE_PREFERRED or rt.top_option() == candidate:
        resistance -= 0.35
    return max(0.0, resistance)


def valid_holdout(state: DialogueState, persona: Persona, candidate: str) -> bool:
    """Whether keeping dissent is more realistic than converting — from this
    sim's own rank, own live concerns, and switch_resistance only."""
    rt = state.runtimes[persona.id]
    current = rt.explicit_vote or rt.current_acceptance or rt.public_lean or rt.top_option() or persona.preferred_option
    if candidate not in state.scenario.option_ids or current == candidate:
        return False
    if rt.rank(candidate) <= STANCE_REJECTED:
        return True
    unanswered = _own_unanswered_concerns(state, persona.id, candidate)
    if persona.sim_params.switch_resistance >= 0.82 and rt.rank(candidate) < STANCE_ACCEPTABLE:
        return True
    if unanswered >= 2 and persona.sim_params.switch_resistance >= 0.65 and rt.rank(candidate) <= STANCE_DISLIKED:
        return True
    return False


# ---------------------------------------------------------------------------
# Simulator decision view (public state + this sim's own private state only)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SimulatorView:
    """Everything one simulator may read to decide its bid.

    ``persona``/``runtime`` are the deciding sim's own objects; every other
    field is public (derived from accepted visible turns) or scenario-level. No
    other simulator's private goal, hidden ranks, or hidden reasons are here.
    """

    persona: Persona
    runtime: ParticipantRuntime
    scenario: Scenario
    phase: Phase
    # Public conversation state.
    public_candidate: str | None
    top_pair: list[str]
    backing: dict[str, set[str]]
    formal_votes: dict[str, str]
    coverage: dict[str, OptionCoverage]
    recent_turns: list[TurnRecord]
    last_turn: TurnRecord | None
    last_speaker_id: str | None
    recent_speaker_ids: list[str]
    active_threads: list  # list[ThreadState]
    social_ledger: dict[str, PublicParticipantState]
    own_contribution_keys: frozenset[str]
    coverage_gap: str | None
    # This sim's participation state.
    silence_streak: int
    expected_share: float
    realized_share: float
    spoke_last: bool
    # Framing.
    stimulus_kind: str

    # -- convenience readers (own private ranks) ----------------------------
    def top(self) -> str:
        return self.runtime.top_option() or self.persona.preferred_option

    def rival_options(self) -> list[str]:
        """Options other than the sim's own top pick, most publicly-backed first."""
        top = self.top()
        return sorted(
            (oid for oid in self.scenario.option_ids if oid != top),
            key=lambda oid: (-len(self.backing.get(oid, set())), oid),
        )

    def public_support_count(self, option_id: str, *, exclude_self: bool = False) -> int:
        backers = self.backing.get(option_id, set())
        return len(backers - {self.persona.id}) if exclude_self else len(backers)


def build_view(
    state: DialogueState,
    participant_id: str,
    *,
    stimulus: DiscussionStimulus | None = None,
) -> SimulatorView:
    persona = state.persona_by_id(participant_id)
    rt = state.runtimes[participant_id]
    ev = public_evidence(state)
    recent = [t for t in state.turns if t.speaker_id != "moderator"]
    last_turn = recent[-1] if recent else None
    recent_ids: list[str] = []
    for turn in reversed(recent):
        if turn.speaker_id not in recent_ids:
            recent_ids.append(turn.speaker_id)
        if len(recent_ids) >= 3:
            break
    total = sum(r.turn_count for r in state.runtimes.values())
    expected = expected_turn_share(state.personas)
    realized = rt.turn_count / total if total else expected[participant_id]
    # A thread stays a live stimulus only until its per-thread contribution cap:
    # this preserves the thread contribution limit (todo 13) as a stimulus
    # filter rather than a routing prescription.
    hard_cap = int(cfg.threads.max_thread_turns_hard)
    active_threads = [
        t for t in state.threads.values()
        if t.status in (ThreadStatus.HOT, ThreadStatus.COOLING)
        and t.contribution_count < hard_cap
    ]
    window = int(cfg.utterances.recent_turns_in_prompt)
    return SimulatorView(
        persona=persona,
        runtime=rt,
        scenario=state.scenario,
        phase=state.phase,
        public_candidate=ev.candidate_leaders[0] if ev.candidate_leaders else None,
        top_pair=list(ev.top_pair),
        backing=ev.backing,
        formal_votes=ev.formal_votes,
        coverage=state.coverage,
        recent_turns=recent[-window:] if window > 0 else [],
        last_turn=last_turn,
        last_speaker_id=last_turn.speaker_id if last_turn else None,
        recent_speaker_ids=recent_ids,
        active_threads=active_threads,
        social_ledger=public_participant_ledger(state),
        own_contribution_keys=frozenset(
            turn.intent.contribution_key
            for turn in state.turns
            if turn.speaker_id == participant_id
            and not turn.state_mutation_blocked
            and turn.text.strip()
            and turn.intent is not None
            and turn.intent.contribution_key
        ),
        coverage_gap=(stimulus.coverage_gap if stimulus else None),
        silence_streak=_silence_streak(state, participant_id),
        expected_share=expected[participant_id],
        realized_share=realized,
        spoke_last=bool(last_turn and last_turn.speaker_id == participant_id),
        stimulus_kind=(stimulus.kind if stimulus else "normal"),
    )


def _silence_streak(state: DialogueState, persona_id: str) -> int:
    streak = 0
    for turn in reversed(state.turns):
        if turn.speaker_id == "moderator":
            continue
        if turn.speaker_id == persona_id:
            break
        streak += 1
    return streak


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def decide_simulator_bid(
    state: DialogueState,
    participant_id: str,
    *,
    obligation: TurnObligation | None = None,
    stimulus: DiscussionStimulus | None = None,
) -> SimulatorBid:
    """The one participant-owned decision entry point.

    Handles the three decision classes: a constrained response to a protocol
    obligation (opening/direct answer/vote/narrowing/repair), and open-floor
    self-selection under a public stimulus.
    """
    view = build_view(state, participant_id, stimulus=stimulus)
    if obligation is not None:
        return _obligation_bid(state, view, obligation)
    return _open_floor_bid(state, view)


# ---------------------------------------------------------------------------
# Open-floor self-selection
# ---------------------------------------------------------------------------

def _open_floor_bid(state: DialogueState, view: SimulatorView) -> SimulatorBid:
    scores = _score_acts(state, view)
    pid = view.persona.id
    positive = {a: s for a, s in scores.items() if s >= _MIN_ACT_SCORE}
    willingness, trigger = _willingness(view, scores)
    if not positive:
        # Nothing grounded to add: decline the floor. This is a legitimate
        # simulated silence, never overwritten by the framework.
        return SimulatorBid(pid, False, min(willingness, _MIN_WILLINGNESS), None,
                            trigger="no grounded contribution", action_scores=scores)
    # Seeded probabilistic self-selection: willingness gates whether to claim,
    # act scores decide which act if it does.
    claims = random.random() < max(willingness, _MIN_WILLINGNESS)
    if not claims:
        return SimulatorBid(pid, False, willingness, None, trigger=trigger, action_scores=scores)
    acts = list(positive)
    act = weighted_choice(acts, [positive[a] for a in acts])
    intent = _build_open_intent(state, view, act)
    if intent is None:
        return SimulatorBid(pid, False, willingness, None,
                            trigger="act not realizable", action_scores=scores)
    return SimulatorBid(pid, True, willingness, intent, trigger=trigger, action_scores=scores)


def _willingness(view: SimulatorView, scores: dict[ActType, float]) -> tuple[float, str]:
    """Participant-local willingness in [0, 1] plus the dominant trigger label.

    Engagement is the baseline only; relevance and personal stake can outweigh
    it, so a low-engagement sim whose option was challenged beats a highly
    engaged sim with nothing new to add.
    """
    p = view.persona.sim_params
    rt = view.runtime
    top = view.top()
    terms: list[tuple[float, str]] = []
    terms.append((_W_ENGAGEMENT * p.engagement, "engagement"))

    challenged = _option_challenged_recently(view, top)
    if challenged:
        terms.append((_W_CHALLENGED * (0.6 + 0.8 * p.stubbornness), "own option challenged"))

    # A disliked/rejected option gaining visible backing is a personal stake.
    rival_gain = any(
        view.public_support_count(oid, exclude_self=True) >= 1
        for oid in (rt.disliked_options() | rt.rejected_options())
    )
    if rival_gain:
        terms.append((_W_RIVAL_SUPPORT, "disliked option gaining support"))

    if view.last_turn is not None and _turn_relevant_to(view, view.last_turn):
        terms.append((_W_RELEVANT_LAST, "relevant recent turn"))

    if _own_concern_engaged(view):
        terms.append((_W_OWN_CONCERN, "own concern engaged"))

    if _answerable_group_question(view) is not None:
        terms.append((_W_ANSWERABLE_Q, "answerable question"))

    if _contribution_available(view, ActType.SUPPORT, [top], "support"):
        terms.append((_W_UNUSED_REASON, "unused grounded reason"))

    deficit = max(0.0, view.expected_share - view.realized_share)
    if deficit > 0:
        terms.append((_W_SILENCE * min(1.0, deficit / max(view.expected_share, 1e-6)), "under expected share"))

    if view.coverage_gap and rt.rank(view.coverage_gap) != STANCE_NEUTRAL:
        terms.append((_W_UNDER_OPTION, "relevant under-discussed option"))

    if view.stimulus_kind in ("narrowing", "repair") and view.public_candidate:
        stake = abs(rt.rank(view.public_candidate) - STANCE_NEUTRAL)
        if stake >= 1:
            terms.append((_W_NARROWING_STAKE, "stake in narrowing candidate"))

    total = sum(w for w, _ in terms)
    if view.spoke_last:
        total -= _W_SPOKE_LAST
    overshoot = max(0.0, view.realized_share - view.expected_share)
    if overshoot > 0:
        total -= _W_OVERSHOOT * min(1.0, overshoot / max(view.expected_share, 1e-6))
    if scores.get(ActType.COMMENT, 0.0) >= max(scores.values(), default=0.0) and max(scores.values(), default=0.0) < 0.6:
        # Only a weak light comment is available: little pull to speak.
        total -= _W_REPEAT * 0.5

    trigger = max(terms, key=lambda t: t[0])[1] if terms else "engagement"
    return max(0.0, min(1.0, total)), trigger


def _score_acts(state: DialogueState, view: SimulatorView) -> dict[ActType, float]:
    """Score useful participant-owned actions from public stimuli and own state.

    A score means that concrete content is available; it is not a speaking
    quota. COMMENT deliberately has no unconditional baseline and silence is
    valid when no score reaches ``_MIN_ACT_SCORE``.
    """
    p = view.persona.sim_params
    rt = view.runtime
    top = view.top()
    scores: dict[ActType, float] = {a: 0.0 for a in _OPEN_FLOOR_ACTS}

    group_question = _answerable_group_question(view)
    if group_question is not None:
        relevance = 0.75 if group_question.focus_options else 0.60
        scores[ActType.ANSWER] = relevance + 0.20 * p.engagement

    challenged = _option_challenged_recently(view, top)
    unused_support = _contribution_available(view, ActType.SUPPORT, [top], "support")
    support = 0.0
    if unused_support:
        support += 0.24
    if challenged:
        support += 0.42 * (0.55 + p.stubbornness)
    if view.public_candidate == top and view.public_support_count(top, exclude_self=True):
        support += 0.16
    if view.stimulus_kind in {"coverage", "stall"} and view.coverage_gap == top:
        support += 0.22
    scores[ActType.SUPPORT] = support

    concern_option = _concern_target_option(view)
    concern = 0.0
    if concern_option:
        rank = rt.rank(concern_option)
        public_weight = view.public_support_count(concern_option, exclude_self=True)
        card = view.scenario.option(concern_option)
        if rank <= STANCE_DISLIKED:
            concern += 0.48
        elif rank == STANCE_NEUTRAL and public_weight:
            concern += 0.24
        elif rank == STANCE_ACCEPTABLE and public_weight and card.concern:
            concern += 0.18
        if public_weight:
            concern += 0.16
        if card.concern and _contribution_available(view, ActType.CONCERN, [concern_option], "listed_drawback"):
            concern += 0.18
        if _public_concern_shared_by_other(view, concern_option):
            concern += 0.15
        if _option_claim_conflicts_with_card(view, concern_option):
            concern += 0.30
    scores[ActType.CONCERN] = concern * (0.70 + 0.45 * p.directness + 0.35 * p.stubbornness)

    ask = 0.0
    if _unexplained_public_owner(view) is not None:
        ask += 0.34
    if _unanswered_public_concern_owner(view) is not None:
        ask += 0.32
    if _unclear_recent_claim_owner(view) is not None:
        ask += 0.25
    if len(view.top_pair) == 2:
        ask += 0.18
    if view.stimulus_kind in {"coverage", "stall", "narrowing"}:
        ask += 0.18
    if _recent_question_count(view) >= 1:
        ask *= 0.45
    scores[ActType.ASK] = ask

    compare = 0.0
    pair = _compare_pair(view)
    if len(pair) == 2 and rt.rank(pair[0]) != rt.rank(pair[1]):
        compare += 0.34
    if any(t.thread_type is ThreadType.COMPARISON and t.status is ThreadStatus.HOT for t in view.active_threads):
        compare += 0.22
    if view.coverage_gap and view.coverage_gap in pair:
        compare += 0.18
    if _contribution_available(view, ActType.COMPARE, pair, "tradeoff") is False:
        compare *= 0.35
    scores[ActType.COMPARE] = compare

    # A comment is useful only as a concrete acknowledgement/interpretation of
    # a visible contribution; it is never a generic filler action.
    if view.last_turn is not None and view.last_turn.speaker_id != view.persona.id:
        if _turn_relevant_to(view, view.last_turn) and _contribution_available(
            view, ActType.COMMENT, view.last_turn.mentioned_options()[:1], f"ack:{view.last_turn.index}"
        ):
            scores[ActType.COMMENT] = 0.24
    if view.stimulus_kind == "stall":
        scores[ActType.COMMENT] = max(scores[ActType.COMMENT], 0.31)
        scores[ActType.PROCESS] = 0.35 + 0.20 * p.engagement
        scores[ActType.ASK] += 0.20
        scores[ActType.COMPARE] += 0.15
    elif view.stimulus_kind == "coverage":
        scores[ActType.ASK] += 0.16
        scores[ActType.COMPARE] += 0.12

    scores[ActType.COMPROMISE] = _compromise_score(state, view)

    if view.spoke_last:
        for act in scores:
            if act is not ActType.ANSWER:
                scores[act] *= 0.55
    return scores

def _compromise_score(state: DialogueState, view: SimulatorView) -> float:
    p = view.persona.sim_params
    rt = view.runtime
    candidate = view.public_candidate or (view.top_pair[0] if view.top_pair else None)
    if candidate is None or candidate == view.top() or hard_blocks(rt, candidate):
        return 0.0
    rank = rt.rank(candidate)
    if rank < STANCE_NEUTRAL or not can_move_to(view.persona, rt, candidate, final=False):
        return 0.0
    support = view.public_support_count(candidate, exclude_self=True)
    if support < 1:
        return 0.0
    own_top_support = view.public_support_count(view.top(), exclude_self=True)
    concerns = _own_unanswered_concerns(state, view.persona.id, candidate)
    flexibility = 1.0 - p.switch_resistance
    score = 0.20 + 0.28 * flexibility + 0.08 * min(2, support)
    if rank >= STANCE_ACCEPTABLE:
        score += 0.15
    if own_top_support == 0:
        score += 0.12
    if concerns == 0:
        score += 0.10
    else:
        score -= 0.18 * concerns
    if not _contribution_available(view, ActType.COMPROMISE, [candidate], "conditional_move"):
        score *= 0.25
    return max(0.0, min(0.85, score))


# ---------------------------------------------------------------------------
# Open-floor intent construction (act -> target/focus/reason/addressee)
# ---------------------------------------------------------------------------

def _build_open_intent(state: DialogueState, view: SimulatorView, act: ActType) -> MoveIntent | None:
    aliases = short_alias_map(view.scenario.options)
    top = view.top()
    top_name = aliases.get(top, top)
    pid = view.persona.id

    if act is ActType.ANSWER:
        question = _answerable_group_question(view)
        if question is None:
            return None
        source = _turn_by_index(view, question.source_turn_index)
        focus = [o for o in question.focus_options if o in view.scenario.option_ids]
        return _intent(
            pid, act,
            "answer the open group question directly from your own current view, then state one decision implication",
            focus=focus, target=source, thread=question, addressee=question.started_by,
            key=f"answer:{question.thread_id}",
        )

    if act is ActType.SUPPORT:
        thread, target = _reactive_target(view, focus_option=top)
        issue_owner = thread.started_by if thread is not None and thread.started_by != pid else None
        reason = (
            f"state unmistakably that {top_name} is a good fit for you and give one grounded personal reason "
            "that has not already been contributed"
        )
        return _intent(pid, act, reason, focus=[top], target=target, thread=thread,
                       addressee=issue_owner, key="support")

    if act is ActType.CONCERN:
        rival = _concern_target_option(view)
        if rival is None:
            return None
        thread, target = _reactive_target(view, focus_option=rival)
        card = view.scenario.option(rival)
        if hard_blocks(view.runtime, rival):
            reason_against = view.runtime.reason_against(rival)
            reason = (
                f"state a clear objection to {aliases.get(rival, rival)} and why it remains unacceptable"
                + (f": {reason_against}" if reason_against else "")
            )
            issue = "hard_block"
        elif card.concern:
            reason = (
                f"raise one clear concern about {aliases.get(rival, rival)} using its listed drawback "
                f"({card.concern}); explain why that matters to your decision"
            )
            issue = "listed_drawback"
        else:
            reason = (
                f"raise one clear, grounded concern or uncertainty about {aliases.get(rival, rival)}; "
                "do not merely imply hesitation"
            )
            issue = "public_risk"
        addressee = _advocate_of(view, rival)
        return _intent(pid, act, reason, focus=[rival], target=target, thread=thread,
                       addressee=addressee, key=issue)

    if act is ActType.ASK:
        focus = _ask_focus(view)
        owner = _question_target_owner(view, focus)
        thread, target = _reactive_target(view, focus_option=focus[0] if focus else None)
        if owner:
            owner_state = view.social_ledger[owner]
            if owner_state.public_position and owner_state.public_position in view.scenario.option_ids:
                named = aliases.get(owner_state.public_position, owner_state.public_position)
                reason = f"ask why {named} works for them or which trade-off matters most; ask one concrete question"
            elif owner_state.concerned_options:
                named = aliases.get(owner_state.concerned_options[0], owner_state.concerned_options[0])
                reason = f"ask what would resolve their public concern about {named}; ask one concrete question"
            else:
                reason = "ask one concrete clarification about the public point they raised"
        else:
            names = ", ".join(aliases.get(o, o) for o in focus) or "the remaining options"
            reason = f"ask one concrete group question that helps settle the trade-off between {names}"
        if owner and view.social_ledger[owner].concerned_options:
            issue_key = f"concern:{owner}"
        elif owner:
            issue_key = f"why:{owner}"
        else:
            issue_key = "group_tradeoff"
        return _intent(pid, act, reason, focus=focus, target=target, thread=thread,
                       addressee=owner, key=issue_key)

    if act is ActType.COMPARE:
        pair = _compare_pair(view)
        if len(pair) < 2:
            return None
        thread, target = _reactive_target(view, focus_option=pair[0])
        names = ", ".join(aliases.get(o, o) for o in pair)
        reason = f"compare {names} explicitly on one grounded trade-off and make your own preference clear"
        return _intent(pid, act, reason, focus=pair, target=target, thread=thread,
                       key="tradeoff")

    if act is ActType.COMPROMISE:
        candidate = view.public_candidate or (view.top_pair[0] if view.top_pair else None)
        if candidate is None:
            return None
        cname = aliases.get(candidate, candidate)
        reason = (
            f"state visible conditional willingness to move toward {cname} as common ground and name the "
            "condition or trade-off that makes it acceptable; do not cast a final vote"
        )
        focus = [candidate, top] if top != candidate else [candidate]
        return _intent(pid, act, reason, focus=focus, key="conditional_move")

    if act is ActType.PROCESS:
        pair = view.top_pair
        if len(pair) == 2:
            names = " and ".join(aliases.get(o, o) for o in pair)
            reason = f"suggest that the group resolve the remaining trade-off between {names} before deciding"
            focus = list(pair)
        else:
            reason = "briefly suggest one useful next discussion step without choosing anyone else's stance"
            focus = [top]
        return _intent(pid, act, reason, focus=focus, length_hint="short",
                       key=view.stimulus_kind)

    if act is ActType.COMMENT:
        if view.last_turn is None or view.last_turn.speaker_id == pid:
            return None
        focus = [o for o in view.last_turn.mentioned_options() if o in view.scenario.option_ids][:1]
        reason = "briefly acknowledge or interpret the specific public point just made without inventing a new stance"
        return _intent(pid, act, reason, focus=focus, target=view.last_turn,
                       addressee=view.last_turn.speaker_id,
                       length_hint="short", key=f"ack:{view.last_turn.index}")
    return None


def _intent(
    pid: str,
    act: ActType,
    reason: str,
    *,
    focus: list[str] | None = None,
    target: TurnRecord | None = None,
    thread=None,
    addressee: str | None = None,
    length_hint: str = "medium",
    key: str | None = None,
) -> MoveIntent:
    return MoveIntent(
        speaker_id=pid,
        act=act,
        reason=reason,
        route_source="self_selection",
        addressee_id=addressee,
        option_focus=list(focus or []),
        respond_to_turn=target.index if target is not None else None,
        thread_id=thread.thread_id if thread is not None else None,
        length_hint=length_hint,  # type: ignore[arg-type]
        contribution_key=_contribution_key(act, list(focus or []), key) if key else None,
    )


def _turn_by_index(view: SimulatorView, index: int | None) -> TurnRecord | None:
    if index is None:
        return None
    return next((turn for turn in view.recent_turns if turn.index == index), None)

def _reactive_target(view: SimulatorView, *, focus_option: str | None):
    """Pick a hot thread this sim reacts to and the source turn to respond to.

    Prefers a hot thread touching ``focus_option``; otherwise reacts to the last
    accepted turn about a relevant option. Returns (thread_or_None, turn_or_None).
    """
    for thread in view.active_threads:
        if thread.status is not ThreadStatus.HOT:
            continue
        if focus_option and focus_option in thread.focus_options and thread.started_by != view.persona.id:
            target = next((t for t in view.recent_turns if t.index == thread.source_turn_index), None)
            return thread, target
    last = view.last_turn
    if last is not None and last.speaker_id != view.persona.id and _turn_relevant_to(view, last):
        return None, last
    return None, None


def _concern_target_option(view: SimulatorView) -> str | None:
    rt = view.runtime
    candidates: list[str] = []
    if view.public_candidate and view.public_candidate != view.top():
        candidates.append(view.public_candidate)
    candidates.extend(oid for oid in view.top_pair if oid != view.top())
    candidates.extend(view.rival_options())
    candidates = list(dict.fromkeys(oid for oid in candidates if oid in view.scenario.option_ids))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda oid: (
            4 if rt.rank(oid) <= STANCE_DISLIKED
            else 3 if oid == view.public_candidate and view.public_support_count(oid, exclude_self=True)
            else 2 if view.public_support_count(oid, exclude_self=True)
            else 1 if rt.rank(oid) == STANCE_NEUTRAL
            else 0,
            view.public_support_count(oid, exclude_self=True),
            1 if view.scenario.option(oid).concern else 0,
            -view.scenario.option_ids.index(oid),
        ),
    )

def _ask_focus(view: SimulatorView) -> list[str]:
    top = view.top()
    if view.top_pair:
        return [o for o in view.top_pair][:2]
    rivals = view.rival_options()
    focus = [top]
    if rivals:
        focus.append(rivals[0])
    return [o for o in focus if o in view.scenario.option_ids][:2]


def _compare_pair(view: SimulatorView) -> list[str]:
    rt = view.runtime
    top = view.top()
    if len(view.top_pair) == 2:
        pair = list(view.top_pair)
    else:
        rivals = view.rival_options()
        pair = [top] + rivals[:1]
    if view.coverage_gap and view.coverage_gap not in pair and rt.rank(view.coverage_gap) != STANCE_NEUTRAL:
        pair = [top, view.coverage_gap]
    return [o for o in dict.fromkeys(pair) if o in view.scenario.option_ids][:2]


def _advocate_of(view: SimulatorView, option_id: str | None) -> str | None:
    """Most recent visible public backer of an option."""
    if not option_id:
        return None
    backers = view.backing.get(option_id, set()) - {view.persona.id}
    if not backers:
        return None
    return max(
        backers,
        key=lambda pid: view.social_ledger.get(pid, PublicParticipantState(pid, pid)).last_turn_index or -1,
    )


def _unexplained_public_owner(view: SimulatorView) -> str | None:
    """Visible position owner worth asking, without reading private reasons."""
    candidates = [
        item for pid, item in view.social_ledger.items()
        if pid != view.persona.id
        and item.public_position in view.scenario.option_ids
        and item.last_act in {ActType.OPENING, ActType.SUPPORT, ActType.COMMENT}
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.last_turn_index or -1, reverse=True)
    for item in candidates:
        focus = [item.public_position] if item.public_position else []
        if _contribution_available(view, ActType.ASK, focus, f"why:{item.participant_id}"):
            return item.participant_id
    return None


def _unanswered_public_concern_owner(view: SimulatorView) -> str | None:
    candidates = [
        item for pid, item in view.social_ledger.items()
        if pid != view.persona.id and item.concerned_options
    ]
    candidates.sort(key=lambda item: item.last_turn_index or -1, reverse=True)
    for item in candidates:
        if _contribution_available(
            view, ActType.ASK, list(item.concerned_options[:1]), f"concern:{item.participant_id}"
        ):
            return item.participant_id
    return None


def _unclear_recent_claim_owner(view: SimulatorView) -> str | None:
    for turn in reversed(view.recent_turns):
        if turn.speaker_id == view.persona.id or turn.evidence is None:
            continue
        unclear = bool(turn.evidence.ambiguous_references)
        if turn.assessment is not None:
            unclear = unclear or any(
                issue.code in {"unsupported_fact", "ambiguous_reference", "ungrounded_inference"}
                for issue in turn.assessment.issues
            )
        if unclear:
            return turn.speaker_id
    return None


def _question_target_owner(view: SimulatorView, focus: list[str]) -> str | None:
    for finder in (_unanswered_public_concern_owner, _unexplained_public_owner, _unclear_recent_claim_owner):
        owner = finder(view)
        if owner is None:
            continue
        item = view.social_ledger.get(owner)
        if not focus or item is None or not item.last_focus_options or set(focus) & set(item.last_focus_options):
            return owner
    return None


def _public_concern_shared_by_other(view: SimulatorView, option_id: str) -> bool:
    return any(
        pid != view.persona.id and option_id in item.concerned_options
        for pid, item in view.social_ledger.items()
    )


def _option_claim_conflicts_with_card(view: SimulatorView, option_id: str) -> bool:
    for turn in reversed(view.recent_turns):
        if option_id not in turn.mentioned_options() or turn.assessment is None:
            continue
        if any(issue.code in {"contradiction", "cross_option_transfer"} for issue in turn.assessment.issues):
            return True
    return False


# ---------------------------------------------------------------------------
# Relevance / repetition helpers (public signals + own stance only)
# ---------------------------------------------------------------------------

def _option_challenged_recently(view: SimulatorView, option_id: str) -> bool:
    for turn in view.recent_turns[-3:]:
        if turn.speaker_id == view.persona.id or turn.evidence is None:
            continue
        if any(c.option_id == option_id for c in turn.evidence.concerns):
            return True
        if any(b.option_id == option_id and b.action == "raised" for b in turn.evidence.blockers):
            return True
    return False


def _turn_relevant_to(view: SimulatorView, turn: TurnRecord) -> bool:
    rt = view.runtime
    for oid in turn.mentioned_options():
        if oid not in view.scenario.option_ids:
            continue
        if rt.rank(oid) != STANCE_NEUTRAL or oid == view.top():
            return True
    return False


def _own_concern_engaged(view: SimulatorView) -> bool:
    for thread in view.active_threads:
        if (
            thread.started_by == view.persona.id
            and thread.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
            and thread.status is ThreadStatus.COOLING
        ):
            return True
    return False


def _answerable_group_question(view: SimulatorView):
    for thread in view.active_threads:
        if (
            thread.thread_type is ThreadType.QUESTION
            and thread.status is ThreadStatus.HOT
            and thread.question_scope == "group"
            and thread.started_by != view.persona.id
        ):
            relevant = not thread.focus_options or any(
                view.runtime.rank(oid) != STANCE_NEUTRAL for oid in thread.focus_options
            )
            if relevant:
                return thread
    return None


def _contribution_key(act: ActType, focus: list[str], issue: str) -> str:
    return f"{act.value}:{':'.join(sorted(o for o in focus if o))}:{issue}"


def _contribution_available(
    view: SimulatorView,
    act: ActType,
    focus: list[str],
    issue: str,
) -> bool:
    return _contribution_key(act, focus, issue) not in view.own_contribution_keys

def _recent_question_count(view: SimulatorView) -> int:
    return sum(1 for t in view.recent_turns[-3:] if "?" in t.text)


# ---------------------------------------------------------------------------
# Protocol obligations (framework fixes speaker+act; sim decides substance)
# ---------------------------------------------------------------------------

def _obligation_bid(state: DialogueState, view: SimulatorView, ob: TurnObligation) -> SimulatorBid:
    pid = view.persona.id
    if ob.kind == "opening":
        intent = _opening_intent(view)
        return SimulatorBid(pid, True, 1.0, intent, trigger="opening protocol",
                            action_scores={ActType.OPENING: 1.0})
    if ob.kind == "direct_answer":
        intent = _answer_intent(state, view, ob)
        return SimulatorBid(pid, True, 1.0, intent, trigger="direct question obligation",
                            action_scores={ActType.ANSWER: 1.0})
    if ob.kind == "vote":
        intent = _vote_intent(state, view, ob, is_repair=False)
        return SimulatorBid(pid, True, 1.0, intent, trigger="vote protocol",
                            action_scores={ActType.VOTE: 1.0})
    if ob.kind in ("final_decision", "repair_vote"):
        intent = _vote_intent(state, view, ob, is_repair=True)
        return SimulatorBid(pid, True, 1.0, intent, trigger="repair vote",
                            action_scores={ActType.VOTE: 1.0})
    if ob.kind in ("reservation", "majority_concern"):
        intent = _reservation_intent(state, view, ob)
        return SimulatorBid(pid, True, 1.0, intent, trigger=ob.kind,
                            action_scores={ActType.ANSWER: 1.0})
    if ob.kind == "reservation_response":
        intent = _reservation_response_intent(state, view, ob)
        return SimulatorBid(pid, True, 1.0, intent, trigger=ob.kind,
                            action_scores={ActType.ANSWER: 1.0})
    if ob.kind == "narrowing_reaction":
        intent = _narrowing_reaction_intent(state, view, ob)
        return SimulatorBid(pid, True, 1.0, intent, trigger="narrowing reaction",
                            action_scores={intent.act: 1.0})
    raise ValueError(f"unknown obligation kind {ob.kind!r}")


def _opening_intent(view: SimulatorView) -> MoveIntent:
    # The opening lean is this sim's own top-ranked option, with its own reason.
    top = view.top()
    reason = (
        "state the current favorite and one grounded personal decision criterion; optionally add one "
        "genuine uncertainty or concern, without making a final vote"
    )
    return MoveIntent(
        speaker_id=view.persona.id,
        act=ActType.OPENING,
        reason=reason,
        route_source="opening_protocol",
        option_focus=[top],
    )


def _answer_intent(state: DialogueState, view: SimulatorView, ob: TurnObligation) -> MoveIntent:
    # The framework fixes speaker/act/target; the sim decides the answer's
    # direction, option focus, and grounded reason from its own stance.
    focus = [o for o in ob.focus_options if o in view.scenario.option_ids] or _focus_from_recent(state)
    return MoveIntent(
        speaker_id=view.persona.id,
        act=ActType.ANSWER,
        reason="answer the direct question you were just asked, then add one implication for the decision",
        route_source="direct_obligation",
        addressee_id=ob.addressee_id,
        option_focus=focus,
        respond_to_turn=ob.respond_to_turn,
        thread_id=ob.thread_id,
    )


def _reservation_intent(state: DialogueState, view: SimulatorView, ob: TurnObligation) -> MoveIntent:
    aliases = short_alias_map(view.scenario.options)
    candidate = ob.candidate
    cname = aliases.get(candidate, candidate) if candidate else "the candidate"
    if ob.kind == "majority_concern":
        reason = (
            f"state your single main concern about {cname}, then say whether you might reasonably move for "
            "the group and why; do not cast the final vote yet"
        )
    else:
        reason = (
            f"say concretely what still makes you hesitate about {cname} — one specific reservation or "
            "condition, grounded in the option facts or what they leave unknown; do not cast a vote yet"
        )
    return MoveIntent(
        speaker_id=view.persona.id,
        act=ActType.ANSWER,
        reason=reason,
        route_source="repair_protocol",
        option_focus=[candidate] if candidate else [],
        respond_to_turn=ob.respond_to_turn,
        length_hint="short",
    )



def _reservation_response_intent(
    state: DialogueState, view: SimulatorView, ob: TurnObligation
) -> MoveIntent:
    aliases = short_alias_map(view.scenario.options)
    candidate = ob.candidate if ob.candidate in view.scenario.option_ids else None
    name = aliases.get(candidate, candidate) if candidate else "the candidate"
    return MoveIntent(
        speaker_id=view.persona.id,
        act=ActType.ANSWER,
        reason=(
            f"answer the public reservation about {name} honestly from your own perspective: acknowledge a "
            "valid limitation, then explain only what the listed facts support; do not pressure the other person"
        ),
        route_source="repair_protocol",
        addressee_id=ob.addressee_id,
        option_focus=[candidate] if candidate else [],
        respond_to_turn=ob.respond_to_turn,
        length_hint="short",
        contribution_key=f"answer:reservation:{ob.respond_to_turn}",
    )

def _narrowing_reaction_intent(state: DialogueState, view: SimulatorView, ob: TurnObligation) -> MoveIntent:
    """The sim reacts to the narrowing candidate by its OWN stance: it can back
    it, raise a remaining concern, or say it is ready — the framework only asks
    for a reaction, never which one."""
    aliases = short_alias_map(view.scenario.options)
    candidate = ob.candidate
    rt = view.runtime
    if candidate and candidate in view.scenario.option_ids:
        name = aliases.get(candidate, candidate)
        rank = rt.rank(candidate)
        if rank >= STANCE_ACCEPTABLE or (rank == STANCE_NEUTRAL and view.persona.sim_params.switch_resistance < 0.5):
            act = ActType.SUPPORT
            reason = (
                f"answer the narrowing question: say honestly whether you could live with {name} and the one "
                "listed fact that makes it workable for you; this is not a final vote"
            )
        else:
            act = ActType.CONCERN
            reason = (
                f"answer the narrowing question by naming the one concrete thing that still blocks {name} for "
                "you; this is not a final vote"
            )
        return MoveIntent(
            speaker_id=view.persona.id, act=act, reason=reason,
            route_source="narrowing_protocol", option_focus=[candidate], length_hint="short",
            respond_to_turn=ob.respond_to_turn,
        )
    focus = [o for o in ob.focus_options if o in view.scenario.option_ids][:2]
    reason = (
        "answer the narrowing question in one short turn: name the strongest remaining concern if you have "
        "one, otherwise say plainly that you are ready to vote; do not cast a final vote yet"
    )
    return MoveIntent(
        speaker_id=view.persona.id, act=ActType.ANSWER, reason=reason,
        route_source="narrowing_protocol", option_focus=focus, length_hint="short",
        respond_to_turn=ob.respond_to_turn,
    )


def _vote_intent(state: DialogueState, view: SimulatorView, ob: TurnObligation, *, is_repair: bool) -> MoveIntent:
    """Simulator-owned vote decision.

    The framework only says a vote is due and which candidate (if any) is being
    tested. The sim selects its own target from its ranks, hard rejections,
    visible lean/acceptance, unresolved concerns, visible candidate support,
    switch_resistance, and the tested candidate — and whether this is a visible
    switch with a grounded reason. Hard blockers and rank-1 options remain
    impossible switches.
    """
    persona = view.persona
    rt = view.runtime
    tested = ob.candidate if ob.candidate in view.scenario.option_ids else None
    current = _current_public_pick(view)

    target = current
    outcome = "stay"
    if is_repair and tested is not None and tested != current:
        # A repair re-vote: the sim decides stay vs switch to the tested option.
        if _should_switch(state, view, tested, current):
            target = tested
            outcome = "switch"
    elif not is_repair:
        # First formal vote: pick the best stance-consistent acceptable target.
        target = _stance_consistent_target(view, tested)

    if hard_blocks(rt, target):
        target = _best_acceptable_alternative(view, avoid=target)

    old_pref = current if target != current else None
    reason = _vote_reason(view, target, current=current, tested=tested, outcome=outcome)
    focus = [target]
    for oid in (tested, current):
        if oid and oid in view.scenario.option_ids and oid not in focus:
            focus.append(oid)
    route_source = "repair_protocol" if is_repair else "vote_protocol"
    if outcome == "switch":
        instruction = (
            "cast a clear visible final vote; this is a genuine compromise switch, so make the change of "
            "mind visible and give your grounded reason"
        )
    else:
        instruction = (
            "cast a clear visible final vote for the option you actually choose now; this formal vote may "
            "replace an earlier discussion commitment"
        )
    return MoveIntent(
        speaker_id=persona.id,
        act=ActType.VOTE,
        reason=instruction,
        route_source=route_source,
        option_focus=focus,
        length_hint="short",
        allow_vote_change=target != current,
        required_vote=target,
        old_preference=old_pref,
        allowed_reason=reason,
    )


def _current_public_pick(view: SimulatorView) -> str:
    rt = view.runtime
    pick = rt.explicit_vote or rt.current_acceptance or rt.public_lean or rt.top_option() or view.persona.preferred_option
    if pick not in view.scenario.option_ids:
        pick = view.persona.preferred_option
    return pick


def _stance_consistent_target(view: SimulatorView, candidate: str | None) -> str:
    """Best final-vote target consistent with this sim's own visible stance."""
    rt = view.runtime
    rejected = set(rt.disliked_options()) | set(rt.rejected_options())

    def ok(oid: str | None) -> bool:
        return bool(oid and oid in view.scenario.option_ids and oid not in rejected)

    for oid in (rt.explicit_vote, rt.current_acceptance, rt.public_lean, rt.top_option(),
                *sorted(rt.acceptable_options()), *view.persona.preferred_options, candidate):
        if ok(oid):
            return str(oid)
    visible = sorted(
        view.scenario.option_ids,
        key=lambda oid: (-view.public_support_count(oid, exclude_self=True), oid),
    )
    for oid in visible:
        if ok(oid):
            return oid
    return _best_acceptable_alternative(view, avoid=None)


def _best_acceptable_alternative(view: SimulatorView, *, avoid: str | None) -> str:
    rt = view.runtime
    for oid in [*view.persona.preferred_options, *view.scenario.option_ids]:
        if oid in view.scenario.option_ids and oid != avoid and oid not in rt.rejected_options():
            return oid
    return view.scenario.option_ids[0]


def _should_switch(state: DialogueState, view: SimulatorView, tested: str, current: str) -> bool:
    """Seeded, conservative repair switch decision owned by one simulator."""
    persona = view.persona
    rt = view.runtime
    if tested == current or not can_move_to(persona, rt, tested, final=True):
        return False
    if valid_holdout(state, persona, tested) or rt.rank(tested) < STANCE_NEUTRAL:
        return False
    unresolved = _own_unanswered_concerns(state, persona.id, tested)
    if unresolved and rt.rank(tested) < STANCE_ACCEPTABLE:
        return False

    votes = view.formal_votes
    tested_votes = sum(1 for vote in votes.values() if vote == tested)
    current_votes = sum(1 for vote in votes.values() if vote == current)
    if tested_votes < current_votes:
        return False

    n = max(1, len(state.personas))
    advantage = max(0.0, (tested_votes - current_votes) / n)
    flexibility = 1.0 - persona.sim_params.switch_resistance
    rank_bonus = max(0.0, (rt.rank(tested) - STANCE_NEUTRAL) / 2.0)
    openness = max(0.0, _visible_candidate_openness(state, persona.id, tested)) / 4.0
    probability = (
        0.06
        + 0.30 * advantage
        + 0.34 * flexibility
        + 0.12 * rank_bonus
        + 0.10 * openness
        - 0.18 * unresolved
    )
    if tested_votes == current_votes:
        probability -= 0.12
    if persona.sim_params.switch_resistance >= 0.70:
        probability -= 0.18
    probability = max(0.02, min(0.78, probability))
    return random.random() < probability

def _vote_reason(view: SimulatorView, target: str, *, current: str, tested: str | None, outcome: str) -> str:
    from utils import clause_fragment, usable_reason_fragment
    rt = view.runtime
    if target not in view.scenario.option_ids:
        return "it is the clearest option left in the visible discussion"
    card = view.scenario.option(target)
    personal_for = usable_reason_fragment(rt.reason_for(target), card.name)
    if outcome == "switch" and personal_for:
        return personal_for
    if outcome == "stay":
        if tested and tested in view.scenario.option_ids and tested != target:
            tcard = view.scenario.option(tested)
            if tcard.concern:
                return f"the listed concern remains: {clause_fragment(tcard.concern, tcard.name)}"
            personal_against = usable_reason_fragment(rt.reason_against(tested), "")
            if personal_against:
                return personal_against
        if card.upside:
            return clause_fragment(card.upside, card.name)
        return "this is still the more defensible option from the listed facts"
    if personal_for:
        return personal_for
    if card.upside:
        return clause_fragment(card.upside, card.name)
    return "this remains your most defensible choice from the visible discussion"


def _focus_from_recent(state: DialogueState) -> list[str]:
    for turn in reversed(state.turns):
        mentioned = turn.mentioned_options()
        if mentioned:
            return mentioned[:2]
    return []
