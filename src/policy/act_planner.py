"""
policy/act_planner.py
----------------------
ActPlanner — selects the optimal DialogueAct for a speaker's next turn.

Stage 7: the simulator stops deciding what to do; it only realizes a planned act.
The orchestrator calls plan_turn() per selected speaker and passes the resulting
TurnPlan into simulator.generate_turn(turn_plan=...).

Priority cascade (stops at first match):
  1. Discourse obligation    — pending question addressed to speaker → ANSWER
  2. Phase obligation        — greeting → GREET; opening → OPEN_PRIORITY;
                               narrowing+no-vote → COMMIT_VOTE;
                               confirmation → CONFIRM or REJECT_WITH_REASON
  3. Emergence soft-path     — candidate ∈ acceptable + phase=emergence → CONDITIONAL_ACCEPT
  4. Hard-blocker path       — is_true_hard_blocker + candidate ∈ rejected → REJECT_WITH_REASON
  5. Personality-biased      — weighted sampling, phase-specific base rates,
                               PersonalityBias applied, cooldowns honored (MUCA §3.3)

Config gate: cfg.act_planner.enabled  (default: false — A/B safety pattern)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from policy.personality_bias import PersonalityBias, derive as derive_bias
from state.turn import DialogueAct

if TYPE_CHECKING:
    from persona import Persona
    from state.dialogue_state import StructuredState
    from state.participant import ParticipantState
    from orchestrator import DialogueState


# ---------------------------------------------------------------------------
# TurnPlan — the output of plan_turn()
# ---------------------------------------------------------------------------

@dataclass
class TurnPlan:
    """A planned next move for one speaker.  rationale is logged but never sent to the LLM."""
    speaker: str
    act: DialogueAct
    target_option: Optional[str]  = None  # option this act concerns
    addressee: Optional[str]      = None  # name to address (None = open turn)
    reply_to: Optional[int]       = None  # turn_id being replied to
    condition: Optional[str]      = None  # for CONDITIONAL_ACCEPT / REJECT_WITH_REASON
    max_words: int                = 36
    rationale: str                = ""   # logged in *_state.jsonl; not in prompt

    def to_prompt_str(self) -> str:
        """
        Compact one-to-three-line description for the YOUR MOVE block.
        Tells the LLM what move to make without dictating exact phrasing.
        """
        label = _ACT_LABELS.get(self.act, self.act.name)
        line1 = f"Planned act: {label}"
        if self.target_option:
            line1 += f" → Option {self.target_option}"
        parts = [line1]
        if self.condition:
            parts.append(f"  Condition: {self.condition}")
        if self.addressee:
            reply_note = f" (reply to turn {self.reply_to})" if self.reply_to else ""
            parts.append(f"  Address: {self.addressee}{reply_note}")
        return "\n".join(parts)


_ACT_LABELS: dict[DialogueAct, str] = {
    DialogueAct.GREET:              "Greet",
    DialogueAct.OPEN_PRIORITY:      "State opening priority",
    DialogueAct.ASSERT_SUPPORT:     "Express support",
    DialogueAct.ASSERT_OPPOSITION:  "Express opposition",
    DialogueAct.ASSERT_AMBIGUOUS:   "Express uncertainty",
    DialogueAct.ASK_CLARIFICATION:  "Ask for clarification",
    DialogueAct.ASK_PREFERENCE:     "Ask for preference",
    DialogueAct.ANSWER:             "Answer the question",
    DialogueAct.CHALLENGE:          "Challenge a claim",
    DialogueAct.CONCEDE:            "Concede or agree",
    DialogueAct.CONDITIONAL_ACCEPT: "Accept conditionally",
    DialogueAct.PROPOSE_COMPROMISE: "Propose a compromise",
    DialogueAct.COMMIT_VOTE:        "State your explicit vote",
    DialogueAct.CONFIRM:            "Confirm agreement",
    DialogueAct.REJECT_WITH_REASON: "Reject with a specific reason",
    DialogueAct.SUMMARIZE:          "Summarize the discussion so far",
    DialogueAct.GOODBYE:            "Say goodbye",
    DialogueAct.SILENCE:            "Remain silent",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plan_turn(
    speaker_name: str,
    persona: "Persona",
    structured: "StructuredState",
    legacy_state: "DialogueState",
    max_words: int = 36,
) -> TurnPlan:
    """
    Select the best DialogueAct + target for this speaker's next turn.

    Reads StructuredState (Stage 5+) for discourse obligations and participant
    state; reads legacy orchestrator state for phase and candidate option.
    Falls back gracefully when structured state is incomplete.
    """
    phase     = legacy_state.phase
    ps        = structured.participants.get(speaker_name)
    beliefs   = persona.beliefs if persona else None
    bias      = derive_bias(persona)
    candidate = legacy_state.candidate_option or legacy_state.current_leading_option

    # ── 1. Discourse obligation ──────────────────────────────────────────────
    if ps is not None:
        for q_id, addressees in list(structured.discourse.pending_questions.items()):
            if speaker_name in addressees:
                return TurnPlan(
                    speaker=speaker_name,
                    act=DialogueAct.ANSWER,
                    reply_to=q_id,
                    max_words=max_words,
                    rationale=f"pending question turn={q_id} addressed to {speaker_name}",
                )

    # ── 2. Phase obligations ─────────────────────────────────────────────────
    if phase == "greeting":
        return TurnPlan(speaker=speaker_name, act=DialogueAct.GREET,
                        max_words=max_words, rationale="greeting phase")

    if phase == "opening":
        return TurnPlan(speaker=speaker_name, act=DialogueAct.OPEN_PRIORITY,
                        max_words=max_words, rationale="opening phase")

    if phase == "closure":
        return TurnPlan(speaker=speaker_name, act=DialogueAct.GOODBYE,
                        max_words=max_words, rationale="closure phase")

    if phase == "narrowing":
        has_voted = bool(ps.public_preference) if ps else False
        if not has_voted:
            return TurnPlan(
                speaker=speaker_name,
                act=DialogueAct.COMMIT_VOTE,
                target_option=beliefs.preferred if beliefs else None,
                max_words=max_words,
                rationale="narrowing phase: no public vote recorded yet",
            )

    if phase == "confirmation" and candidate:
        rejected = beliefs.rejected if beliefs else []
        if candidate in rejected:
            return TurnPlan(
                speaker=speaker_name,
                act=DialogueAct.REJECT_WITH_REASON,
                target_option=candidate,
                condition=beliefs.concession if beliefs else None,
                max_words=max_words,
                rationale=f"confirmation: candidate {candidate} in beliefs.rejected",
            )
        return TurnPlan(
            speaker=speaker_name,
            act=DialogueAct.CONFIRM,
            target_option=candidate,
            max_words=max_words,
            rationale=f"confirmation: candidate {candidate} acceptable",
        )

    # ── 3. Emergence soft-path ───────────────────────────────────────────────
    if phase == "emergence" and candidate and beliefs:
        acceptable = beliefs.acceptable or []
        if candidate in acceptable and candidate != beliefs.preferred:
            return TurnPlan(
                speaker=speaker_name,
                act=DialogueAct.CONDITIONAL_ACCEPT,
                target_option=candidate,
                condition=beliefs.concession or None,
                max_words=max_words,
                rationale=f"emergence: candidate {candidate} within acceptable range",
            )

    # ── 4. Hard-blocker path ─────────────────────────────────────────────────
    if ps and ps.is_true_hard_blocker and candidate and beliefs:
        if candidate in (beliefs.rejected or []):
            return TurnPlan(
                speaker=speaker_name,
                act=DialogueAct.REJECT_WITH_REASON,
                target_option=candidate,
                condition=beliefs.concession or None,
                max_words=max_words,
                rationale=f"hard blocker: {speaker_name} rejects candidate {candidate}",
            )

    # ── 5. Personality-biased sampling ───────────────────────────────────────
    return _sample_biased_act(
        speaker_name=speaker_name,
        bias=bias,
        ps=ps,
        phase=phase,
        candidate=candidate,
        beliefs=beliefs,
        max_words=max_words,
    )


# ---------------------------------------------------------------------------
# Weighted act sampling
# ---------------------------------------------------------------------------

# Base act weights per phase — personality bias is applied on top of these.
# Tuple: (DialogueAct, base_weight)
_PHASE_BASE_WEIGHTS: dict[str, list[tuple[DialogueAct, float]]] = {
    "negotiation": [
        (DialogueAct.ASSERT_SUPPORT,     0.25),
        (DialogueAct.ASSERT_OPPOSITION,  0.18),
        (DialogueAct.ASK_CLARIFICATION,  0.15),
        (DialogueAct.CONCEDE,            0.12),
        (DialogueAct.PROPOSE_COMPROMISE, 0.10),
        (DialogueAct.ASSERT_AMBIGUOUS,   0.20),
    ],
    "emergence": [
        (DialogueAct.ASSERT_SUPPORT,     0.18),
        (DialogueAct.ASSERT_OPPOSITION,  0.12),
        (DialogueAct.CONCEDE,            0.22),
        (DialogueAct.CONDITIONAL_ACCEPT, 0.28),
        (DialogueAct.PROPOSE_COMPROMISE, 0.12),
        (DialogueAct.ASSERT_AMBIGUOUS,   0.08),
    ],
}


def _sample_biased_act(
    speaker_name: str,
    bias: PersonalityBias,
    ps: Optional["ParticipantState"],
    phase: str,
    candidate: Optional[str],
    beliefs,
    max_words: int,
) -> TurnPlan:
    base = list(_PHASE_BASE_WEIGHTS.get(phase, _PHASE_BASE_WEIGHTS["negotiation"]))
    weighted = _apply_bias(base, bias)

    # Remove acts currently on cooldown (MUCA-style strategy throttle)
    if ps is not None:
        weighted = [(act, w) for act, w in weighted if not ps.on_cooldown(act)]

    chosen = _weighted_choice(weighted) if weighted else DialogueAct.ASSERT_AMBIGUOUS

    # Infer target option from the chosen act
    target: Optional[str] = None
    if chosen in (DialogueAct.ASSERT_SUPPORT, DialogueAct.CONCEDE):
        target = beliefs.preferred if beliefs else candidate
    elif chosen in (DialogueAct.ASSERT_OPPOSITION, DialogueAct.CONDITIONAL_ACCEPT,
                    DialogueAct.CHALLENGE):
        target = candidate

    return TurnPlan(
        speaker=speaker_name,
        act=chosen,
        target_option=target,
        max_words=max_words,
        rationale=f"personality-biased sampling (phase={phase})",
    )


def _apply_bias(
    weights: list[tuple[DialogueAct, float]],
    bias: PersonalityBias,
) -> list[tuple[DialogueAct, float]]:
    """Scale base weights by personality bias multipliers."""
    _multipliers: dict[DialogueAct, float] = {
        DialogueAct.CONCEDE:            0.5 + bias.concession_propensity,
        DialogueAct.ASSERT_OPPOSITION:  0.5 + bias.objection_propensity,
        DialogueAct.ASK_CLARIFICATION:  0.5 + bias.clarification_propensity,
        DialogueAct.PROPOSE_COMPROMISE: 0.5 + bias.reframing_propensity,
        DialogueAct.ASSERT_SUPPORT:     0.5 + bias.detail_orientation,
        DialogueAct.CONDITIONAL_ACCEPT: 0.5 + bias.concession_propensity * 0.7,
        DialogueAct.CHALLENGE:          0.5 + bias.objection_propensity * 0.6,
    }
    return [
        (act, max(0.01, w * _multipliers.get(act, 1.0)))
        for act, w in weights
    ]


def _weighted_choice(weights: list[tuple[DialogueAct, float]]) -> DialogueAct:
    total = sum(w for _, w in weights)
    r = random.uniform(0.0, total)
    cumulative = 0.0
    for act, w in weights:
        cumulative += w
        if cumulative >= r:
            return act
    return weights[-1][0]
