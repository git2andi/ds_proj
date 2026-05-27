"""
policy.py
---------
Speaker selection (SSJ) + per-turn act planning + personality bias derivation.
Merged from the former policy/ subpackage (turn_policy, act_planner,
personality_bias, stubbornness).

Public API:
  - select_next_speakers()   speaker selection cascade (SSJ 1a..1c)
  - plan_turn() -> TurnPlan  per-speaker act selection
  - derive_bias()            Big-Five -> probabilistic floats
  - sample_hard_blockers()   rare latent-trait sampler at dialogue start
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from config_loader import cfg
from state import DialogueAct

if TYPE_CHECKING:
    from persona import Persona
    from state import DiscourseGraph, ParticipantState, StructuredState
    from simulator import Simulator
    from orchestrator import DialogueState


# =============================================================================
# Personality bias (McCrae & John 1992 -- orthogonal factor model)
# =============================================================================

@dataclass
class PersonalityBias:
    talkativeness: float
    self_selection_propensity: float
    concession_propensity: float
    objection_propensity: float
    risk_salience: float
    detail_orientation: float
    consistency: float
    reframing_propensity: float
    clarification_propensity: float


def derive_bias(persona: "Persona") -> PersonalityBias:
    def n(v: int) -> float:
        return (max(1, min(5, int(v))) - 1) / 4.0

    e = n(persona.extraversion)
    a = n(persona.agreeableness)
    o = n(persona.openness)
    c = n(persona.conscientiousness)
    neuro = n(persona.neuroticism)

    return PersonalityBias(
        talkativeness             = e,
        self_selection_propensity = e * 0.6 + o * 0.2 + a * 0.2,
        concession_propensity     = a * 0.7 + (1.0 - neuro) * 0.3,
        objection_propensity      = (1.0 - a) * 0.6 + neuro * 0.4,
        risk_salience             = neuro,
        detail_orientation        = c,
        consistency               = c * 0.6 + (1.0 - o) * 0.4,
        reframing_propensity      = o,
        clarification_propensity  = neuro * 0.5 + o * 0.3 + (1.0 - c) * 0.2,
    )


# =============================================================================
# Hard-blocker sampling (rare, at dialogue start)
# =============================================================================

def sample_hard_blockers(participant_states: list["ParticipantState"]) -> list[str]:
    prob = cfg.stubbornness.hard_blocker_dialogue_probability
    max_blockers = cfg.stubbornness.max_hard_blockers_per_dialogue

    if random.random() >= prob:
        return []

    candidates_with_rejection = [
        ps for ps in participant_states
        if ps.persona_ref and ps.persona_ref.beliefs and ps.persona_ref.beliefs.rejected
    ]
    candidates = candidates_with_rejection or list(participant_states)
    chosen = random.sample(candidates, min(max_blockers, len(candidates)))
    for ps in chosen:
        ps.is_true_hard_blocker = True
    return [ps.name for ps in chosen]


# =============================================================================
# Speaker selection -- SSJ rule cascade
# =============================================================================

def _norm(value: int) -> float:
    return (max(1, min(5, int(value))) - 1) / 4.0


def _turn_count_for(name: str, history: list[str]) -> int:
    return sum(1 for line in history if line.startswith(f"{name}:"))


def _has_spoken(name: str, history: list[str]) -> bool:
    return any(line.startswith(f"{name}:") for line in history)


def _last_participant_speaker(history: list[str]) -> Optional[str]:
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker = line.split(":", 1)[0].strip()
        if speaker not in cfg.EXCLUDED_SPEAKERS:
            return speaker
    return None


def _recent_speakers(history: list[str], n: int = 4) -> list[str]:
    speakers: list[str] = []
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker = line.split(":", 1)[0].strip()
        if speaker not in cfg.EXCLUDED_SPEAKERS:
            speakers.append(speaker)
            if len(speakers) >= n:
                break
    return speakers


def _own_recent_repetition(name: str, history: list[str]) -> bool:
    turns: list[str] = []
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        if speaker.strip() == name:
            turns.append(msg.strip().lower())
            if len(turns) >= 2:
                break
    if len(turns) < 2:
        return False
    a = set(re.sub(r"[^\w\s]", "", turns[0]).split())
    b = set(re.sub(r"[^\w\s]", "", turns[1]).split())
    if not a or not b:
        return False
    return len(a & b) / max(1, min(len(a), len(b))) >= cfg.repetition.jaccard_threshold_self


def select_next_speakers(
    sims: list["Simulator"],
    history: list[str],
    state: "DialogueState",
    discourse: Optional["DiscourseGraph"],
    max_speakers: int = 2,
) -> list["Simulator"]:
    """SSJ rule cascade. Returns up to max_speakers Simulators."""
    if not sims:
        return []

    if state.phase == "greeting":
        missing = [s for s in sims if _turn_count_for(s.name, history) == 0]
        sims = missing or sims
    elif state.phase == "opening":
        missing = [s for s in sims if _turn_count_for(s.name, history) < 2]
        sims = missing or sims

    if state.phase == "confirmation":
        ordered = sorted(sims, key=lambda s: (0 if s.persona.is_primary else 1))
        return ordered[:max_speakers]

    # Rule 1a -- obligated addressees of pending questions (directed or open)
    if discourse is not None:
        obligated = [s for s in sims if discourse.has_obligation_for(s.name)]
        if obligated:
            oldest = discourse.oldest_pending_addressees()
            # Sentinel "!asker" means anyone-but-asker (open invitation)
            ban = {n[1:] for n in oldest if isinstance(n, str) and n.startswith("!")}
            named = {n for n in oldest if isinstance(n, str) and not n.startswith("!")}
            if named:
                top = [s for s in obligated if s.name in named]
            else:
                top = [s for s in obligated if s.name not in ban]
            top = top or obligated
            return _fill_to(top, sims, history, state, max_speakers)

    # Rule 1a' -- recently name-mentioned (no question pending)
    if state.last_addressed and not state.pending_question_target:
        addr = [s for s in sims if s.name == state.last_addressed]
        if addr:
            return _fill_to(addr, sims, history, state, max_speakers)

    # Rule 1b -- self-selection
    return _self_select(sims, history, state, count=max_speakers)


def _fill_to(priority: list["Simulator"], all_sims: list["Simulator"],
             history: list[str], state: "DialogueState",
             max_speakers: int) -> list["Simulator"]:
    result = list(priority[:max_speakers])
    if len(result) < max_speakers:
        rest = [s for s in all_sims if s not in result]
        result.extend(_self_select(rest, history, state, count=max_speakers - len(result)))
    return result[:max_speakers]


def _self_select(sims: list["Simulator"], history: list[str],
                 state: "DialogueState", count: int = 1) -> list["Simulator"]:
    available = list(sims)
    picked: list["Simulator"] = []

    for _ in range(min(count, len(available))):
        scored = [(_score(s, history, state), s) for s in available]
        total = sum(sc for sc, _ in scored)
        if total <= 0:
            choice = random.choice(available)
        else:
            r = random.uniform(0, total)
            upto = 0.0
            choice = available[0]
            for sc, s in scored:
                upto += sc
                if upto >= r:
                    choice = s
                    break
        picked.append(choice)
        available.remove(choice)

    return picked


def _score(sim: "Simulator", history: list[str], state: "DialogueState") -> float:
    w = cfg.turn_policy.weights
    p = cfg.turn_policy.penalties
    score = 0.0

    score += w.extraversion * _norm(sim.persona.extraversion)

    if state.phase in {"negotiation", "emergence"}:
        score += w.openness_negotiation * _norm(sim.persona.openness)
        score += w.conscientiousness_negotiation * _norm(sim.persona.conscientiousness)

    if state.phase not in {"greeting", "opening"}:
        score += w.neuroticism_pressure * _norm(sim.persona.neuroticism) * state.repetition_pressure

    score -= w.agreeableness_off * _norm(sim.persona.agreeableness)

    if sim.persona.is_primary:
        score += w.primary_boost

    recent_6 = _recent_speakers(history, n=6)
    if not _has_spoken(sim.name, history):
        score += w.unspoken_boost
    elif sim.name not in recent_6:
        # Has spoken before but has been silent in the last 6 turns — pull them back in.
        score += w.unspoken_boost * 0.70

    if _last_participant_speaker(history) == sim.name:
        score -= p.last_speaker
    recent_count = sum(1 for n in recent_6 if n == sim.name)
    score -= p.recent_speaker_per_turn * recent_count

    if sim.persona.extraversion <= 2:
        score -= p.introvert_off_turn

    if _own_recent_repetition(sim.name, history):
        score -= p.own_repetition

    return max(0.01, score)


# =============================================================================
# Repetition + discourse extraction (used by orchestrator)
# =============================================================================

def repetition_pressure(history: list[str]) -> float:
    window = cfg.repetition.pressure_window
    min_len = cfg.repetition.min_word_length
    texts: list[str] = []
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
            continue
        texts.append(msg.strip().lower())
        if len(texts) >= window:
            break

    if len(texts) < 3:
        return 0.0

    token_sets = [
        {w.strip(".,!?;:'\"()[]{}") for w in t.split() if len(w) > min_len}
        for t in texts
    ]
    overlaps: list[float] = []
    for i in range(len(token_sets) - 1):
        a, b = token_sets[i], token_sets[i + 1]
        if a and b:
            overlaps.append(len(a & b) / max(1, min(len(a), len(b))))
        else:
            overlaps.append(0.0)
    return max(0.0, min(1.0, sum(overlaps) / len(overlaps))) if overlaps else 0.0


def extract_discourse(history: list[str], sim_names: set[str]) -> dict[str, Optional[str]]:
    """Last addressed + pending question target from the most recent participant line."""
    result: dict[str, Optional[str]] = {"last_addressed": None, "pending_question_target": None}
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        speaker = speaker.strip()
        if speaker in cfg.EXCLUDED_SPEAKERS:
            continue
        for name in sim_names:
            if name != speaker and re.search(rf"\b{re.escape(name)}\b", msg, re.IGNORECASE):
                result["last_addressed"] = name
                if "?" in msg:
                    result["pending_question_target"] = name
        break
    return result


# =============================================================================
# Act planner -- TurnPlan + plan_turn()
# =============================================================================

@dataclass
class TurnPlan:
    speaker: str
    act: DialogueAct
    target_option: Optional[str]  = None
    addressee: Optional[str]      = None
    reply_to: Optional[int]       = None
    condition: Optional[str]      = None
    max_words: int                = 24
    rationale: str                = ""

    def to_prompt_str(self) -> str:
        label = _ACT_LABELS.get(self.act, self.act.name)
        line1 = f"Planned act: {label}"
        if self.target_option:
            line1 += f" -> Option {self.target_option}"
        parts = [line1]
        if self.condition:
            parts.append(f"  Condition: {self.condition}")
        if self.addressee:
            parts.append(f"  Address: {self.addressee}")
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


_PHASE_BASE_WEIGHTS: dict[str, list[tuple[DialogueAct, float]]] = {
    "negotiation": [
        (DialogueAct.ASSERT_SUPPORT,     0.10),  # reduced -- don't just restate position
        (DialogueAct.ASSERT_OPPOSITION,  0.16),
        (DialogueAct.ASK_CLARIFICATION,  0.10),
        (DialogueAct.CHALLENGE,          0.22),  # push back on what others actually said
        (DialogueAct.CONCEDE,            0.16),  # show real flexibility
        (DialogueAct.PROPOSE_COMPROMISE, 0.10),
        (DialogueAct.ASSERT_AMBIGUOUS,   0.10),  # reduced
        (DialogueAct.ANSWER,             0.06),  # follow-up on unanswered points
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


def plan_turn(
    speaker_name: str,
    persona: "Persona",
    structured: "StructuredState",
    legacy_state: "DialogueState",
    max_words: int = 24,
) -> TurnPlan:
    phase     = legacy_state.phase
    ps        = structured.participants.get(speaker_name)
    beliefs   = persona.beliefs if persona else None
    bias      = derive_bias(persona)
    candidate = legacy_state.candidate_option or legacy_state.current_leading_option

    # 1. Discourse obligation
    if ps is not None:
        for q_id, addressees in list(structured.discourse.pending_questions.items()):
            if speaker_name in addressees:
                return TurnPlan(
                    speaker=speaker_name, act=DialogueAct.ANSWER,
                    reply_to=q_id, max_words=max_words,
                    rationale=f"pending question turn={q_id}",
                )

    # 2. Phase obligations
    if phase == "greeting":
        return TurnPlan(speaker=speaker_name, act=DialogueAct.GREET, max_words=max_words)
    if phase == "opening":
        return TurnPlan(speaker=speaker_name, act=DialogueAct.OPEN_PRIORITY, max_words=max_words)
    if phase == "closure":
        return TurnPlan(speaker=speaker_name, act=DialogueAct.GOODBYE, max_words=max_words)

    if phase == "narrowing":
        has_voted = bool(ps.public_preference) if ps else False
        if not has_voted:
            return TurnPlan(
                speaker=speaker_name, act=DialogueAct.COMMIT_VOTE,
                target_option=beliefs.preferred if beliefs else None,
                max_words=max_words,
                rationale="narrowing: no public vote yet",
            )

    if phase == "confirmation" and candidate:
        rejected = beliefs.rejected if beliefs else []
        if candidate in rejected:
            return TurnPlan(
                speaker=speaker_name, act=DialogueAct.REJECT_WITH_REASON,
                target_option=candidate, condition=beliefs.concession if beliefs else None,
                max_words=max_words,
                rationale=f"confirmation: {candidate} in rejected",
            )
        return TurnPlan(
            speaker=speaker_name, act=DialogueAct.CONFIRM,
            target_option=candidate, max_words=max_words,
            rationale=f"confirmation: {candidate} acceptable",
        )

    # 3. Emergence soft-path
    if phase == "emergence" and candidate and beliefs:
        if candidate in (beliefs.acceptable or []) and candidate != beliefs.preferred:
            return TurnPlan(
                speaker=speaker_name, act=DialogueAct.CONDITIONAL_ACCEPT,
                target_option=candidate, condition=beliefs.concession or None,
                max_words=max_words,
                rationale=f"emergence: {candidate} within acceptable",
            )

    # 4. Hard-blocker path
    if ps and ps.is_true_hard_blocker and candidate and beliefs and candidate in (beliefs.rejected or []):
        return TurnPlan(
            speaker=speaker_name, act=DialogueAct.REJECT_WITH_REASON,
            target_option=candidate, condition=beliefs.concession or None,
            max_words=max_words,
            rationale="hard blocker rejects candidate",
        )

    # 5. Personality-biased sampling
    base = list(_PHASE_BASE_WEIGHTS.get(phase, _PHASE_BASE_WEIGHTS["negotiation"]))
    multipliers: dict[DialogueAct, float] = {
        DialogueAct.CONCEDE:            0.5 + bias.concession_propensity,
        DialogueAct.ASSERT_OPPOSITION:  0.5 + bias.objection_propensity,
        # CHALLENGE: objection-driven but also high-openness sims like to probe claims
        DialogueAct.CHALLENGE:          0.5 + bias.objection_propensity * 0.6 + bias.reframing_propensity * 0.4,
        DialogueAct.ASK_CLARIFICATION:  0.5 + bias.clarification_propensity,
        DialogueAct.PROPOSE_COMPROMISE: 0.5 + bias.reframing_propensity,
        DialogueAct.ASSERT_SUPPORT:     0.5 + bias.detail_orientation,
        DialogueAct.CONDITIONAL_ACCEPT: 0.5 + bias.concession_propensity * 0.7,
    }
    weighted = [(act, max(0.01, w * multipliers.get(act, 1.0))) for act, w in base]
    if ps is not None:
        weighted = [(act, w) for act, w in weighted if not ps.on_cooldown(act)]

    chosen = _weighted_choice(weighted) if weighted else DialogueAct.ASSERT_AMBIGUOUS

    target: Optional[str] = None
    addressee: Optional[str] = None

    if chosen in (DialogueAct.ASSERT_SUPPORT, DialogueAct.CONCEDE):
        target = beliefs.preferred if beliefs else candidate
    elif chosen in (DialogueAct.ASSERT_OPPOSITION, DialogueAct.CONDITIONAL_ACCEPT, DialogueAct.CHALLENGE):
        target = candidate

    if chosen == DialogueAct.CHALLENGE:
        # Direct the challenge at the last other speaker and their option claim
        last_other = next(
            (t for t in reversed(structured.turns)
             if not t.is_moderator and t.speaker != speaker_name),
            None,
        )
        if last_other:
            addressee = last_other.speaker
            if not target and last_other.mentioned_options:
                target = last_other.mentioned_options[0]

    return TurnPlan(
        speaker=speaker_name, act=chosen, target_option=target,
        addressee=addressee,
        max_words=max_words, rationale=f"sampled act in {phase}",
    )


def _weighted_choice(weights: list[tuple[DialogueAct, float]]) -> DialogueAct:
    total = sum(w for _, w in weights)
    r = random.uniform(0.0, total)
    cumulative = 0.0
    for act, w in weights:
        cumulative += w
        if cumulative >= r:
            return act
    return weights[-1][0]
