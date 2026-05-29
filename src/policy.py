"""
policy.py
---------
Speaker selection (SSJ cascade), repetition / discourse signals.

Public API
  - select_next_speakers()  : SSJ priority cascade, returns ONE speaker per round
                              (Sacks/Schegloff/Jefferson 1974).
  - sample_hard_blocker()   : rare per-dialogue stubbornness sampler.
  - repetition_pressure()   : rolling Jaccard overlap signal.
  - extract_discourse()     : last-addressed + pending-question target (Ouchi/Tsuboi).
"""

from __future__ import annotations

import random
import re
from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from state import DiscourseGraph, ParticipantState, StructuredState
    from simulator import Simulator
    from orchestrator import DialogueState


# =============================================================================
# Hard-blocker sampling -- rare, at dialogue start
# =============================================================================

def sample_hard_blocker(participant_states: list["ParticipantState"]) -> Optional[str]:
    """With cfg.stubbornness.hard_blocker_dialogue_probability, flag ONE sim."""
    if random.random() >= cfg.stubbornness.hard_blocker_dialogue_probability:
        return None
    candidates = [
        ps for ps in participant_states
        if ps.persona_ref and ps.persona_ref.beliefs and ps.persona_ref.beliefs.rejected
    ] or list(participant_states)
    if not candidates:
        return None
    chosen = random.choice(candidates)
    chosen.is_true_hard_blocker = True
    return chosen.name


# =============================================================================
# Speaker selection -- SSJ rule cascade (Sacks/Schegloff/Jefferson 1974)
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


def _recent_speakers(history: list[str], n: int) -> list[str]:
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
) -> list["Simulator"]:
    """SSJ rule cascade. Returns a single speaker (one-at-a-time)."""
    if not sims:
        return []

    # Opening covers everyone once before repeats.
    if state.phase == "opening":
        missing = [s for s in sims if _turn_count_for(s.name, history) == 0]
        sims = missing or sims

    if state.phase == "confirmation":
        ordered = sorted(sims, key=lambda s: (0 if s.persona.is_primary else 1))
        return [ordered[0]]

    # Rule 1a -- obligated addressees of pending questions (directed or open).
    if discourse is not None:
        obligated = [s for s in sims if discourse.has_obligation_for(s.name)]
        if obligated:
            oldest = discourse.oldest_pending_addressees()
            ban = {n[1:] for n in oldest if isinstance(n, str) and n.startswith("!")}
            named = {n for n in oldest if isinstance(n, str) and not n.startswith("!")}
            if named:
                top = [s for s in obligated if s.name in named]
            else:
                top = [s for s in obligated if s.name not in ban]
            top = top or obligated
            return [top[0]]

    # Rule 1a' -- recently name-mentioned (no question pending).
    if state.last_addressed and not state.pending_question_target:
        addr = [s for s in sims if s.name == state.last_addressed]
        if addr:
            return [addr[0]]

    # Rule 1b -- self-selection. Hard-exclude the last speaker so two
    # consecutive turns from the same sim are impossible under self-select.
    return [_self_select(sims, history, state)]


def _self_select(sims: list["Simulator"], history: list[str],
                 state: "DialogueState") -> "Simulator":
    last = _last_participant_speaker(history)
    pool = [s for s in sims if s.name != last] or sims  # never empty
    scored = [(_score(s, history, state), s) for s in pool]
    total = sum(sc for sc, _ in scored)
    if total <= 0:
        return random.choice(pool)
    r = random.uniform(0, total)
    upto = 0.0
    for sc, s in scored:
        upto += sc
        if upto >= r:
            return s
    return pool[-1]


def _score(sim: "Simulator", history: list[str], state: "DialogueState") -> float:
    w = cfg.turn_policy.weights
    p = cfg.turn_policy.penalties
    score = 0.0

    score += w.extraversion * _norm(sim.persona.extraversion)

    if state.phase in {"negotiation", "emergence"}:
        score += w.openness_negotiation * _norm(sim.persona.openness)
        score += w.conscientiousness_negotiation * _norm(sim.persona.conscientiousness)

    if state.phase != "opening":
        score += w.neuroticism_pressure * _norm(sim.persona.neuroticism) * state.repetition_pressure

    score -= w.agreeableness_off * _norm(sim.persona.agreeableness)

    if sim.persona.is_primary:
        score += w.primary_boost

    window = cfg.turn_policy.recent_speaker_window
    recent = _recent_speakers(history, n=window)
    if not _has_spoken(sim.name, history):
        score += w.unspoken_boost
    elif sim.name not in recent:
        score += w.unspoken_boost * w.unspoken_recent_factor

    if _last_participant_speaker(history) == sim.name:
        score -= p.last_speaker
    recent_count = sum(1 for n in recent if n == sim.name)
    score -= p.recent_speaker_per_turn * recent_count

    if sim.persona.extraversion <= cfg.turn_policy.introvert_threshold:
        score -= p.introvert_off_turn

    if _own_recent_repetition(sim.name, history):
        score -= p.own_repetition

    return max(cfg.turn_policy.min_score_floor, score)


# =============================================================================
# Repetition pressure + discourse extraction
# =============================================================================

def repetition_pressure(history: list[str]) -> float:
    """Single Jaccard-overlap signal over the last `pressure_window` participant turns."""
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
    """Addressee tracking.

    Explicit names win. If a participant asks an unnamed question that repeats a
    keyword from the immediately previous participant turn, route it to that
    previous speaker. Otherwise leave it open; open participant questions no
    longer force a random answer.
    """
    from state import _is_addressing, _CHALLENGE_MARKERS  # avoid module-load cycle

    result: dict[str, Optional[str]] = {"last_addressed": None, "pending_question_target": None}
    participant_lines: list[tuple[str, str]] = []
    for line in history:
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        speaker = speaker.strip()
        if speaker not in cfg.EXCLUDED_SPEAKERS:
            participant_lines.append((speaker, msg.strip()))

    if not participant_lines:
        return result

    speaker, msg = participant_lines[-1]
    is_question = "?" in msg
    has_challenge = bool(_CHALLENGE_MARKERS.search(msg))
    for name in sim_names:
        if name == speaker:
            continue
        if _is_addressing(name, msg, is_question=is_question, has_challenge=has_challenge):
            result["last_addressed"] = name
            if is_question:
                result["pending_question_target"] = name
            return result

    if is_question and len(participant_lines) >= 2:
        prev_speaker, prev_msg = participant_lines[-2]
        if prev_speaker != speaker and _shares_content_keyword(msg, prev_msg):
            result["last_addressed"] = prev_speaker
            result["pending_question_target"] = prev_speaker
    return result


def _shares_content_keyword(a: str, b: str) -> bool:
    stop = {
        "what", "about", "which", "option", "important", "choice", "really",
        "still", "think", "would", "could", "should", "there", "their", "your",
        "with", "that", "this", "from", "have", "does", "need", "want", "like",
        "nice", "spot", "place", "decision", "best", "good", "lean", "toward",
    }
    def toks(x: str) -> set[str]:
        return {
            re.sub(r"[^a-z0-9]", "", w.lower())
            for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", x)
            if re.sub(r"[^a-z0-9]", "", w.lower()) not in stop
        }
    return bool(toks(a) & toks(b))
