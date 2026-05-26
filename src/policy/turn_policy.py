"""
policy/turn_policy.py
----------------------
SSJTurnPolicy — speaker selection via the Sacks/Schegloff/Jefferson (1974) rule cascade.

Replaces TurnManager.select_speakers() when cfg.turn_policy.use_legacy = false.
The old TurnManager remains available and is still used during the A/B transition.

SSJ priority order (strict, stop at first tier that fires):
  1a. Obligated addressees of pending questions  (DiscourseGraph.pending_questions)
  1a'. Recently name-mentioned participant without a question
  1b. Self-selection with Big-Five personality bias
  1c. Current speaker continues OR moderator intervenes (handled by orchestrator)

Key difference from TurnManager's weighted-sum approach:
  - Discourse obligations are MANDATORY (not just high-weight), grounded in SSJ 1a
  - Q→A pair tracking persists via DiscourseGraph until answered (not one-shot)
  - Self-selection uses the same config weights as TurnManager for compatibility

Acceptance criterion (Stage 6):
  question_answer_rate >= 85% in the new path vs. baseline.
"""

from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator
    from state.discourse import DiscourseGraph


# ---------------------------------------------------------------------------
# Main policy class
# ---------------------------------------------------------------------------

class SSJTurnPolicy:
    """SSJ rule-cascade speaker selector."""

    def select_next_speakers(
        self,
        sims: list["Simulator"],
        history: list[str],
        state: "DialogueState",
        discourse: Optional["DiscourseGraph"],
        max_speakers: int = 2,
    ) -> list["Simulator"]:
        """
        Return up to max_speakers sims following the SSJ priority cascade.
        Falls back gracefully if discourse graph is not yet available.
        """
        if not sims:
            return []

        # Phase-specific round-robin (greeting / opening / closure)
        round_limited = self._phase_round_candidates(sims, history, state)
        if round_limited:
            sims = round_limited

        # Confirmation phase: ensure everyone gets one shot (handled here, not in cascade)
        if state.phase == "confirmation":
            return self._confirmation_candidates(sims, history, max_speakers)

        # ── Rule 1a — obligated addressees of pending questions ────────────
        if discourse is not None:
            obligated = [
                s for s in sims
                if discourse.has_obligation_for(s.name)
            ]
            if obligated:
                # Priority among obligated: prefer those in oldest pending question
                oldest = discourse.oldest_pending_addressees()
                top = [s for s in obligated if s.name in oldest]
                if not top:
                    top = obligated
                return self._fill_to(top, sims, history, state, max_speakers)

        # ── Rule 1a' — recently name-mentioned (no question pending) ───────
        # Use state.last_addressed (set by _update_discourse via TurnManager helper)
        last_addr = state.last_addressed
        last_q_target = state.pending_question_target  # may duplicate Rule 1a
        if last_addr and not last_q_target:
            addr_sims = [s for s in sims if s.name == last_addr]
            if addr_sims:
                return self._fill_to(addr_sims, sims, history, state, max_speakers)

        # ── Rule 1b — self-selection ────────────────────────────────────────
        return self._self_select(sims, history, state, count=max_speakers)

    # ------------------------------------------------------------------
    # Phase-specific candidacy restrictions
    # ------------------------------------------------------------------

    def _phase_round_candidates(
        self, sims: list["Simulator"], history: list[str], state: "DialogueState"
    ) -> list["Simulator"]:
        """During greeting/opening, prefer participants who haven't had their round yet."""
        if state.phase == "greeting":
            missing = [s for s in sims if _turn_count_for(s.name, history) == 0]
            return missing or sims
        if state.phase == "opening":
            missing = [s for s in sims if _turn_count_for(s.name, history) < 2]
            return missing or sims
        return []  # no restriction

    def _confirmation_candidates(
        self, sims: list["Simulator"], history: list[str], max_speakers: int
    ) -> list["Simulator"]:
        """Closure/confirmation: primary first, then others. All get one shot."""
        ordered = sorted(sims, key=lambda s: (0 if s.persona.is_primary else 1))
        return ordered[:max_speakers]

    # ------------------------------------------------------------------
    # Self-selection (Rule 1b)
    # ------------------------------------------------------------------

    def _self_select(
        self,
        sims: list["Simulator"],
        history: list[str],
        state: "DialogueState",
        count: int = 1,
    ) -> list["Simulator"]:
        """Weighted sample using personality bias scores (same weights as TurnManager)."""
        available = list(sims)
        picked: list["Simulator"] = []

        for _ in range(min(count, len(available))):
            scored = [(self._score(s, history, state), s) for s in available]
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

    def _score(
        self, sim: "Simulator", history: list[str], state: "DialogueState"
    ) -> float:
        """
        Big-Five personality bias score for self-selection probability.
        Uses the same config weights as TurnManager for a clean A/B comparison.
        """
        w = cfg.turn_policy.weights
        p = cfg.turn_policy.penalties
        score = 0.0

        # Trait-based propensities
        score += w.extraversion * _norm(sim.persona.extraversion)

        if state.phase in {"negotiation", "emergence"}:
            score += w.openness_negotiation * _norm(sim.persona.openness)
            score += w.conscientiousness_negotiation * _norm(sim.persona.conscientiousness)

        if state.phase not in {"greeting", "opening"}:
            score += w.neuroticism_pressure * _norm(sim.persona.neuroticism) * state.repetition_pressure

        # Agreeableness off-turn: high-A sims yield the floor unless addressed
        score -= w.agreeableness_off * _norm(sim.persona.agreeableness)

        # Primary participant slight boost
        if sim.persona.is_primary:
            score += w.primary_boost

        # Novelty: boost sims who haven't spoken yet
        if not _has_spoken(sim.name, history):
            score += w.unspoken_boost

        # Participation balance: favour less-recently-active sims
        if _last_participant_speaker(history) == sim.name:
            score -= p.last_speaker
        recent_count = sum(1 for n in _recent_speakers(history, n=4) if n == sim.name)
        score -= p.recent_speaker_per_turn * recent_count

        # Introvert off-turn penalty
        if sim.persona.extraversion <= 2:
            score -= p.introvert_off_turn

        # Repetition penalty: don't boost a sim that just repeated itself
        if _own_recent_repetition(sim.name, history):
            score -= p.own_repetition

        return max(0.01, score)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fill_to(
        self,
        priority: list["Simulator"],
        all_sims: list["Simulator"],
        history: list[str],
        state: "DialogueState",
        max_speakers: int,
    ) -> list["Simulator"]:
        """Return priority sims first, then fill remaining slots with self-select."""
        result = list(priority[:max_speakers])
        if len(result) < max_speakers:
            rest = [s for s in all_sims if s not in result]
            extras = self._self_select(rest, history, state, count=max_speakers - len(result))
            result.extend(extras)
        return result[:max_speakers]


# ---------------------------------------------------------------------------
# History-scanning utilities (stateless, mirrored from TurnManager)
# ---------------------------------------------------------------------------

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
    import re
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
