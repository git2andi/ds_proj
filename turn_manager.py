"""
turn_manager.py
---------------
TurnManager — speaker selection and repetition pressure tracking.

Simplified from the original:
- Scoring uses ~6 factors instead of ~25.
- Discourse event extraction is removed; the orchestrator handles those needs directly.
- EXCLUDED_SPEAKERS sourced from config_loader.
"""

from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator


class TurnManager:

    # ------------------------------------------------------------------
    # Speaker selection
    # ------------------------------------------------------------------

    def select_speakers(
        self,
        sims: list["Simulator"],
        history: list[str],
        state: "DialogueState",
        max_speakers: int = 2,
    ) -> list["Simulator"]:
        """Return up to max_speakers sims, weighted by score."""
        if not sims:
            return []

        # If a specific participant was addressed or asked a question, force them first.
        forced = self._forced_speakers(sims, state)
        if forced:
            best = max(forced, key=lambda s: self._score(s, history, state))
            remaining = [s for s in sims if s is not best]
            selected = [best]
            if max_speakers > 1 and remaining:
                selected += self._weighted_pick(remaining, history, state, count=max_speakers - 1)
            return selected[:max_speakers]

        return self._weighted_pick(sims, history, state, count=max_speakers)

    def _forced_speakers(
        self, sims: list["Simulator"], state: "DialogueState"
    ) -> list["Simulator"]:
        forced_names = {state.last_addressed, state.pending_question_target}
        forced_names.discard(None)
        return [s for s in sims if s.name in forced_names]

    def _score(
        self, sim: "Simulator", history: list[str], state: "DialogueState"
    ) -> float:
        score = 0.0

        # Discourse obligations
        if state.last_addressed == sim.name:
            score += 0.80
        if state.pending_question_target == sim.name:
            score += 0.90

        # Persona traits
        score += 0.12 * _norm(sim.persona.talkativeness)
        score += 0.10 * _norm(sim.persona.assertiveness)
        score += 0.08 * _norm(sim.persona.agreeableness)

        if sim.persona.is_primary:
            score += 0.10

        # Speaking balance — penalise recent speakers
        if self._last_participant_speaker(history) == sim.name:
            score -= 0.50
        recent_count = sum(
            1 for s in self._recent_speakers(history, n=4) if s == sim.name
        )
        score -= 0.12 * recent_count

        # Boost participants who haven't spoken yet
        if not self._has_spoken(sim.name, history):
            score += 0.20

        return max(0.01, score)

    def _weighted_pick(
        self,
        sims: list["Simulator"],
        history: list[str],
        state: "DialogueState",
        count: int = 1,
    ) -> list["Simulator"]:
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

    # ------------------------------------------------------------------
    # Repetition pressure
    # ------------------------------------------------------------------

    def repetition_pressure(self, history: list[str]) -> float:
        """
        Returns 0.0–1.0 based on vocabulary overlap in recent participant messages.
        High values indicate the discussion is going in circles.
        """
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

    # ------------------------------------------------------------------
    # Discourse extraction — simplified
    # ------------------------------------------------------------------

    def extract_discourse(self, history: list[str], sim_names: set[str]) -> dict:
        """
        Scan the most recent participant line for:
        - who was addressed by name
        - whether a question was asked (and at whom)
        Returns a dict with keys: last_addressed, pending_question_target.
        """
        result = {"last_addressed": None, "pending_question_target": None}
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in cfg.EXCLUDED_SPEAKERS:
                continue

            import re
            for name in sim_names:
                if name != speaker and re.search(rf"\b{re.escape(name)}\b", msg, re.IGNORECASE):
                    result["last_addressed"] = name
                    if "?" in msg:
                        result["pending_question_target"] = name
            break  # only the most recent participant line matters

        return result

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _recent_speakers(self, history: list[str], n: int = 4) -> list[str]:
        speakers: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker = line.split(":", 1)[0].strip()
            if speaker in cfg.EXCLUDED_SPEAKERS:
                continue
            speakers.append(speaker)
            if len(speakers) >= n:
                break
        return speakers

    def _last_participant_speaker(self, history: list[str]) -> Optional[str]:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker = line.split(":", 1)[0].strip()
            if speaker not in cfg.EXCLUDED_SPEAKERS:
                return speaker
        return None

    def _has_spoken(self, name: str, history: list[str]) -> bool:
        return any(line.startswith(f"{name}:") for line in history)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _norm(value: int) -> float:
    return (max(1, min(5, int(value))) - 1) / 4.0
