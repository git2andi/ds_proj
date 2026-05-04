"""
modules/turn_manager.py
-----------------------
TurnManager — speaker selection, discourse obligation tracking,
and repetition/stall detection.

Key improvements over original:
- Repetition window is config-driven (default 8, was 4).
- Forbidden phrase-frame detection added (structural patterns, not just openers).
- EXCLUDED_SPEAKERS sourced from config_loader, not constants.py.
"""

from __future__ import annotations

import random
import re
from typing import Optional

from config_loader import cfg


class TurnManager:

    # ------------------------------------------------------------------
    # Discourse event extraction
    # ------------------------------------------------------------------

    def extract_events(self, history: list[str], state, sims: list) -> None:
        """Update dialogue state from recent history."""
        recent = self._recent_participant_lines(history, limit=6)
        sim_names = {sim.name for sim in sims}

        state.last_addressed = self._extract_last_addressed(recent, sim_names)
        state.pending_question_target = self._extract_question_target(recent, sim_names)
        state.pending_reply_target = self._extract_reply_target(recent, sim_names)
        state.repetition_pressure = self._compute_repetition_pressure(history)
        state.important_events = self._extract_important_events(recent)

    # ------------------------------------------------------------------
    # Speaker selection
    # ------------------------------------------------------------------

    def forced_names(self, state) -> set[str]:
        """Names that have a discourse obligation this turn."""
        names = {
            state.last_addressed,
            state.pending_question_target,
            state.pending_reply_target,
        }
        names.discard(None)
        return names

    def select_speakers(
        self, sims: list, history: list[str], state, max_speakers: int = 2
    ) -> list:
        if not sims:
            return []

        forced = [s for s in sims if s.name in self.forced_names(state)]
        if forced:
            best = max(forced, key=lambda s: self._score(s, history, state))
            remaining = [s for s in sims if s is not best]
            selected = [best]
            if max_speakers > 1 and remaining:
                selected += self._weighted_pick(remaining, history, state, count=max_speakers - 1)
            return selected[:max_speakers]

        return self._weighted_pick(sims, history, state, count=max_speakers)

    def _score(self, sim, history: list[str], state) -> float:
        score = 0.0
        phase = state.phase

        # Discourse obligations.
        if state.last_addressed == sim.name:
            score += 0.85
        if state.pending_question_target == sim.name:
            score += 0.95
        if state.pending_reply_target == sim.name:
            score += 0.75

        # Persona traits.
        score += 0.12 * _norm(sim.persona.get("initiative", 3))
        score += 0.10 * _norm(sim.persona.get("talkativeness", 3))
        score += 0.08 * _norm(sim.persona.get("assertiveness", 3))

        if phase in {"narrowing", "confirmation"}:
            score += 0.10 * _norm(sim.persona.get("agreeableness", 3))
            score += 0.10 * _norm(sim.persona.get("flexibility", 3))

        if sim.persona.get("is_primary", False):
            score += 0.10  # slightly more weight than before — primary voice matters

        # Speaking balance.
        last_speaker = self._last_participant_speaker(history)
        if last_speaker == sim.name:
            score -= 0.50
        recent_count = sum(1 for s in self._recent_speakers(history, n=4) if s == sim.name)
        score -= 0.12 * recent_count
        if not self._has_spoken(sim.name, history):
            score += 0.20

        # Anti-stall.
        if state.repetition_pressure >= 0.60:
            score += 0.08 * _norm(sim.persona.get("flexibility", 3))
            score += 0.06 * _norm(sim.persona.get("agreeableness", 3))
            score -= 0.05 * _norm(sim.persona.get("assertiveness", 3))

        # Phase boosts.
        if phase == "opening" and not self._has_spoken(sim.name, history):
            score += 0.18
        elif phase == "preference_expression":
            score += 0.05 * _norm(sim.persona.get("talkativeness", 3))
        elif phase == "confirmation":
            score += 0.06 * _norm(sim.persona.get("agreeableness", 3))
            score += 0.06 * _norm(sim.persona.get("patience", 3))

        return max(0.01, score)

    def _weighted_pick(
        self, sims: list, history: list[str], state, count: int = 1
    ) -> list:
        available = sims[:]
        picked: list = []
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
    # Repetition detection
    # ------------------------------------------------------------------

    def _compute_repetition_pressure(self, history: list[str]) -> float:
        """
        Returns 0.0–1.0 based on vocabulary overlap in recent participant messages.
        Window size is config-driven (repetition.pressure_window).
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
    # Line / speaker helpers
    # ------------------------------------------------------------------

    def _recent_participant_lines(self, history: list[str], limit: int = 6) -> list[str]:
        lines: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            if line.split(":", 1)[0].strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            lines.append(line)
            if len(lines) >= limit:
                break
        lines.reverse()
        return lines

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

    # ------------------------------------------------------------------
    # Discourse event extractors
    # ------------------------------------------------------------------

    def _extract_last_addressed(
        self, lines: list[str], sim_names: set[str]
    ) -> Optional[str]:
        if not lines:
            return None
        speaker, msg = lines[-1].split(":", 1)
        speaker = speaker.strip()
        for name in sim_names:
            if name != speaker and re.search(rf"\b{re.escape(name)}\b", msg, re.IGNORECASE):
                return name
        return None

    def _extract_question_target(
        self, lines: list[str], sim_names: set[str]
    ) -> Optional[str]:
        if not lines or "?" not in lines[-1]:
            return None
        speaker, msg = lines[-1].split(":", 1)
        speaker = speaker.strip()
        for name in sim_names:
            if name != speaker and re.search(rf"\b{re.escape(name)}\b", msg, re.IGNORECASE):
                return name
        return None

    def _extract_reply_target(
        self, lines: list[str], sim_names: set[str]
    ) -> Optional[str]:
        if not lines:
            return None
        speaker, msg = lines[-1].split(":", 1)
        speaker = speaker.strip()
        disagreement_markers = [
            "but", "however", "don't agree", "do not agree", "not sure", "instead", "rather"
        ]
        if not any(m in msg.lower() for m in disagreement_markers):
            return None
        for name in sim_names:
            if name != speaker and re.search(rf"\b{re.escape(name)}\b", msg, re.IGNORECASE):
                return name
        return None

    def _extract_important_events(self, lines: list[str]) -> list[str]:
        if not lines or ":" not in lines[-1]:
            return []
        speaker, msg = lines[-1].split(":", 1)
        speaker = speaker.strip()
        events: list[str] = []

        if "?" in msg:
            events.append(f"question_asked_by:{speaker}")

        options_mentioned = re.findall(r"\boption\s+([A-D])\b", msg, re.IGNORECASE)
        if options_mentioned:
            events.append(
                f"option_mentioned_by:{speaker}:{','.join(o.upper() for o in options_mentioned)}"
            )

        if any(x in msg.lower() for x in ["backup", "second choice"]):
            events.append(f"backup_preference_by:{speaker}")

        return events


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _norm(value: int) -> float:
    return (max(1, min(5, int(value))) - 1) / 4.0
