import random
import re
from typing import List, Optional

from constants import EXCLUDED_SPEAKERS


class TurnManager:
    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_events(self, history: List[str], state, sims: List) -> None:
        """Update dialogue state from recent history."""
        recent_lines = self._recent_participant_lines(history, limit=6)
        sim_names = {sim.name for sim in sims}

        state.last_addressed = self._extract_last_addressed(recent_lines, sim_names)
        state.pending_question_target = self._extract_question_target(recent_lines, sim_names)
        state.pending_reply_target = self._extract_reply_target(recent_lines, sim_names)
        state.repetition_pressure = self._compute_repetition_pressure(recent_lines)
        state.important_events = self._extract_important_events(recent_lines)

    def forced_names(self, state) -> set:
        """Return the set of participant names that have a discourse obligation this turn."""
        names = {
            getattr(state, "pending_question_target", None),
            getattr(state, "pending_reply_target", None),
            getattr(state, "last_addressed", None),
        }
        names.discard(None)
        return names

    def select_speakers(self, sims: List, history: List[str], state, max_speakers: int = 2) -> List:
        """
        Speaker selection:
        1. Discourse obligations first (question/reply targets).
        2. Weighted randomness for remaining slots.
        """
        if not sims:
            return []

        forced = self._forced_candidates(sims, state)
        if forced:
            best_forced = max(forced, key=lambda s: self.score_sim(s, history, state))
            selected = [best_forced]
            remaining = [s for s in sims if s is not best_forced]
            if max_speakers > 1 and remaining:
                selected.extend(self._weighted_pick(remaining, history, state, count=max_speakers - 1))
            return selected[:max_speakers]

        return self._weighted_pick(sims, history, state, count=max_speakers)

    def score_sim(self, sim, history: List[str], state) -> float:
        """Score one participant for the next turn."""
        score = 0.0
        last_speaker = self._last_participant_speaker(history)
        phase = getattr(state, "phase", "negotiation")

        # Discourse obligations.
        if state.last_addressed == sim.name:
            score += 0.85
        if state.pending_question_target == sim.name:
            score += 0.95
        if state.pending_reply_target == sim.name:
            score += 0.75

        # Persona traits (1-5 normalized to 0-1).
        score += 0.12 * self._norm(sim.persona.get("initiative", 3))
        score += 0.10 * self._norm(sim.persona.get("talkativeness", 3))
        score += 0.08 * self._norm(sim.persona.get("assertiveness", 3))

        if phase in {"narrowing", "confirmation"}:
            score += 0.10 * self._norm(sim.persona.get("agreeableness", 3))
            score += 0.10 * self._norm(sim.persona.get("flexibility", 3))

        if sim.persona.get("is_primary", False):
            score += 0.08

        # Speaking balance — penalise recent over-representation.
        if last_speaker == sim.name:
            score -= 0.50
        recent_count = sum(1 for s in self._recent_speakers(history, max_count=4) if s == sim.name)
        score -= 0.12 * recent_count
        if not self._has_spoken_before(sim.name, history):
            score += 0.20

        # Anti-stall.
        repetition_pressure = getattr(state, "repetition_pressure", 0.0)
        if repetition_pressure >= 0.60:
            score += 0.08 * self._norm(sim.persona.get("flexibility", 3))
            score += 0.06 * self._norm(sim.persona.get("agreeableness", 3))
            score -= 0.05 * self._norm(sim.persona.get("assertiveness", 3))

        # Phase boosts.
        if phase == "opening" and not self._has_spoken_before(sim.name, history):
            score += 0.18
        elif phase == "preference_expression":
            score += 0.05 * self._norm(sim.persona.get("talkativeness", 3))
        elif phase == "confirmation":
            score += 0.06 * self._norm(sim.persona.get("agreeableness", 3))
            score += 0.06 * self._norm(sim.persona.get("patience", 3))

        return max(0.01, score)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _weighted_pick(self, sims: List, history: List[str], state, count: int = 1) -> List:
        available = sims[:]
        picked = []

        for _ in range(min(count, len(available))):
            scored = [(self.score_sim(s, history, state), s) for s in available]
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

    def _forced_candidates(self, sims: List, state) -> List:
        targets = {
            getattr(state, "pending_question_target", None),
            getattr(state, "pending_reply_target", None),
            getattr(state, "last_addressed", None),
        }
        targets.discard(None)
        return [s for s in sims if s.name in targets]

    def _recent_participant_lines(self, history: List[str], limit: int = 6) -> List[str]:
        lines = []
        for line in reversed(history):
            if ":" not in line:
                continue
            if line.split(":", 1)[0].strip() in EXCLUDED_SPEAKERS:
                continue
            lines.append(line)
            if len(lines) >= limit:
                break
        lines.reverse()
        return lines

    def _recent_speakers(self, history: List[str], max_count: int = 4) -> List[str]:
        speakers = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker = line.split(":", 1)[0].strip()
            if speaker in EXCLUDED_SPEAKERS:
                continue
            speakers.append(speaker)
            if len(speakers) >= max_count:
                break
        return speakers

    def _last_participant_speaker(self, history: List[str]) -> Optional[str]:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker = line.split(":", 1)[0].strip()
            if speaker not in EXCLUDED_SPEAKERS:
                return speaker
        return None

    def _has_spoken_before(self, name: str, history: List[str]) -> bool:
        return any(line.startswith(f"{name}:") for line in history)

    def _extract_last_addressed(self, lines: List[str], sim_names: set) -> Optional[str]:
        if not lines:
            return None
        speaker, msg = lines[-1].split(":", 1)
        speaker = speaker.strip()
        for name in sim_names:
            if name != speaker and re.search(rf"\b{re.escape(name)}\b", msg, re.IGNORECASE):
                return name
        return None

    def _extract_question_target(self, lines: List[str], sim_names: set) -> Optional[str]:
        if not lines:
            return None
        speaker, msg = lines[-1].split(":", 1)
        speaker = speaker.strip()
        if "?" not in msg:
            return None
        for name in sim_names:
            if name != speaker and re.search(rf"\b{re.escape(name)}\b", msg, re.IGNORECASE):
                return name
        return None

    def _extract_reply_target(self, lines: List[str], sim_names: set) -> Optional[str]:
        if not lines:
            return None
        speaker, msg = lines[-1].split(":", 1)
        speaker = speaker.strip()
        msg_lower = msg.strip().lower()

        disagreement_markers = ["but", "however", "don't agree", "do not agree", "not sure", "instead", "rather"]
        if not any(m in msg_lower for m in disagreement_markers):
            return None

        for name in sim_names:
            if name != speaker and re.search(rf"\b{re.escape(name)}\b", msg_lower, re.IGNORECASE):
                return name
        return None

    def _compute_repetition_pressure(self, lines: List[str]) -> float:
        """Returns 0.0–1.0 based on vocabulary overlap in recent messages."""
        texts = []
        for line in lines[-4:]:
            if ":" not in line:
                continue
            _, msg = line.split(":", 1)
            texts.append(msg.strip().lower())

        if len(texts) < 3:
            return 0.0

        token_sets = [
            {w.strip(".,!?;:'\"()[]{}")
             for w in t.split()
             if len(w.strip(".,!?;:'\"()[]{}")) > 3}
            for t in texts
        ]

        overlaps = []
        for i in range(len(token_sets) - 1):
            a, b = token_sets[i], token_sets[i + 1]
            if a and b:
                overlaps.append(len(a & b) / max(1, min(len(a), len(b))))
            else:
                overlaps.append(0.0)

        return max(0.0, min(1.0, sum(overlaps) / len(overlaps))) if overlaps else 0.0

    def _extract_important_events(self, lines: List[str]) -> List[str]:
        if not lines or ":" not in lines[-1]:
            return []

        speaker, msg = lines[-1].split(":", 1)
        speaker = speaker.strip()
        events = []

        if "?" in msg:
            events.append(f"question_asked_by:{speaker}")

        option_mentions = re.findall(r"\boption\s+([A-D])\b", msg, re.IGNORECASE)
        if option_mentions:
            events.append(f"option_mentioned_by:{speaker}:{','.join(o.upper() for o in option_mentions)}")

        if any(x in msg.lower() for x in ["backup", "second choice"]):
            events.append(f"backup_preference_by:{speaker}")

        return events

    @staticmethod
    def _norm(value: int) -> float:
        return (max(1, min(5, int(value))) - 1) / 4.0
