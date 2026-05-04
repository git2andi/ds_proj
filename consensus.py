"""
consensus.py
------------
ConsensusDetector — three-tier consensus detection:
  1. Soft    — natural agreement language in recent turns
  2. Regex   — explicit option-letter votes from all participants
  3. LLM     — model call as reliable fallback (runs every N turns)

Duplicate helpers from the old orchestrator.py are gone — this is the
single place that knows how to read the history for consensus signals.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional, TYPE_CHECKING

import prompts as prompts
from config_loader import cfg
from llm_client import get_llm_client

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator


class ConsensusDetector:

    def __init__(
        self,
        sims: list["Simulator"],
        options: list[str],
        moderator_style: str,
    ) -> None:
        self.sims = sims
        self.options = options
        self.moderator_style = moderator_style
        self._llm = get_llm_client()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        history: list[str],
        state: "DialogueState",
    ) -> Optional[tuple[str, Optional[str]]]:
        """
        Run all tiers in order. Returns (preferred_option, backup_option) or None.
        Guard: everyone must have spoken at least twice before we can detect consensus.
        """
        if self._participant_turn_count(history) < len(self.sims) * 2:
            return None

        result = self._soft(history, state)
        if result:
            return result

        result = self._regex(history)
        if result:
            return result

        if state.phase in {"negotiation", "narrowing", "confirmation"}:
            state.llm_check_countdown -= 1
            if state.llm_check_countdown <= 0:
                state.llm_check_countdown = cfg.consensus.llm_check_every_n_turns
                return self.llm_check(history)

        return None

    def llm_check(self, history: list[str]) -> Optional[tuple[str, Optional[str]]]:
        """Direct LLM check — also callable from the orchestrator on stall."""
        n_needed = max(2, len(self.sims) - 1)
        recent = "\n".join(history[-20:])
        names = [s.name for s in self.sims]
        try:
            data = self._llm.generate_json(
                prompts.consensus_check(names, self.options, recent, n_needed, len(self.sims))
            )
            if not data.get("consensus_reached"):
                return None
            opt = str(data.get("preferred_option") or "").strip().upper()
            bak_raw = str(data.get("backup_option") or "").strip().upper()
            bak = bak_raw if bak_raw in {"A", "B", "C", "D"} and bak_raw != opt else None
            if opt in {"A", "B", "C", "D"}:
                return opt, bak
        except Exception as exc:
            print(f"!! LLM consensus check error: {exc}")
        return None

    # ------------------------------------------------------------------
    # Tier 1 — Soft (agreement language)
    # ------------------------------------------------------------------

    def _soft(
        self, history: list[str], state: "DialogueState"
    ) -> Optional[tuple[str, Optional[str]]]:
        if state.phase not in {"negotiation", "narrowing", "confirmation"}:
            return None
        leading = state.current_leading_option
        if not leading:
            return None

        agreement_signals = [
            "sounds good", "sounds great", "sounds perfect", "that works",
            "i'm in", "i'm good with", "i agree", "let's go", "let's do",
            "works for me", "i confirm", "i'm happy with", "i'm on board",
            "on board", "perfect", "absolutely", "definitely",
        ]
        dissent_signals = [
            "not sure", "don't agree", "do not agree", "i disagree",
            "still think", "what about option", "not convinced",
        ]

        latest = self._latest_turn_per_speaker(history)
        if len(latest) < len(self.sims):
            return None

        agree_count = sum(
            1 for msg in latest.values()
            if any(s in msg for s in agreement_signals)
            and not any(s in msg for s in dissent_signals)
        )

        required = (
            max(2, len(self.sims) - 1)
            if self.moderator_style == "active"
            else len(self.sims)
        )
        return (leading, None) if agree_count >= required else None

    # ------------------------------------------------------------------
    # Tier 2 — Regex (explicit option letters)
    # ------------------------------------------------------------------

    def _regex(self, history: list[str]) -> Optional[tuple[str, Optional[str]]]:
        window = max(cfg.consensus.regex_window, len(self.sims) * 3)
        recent = self._recent_participant_lines(history, limit=window)

        if len(recent) < len(self.sims):
            return None

        latest_vote: dict[str, str] = {}
        for line in reversed(recent):
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            mentions = self._extract_option_letters(msg)
            if mentions and speaker not in latest_vote:
                latest_vote[speaker] = mentions[0]

        if len(latest_vote) < len(self.sims):
            return None

        # Primary's vote counts double
        primary = self._primary_sim()
        weighted = list(latest_vote.values())
        if primary and primary.name in latest_vote:
            weighted.append(latest_vote[primary.name])

        counts = Counter(weighted)
        top_option, top_count = counts.most_common(1)[0]

        n = len(self.sims)
        max_dissenters = (
            cfg.consensus.max_dissenters_active
            if self.moderator_style == "active"
            else cfg.consensus.max_dissenters_other
        )
        if top_count < max(2, n - max_dissenters):
            return None

        return top_option, None

    # ------------------------------------------------------------------
    # Helpers — no duplicates with orchestrator
    # ------------------------------------------------------------------

    def _extract_option_letters(self, text: str) -> list[str]:
        return [m.upper() for m in re.findall(r"\boption\s+([a-d])\b", text.lower())]

    def _primary_sim(self) -> Optional["Simulator"]:
        return next((s for s in self.sims if s.persona.is_primary), None)

    def _participant_turn_count(self, history: list[str]) -> int:
        return sum(
            1 for line in history
            if ":" in line
            and line.split(":", 1)[0].strip() not in cfg.EXCLUDED_SPEAKERS
        )

    def _recent_participant_lines(self, history: list[str], limit: int) -> list[str]:
        lines: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            if line.split(":", 1)[0].strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    def _latest_turn_per_speaker(self, history: list[str]) -> dict[str, str]:
        latest: dict[str, str] = {}
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in cfg.EXCLUDED_SPEAKERS:
                continue
            if speaker not in latest:
                latest[speaker] = msg.lower()
            if len(latest) == len(self.sims):
                break
        return latest
