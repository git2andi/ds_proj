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

import prompts
from config_loader import cfg
from llm_client import get_llm_client

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator


def extract_preference_vote(msg: str) -> Optional[str]:
    """
    Return the option letter (A-D) this message votes for, or None if ambiguous.

    Counts:  first-person preference phrases ("I prefer Option B"),
             option letters in the first 5 words of a non-question message.
    Ignores: question-starting messages, comparisons ("Option A or B"),
             option mentions that are references to others' positions.
    """
    text = msg.strip().lower()
    words = text.split()

    # Explicit first-person preference — unambiguous vote
    fp_patterns = [
        r"\bi\s+prefer\s+option\s+([a-d])\b",
        r"\bi(?:'m|\s+am)\s+(?:going\s+with|for)\s+option\s+([a-d])\b",
        r"\bi(?:'d|\s+would)\s+(?:go\s+with|choose|prefer)\s+option\s+([a-d])\b",
        r"\bi\s+(?:choose|pick|want|vote\s+for)\s+option\s+([a-d])\b",
        r"\bmy\s+(?:choice|pick|preference)\s+is\s+(?:option\s+)?([a-d])\b",
    ]
    for pat in fp_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()

    # Option letter in first 5 words — likely a stated position
    first5 = " ".join(words[:5])
    early = re.search(r"\boption\s+([a-d])\b", first5)
    if early:
        # Skip question-opening messages ("What about Option A?")
        lead = words[0].rstrip(",'") if words else ""
        if lead in {"what", "how", "why", "when", "where", "is", "are", "does", "can", "could"}:
            return None
        # Skip comparisons: "Option A or B" — multiple options in first 10 words
        nearby = " ".join(words[:10])
        if len(re.findall(r"\boption\s+[a-d]\b", nearby)) > 1:
            return None
        # Skip possessive usage: "Option D's limited job opportunities are a red flag…"
        # The speaker is describing the option's properties, not voting for it.
        letter = early.group(1)
        pos = text.find(f"option {letter}")
        char_after = text[pos + len(f"option {letter}"):].lstrip()[:1]
        if char_after == "'":
            return None
        return letter.upper()

    return None


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

        # Pass the currently expected option so stale votes for a different option
        # don't falsely trigger confirmation of something the group has moved away from.
        result = self._regex(history, expected_option=state.preferred_option)
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

    def _regex(
        self,
        history: list[str],
        expected_option: Optional[str] = None,
    ) -> Optional[tuple[str, Optional[str]]]:
        """
        Count each participant's MOST RECENT committed vote, then tally unique voters.

        Window scales with group size: max(config_window, n_sims * 4) to ensure
        at least 4 full rounds per sim are covered even in large groups.

        Cross-check guard: if expected_option is provided (i.e. a preferred_option
        was already set from a previous confirmation attempt), the regex result must
        match it — this prevents stale old votes from triggering a confirmation for
        the wrong option when the group has already moved on.
        """
        window = max(cfg.consensus.regex_window, len(self.sims) * 4)
        recent = self._recent_participant_lines(history, limit=window)

        if len(recent) < len(self.sims):
            return None

        # Walk history newest-first; one vote per speaker (their most recent).
        latest_vote: dict[str, str] = {}
        for line in recent:                     # recent is newest-first from _recent_participant_lines
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in latest_vote:
                continue
            vote = extract_preference_vote(msg)
            if vote:
                latest_vote[speaker] = vote

        if len(latest_vote) < len(self.sims):
            return None

        # Tally unique voters per option (primary counts double)
        primary = self._primary_sim()
        vote_counts: Counter = Counter()
        for speaker, opt in latest_vote.items():
            weight = 2 if (primary and speaker == primary.name) else 1
            vote_counts[opt] += weight

        top_option, _ = vote_counts.most_common(1)[0]

        # Cross-check: if an expected option is set, only accept a matching result.
        # This prevents stale old votes from overriding the current group direction.
        if expected_option and top_option != expected_option:
            return None

        unique_voters_for_top = sum(
            1 for speaker, opt in latest_vote.items() if opt == top_option
        )
        n = len(self.sims)
        max_dissenters = (
            cfg.consensus.max_dissenters_active
            if self.moderator_style == "active"
            else cfg.consensus.max_dissenters_other
        )
        required_voters = n - max_dissenters

        if unique_voters_for_top < required_voters:
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