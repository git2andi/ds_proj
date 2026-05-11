"""
consensus.py
------------
ConsensusDetector — three-tier consensus detection:
  1. Soft    — natural agreement language in recent turns
  2. Regex   — explicit option-letter votes from all participants
  3. LLM     — model call as reliable fallback (runs every N turns)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from utils import (
    extract_preference_vote,
    last_n_turns_for,
    participant_turn_count,
    recent_participant_lines,
    latest_turn_per_speaker,
)

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
        if participant_turn_count(history) < len(self.sims) * 2:
            return None

        result = self._soft(history, state)
        if result:
            return result

        result = self._regex(history, expected_option=state.preferred_option)
        if result:
            return result

        # Reduced-opposition tier: emergence-phase consensus detected by silence of dissent
        result = self._reduced_opposition(history, state)
        if result:
            return result

        if state.phase in {"negotiation", "narrowing", "emergence", "confirmation"}:
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

        latest = latest_turn_per_speaker(history, self.sims)
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
        Count each participant's most recent committed vote, then tally unique voters.

        Cross-check guard: if expected_option is provided, the regex result must
        match it — this prevents stale old votes from triggering a confirmation
        for the wrong option when the group has already moved on.
        """
        window = max(cfg.consensus.regex_window, len(self.sims) * 4)
        recent = recent_participant_lines(history, limit=window)

        if len(recent) < len(self.sims):
            return None

        latest_vote: dict[str, str] = {}
        for line in recent:
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in latest_vote:
                continue
            vote = extract_preference_vote(msg)
            if vote:
                latest_vote[speaker] = vote

        if len(latest_vote) < len(self.sims):
            return None

        primary = self._primary_sim()
        vote_counts: Counter = Counter()
        for speaker, opt in latest_vote.items():
            weight = 2 if (primary and speaker == primary.name) else 1
            vote_counts[opt] += weight

        top_option, _ = vote_counts.most_common(1)[0]

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
    # Tier 3 — Reduced opposition (emergence-phase consensus)
    # ------------------------------------------------------------------

    def _reduced_opposition(
        self, history: list[str], state: "DialogueState"
    ) -> Optional[tuple[str, Optional[str]]]:
        """
        Fisher's emergence signal: the candidate option has plurality votes AND
        dissenters' most recent turns show ambiguity rather than active objection.
        Only runs in the emergence phase.
        """
        if state.phase != "emergence":
            return None
        candidate = state.candidate_option
        if not candidate:
            return None

        window = max(cfg.consensus.regex_window, len(self.sims) * 4)
        recent_lines = recent_participant_lines(history, limit=window)
        votes: dict[str, str] = {}
        for line in recent_lines:
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in votes:
                continue
            vote = extract_preference_vote(msg)
            if vote:
                votes[speaker] = vote

        n = len(self.sims)
        candidate_voters = sum(1 for v in votes.values() if v == candidate)
        required = max(2, n - 1) if self.moderator_style == "active" else n
        if candidate_voters < required:
            return None

        # Check dissenters have gone ambiguous — no active objection in their last 2 turns
        dissenters = [name for name, vote in votes.items() if vote != candidate]
        if not dissenters:
            return candidate, None

        objection_signals = [
            "still think", "not convinced", "don't agree", "disagree",
            "still prefer", "rather have", "not sure about", "still on",
        ]
        for dissenter in dissenters:
            turns = last_n_turns_for(dissenter, history, n=2)
            if not turns:
                continue
            if any(sig in turns[0] for sig in objection_signals):
                return None

        return candidate, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _primary_sim(self) -> Optional["Simulator"]:
        return next((s for s in self.sims if s.persona.is_primary), None)
