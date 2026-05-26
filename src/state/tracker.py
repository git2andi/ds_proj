"""
state/tracker.py
-----------------
StateTracker — processes each raw turn line and maintains StructuredState.

Stage 5: runs in PARALLEL with the legacy orchestrator logic.
         All decisions still come from orchestrator.DialogueState.
         StateTracker output is logged to *_state.jsonl for comparison.

Stage 6+: discourse.pending_questions drives SSJ turn selection.
Stage 9+: stance_table drives consensus and phase detection.

Deterministic rules first; LLM stance classifier deferred to Stage 9.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

from config_loader import cfg
from state.dialogue_state import StructuredState, PhaseEvidence
from state.discourse import DiscourseGraph
from state.participant import ParticipantState
from state.stance import OptionState, StanceTable
from state.turn import DialogueAct, StanceUpdate, TurnRecord

if TYPE_CHECKING:
    pass


class StateTracker:
    """
    Maintains a StructuredState by processing turns one at a time.
    Called from orchestrator._store_line() after each turn is appended to history.
    """

    def __init__(self, participant_names: list[str], options: list[str]) -> None:
        self._participant_names: set[str] = set(participant_names)
        self._all_names: list[str] = list(participant_names)
        self._option_letters: list[str] = []
        self._option_map: dict[str, str] = {}  # letter → full text

        for opt in options:
            m = re.match(r"^Option\s+([A-D])\b", opt, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                self._option_letters.append(letter)
                self._option_map[letter] = opt

        self.state = StructuredState(
            participants={
                name: ParticipantState(name=name)
                for name in participant_names
            },
            options={
                letter: OptionState(option_id=letter, text=text)
                for letter, text in self._option_map.items()
            },
        )

    def attach_personas(self, sims: list) -> None:
        """Attach Persona objects to ParticipantState entries (called after sims are built)."""
        for sim in sims:
            ps = self.state.participants.get(sim.name)
            if ps:
                ps.persona_ref = sim.persona

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def update(
        self,
        line: str,
        phase: str,
        selected_reason: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        prompt_type: str = "sim_turn",
    ) -> Optional[TurnRecord]:
        """
        Process one raw 'Speaker: text' line.
        Returns a TurnRecord (for JSONL logging), or None if line is unparseable.
        """
        if ":" not in line:
            return None

        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        text = text.strip()
        is_moderator = speaker not in self._participant_names

        self.state.turn_id_counter += 1
        turn_id = self.state.turn_id_counter

        # --- Extraction ------------------------------------------------
        addressees = self._extract_addressees(text, speaker, is_moderator)
        is_question = "?" in text
        mentioned_options = self._extract_option_letters(text)
        act = self._estimate_act(text, phase, is_moderator, is_question)
        reply_to = self._resolve_reply_to(speaker, is_moderator)
        answers_id = self._resolve_answers_question(turn_id, speaker, is_moderator)
        stance_updates = self._extract_stances(text, speaker, phase, act, mentioned_options)

        # --- Discourse update ------------------------------------------
        if not is_moderator:
            if is_question and addressees:
                self.state.discourse.add_question(turn_id, addressees)
                self.state.discourse.last_addressed = addressees[0]
            elif is_question:
                self.state.discourse.add_question(turn_id, [])  # open invitation
            elif addressees:
                self.state.discourse.last_addressed = addressees[0]

        # --- Stance update --------------------------------------------
        for su in stance_updates:
            opt_state = self.state.options.get(su.option)
            self.state.stance_table.apply(su, opt_state)
            if opt_state and mentioned_options:
                opt_state.last_mentioned_turn = turn_id

        # --- Participant state update ----------------------------------
        if not is_moderator:
            ps = self.state.participants.get(speaker)
            if ps:
                ps.turn_count += 1
                ps.last_spoke_turn = turn_id
                ps.record_act(act)
                ps.decrement_cooldowns()

                # Update committed public preference from COMMIT_VOTE
                if act == DialogueAct.COMMIT_VOTE:
                    vote = self._extract_vote(text)
                    if vote:
                        ps.public_preference = vote

                # Update participation debt: speaker −1, others +1/(n−1)
                n = len(self._all_names)
                if n > 1:
                    for name, other_ps in self.state.participants.items():
                        if name == speaker:
                            other_ps.participation_debt -= 1.0
                        else:
                            other_ps.participation_debt += 1.0 / (n - 1)

        # --- Build and store TurnRecord --------------------------------
        record = TurnRecord(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            phase=phase,
            is_moderator=is_moderator,
            addressees=addressees,
            reply_to=reply_to,
            is_question=is_question,
            answers_question_id=answers_id,
            dialogue_act=act,
            mentioned_options=mentioned_options,
            stance_updates=stance_updates,
            selected_reason=selected_reason,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            prompt_type="moderator" if is_moderator else prompt_type,
        )
        self.state.turns.append(record)
        return record

    # ------------------------------------------------------------------
    # Addressee extraction
    # ------------------------------------------------------------------

    def _extract_addressees(
        self, text: str, speaker: str, is_moderator: bool
    ) -> list[str]:
        """
        Ouchi & Tsuboi (2016) convention: check first 1-3 words for direct address,
        then fall back to name-mentions anywhere in the turn.
        """
        result: list[str] = []
        targets = self._participant_names if is_moderator else (self._participant_names - {speaker})

        first_three = " ".join(text.split()[:3])
        for name in targets:
            if re.search(rf"\b{re.escape(name)}\b", first_three, re.IGNORECASE):
                result.append(name)

        if not result:
            for name in targets:
                if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                    result.append(name)

        return result

    # ------------------------------------------------------------------
    # Option extraction
    # ------------------------------------------------------------------

    def _extract_option_letters(self, text: str) -> list[str]:
        return [m.upper() for m in re.findall(r"\boption\s+([a-d])\b", text.lower())]

    # ------------------------------------------------------------------
    # Dialogue act estimation (deterministic)
    # ------------------------------------------------------------------

    def _estimate_act(
        self, text: str, phase: str, is_moderator: bool, is_question: bool
    ) -> DialogueAct:
        if is_moderator:
            return DialogueAct.MODERATOR

        lower = text.lower().strip()
        stripped = lower.rstrip(".,!?")

        if phase == "greeting":
            return DialogueAct.GREET
        if phase == "closure":
            return DialogueAct.GOODBYE

        # Explicit vote
        if self._extract_vote(text):
            return DialogueAct.COMMIT_VOTE

        # Confirmation phase signals
        if phase == "confirmation":
            confirm_words = {"yes", "yeah", "sure", "ok", "okay", "yep",
                             "absolutely", "fine", "agreed", "sounds good"}
            if stripped in confirm_words or stripped.startswith("yes "):
                return DialogueAct.CONFIRM
            if stripped in {"no", "nope", "nah"} or lower.startswith("no "):
                return DialogueAct.REJECT_WITH_REASON

        # Conditional acceptance
        if re.search(r"\bcould\s+(accept|live\s+with|work\s+with)\b.*\bif\b", lower):
            return DialogueAct.CONDITIONAL_ACCEPT
        if re.search(r"\bif\b.{0,60}\b(work|accept|fine|okay|deal)\b", lower):
            if re.search(r"\boption\s+[a-d]\b", lower):
                return DialogueAct.CONDITIONAL_ACCEPT

        # Questions
        if is_question:
            if any(w in lower for w in ("prefer", "choice", "pick", "vote", "which option")):
                return DialogueAct.ASK_PREFERENCE
            return DialogueAct.ASK_CLARIFICATION

        # Concession
        concede_phrases = (
            "i agree", "sounds good", "works for me", "i'm in", "that works",
            "on board", "fair enough", "good point", "you're right",
        )
        if any(p in lower for p in concede_phrases):
            return DialogueAct.CONCEDE

        # Opposition
        opposition_phrases = (
            "but ", "however", "actually ", "not really", "disagree",
            "don't think", "doesn't work", "that's not", "won't work",
            "problem with", "issue with",
        )
        if any(p in lower for p in opposition_phrases):
            return DialogueAct.ASSERT_OPPOSITION

        if phase == "opening":
            return DialogueAct.OPEN_PRIORITY

        return DialogueAct.ASSERT_AMBIGUOUS

    # ------------------------------------------------------------------
    # Vote extraction helper
    # ------------------------------------------------------------------

    def _extract_vote(self, text: str) -> Optional[str]:
        """Extract explicit first-person vote from text (same patterns as utils.extract_preference_vote)."""
        lower = text.lower()
        patterns = [
            r"\bi\s+prefer\s+option\s+([a-d])\b",
            r"\bi(?:'m|\s+am)\s+(?:going\s+with|for)\s+option\s+([a-d])\b",
            r"\bi(?:'d|\s+would)\s+(?:go\s+with|choose|prefer)\s+option\s+([a-d])\b",
            r"\bi\s+(?:choose|pick|want|vote\s+for)\s+option\s+([a-d])\b",
            r"\bmy\s+(?:choice|pick|preference)\s+is\s+(?:option\s+)?([a-d])\b",
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                return m.group(1).upper()
        return None

    # ------------------------------------------------------------------
    # Reply / answer resolution
    # ------------------------------------------------------------------

    def _resolve_reply_to(self, speaker: str, is_moderator: bool) -> Optional[int]:
        """Check if this speaker is answering a pending question; return question turn_id."""
        if is_moderator:
            return None
        for q_id, addressees in self.state.discourse.pending_questions.items():
            if speaker in addressees:
                return q_id
        return None

    def _resolve_answers_question(
        self, turn_id: int, speaker: str, is_moderator: bool
    ) -> Optional[int]:
        """Mark the pending question as answered and return its turn_id."""
        if is_moderator:
            return None
        return self.state.discourse.resolve_question(turn_id, speaker)

    # ------------------------------------------------------------------
    # Stance extraction (deterministic)
    # ------------------------------------------------------------------

    def _extract_stances(
        self,
        text: str,
        speaker: str,
        phase: str,
        act: DialogueAct,
        mentioned_options: list[str],
    ) -> list[StanceUpdate]:
        """
        Map dialogue act + mentioned options to StanceUpdate records.
        Deterministic rules only; LLM classifier deferred to Stage 9.
        """
        updates: list[StanceUpdate] = []

        if act == DialogueAct.COMMIT_VOTE:
            vote = self._extract_vote(text)
            if vote:
                updates.append(StanceUpdate(
                    speaker=speaker, option=vote, stance="support", confidence=1.0,
                ))

        elif act == DialogueAct.CONFIRM:
            for opt in mentioned_options:
                updates.append(StanceUpdate(
                    speaker=speaker, option=opt, stance="support", confidence=0.9,
                ))

        elif act == DialogueAct.REJECT_WITH_REASON:
            condition = self._extract_condition(text)
            for opt in mentioned_options:
                updates.append(StanceUpdate(
                    speaker=speaker, option=opt, stance="blocker",
                    confidence=0.8, condition=condition,
                ))

        elif act == DialogueAct.CONDITIONAL_ACCEPT:
            condition = self._extract_condition(text)
            for opt in mentioned_options:
                updates.append(StanceUpdate(
                    speaker=speaker, option=opt, stance="conditional_support",
                    confidence=0.8, condition=condition,
                ))

        elif act == DialogueAct.CONCEDE:
            for opt in mentioned_options:
                updates.append(StanceUpdate(
                    speaker=speaker, option=opt, stance="conditional_support",
                    confidence=0.6,
                ))

        elif act == DialogueAct.ASSERT_OPPOSITION:
            for opt in mentioned_options:
                updates.append(StanceUpdate(
                    speaker=speaker, option=opt, stance="oppose", confidence=0.5,
                ))

        elif act == DialogueAct.ASSERT_SUPPORT:
            for opt in mentioned_options:
                updates.append(StanceUpdate(
                    speaker=speaker, option=opt, stance="support", confidence=0.4,
                ))

        # For ambiguous acts, only record if there's a clear option mention
        elif act in (DialogueAct.ASSERT_AMBIGUOUS, DialogueAct.OPEN_PRIORITY):
            for opt in mentioned_options:
                updates.append(StanceUpdate(
                    speaker=speaker, option=opt, stance="ambiguous", confidence=0.3,
                ))

        return updates

    def _extract_condition(self, text: str) -> Optional[str]:
        """Extract a condition clause ('if ...') from text for conditional stances."""
        m = re.search(r"\bif\b(.{5,100})", text.lower())
        return m.group(1).strip()[:100] if m else None
