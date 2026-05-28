"""
state.py
--------
Structured dialogue state — the single source of truth the orchestrator
decides from.

  - DialogueAct enum (only the acts actually used in act planning / detection)
  - TurnRecord, StanceUpdate dataclasses
  - StanceTable + OptionState (per speaker x option public stance)
  - DiscourseGraph (pending questions, reply edges)
  - ParticipantState (public stance, cooldowns, debt)
  - StructuredState container
  - StateTracker — deterministic raw line -> structured update
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, TYPE_CHECKING

from utils import OptionResolver

if TYPE_CHECKING:
    from persona import Persona


# =============================================================================
# Dialogue acts
# =============================================================================

class DialogueAct(Enum):
    OPEN_PRIORITY      = "OPEN_PRIORITY"
    ASSERT_SUPPORT     = "ASSERT_SUPPORT"
    ASSERT_OPPOSITION  = "ASSERT_OPPOSITION"
    ASSERT_AMBIGUOUS   = "ASSERT_AMBIGUOUS"
    ASK_CLARIFICATION  = "ASK_CLARIFICATION"
    ANSWER             = "ANSWER"
    CONCEDE            = "CONCEDE"
    CONDITIONAL_ACCEPT = "CONDITIONAL_ACCEPT"
    COMMIT_VOTE        = "COMMIT_VOTE"
    CONFIRM            = "CONFIRM"
    REJECT_WITH_REASON = "REJECT_WITH_REASON"
    GOODBYE            = "GOODBYE"
    MODERATOR          = "MODERATOR"


@dataclass
class StanceUpdate:
    speaker: str
    option: str                        # support|oppose|ambiguous|conditional_support|blocker|neutral
    stance: str
    confidence: float = 1.0
    condition: Optional[str] = None


@dataclass
class TurnRecord:
    turn_id: int
    speaker: str
    text: str
    phase: str
    is_moderator: bool

    addressees: list[str]              = field(default_factory=list)
    reply_to: Optional[int]            = None
    is_question: bool                  = False
    answers_question_id: Optional[int] = None

    dialogue_act: DialogueAct          = DialogueAct.ASSERT_AMBIGUOUS
    mentioned_options: list[str]       = field(default_factory=list)
    stance_updates: list[StanceUpdate] = field(default_factory=list)

    selected_reason: str = ""
    tokens_in: int       = 0
    tokens_out: int      = 0


# =============================================================================
# OptionState + StanceTable
# =============================================================================

@dataclass
class OptionState:
    option_id: str
    text: str

    supporters: set[str]                       = field(default_factory=set)
    opponents: set[str]                        = field(default_factory=set)
    ambiguous: set[str]                        = field(default_factory=set)
    conditional_supporters: dict[str, str]     = field(default_factory=dict)
    hard_blockers: dict[str, str]              = field(default_factory=dict)

    last_mentioned_turn: Optional[int]         = None

    def apply_update(self, update: StanceUpdate) -> None:
        name = update.speaker
        self.supporters.discard(name)
        self.opponents.discard(name)
        self.ambiguous.discard(name)
        self.conditional_supporters.pop(name, None)
        self.hard_blockers.pop(name, None)

        if update.stance == "support":
            self.supporters.add(name)
        elif update.stance == "oppose":
            self.opponents.add(name)
        elif update.stance == "blocker":
            self.opponents.add(name)
            self.hard_blockers[name] = update.condition or "unspecified"
        elif update.stance == "conditional_support":
            self.conditional_supporters[name] = update.condition or ""
        elif update.stance == "ambiguous":
            self.ambiguous.add(name)


class StanceTable:

    def __init__(self) -> None:
        self._current: dict[tuple[str, str], StanceUpdate] = {}
        self._history: list[StanceUpdate] = []

    def current_stance_label(self, speaker: str, option: str) -> str:
        su = self._current.get((speaker, option))
        return su.stance if su else "neutral"

    def apply(self, update: StanceUpdate, option_state: Optional[OptionState] = None) -> None:
        self._current[(update.speaker, update.option)] = update
        self._history.append(update)
        if option_state is not None:
            option_state.apply_update(update)

    def history(self) -> list[StanceUpdate]:
        return list(self._history)

    def current_items(self) -> list[tuple[tuple[str, str], StanceUpdate]]:
        return list(self._current.items())

    def unresolved_blockers(self, option: str) -> dict[str, str]:
        return {
            speaker: su.condition or "unspecified"
            for (speaker, opt), su in self._current.items()
            if opt == option and su.stance == "blocker"
        }


# =============================================================================
# DiscourseGraph
# =============================================================================

@dataclass
class DiscourseGraph:
    pending_questions: dict[int, list[str]] = field(default_factory=dict)
    reply_edges: dict[int, int]             = field(default_factory=dict)
    # open_invitations: question turn_id -> asker name. Anyone-but-asker is obligated.
    open_invitations: dict[int, str]        = field(default_factory=dict)
    last_addressed: Optional[str]           = None

    def has_obligation_for(self, name: str) -> bool:
        if any(name in addressees for addressees in self.pending_questions.values()):
            return True
        return any(asker != name for asker in self.open_invitations.values())

    def oldest_pending_addressees(self) -> list[str]:
        """Addressees of the oldest pending or open question (FIFO priority)."""
        candidates: list[tuple[int, list[str]]] = []
        for q_id, addressees in self.pending_questions.items():
            candidates.append((q_id, addressees))
        for q_id, asker in self.open_invitations.items():
            candidates.append((q_id, [f"!{asker}"]))    # sentinel: "anyone but asker"
        if not candidates:
            return []
        return min(candidates, key=lambda c: c[0])[1]

    def resolve_question(self, answering_turn_id: int, speaker: str) -> Optional[int]:
        for q_id, addressees in list(self.pending_questions.items()):
            if speaker in addressees:
                self.reply_edges[answering_turn_id] = q_id
                addressees.remove(speaker)
                if not addressees:
                    del self.pending_questions[q_id]
                else:
                    self.pending_questions[q_id] = addressees
                return q_id
        for q_id in sorted(self.open_invitations):
            asker = self.open_invitations[q_id]
            if speaker != asker:
                self.reply_edges[answering_turn_id] = q_id
                del self.open_invitations[q_id]
                return q_id
        return None

    def add_question(self, turn_id: int, addressees: list[str],
                     asker: Optional[str] = None) -> None:
        if addressees:
            self.pending_questions[turn_id] = list(addressees)
        elif asker:
            self.open_invitations[turn_id] = asker


# =============================================================================
# ParticipantState
# =============================================================================

@dataclass
class ParticipantState:
    name: str

    public_preference: Optional[str]          = None

    turn_count: int                           = 0
    last_spoke_turn: Optional[int]            = None
    participation_debt: float                 = 0.0

    recent_dialogue_acts: list[DialogueAct]   = field(default_factory=list)
    strategy_cooldowns: dict[str, int]        = field(default_factory=dict)

    is_true_hard_blocker: bool                = False

    persona_ref: Optional["Persona"]          = field(default=None, repr=False, compare=False)

    def decrement_cooldowns(self) -> None:
        self.strategy_cooldowns = {
            act: max(0, remaining - 1)
            for act, remaining in self.strategy_cooldowns.items()
            if remaining > 1
        }

    def on_cooldown(self, act: DialogueAct) -> bool:
        return self.strategy_cooldowns.get(act.value, 0) > 0

    def record_act(self, act: DialogueAct, max_history: int = 10) -> None:
        self.recent_dialogue_acts.append(act)
        if len(self.recent_dialogue_acts) > max_history:
            self.recent_dialogue_acts = self.recent_dialogue_acts[-max_history:]


# =============================================================================
# StructuredState
# =============================================================================

@dataclass
class StructuredState:
    turns: list[TurnRecord]                          = field(default_factory=list)
    turn_id_counter: int                             = 0
    participants: dict[str, ParticipantState]        = field(default_factory=dict)
    options: dict[str, OptionState]                  = field(default_factory=dict)
    stance_table: StanceTable                        = field(default_factory=StanceTable)
    discourse: DiscourseGraph                        = field(default_factory=DiscourseGraph)
    consensus_state: Literal[
        "none", "candidate_emerging", "majority_candidate",
        "conditional_consensus", "full_consensus", "blocked", "failed"
    ] = "none"
    candidate_option: Optional[str]                  = None


# =============================================================================
# StateTracker — deterministic raw-line -> structured update
# =============================================================================

class StateTracker:

    def __init__(self, participant_names: list[str], options: list[str],
                 resolver: OptionResolver) -> None:
        self._participant_names: set[str] = set(participant_names)
        self._all_names: list[str] = list(participant_names)
        self._resolver = resolver
        self._option_map: dict[str, str] = {}

        for opt in options:
            m = re.match(r"^Option\s+([A-D])\b", opt, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                self._option_map[letter] = opt

        self.state = StructuredState(
            participants={n: ParticipantState(name=n) for n in participant_names},
            options={l: OptionState(option_id=l, text=t) for l, t in self._option_map.items()},
        )

    def attach_personas(self, sims: list) -> None:
        for sim in sims:
            ps = self.state.participants.get(sim.name)
            if ps:
                ps.persona_ref = sim.persona

    def update(self, line: str, phase: str, selected_reason: str = "",
               tokens_in: int = 0, tokens_out: int = 0) -> Optional[TurnRecord]:
        if ":" not in line:
            return None

        speaker, text = line.split(":", 1)
        speaker, text = speaker.strip(), text.strip()
        is_moderator = speaker not in self._participant_names

        self.state.turn_id_counter += 1
        turn_id = self.state.turn_id_counter

        addressees = self._extract_addressees(text, speaker, is_moderator)
        is_question = "?" in text
        mentioned_options = self._resolver.options_in(text)
        act = self._estimate_act(text, phase, is_moderator, is_question, mentioned_options)
        answers_id = None if is_moderator else self.state.discourse.resolve_question(turn_id, speaker)
        reply_to = None
        if not is_moderator:
            for q_id, addressed in self.state.discourse.pending_questions.items():
                if speaker in addressed:
                    reply_to = q_id
                    break
        stance_updates = self._extract_stances(text, speaker, act, mentioned_options)

        if is_question and addressees:
            self.state.discourse.add_question(turn_id, addressees)
            if not is_moderator:
                self.state.discourse.last_addressed = addressees[0]
        elif is_question and not is_moderator:
            self.state.discourse.add_question(turn_id, [], asker=speaker)
        elif is_question and is_moderator:
            self.state.discourse.add_question(turn_id, list(self._participant_names))
        elif addressees and not is_moderator:
            self.state.discourse.last_addressed = addressees[0]

        for su in stance_updates:
            opt_state = self.state.options.get(su.option)
            self.state.stance_table.apply(su, opt_state)
            if opt_state and mentioned_options:
                opt_state.last_mentioned_turn = turn_id

        if not is_moderator:
            ps = self.state.participants.get(speaker)
            if ps:
                ps.turn_count += 1
                ps.last_spoke_turn = turn_id
                ps.record_act(act)
                ps.decrement_cooldowns()
                if act == DialogueAct.COMMIT_VOTE:
                    vote = self._resolver.vote_in(text)
                    if vote:
                        ps.public_preference = vote
                n = len(self._all_names)
                if n > 1:
                    for nm, other_ps in self.state.participants.items():
                        if nm == speaker:
                            other_ps.participation_debt -= 1.0
                        else:
                            other_ps.participation_debt += 1.0 / (n - 1)

        record = TurnRecord(
            turn_id=turn_id, speaker=speaker, text=text, phase=phase,
            is_moderator=is_moderator, addressees=addressees, reply_to=reply_to,
            is_question=is_question, answers_question_id=answers_id,
            dialogue_act=act, mentioned_options=mentioned_options,
            stance_updates=stance_updates, selected_reason=selected_reason,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
        self.state.turns.append(record)
        return record

    # ── helpers ────────────────────────────────────────────────────────

    def _extract_addressees(self, text: str, speaker: str, is_moderator: bool) -> list[str]:
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

    def _estimate_act(self, text: str, phase: str, is_moderator: bool,
                      is_question: bool, mentioned_options: list[str]) -> DialogueAct:
        if is_moderator:
            return DialogueAct.MODERATOR

        lower = text.lower().strip()
        stripped = lower.rstrip(".,!?")

        if phase == "closure":
            return DialogueAct.GOODBYE

        if self._resolver.vote_in(text):
            return DialogueAct.COMMIT_VOTE

        if phase == "confirmation":
            if stripped in {"yes", "yeah", "sure", "ok", "okay", "yep", "fine", "agreed"} or stripped.startswith("yes "):
                return DialogueAct.CONFIRM
            if stripped in {"no", "nope", "nah"} or lower.startswith("no "):
                return DialogueAct.REJECT_WITH_REASON

        if re.search(r"\bcould\s+(accept|live\s+with|work\s+with)\b.*\bif\b", lower):
            return DialogueAct.CONDITIONAL_ACCEPT
        if re.search(r"\bif\b.{0,60}\b(work|accept|fine|okay|deal)\b", lower) and mentioned_options:
            return DialogueAct.CONDITIONAL_ACCEPT

        if is_question:
            return DialogueAct.ASK_CLARIFICATION

        if any(p in lower for p in ("i agree", "sounds good", "works for me", "i'm in",
                                     "that works", "on board", "fair enough", "you're right")):
            return DialogueAct.CONCEDE

        if any(p in lower for p in ("but ", "however", "actually ", "not really", "disagree",
                                     "don't think", "doesn't work", "that's not", "won't work",
                                     "problem with", "issue with")):
            return DialogueAct.ASSERT_OPPOSITION

        if phase == "opening":
            return DialogueAct.OPEN_PRIORITY
        return DialogueAct.ASSERT_AMBIGUOUS

    def _extract_stances(self, text: str, speaker: str, act: DialogueAct,
                         mentioned_options: list[str]) -> list[StanceUpdate]:
        updates: list[StanceUpdate] = []
        cond = _extract_condition(text)

        if act == DialogueAct.COMMIT_VOTE:
            vote = self._resolver.vote_in(text)
            if vote:
                updates.append(StanceUpdate(speaker, vote, "support", 1.0))

        elif act == DialogueAct.CONFIRM:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "support", 0.9))

        elif act == DialogueAct.REJECT_WITH_REASON:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "blocker", 0.8, cond))

        elif act == DialogueAct.CONDITIONAL_ACCEPT:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "conditional_support", 0.8, cond))

        elif act == DialogueAct.CONCEDE:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "conditional_support", 0.6))

        elif act == DialogueAct.ASSERT_OPPOSITION:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "oppose", 0.5))

        elif act == DialogueAct.ASSERT_SUPPORT:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "support", 0.4))

        elif act in (DialogueAct.ASSERT_AMBIGUOUS, DialogueAct.OPEN_PRIORITY):
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "ambiguous", 0.3))

        return updates


def _extract_condition(text: str) -> Optional[str]:
    m = re.search(r"\bif\b(.{5,100})", text.lower())
    return m.group(1).strip()[:100] if m else None
