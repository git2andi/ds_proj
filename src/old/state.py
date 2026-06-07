"""
state.py
--------
Small dialogue memory used for prompt context, routing, verification, and logs.

This module intentionally does not contain stance tables, consensus engines,
challenge graphs, or abstract decision gates. Live decisions are owned by
orchestrator.DialogueState through explicit votes, accepts, and rejects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from config_loader import cfg
from utils import OptionResolver

if TYPE_CHECKING:
    from persona import Persona


@dataclass
class TurnRecord:
    turn_id: int
    speaker: str
    text: str
    phase: str
    is_moderator: bool
    mentioned_options: list[str] = field(default_factory=list)
    addressees: list[str] = field(default_factory=list)
    is_question: bool = False
    selected_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class ParticipantState:
    name: str
    public_preference: Optional[str] = None
    turn_count: int = 0
    last_spoke_turn: Optional[int] = None
    is_true_hard_blocker: bool = False
    stated_priority: Optional[str] = None
    points_made: list[str] = field(default_factory=list)
    persona_ref: Optional["Persona"] = field(default=None, repr=False, compare=False)

    def record_point(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.points_made.append(text)
        cap = int(cfg.memory.points_made_max)
        if len(self.points_made) > cap:
            self.points_made = self.points_made[-cap:]


class DialogueMemory:
    """Minimal structured memory extracted from emitted transcript lines."""

    def __init__(
        self,
        participant_names: list[str],
        options: list[str],
        resolver: OptionResolver,
    ) -> None:
        self.participants = {
            name: ParticipantState(name=name) for name in participant_names
        }
        self.turns: list[TurnRecord] = []
        self.turn_id_counter = 0
        self.options = options
        self._resolver = resolver
        self._participant_names = set(participant_names)

    def attach_personas(self, sims: list) -> None:
        for sim in sims:
            ps = self.participants.get(sim.name)
            if ps:
                ps.persona_ref = sim.persona

    def update(
        self,
        line: str,
        phase: str,
        selected_reason: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> Optional[TurnRecord]:
        if ":" not in line:
            return None

        speaker, text = line.split(":", 1)
        speaker, text = speaker.strip(), text.strip()
        is_moderator = speaker in cfg.EXCLUDED_SPEAKERS

        self.turn_id_counter += 1
        turn_id = self.turn_id_counter
        mentioned = self._resolver.options_in(text)
        is_question = "?" in text
        addressees = self._extract_addressees(text, speaker, is_moderator, is_question)

        record = TurnRecord(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            phase=phase,
            is_moderator=is_moderator,
            mentioned_options=mentioned,
            addressees=addressees,
            is_question=is_question,
            selected_reason=selected_reason,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        self.turns.append(record)

        if not is_moderator:
            ps = self.participants.get(speaker)
            if ps:
                ps.turn_count += 1
                ps.last_spoke_turn = turn_id
                vote = self._resolver.vote_in(text)
                if vote:
                    ps.public_preference = vote
                if phase == "opening" and not ps.stated_priority:
                    ps.stated_priority = self._extract_priority(text)
                if self._is_memory_point(text, mentioned, phase):
                    ps.record_point(_summarise_point(text))
        return record

    def _extract_addressees(
        self,
        text: str,
        speaker: str,
        is_moderator: bool,
        is_question: bool,
    ) -> list[str]:
        targets = self._participant_names if is_moderator else self._participant_names - {speaker}
        addressees = [name for name in targets if _is_addressing(name, text, is_question)]
        if is_question and not addressees and not is_moderator:
            implicit = self._implicit_previous_speaker_target(text, speaker)
            if implicit:
                addressees.append(implicit)
        return addressees

    def _implicit_previous_speaker_target(self, question_text: str, speaker: str) -> Optional[str]:
        previous = next(
            (
                turn for turn in reversed(self.turns)
                if not turn.is_moderator and turn.speaker != speaker
            ),
            None,
        )
        if previous is None:
            return None
        return previous.speaker if _shares_content_keyword(question_text, previous.text) else None

    def _extract_priority(self, text: str) -> Optional[str]:
        m = _PRIORITY_MARKERS.search(text)
        if m:
            tail = text[m.end():].lstrip(" ,:;")
            phrase = re.split(r"[.;!?]", tail, maxsplit=1)[0].strip()
            return _trim(phrase, int(cfg.memory.perceived_priorities_max_chars)) or None
        cleaned = re.sub(r"^\W*(hi|hey|hello)[,!\s]+", "", text, flags=re.I).strip()
        first = re.split(r"[.;!?]", cleaned, maxsplit=1)[0].strip()
        return _trim(first, int(cfg.memory.perceived_priorities_max_chars)) or None

    def _is_memory_point(self, text: str, mentioned_options: list[str], phase: str) -> bool:
        if phase in {"opening", "closure", "confirmation"}:
            return False
        if len(text.split()) < int(cfg.memory.point_min_words):
            return False
        return bool(mentioned_options or _REASON_MARKERS.search(text))


_PRIORITY_MARKERS = re.compile(
    r"\b(care\s+about|matters?\s+(?:to\s+me|most)|priorit|important|"
    r"hoping\s+(?:for|to)|looking\s+for|want\s+(?:to|something)|"
    r"need\s+something|worried)\b",
    re.I,
)

_REASON_MARKERS = re.compile(
    r"\b(because|since|given|due\s+to|reason|tradeoff|drawback|concern|"
    r"helps?|fits?|matters?|works?|risk|cost|price|time|travel|noise|"
    r"menu|safety|local|comfort|reliable|flexib)\b",
    re.I,
)

_ADDRESS_CITATION_PATTERN = r"\b(?:with|like|as|what)\s+{name}\b|{name}'s\b"


def _is_addressing(name: str, text: str, is_question: bool) -> bool:
    n = re.escape(name)
    citation = re.compile(_ADDRESS_CITATION_PATTERN.format(name=n), re.I)
    if citation.search(text):
        return False
    if re.match(rf"\s*{n}\s*[,:]", text, re.I):
        return True
    if re.search(rf",\s*{n}\s*[.?!]?\s*$", text, re.I):
        return True
    return bool(is_question and re.search(rf"\b{n}\b", text, re.I))


def _shares_content_keyword(a: str, b: str) -> bool:
    stop = {
        "what", "about", "which", "option", "important", "choice", "really",
        "still", "think", "would", "could", "should", "there", "their", "your",
        "with", "that", "this", "from", "have", "does", "need", "want", "like",
        "nice", "spot", "place", "decision", "best", "good", "lean", "toward",
    }

    def toks(text: str) -> set[str]:
        return {
            re.sub(r"[^a-z0-9]", "", word.lower())
            for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text)
            if re.sub(r"[^a-z0-9]", "", word.lower()) not in stop
        }

    return bool(toks(a) & toks(b))


def _summarise_point(text: str) -> str:
    cleaned = re.sub(r"^\s*[A-Z][a-zA-Z]*,\s*", "", text).strip()
    first = re.split(r"[.;!?]", cleaned, maxsplit=1)[0].strip()
    return " ".join(first.split()[: int(cfg.memory.point_summary_words)])


def _trim(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."
