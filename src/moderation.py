"""
moderation.py
-------------
ModerationEngine — intervention timing + LLM-generated moderator lines.

The orchestrator owns phase transitions, votes, consensus, closure.
This module owns when/how to intervene and which prompt to use.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from utils import OptionResolver, current_votes, last_n_turns_for

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator


# (text, tokens_in, tokens_out)
StoreFn = Callable[[str, int, int], None]


def _strip_wrapping_quotes(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] in {'"', "'", "“", "‘"} and t[-1] in {'"', "'", "”", "’"}:
        return t[1:-1].strip()
    return t


# Sims fishing for information that the options don't contain. High-precision
# laments + a fallback "pile of questions" signal.
_INFO_GAP = re.compile(
    r"(isn'?t|not)\s+specified|wish\s+we\s+had|still\s+waiting|"
    r"don'?t\s+have\s+(?:the\s+)?(?:cost|price|detail|info|number|figure)|"
    r"no\s+(?:cost|price|detail|info|number|figure)s?\b|"
    r"do\s+we\s+(?:know|have)\b|"
    r"what'?s\s+the\s+(?:exact\s+)?(?:cost|price|time|fee|rate|detail)",
    re.I,
)


def clean_moderator_line(text: str, participant_names: list[str]) -> str:
    """Drop the turn if any line starts with a participant name + colon."""
    t = _strip_wrapping_quotes(text)
    name_set = {n.lower() for n in participant_names}
    for line in t.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^([A-Za-z][a-zA-Z]*):\s+", stripped)
        if m and m.group(1).lower() in name_set:
            return ""
    return t


class ModerationEngine:

    def __init__(self, topic: str, options: list[str], sims: list["Simulator"],
                 resolver: OptionResolver) -> None:
        self.topic = topic
        self.options = list(options)
        self.sims = sims
        self._resolver = resolver
        self._llm = get_llm_client()

    # ------------------------------------------------------------------

    def escalation_level(self, state: "DialogueState") -> int:
        # Give the group n rounds of grace to collect votes before the patience
        # clock effectively starts, so larger groups aren't forced prematurely.
        r = state.post_narrowing_rounds
        grace = len(self.sims)
        if r < cfg.turns.escalation_level_1 + grace:
            return 0
        if r < cfg.turns.escalation_level_2 + grace:
            return 1
        if r < cfg.turns.escalation_level_3 + grace:
            return 2
        return 3

    def should_narrow(self, state: "DialogueState", participant_turn_count: int) -> bool:
        if state.has_asked_narrowing:
            return False
        n = len(self.sims)
        min_turns = n * cfg.turns.min_before_narrowing_per_participant
        if participant_turn_count < min_turns:
            return False
        stalling = (state.repetition_pressure >= cfg.repetition.stall_increment_threshold
                    and state.stall_rounds >= 1)
        talked_plenty = participant_turn_count >= n * cfg.turns.narrow_after_per_participant
        return stalling or talked_plenty

    def _recent_participant_lines(self, history: list[str], n: int) -> list[str]:
        out: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in {"Moderator"}:
                continue
            out.append(msg.strip())
            if len(out) >= n:
                break
        return out

    def detect_info_gap(self, history: list[str]) -> bool:
        """True when the group is fishing for a detail the options don't hold —
        an explicit lament in the last two turns, or a pile-up of questions."""
        recent = self._recent_participant_lines(history, 4)
        if not recent:
            return False
        if any(_INFO_GAP.search(line) for line in recent[:2]):
            return True
        return sum(1 for line in recent if "?" in line) >= 2

    def should_intervene(self, state: "DialogueState", history: list[str],
                         any_sim_stuck: bool, participant_turn_count: int) -> Optional[str]:
        if participant_turn_count < len(self.sims):
            return None

        outlier = self._detect_outlier(state, history)
        if outlier:
            return f"outlier:{outlier}"

        if state.has_asked_narrowing and any_sim_stuck:
            return "stall"
        if (state.repetition_pressure >= cfg.repetition.stall_increment_threshold
                and state.stall_rounds >= 2):
            return "stall"
        return None

    def run_intervention(self, reason: str, state: "DialogueState",
                         history: list[str], store_fn: StoreFn) -> None:
        names = [s.name for s in self.sims]
        recent = "\n".join(history[-10:])
        level = self.escalation_level(state)

        if reason == "clarify":
            line = self._llm.generate(
                prompts.moderator_clarify_info(
                    topic=self.topic, participant_names=names,
                    options=self.options, recent_dialogue=recent,
                )
            ).strip()
        elif reason.startswith("outlier:"):
            outlier_name = reason.split(":", 1)[1]
            state.nudged_participants.add(outlier_name)
            line = self._llm.generate(
                prompts.moderator_stall(
                    topic=self.topic, participant_names=names,
                    recent_dialogue=recent,
                    current_votes=current_votes(history, self._resolver),
                    escalation_level=level,
                )
            ).strip()
        else:  # stall
            candidate = getattr(state, "candidate_option", None) or state.current_leading_option
            if state.phase == "emergence" and level < 2 and candidate:
                line = self._llm.generate(
                    prompts.moderator_emergence(
                        topic=self.topic, participant_names=names,
                        recent_dialogue=recent, candidate_option=candidate,
                    )
                ).strip()
            else:
                line = self._llm.generate(
                    prompts.moderator_stall(
                        topic=self.topic, participant_names=names,
                        recent_dialogue=recent,
                        current_votes=current_votes(history, self._resolver),
                        escalation_level=level,
                    )
                ).strip()

        cleaned = clean_moderator_line(line, names)
        if cleaned:
            store_fn(cleaned, self._llm.last_tokens_in, self._llm.last_tokens_out)

    # ------------------------------------------------------------------

    def _detect_outlier(self, state: "DialogueState", history: list[str]) -> Optional[str]:
        """A participant who repeats themselves verbatim after narrowing."""
        if not state.has_asked_narrowing:
            return None
        for sim in self.sims:
            turns = last_n_turns_for(sim.name, history, n=2)
            if len(turns) < 2:
                continue
            words0 = set(turns[0].split())
            if not words0:
                continue
            ratio = len(words0 & set(turns[1].split())) / len(words0)
            if ratio >= 0.55:
                return sim.name
        return None
