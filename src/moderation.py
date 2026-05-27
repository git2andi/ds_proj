"""
moderation.py
-------------
ModerationEngine -- intervention timing and LLM-generated moderator lines.

  Orchestrator owns: phase transitions, votes, consensus, closure.
  ModerationEngine owns: when/how to intervene, escalation level, outlier
                         detection, the prompt selection for each reason.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from utils import current_votes, last_n_turns_for

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator


# (text, tokens_in, tokens_out)
StoreFn = Callable[[str, int, int], None]


def _strip_wrapping_quotes(text: str) -> str:
    """LLMs sometimes wrap moderator lines in quote marks. Strip outer quotes."""
    t = text.strip()
    if len(t) >= 2 and t[0] in {'"', "'", "“", "‘"} and t[-1] in {'"', "'", "”", "’"}:
        return t[1:-1].strip()
    return t


def _clean_moderator_line(text: str, participant_names: list[str]) -> str:
    """
    LLMs occasionally impersonate a participant ("Lena: yes if seafood is fresh").
    This can appear at the start of the string OR mid-text (after a preamble line
    the LLM added before the actual response).  Either way: drop the whole turn.
    """
    import re
    t = _strip_wrapping_quotes(text)
    name_set = {n.lower() for n in participant_names}
    # Check every non-empty line -- catches preamble + attribution patterns
    for line in t.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^([A-Za-z][a-zA-Z]*):\s+", stripped)
        if m and m.group(1).lower() in name_set:
            return ""
    return t


class ModerationEngine:

    def __init__(self, topic: str, options: list[str],
                 moderator_style: str, sims: list["Simulator"]) -> None:
        self.topic = topic
        self.options = list(options)
        self.moderator_style = moderator_style
        self.sims = sims
        self._llm = get_llm_client()

    # ------------------------------------------------------------------

    def escalation_level(self, state: "DialogueState") -> int:
        r = state.post_narrowing_rounds
        if r < cfg.turns.escalation_level_1:
            return 0
        if r < cfg.turns.escalation_level_2:
            return 1
        if r < cfg.turns.escalation_level_3:
            return 2
        return 3

    def should_narrow(self, state: "DialogueState", participant_turn_count: int) -> bool:
        if state.has_asked_narrowing or self.moderator_style == "passive":
            return False
        n = len(self.sims)
        if participant_turn_count < max(n * 2, cfg.turns.min_before_narrowing):
            return False
        stalling = (state.repetition_pressure >= cfg.moderation.narrowing.stalling_repetition_threshold
                    and state.stall_rounds >= 1)
        talked_plenty = participant_turn_count >= n * 5
        if self.moderator_style == "minimal":
            return stalling and talked_plenty
        return stalling or talked_plenty

    def should_intervene(self, state: "DialogueState", history: list[str],
                         any_sim_stuck: bool, participant_turn_count: int) -> Optional[str]:
        if self.moderator_style == "passive":
            return None
        if participant_turn_count < len(self.sims):
            return None

        outlier = self._detect_outlier(state, history)
        if outlier:
            return f"outlier:{outlier}"

        if state.has_asked_narrowing and any_sim_stuck:
            return "stall"
        if (state.repetition_pressure >= cfg.moderation.interventions.stall_repetition_threshold
                and state.stall_rounds >= cfg.moderation.interventions.stall_rounds_required):
            return "stall"
        return None

    def run_intervention(self, reason: str, state: "DialogueState",
                         history: list[str], store_fn: StoreFn) -> None:
        names = [s.name for s in self.sims]
        recent = "\n".join(history[-10:])
        level = self.escalation_level(state)

        try:
            if reason.startswith("outlier:"):
                outlier_name = reason.split(":", 1)[1]
                state.nudged_participants.add(outlier_name)
                line = self._llm.generate(
                    prompts.moderator_intervention(
                        topic=self.topic, participant_names=names,
                        recent_dialogue=recent,
                        reason=f"{outlier_name} keeps repeating the same position.",
                        target_participant=outlier_name, escalation_level=level,
                    )
                ).strip()

            elif reason == "compromise":
                candidate = (getattr(state, "compromise_option", None)
                             or getattr(state, "candidate_option", None) or "A")
                holdouts = [s.name for s in self.sims
                            if current_votes(history, self.sims).get(s.name) != candidate]
                line = self._llm.generate(
                    prompts.moderator_compromise_test(
                        topic=self.topic, participant_names=names,
                        options=self.options, recent_dialogue=recent,
                        compromise_option=candidate, holdout_names=holdouts,
                    )
                ).strip()

            else:  # stall
                votes = current_votes(history, self.sims)
                candidate = getattr(state, "candidate_option", None) or state.current_leading_option
                if state.phase == "emergence" and level < 2 and candidate:
                    line = self._llm.generate(
                        prompts.moderator_emergence(
                            topic=self.topic, participant_names=names,
                            options=self.options, recent_dialogue=recent,
                            candidate_option=candidate,
                        )
                    ).strip()
                else:
                    line = self._llm.generate(
                        prompts.moderator_stall(
                            topic=self.topic, participant_names=names,
                            recent_dialogue=recent, current_votes=votes,
                            escalation_level=level,
                        )
                    ).strip()

            cleaned = _clean_moderator_line(line, [s.name for s in self.sims])
            if cleaned:
                store_fn(cleaned, self._llm.last_tokens_in, self._llm.last_tokens_out)

        except Exception as exc:
            print(f"!! Moderator intervention error ({reason}): {exc}")

    # ------------------------------------------------------------------

    def _detect_outlier(self, state: "DialogueState", history: list[str]) -> Optional[str]:
        if not state.has_asked_narrowing:
            return None
        for sim in self.sims:
            turns = last_n_turns_for(sim.name, history, n=2)
            if len(turns) < 2:
                continue
            words0 = set(turns[0].split())
            ratio = len(words0 & set(turns[1].split())) / max(1, len(words0))
            if ratio >= cfg.moderation.interventions.outlier_overlap_threshold:
                return sim.name
        return None
