"""
logger.py
---------
DialogueLogger — writes two files per dialogue:

  .txt          chat with persona block at the top, Outcome/Tokens footer.
  .eval.json    everything else for evaluation:
                  - dialogue metadata + outcome + tokens
                  - per-speaker / per-phase turn counts + Gini
                  - vote flips, confirmation rejections
                  - Fisher (1970) ratios over public stances
                  - full personas (traits + beliefs)
                  - structured per-turn trace (acts, addressees, stance updates)
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, TYPE_CHECKING

from config_loader import cfg
from reasoning import fisher_ratios

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from persona import Persona
    from simulator import Simulator
    from state import StructuredState


def _gini(values: list[int]) -> float:
    """Gini coefficient of participation (0 = equal, 1 = one speaker dominates)."""
    if not values or len(values) <= 1:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    numer = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    return round(numer / (n * total), 4)


def _persona_block(personas: list["Persona"]) -> str:
    """Human-readable participant introductions for the top of the chat file."""
    lines: list[str] = ["Participants"]
    lines.append("-" * 50)
    for p in personas:
        primary_tag = " (primary)" if p.is_primary else ""
        lines.append(f"{p.name}{primary_tag} — {p.role}")
        lines.append(
            f"  traits: open={p.openness} consc={p.conscientiousness} "
            f"extra={p.extraversion} agree={p.agreeableness} "
            f"neuro={p.neuroticism} length={p.response_length}"
        )
        if p.goal:
            lines.append(f"  goal: {p.goal}")
        if p.backstory:
            lines.append(f"  backstory: {p.backstory}")
        if p.beliefs:
            b = p.beliefs
            accept_others = [x for x in b.acceptable if x != b.preferred]
            accept_str = f", accepts {accept_others}" if accept_others else ""
            rejects_str = f", rejects {b.rejected}" if b.rejected else ""
            lines.append(f"  prefers Option {b.preferred}{accept_str}{rejects_str}")
            lines.append(f"  concern: {b.key_concern}")
    lines.append("-" * 50)
    return "\n".join(lines)


class DialogueLogger:

    def __init__(self, dialogue_id: str, topic: str) -> None:
        self.dialogue_id = dialogue_id
        self.topic = topic

        log_dir = cfg.output.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.chat_file = os.path.join(log_dir, f"{dialogue_id}.txt")
        self.eval_file = os.path.join(log_dir, f"{dialogue_id}.eval.json")

        self._turn_records: list[dict[str, Any]] = []
        self._speaker_turn_counts: dict[str, int] = {}
        self._phase_turn_counts: dict[str, int] = {}

    # ------------------------------------------------------------------

    def write_header(self, participant_names: list[str],
                     personas: list["Persona"], opening_lines: list[str]) -> None:
        header = (
            f"Dialogue ID : {self.dialogue_id}\n"
            f"Participants: {', '.join(participant_names)}\n"
            f"Topic       : {self.topic}\n"
            + "=" * 50 + "\n\n"
            + _persona_block(personas) + "\n\n"
            + "=" * 50 + "\n"
        )
        with open(self.chat_file, "w", encoding="utf-8") as f:
            f.write(header)
            for line in opening_lines:
                f.write(f"{line}\n")
            f.write("\n")

    def append_chat_line(self, line: str) -> None:
        with open(self.chat_file, "a", encoding="utf-8") as f:
            f.write(f"{line}\n\n")

    def buffer(self, line: str, selected_reason: str, state: "DialogueState",
               tokens_in: int = 0, tokens_out: int = 0) -> None:
        if ":" not in line:
            return
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        is_moderator = speaker == "Moderator"

        self._turn_records.append({
            "turn_index": state.turn_index,
            "phase": state.phase,
            "speaker": speaker,
            "is_moderator": is_moderator,
            "text": text.strip(),
            "selected_reason": selected_reason,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        })
        if not is_moderator:
            self._speaker_turn_counts[speaker] = self._speaker_turn_counts.get(speaker, 0) + 1
            self._phase_turn_counts[state.phase] = self._phase_turn_counts.get(state.phase, 0) + 1

    # ------------------------------------------------------------------

    def flush(
        self,
        outcome: str,
        sims: list["Simulator"],
        state: "DialogueState",
        structured: Optional["StructuredState"],
        setup_tokens_in: int,
        setup_tokens_out: int,
        dialogue_tokens_in: int,
        dialogue_tokens_out: int,
    ) -> None:
        total_in  = setup_tokens_in + dialogue_tokens_in
        total_out = setup_tokens_out + dialogue_tokens_out

        # --- .txt footer ------------------------------------------------
        with open(self.chat_file, "a", encoding="utf-8") as f:
            f.write(
                f"\n--- Outcome: {outcome} ---\n"
                f"--- Tokens : setup={setup_tokens_in}/{setup_tokens_out}  "
                f"dialogue={dialogue_tokens_in}/{dialogue_tokens_out}  "
                f"total={total_in}/{total_out} (in/out) ---\n"
            )

        # --- .eval.json -------------------------------------------------
        meta: dict[str, Any] = {
            "dialogue_id": self.dialogue_id,
            "topic": self.topic,
            "outcome": outcome,
            "tokens": {
                "setup_in": setup_tokens_in,
                "setup_out": setup_tokens_out,
                "dialogue_in": dialogue_tokens_in,
                "dialogue_out": dialogue_tokens_out,
                "total_in": total_in,
                "total_out": total_out,
            },
            "participation": {
                "gini": _gini(list(self._speaker_turn_counts.values())),
                "speaker_turn_counts": self._speaker_turn_counts,
                "phase_turn_counts": self._phase_turn_counts,
            },
            "dynamics": {
                "vote_flips_per_speaker": dict(state.vote_changes),
                "confirmation_rejection_count": state.confirmation_rejection_count,
                "rejected_options_by_speaker": dict(state.rejected_options_by_speaker),
                "final_candidate_option": state.candidate_option,
                "final_preferred_option": state.preferred_option,
            },
            "personas": [s.persona.as_dict() for s in sims],
            "turns": self._turn_records,
        }

        if structured is not None:
            meta["fisher_ratios"] = fisher_ratios(structured)
            meta["consensus_state_final"] = structured.consensus_state
            meta["is_hard_blocker"] = {
                name: ps.is_true_hard_blocker
                for name, ps in structured.participants.items()
            }
            meta["public_preferences_final"] = {
                name: ps.public_preference
                for name, ps in structured.participants.items()
            }
            meta["structured_turns"] = [
                {
                    "id": t.turn_id,
                    "speaker": t.speaker,
                    "phase": t.phase,
                    "act": t.dialogue_act.value,
                    "mentioned_options": t.mentioned_options,
                    "is_question": t.is_question,
                    "addressees": t.addressees,
                    "reply_to": t.reply_to,
                    "answers_question_id": t.answers_question_id,
                    "stance_updates": [
                        {"option": su.option, "stance": su.stance,
                         "confidence": su.confidence, "condition": su.condition}
                        for su in t.stance_updates
                    ],
                }
                for t in structured.turns
            ]

        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    @property
    def paths(self) -> tuple[str, str]:
        return self.chat_file, self.eval_file
