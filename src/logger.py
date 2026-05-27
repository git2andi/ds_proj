"""
logger.py
---------
DialogueLogger — writes three files per dialogue:

  .txt          clean human-readable transcript, with a Tokens/Outcome footer
  .csv          one row per turn (phase, speaker, text, persona traits, tokens)
  _meta.json    everything else worth keeping, in one place:
                  - dialogue metadata + outcome + tokens
                  - participation stats (gini, per-speaker counts, vote flips)
                  - phase durations, consensus-cycle metrics
                  - full personas (traits + beliefs)
                  - structured per-turn state (only when StateTracker is wired)
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from typing import Any, Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator
    from state import StructuredState


CSV_COLUMNS = [
    "dialogue_id", "turn_index", "phase", "speaker", "is_moderator", "text",
    "selected_reason", "tokens_in", "tokens_out",
    "role", "is_primary",
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "response_length",
]

_PERSONA_FIELDS = [
    "role", "is_primary",
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "response_length",
]


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


class DialogueLogger:

    def __init__(self, dialogue_id: str, topic: str, moderator_style: str) -> None:
        self.dialogue_id = dialogue_id
        self.topic = topic
        self.moderator_style = moderator_style

        log_dir = cfg.output.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file  = os.path.join(log_dir, f"{dialogue_id}.txt")
        self.csv_file  = os.path.join(log_dir, f"{dialogue_id}.csv")
        self.meta_file = os.path.join(log_dir, f"{dialogue_id}_meta.json")

        self._csv_rows: list[dict[str, Any]] = []
        self._speaker_turn_counts: dict[str, int] = {}
        self._phase_turn_counts: dict[str, int] = {}

    # ------------------------------------------------------------------

    def write_header(self, participant_names: list[str], opening_lines: list[str]) -> None:
        header = (
            f"Dialogue ID : {self.dialogue_id}\n"
            f"Participants: {', '.join(participant_names)}\n"
            f"Topic       : {self.topic}\n"
            f"Moderator   : {self.moderator_style}\n"
            + "=" * 50 + "\n"
        )
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(header)
            for line in opening_lines:
                f.write(f"{line}\n")
            f.write("\n")

    def append_line(self, line: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{line}\n\n")

    def buffer(
        self, line: str, selected_reason: str,
        state: "DialogueState", sims: list["Simulator"],
        tokens_in: int = 0, tokens_out: int = 0,
    ) -> None:
        if ":" not in line:
            return
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        is_moderator = speaker == "Moderator"

        row: dict[str, Any] = {
            "dialogue_id": self.dialogue_id,
            "turn_index": state.turn_index,
            "phase": state.phase,
            "speaker": speaker,
            "is_moderator": is_moderator,
            "text": text.strip(),
            "selected_reason": selected_reason,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            **{col: "" for col in _PERSONA_FIELDS},
        }
        if not is_moderator:
            sim = next((s for s in sims if s.name == speaker), None)
            if sim:
                for fld in _PERSONA_FIELDS:
                    row[fld] = sim.persona.get(fld, "")
            self._speaker_turn_counts[speaker] = self._speaker_turn_counts.get(speaker, 0) + 1
            self._phase_turn_counts[state.phase] = self._phase_turn_counts.get(state.phase, 0) + 1

        self._csv_rows.append(row)

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
        # --- CSV ---------------------------------------------------------
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._csv_rows)

        # --- .txt footer (tokens + outcome) ------------------------------
        total_in  = setup_tokens_in + dialogue_tokens_in
        total_out = setup_tokens_out + dialogue_tokens_out
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(
                f"\n--- Outcome: {outcome} ---\n"
                f"--- Tokens : setup={setup_tokens_in}/{setup_tokens_out}  "
                f"dialogue={dialogue_tokens_in}/{dialogue_tokens_out}  "
                f"total={total_in}/{total_out} (in/out) ---\n"
            )

        # --- _meta.json --------------------------------------------------
        meta: dict[str, Any] = {
            "dialogue_id": self.dialogue_id,
            "topic": self.topic,
            "moderator_style": self.moderator_style,
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
                "compromise_tested": bool(state.compromise_option),
                "compromise_option": state.compromise_option,
                "final_candidate_option": state.candidate_option,
                "final_preferred_option": state.preferred_option,
            },
            "personas": [p.persona.as_dict() for p in sims],
        }

        # Structured per-turn state — compact, optional
        if structured is not None:
            meta["structured"] = {
                "consensus_state_final": structured.consensus_state,
                "phase_evidence_final": dataclasses.asdict(structured.phase_evidence),
                "is_hard_blocker": {
                    name: ps.is_true_hard_blocker
                    for name, ps in structured.participants.items()
                },
                "public_preferences_final": {
                    name: ps.public_preference
                    for name, ps in structured.participants.items()
                },
                "turns": [
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
                             "confidence": su.confidence,
                             "condition": su.condition}
                            for su in t.stance_updates
                        ],
                    }
                    for t in structured.turns
                ],
            }

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    @property
    def paths(self) -> tuple[str, str, str]:
        return self.log_file, self.csv_file, self.meta_file
