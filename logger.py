"""
logger.py
---------
DialogueLogger — handles all output for a single dialogue run.
Writes a .txt transcript and buffers/flushes a .csv data file.
"""

from __future__ import annotations

import csv
import os
from typing import Any, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator


CSV_COLUMNS = [
    "dialogue_id", "turn_index", "phase", "speaker", "is_moderator", "text",
    "selected_reason", "last_addressed", "pending_question_target",
    "repetition_pressure",
    # Persona fields (empty for moderator lines)
    "role", "is_primary",
    "assertiveness", "friendliness", "talkativeness", "agreeableness",
    "patience", "contrarian", "response_length",
]

_PERSONA_FIELDS = [
    "role", "is_primary",
    "assertiveness", "friendliness", "talkativeness", "agreeableness",
    "patience", "contrarian", "response_length",
]


class DialogueLogger:

    def __init__(self, dialogue_id: str, topic: str, moderator_style: str) -> None:
        self.dialogue_id = dialogue_id
        self.topic = topic
        self.moderator_style = moderator_style

        log_dir = cfg.output.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{dialogue_id}.txt")
        self.csv_file = os.path.join(log_dir, f"{dialogue_id}.csv")
        self._csv_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface
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
        if cfg.output.save_txt:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{line}\n\n")

    def buffer(
        self,
        line: str,
        selected_reason: str,
        state: "DialogueState",
        sims: list["Simulator"],
    ) -> None:
        """Parse a dialogue line and append a CSV row."""
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
            "last_addressed": state.last_addressed or "",
            "pending_question_target": state.pending_question_target or "",
            "repetition_pressure": round(state.repetition_pressure, 3),
            **{col: "" for col in _PERSONA_FIELDS},
        }

        if not is_moderator:
            sim = next((s for s in sims if s.name == speaker), None)
            if sim:
                for field in _PERSONA_FIELDS:
                    row[field] = sim.persona.get(field, "")

        self._csv_rows.append(row)

    def flush(self) -> None:
        if not cfg.output.save_csv:
            return
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._csv_rows)

    @property
    def paths(self) -> tuple[str, str]:
        return self.log_file, self.csv_file
