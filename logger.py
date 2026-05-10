"""
logger.py
---------
DialogueLogger — handles all output for a single dialogue run.
Writes a .txt transcript and buffers/flushes a .csv data file.

Token logging
  flush() receives setup and dialogue token totals and appends one line
  to token_log.txt in the project root for cross-dialogue tracking.
  Format: date | dialogue_id | setup=in/out | dialogue=in/out | total=in/out | topic
"""

from __future__ import annotations

import csv
import datetime
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator


CSV_COLUMNS = [
    "dialogue_id", "turn_index", "phase", "speaker", "is_moderator", "text",
    "selected_reason", "last_addressed", "pending_question_target",
    "repetition_pressure", "tokens_in", "tokens_out",
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

_TOKEN_LOG = Path(__file__).parent / "token_log.txt"


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
        tokens_in: int = 0,
        tokens_out: int = 0,
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
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            **{col: "" for col in _PERSONA_FIELDS},
        }

        if not is_moderator:
            sim = next((s for s in sims if s.name == speaker), None)
            if sim:
                for field in _PERSONA_FIELDS:
                    row[field] = sim.persona.get(field, "")

        self._csv_rows.append(row)

    def flush(
        self,
        setup_tokens_in: int = 0,
        setup_tokens_out: int = 0,
        dialogue_tokens_in: int = 0,
        dialogue_tokens_out: int = 0,
        outcome: str = "pending",
    ) -> None:
        if cfg.output.save_csv:
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(self._csv_rows)

        if cfg.output.save_txt:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- Outcome: {outcome} ---\n")

        self._write_token_log(setup_tokens_in, setup_tokens_out, dialogue_tokens_in, dialogue_tokens_out, outcome)

    @property
    def paths(self) -> tuple[str, str]:
        return self.log_file, self.csv_file

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_token_log(
        self,
        setup_in: int,
        setup_out: int,
        dialogue_in: int,
        dialogue_out: int,
        outcome: str = "pending",
    ) -> None:
        total_in = setup_in + dialogue_in
        total_out = setup_out + dialogue_out
        date_str = datetime.date.today().isoformat()
        topic_short = self.topic[:60].replace("|", "/")
        line = (
            f"{date_str} | {self.dialogue_id} | "
            f"setup={setup_in}/{setup_out} | "
            f"dialogue={dialogue_in}/{dialogue_out} | "
            f"total={total_in}/{total_out} | "
            f"outcome={outcome} | "
            f"{topic_short}\n"
        )
        with open(_TOKEN_LOG, "a", encoding="utf-8") as f:
            f.write(line)
