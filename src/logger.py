"""
logger.py
---------
DialogueLogger — handles all output for a single dialogue run.

Outputs per dialogue:
  .txt          — clean human-readable transcript
  .csv          — structured row-per-turn data
  .jsonl        — per-turn trace for diagnostics (Stage 1)
  _summary.json — per-dialogue aggregate stats (Stage 1)

Token logging
  flush() receives setup and dialogue token totals and appends one line
  to token_log.txt in the project root for cross-dialogue tracking.
  Format: date | dialogue_id | setup=in/out | dialogue=in/out | total=in/out | outcome | topic
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

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
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "response_length",
]

_PERSONA_FIELDS = [
    "role", "is_primary",
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "response_length",
]

# token_log.txt lives in the project root, one level above src/
_TOKEN_LOG = Path(__file__).parent.parent / "token_log.txt"


# ---------------------------------------------------------------------------
# Stage 1 — per-turn trace record
# ---------------------------------------------------------------------------

@dataclass
class TurnTrace:
    """Lightweight per-turn diagnostic record written to .jsonl.
    Will be superseded by TurnRecord in Stage 5 (full state package).
    """
    turn_id: int
    phase: str
    speaker: str
    is_moderator: bool
    text: str
    selected_reason: str
    # Discourse
    addressees: list          # list[str] — participant names addressed
    is_question: bool
    mentioned_options: list   # list[str] — option letters A-D
    dialogue_act_estimated: str
    # Pressure & consensus
    repetition_pressure: float
    consensus_tier_used: str  # "soft"|"regex"|"reduced_opposition"|"llm"|"none"
    moderator_intervention_type: Optional[str]
    # Tokens & prompt metadata
    tokens_in: int
    tokens_out: int
    prompt_type: str          # "sim_turn"|"moderator"|"consensus_check"|"setup"


# ---------------------------------------------------------------------------
# Gini coefficient helper
# ---------------------------------------------------------------------------

def _compute_gini(values: list[int]) -> float:
    """Gini coefficient of participation (0=equal, 1=one person speaks all)."""
    if not values or len(values) <= 1:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    gini_num = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    return round(gini_num / (n * total), 4)


# ---------------------------------------------------------------------------
# DialogueLogger
# ---------------------------------------------------------------------------

class DialogueLogger:

    def __init__(self, dialogue_id: str, topic: str, moderator_style: str) -> None:
        self.dialogue_id = dialogue_id
        self.topic = topic
        self.moderator_style = moderator_style

        log_dir = cfg.output.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file     = os.path.join(log_dir, f"{dialogue_id}.txt")
        self.csv_file     = os.path.join(log_dir, f"{dialogue_id}.csv")
        self.jsonl_file   = os.path.join(log_dir, f"{dialogue_id}.jsonl")
        self.state_jsonl_file = os.path.join(log_dir, f"{dialogue_id}_state.jsonl")
        self.summary_file = os.path.join(log_dir, f"{dialogue_id}_summary.json")

        self._csv_rows: list[dict[str, Any]] = []
        self._turn_trace_id: int = 0
        self._speaker_turn_counts: dict[str, int] = {}   # for Gini

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
                for fld in _PERSONA_FIELDS:
                    row[fld] = sim.persona.get(fld, "")
            # Track participation for Gini
            self._speaker_turn_counts[speaker] = self._speaker_turn_counts.get(speaker, 0) + 1

        self._csv_rows.append(row)

    def trace_turn(self, trace: Optional[TurnTrace]) -> None:
        """Write one TurnTrace record to the Stage 1 JSONL file."""
        if trace is None:
            return
        if not cfg.logging.write_jsonl_trace:
            return
        record = dataclasses.asdict(trace)
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def trace_turn_record(self, record: Any) -> None:
        """Write one Stage 5 TurnRecord to *_state.jsonl (structured state layer)."""
        if record is None:
            return
        if not cfg.logging.write_jsonl_trace:
            return
        with open(self.state_jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def next_trace_id(self) -> int:
        self._turn_trace_id += 1
        return self._turn_trace_id

    def flush(
        self,
        setup_tokens_in: int = 0,
        setup_tokens_out: int = 0,
        dialogue_tokens_in: int = 0,
        dialogue_tokens_out: int = 0,
        outcome: str = "pending",
        summary_extras: Optional[dict[str, Any]] = None,
    ) -> None:
        if cfg.output.save_csv:
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(self._csv_rows)

        if cfg.output.save_txt:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- Outcome: {outcome} ---\n")

        self._write_token_log(
            setup_tokens_in, setup_tokens_out,
            dialogue_tokens_in, dialogue_tokens_out,
            outcome,
        )

        if cfg.logging.write_summary_json:
            self._write_summary_json(
                setup_tokens_in, setup_tokens_out,
                dialogue_tokens_in, dialogue_tokens_out,
                outcome,
                summary_extras or {},
            )

    @property
    def paths(self) -> tuple[str, str]:
        return self.log_file, self.csv_file

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_summary_json(
        self,
        setup_in: int,
        setup_out: int,
        dialogue_in: int,
        dialogue_out: int,
        outcome: str,
        extras: dict[str, Any],
    ) -> None:
        gini = _compute_gini(list(self._speaker_turn_counts.values()))
        summary: dict[str, Any] = {
            "dialogue_id": self.dialogue_id,
            "topic": self.topic,
            "moderator_style": self.moderator_style,
            "outcome": outcome,
            "participation_gini": gini,
            "speaker_turn_counts": self._speaker_turn_counts,
            "tokens_setup_in": setup_in,
            "tokens_setup_out": setup_out,
            "tokens_dialogue_in": dialogue_in,
            "tokens_dialogue_out": dialogue_out,
            "total_in": setup_in + dialogue_in,
            "total_out": setup_out + dialogue_out,
        }
        summary.update(extras)
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _write_token_log(
        self,
        setup_in: int,
        setup_out: int,
        dialogue_in: int,
        dialogue_out: int,
        outcome: str = "pending",
    ) -> None:
        if not cfg.logging.write_token_log:
            return
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
