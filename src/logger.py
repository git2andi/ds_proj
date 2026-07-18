"""Human-readable and structured run logging."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from config_loader import cfg
from models import DialogueState, RunOutcome


class DialogueLogger:
    def __init__(self, topic: str) -> None:
        root = cfg.root / str(cfg.output.log_dir)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in topic).split())[:45] or "run"
        self.directory = root / f"{stamp}_{slug}"
        self.directory.mkdir(parents=True, exist_ok=False)

    def write_run(
        self,
        state: DialogueState,
        outcome: RunOutcome,
        *,
        seed: int,
        llm: Any | None = None,
    ) -> dict[str, str]:
        metrics = metrics_for(state, outcome)
        payload = {
            "provenance": _provenance(seed=seed, llm=llm),
            "scenario": _jsonable(state.scenario),
            "personas": [_jsonable(persona) for persona in state.personas],
            "runtime": {pid: _jsonable(runtime) for pid, runtime in state.runtimes.items()},
            "phase_history": list(state.phase_history),
            "turns": [_turn_payload(turn) for turn in state.turns],
            "active_thread": _jsonable(state.active_thread),
            "closed_thread_keys": [list(key) for key in sorted(state.closed_thread_keys)],
            "public_point_counts": {
                f"{option_id}:{attribute}": count
                for (option_id, attribute), count in sorted(state.public_point_counts.items())
            },
            "recent_point_keys": [list(key) for key in state.recent_point_keys],
            "votes": dict(state.votes),
            "vote_records": _jsonable(state.vote_records),
            "outcome": _jsonable(outcome),
            "stats": _jsonable(state.stats),
            "metrics": metrics,
            "generation_attempts": [_jsonable(attempt) for attempt in state.generation_attempts],
            "validation_failures": dict(state.validation_failures),
            "protocol_errors": list(state.protocol_errors),
            "quality_flags": _quality_flags(state),
            "needs_review": bool(_quality_flags(state)),
        }

        json_path = self.directory / str(cfg.output.json_file)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        transcript_path = self.directory / str(cfg.output.transcript_file)
        transcript_path.write_text(
            "\n".join(self._transcript_lines(state, outcome, metrics)),
            encoding="utf-8",
        )

        metrics_path = cfg.root / str(cfg.output.log_dir) / str(cfg.output.metrics_csv)
        self._append_metrics(metrics_path, metrics, outcome)
        return {
            "dir": str(self.directory),
            "json": str(json_path),
            "transcript": str(transcript_path),
            "metrics": str(metrics_path),
        }

    @staticmethod
    def _append_metrics(path: Path, metrics: dict[str, Any], outcome: RunOutcome) -> None:
        row = {**metrics, "outcome": outcome.status, "final_option": outcome.final_option or ""}
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _transcript_lines(
        state: DialogueState,
        outcome: RunOutcome,
        metrics: dict[str, Any],
    ) -> list[str]:
        lines = [
            f"# {state.scenario.topic}",
            "",
            state.scenario.context_text,
            "",
            "## Options",
            "",
            *[f"- {option.public_line()}" for option in state.scenario.options],
            "",
            "## Participants",
            "",
        ]
        for persona in state.personas:
            traits = persona.sim_params
            lines.append(
                f"- **{persona.name}**: preferred {persona.preferred_option}; "
                f"engagement {traits.engagement}, verbosity {traits.verbosity}, "
                f"directness {traits.directness}, stubbornness {traits.stubbornness}"
            )
        lines.extend(["", "## Dialogue", ""])
        for turn in state.turns:
            lines.append(f"**{turn.speaker_name}:** {turn.text}")
        lines.extend(
            [
                "",
                "## Outcome",
                "",
                f"- Status: {outcome.status}",
                f"- Final option: {outcome.final_option or 'none'}",
                f"- Votes: {outcome.votes}",
                f"- Reason: {outcome.reason}",
                "",
                "## Core metrics",
                "",
            ]
        )
        lines.extend(f"- {key}: {value}" for key, value in metrics.items())
        return lines


def metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    participants = state.participant_turns
    moderator_turns = sum(turn.moderator for turn in state.turns)
    word_counts = [turn.word_count for turn in participants]
    visible_changes = sum(runtime.visible_switches for runtime in state.runtimes.values())
    fallbacks = sum(attempt.final_status == "fallback" for attempt in state.generation_attempts)
    repairs = sum(attempt.repair_text is not None for attempt in state.generation_attempts)
    dropped = sum(attempt.final_status == "dropped" for attempt in state.generation_attempts)
    vote_consistent = _vote_outcome_consistent(state, outcome)
    counts = Counter(turn.speaker_id for turn in participants if turn.voluntary)
    return {
        "participant_count": len(state.personas),
        "participant_turns": len(participants),
        "voluntary_turns": state.stats.voluntary_turns,
        "moderator_turns": moderator_turns,
        "moderator_ratio": round(moderator_turns / max(1, len(state.turns)), 4),
        "avg_words_per_participant_turn": round(sum(word_counts) / max(1, len(word_counts)), 2),
        "visible_preference_changes": visible_changes,
        "repair_turns": repairs,
        "dropped_turns": dropped,
        "fallback_turns": fallbacks,
        "response_failures": state.stats.response_failures,
        "protocol_errors": len(state.protocol_errors),
        "vote_outcome_consistent": vote_consistent,
        "input_tokens": state.stats.input_tokens,
        "output_tokens": state.stats.output_tokens,
        "llm_calls": state.stats.llm_calls + state.stats.setup_llm_calls,
        "voluntary_turns_by_persona": dict(counts),
    }


def _vote_outcome_consistent(state: DialogueState, outcome: RunOutcome) -> bool:
    counts = Counter(option_id for option_id in state.votes.values() if option_id)
    if outcome.status == "successful":
        return bool(counts) and max(counts.values()) == len(state.personas)
    if outcome.status == "majority":
        return bool(outcome.final_option) and counts[outcome.final_option] > len(state.personas) / 2
    return not counts or max(counts.values(), default=0) <= len(state.personas) / 2


def _turn_payload(turn: Any) -> dict[str, Any]:
    payload = _jsonable(turn)
    if not bool(cfg.output.get("write_action_trace", True)):
        payload.pop("action", None)
    return payload


def _provenance(*, seed: int, llm: Any | None) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "dialogue_provider": getattr(
            llm,
            "provider",
            str(cfg.llm.dialogue),
        ),
        "dialogue_model": getattr(
            llm,
            "model_id",
            str(cfg.llm.models.get(cfg.llm.dialogue)),
        ),
        "scenario_mode": str(cfg.environment.mode),
        "participant_mode": str(cfg.participants.mode),
        "action_trace_enabled": bool(
            cfg.output.get("write_action_trace", True)
        ),
    }


def _quality_flags(state: DialogueState) -> list[str]:
    flags: list[str] = []
    if state.stats.dropped_turns:
        flags.append(f"dropped_turns:{state.stats.dropped_turns}")
    if state.stats.fallback_turns:
        flags.append(f"fallback_turns:{state.stats.fallback_turns}")
    if state.stats.response_failures:
        flags.append(f"response_failures:{state.stats.response_failures}")
    if state.protocol_errors:
        flags.append(f"protocol_errors:{len(state.protocol_errors)}")
    return flags


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value
