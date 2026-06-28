"""Run logging: transcript, JSON payload, and optional prompt files."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from config_loader import cfg
from models import DialogueState, RunOutcome


class DialogueLogger:
    def __init__(self, run_id: str, topic: str) -> None:
        self.run_id = run_id
        self.topic = topic
        base = Path(str(cfg.output.log_dir))
        if not base.is_absolute():
            base = cfg.root / base
        self.log_root = base
        self.base_dir = base / run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_counter = 0
        # All prompts for a run go to a single JSONL, written only when enabled.
        self.prompt_path = self.base_dir / str(cfg.output.prompt_file)

    def write_prompt(self, prompt: str, kind: str) -> str:
        if not bool(cfg.output.write_prompts):
            return ""
        self.prompt_counter += 1
        record = {"seq": self.prompt_counter, "kind": kind, "prompt": prompt}
        with self.prompt_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(self.prompt_path)

    def finish(self, state: DialogueState, outcome: RunOutcome) -> dict[str, str]:
        transcript_path = self.base_dir / str(cfg.output.transcript_file)
        transcript_path.write_text("\n".join(self._transcript_lines(state, outcome)) + "\n", encoding="utf-8")
        json_path = self.base_dir / str(cfg.output.json_file)
        json_path.write_text(json.dumps(self._json_payload(state, outcome), ensure_ascii=False, indent=2), encoding="utf-8")
        csv_path = self._append_metrics_csv(state, outcome)
        return {"dir": str(self.base_dir), "transcript": str(transcript_path), "json": str(json_path), "metrics_csv": str(csv_path)}

    def _append_metrics_csv(self, state: DialogueState, outcome: RunOutcome) -> Path:
        path = self.log_root / str(cfg.output.metrics_csv)
        row = flat_metrics_for(self.run_id, state, outcome)
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return path

    def _transcript_lines(self, state: DialogueState, outcome: RunOutcome) -> list[str]:
        lines = [f"# Dialogue run {self.run_id}", "", f"Topic: {state.scenario.topic}", "", "## Options", ""]
        for option in state.scenario.options:
            lines.append(f"- {option.public_line()}")
        lines += ["", "## Participants", ""]
        for persona in state.personas:
            stubborn = " stubborn" if persona.traits.agreeableness == 1 else ""
            lines.append(f"### {persona.name} ({persona.role}){stubborn}")
            lines.append(
                f"traits: open={persona.traits.openness} consc={persona.traits.conscientiousness} "
                f"extra={persona.traits.extraversion} agree={persona.traits.agreeableness} "
                f"neuro={persona.traits.neuroticism} length={persona.traits.response_length} "
                f"compromise={persona.traits.compromise_willingness:.2f} initiative={persona.traits.initiative:.2f}"
            )
            lines.append(f"goal: {persona.private_goal}")
            lines.append(f"prefers Option {persona.preferred_option}; accepts {persona.acceptable_options}; soft concerns {persona.soft_rejections}; hard rejects {persona.hard_rejections}")
            if persona.option_scores:
                lines.append("hidden scores: " + ", ".join(f"{opt}={persona.option_scores[opt]}" for opt in sorted(persona.option_scores)))
            lines.append(f"concern: {persona.main_concern}")
            for option_id, reasons in persona.reasons.items():
                if reasons:
                    lines.append(f"reasons for {option_id}: {' | '.join(reasons)}")
            if persona.reservation:
                lines.append(f"reservation: {persona.reservation}")
            if persona.reconsider_if:
                lines.append(f"would reconsider if: {persona.reconsider_if}")
            lines.append("")
        lines += ["", "## Transcript", ""]
        for turn in state.turns:
            lines.append(f"**{turn.speaker_name}:** {turn.text}")
        lines += ["", "## Outcome", "", f"Status: {outcome.status}", f"Final option: {outcome.final_option}", f"Reason: {outcome.reason}"]
        lines += ["", "## Metrics", ""]
        for key, value in metrics_for(state, outcome).items():
            lines.append(f"- {key}: {value}")
        tokens = token_summary_for(state)
        lines += ["", f"--- Tokens : setup={tokens['setup_tokens_in']}/{tokens['setup_tokens_out']} dialogue={tokens['dialogue_tokens_in']}/{tokens['dialogue_tokens_out']} total={tokens['total_tokens_in']}/{tokens['total_tokens_out']} (in/out) ---"]
        return lines

    def _json_payload(self, state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "scenario": _to_jsonable(state.scenario),
            "personas": [_to_jsonable(p) for p in state.personas],
            "turns": [_to_jsonable(t) for t in state.turns],
            "outcome": _to_jsonable(outcome),
            "metrics": metrics_for(state, outcome),
            "tokens": token_summary_for(state),
        }


def token_summary_for(state: DialogueState) -> dict[str, int]:
    return {
        "setup_tokens_in": int(state.setup_tokens_in),
        "setup_tokens_out": int(state.setup_tokens_out),
        "dialogue_tokens_in": int(state.dialogue_tokens_in),
        "dialogue_tokens_out": int(state.dialogue_tokens_out),
        "total_tokens_in": int(state.setup_tokens_in + state.dialogue_tokens_in),
        "total_tokens_out": int(state.setup_tokens_out + state.dialogue_tokens_out),
    }


def metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
    moderator_turns = [t for t in state.turns if t.speaker_id == "moderator"]
    turn_counts = {p.name: state.runtimes[p.id].turn_count for p in state.personas}
    n_part = max(1, len(participant_turns))
    question_density = round(sum(1 for t in participant_turns if "?" in t.text) / n_part, 3)
    avg_words = round(sum(len(t.text.split()) for t in participant_turns) / n_part, 1)
    repaired = sum(1 for t in participant_turns if t.repaired)
    flagged = sum(1 for t in participant_turns if t.validation_issues)
    avg_words_by_persona = {
        p.name: round(
            sum(len(t.text.split()) for t in participant_turns if t.speaker_id == p.id)
            / max(1, state.runtimes[p.id].turn_count), 1
        )
        for p in state.personas
    }
    return {
        "participant_turns": len(participant_turns),
        "moderator_turns": len(moderator_turns),
        "moderator_ratio": round(len(moderator_turns) / max(1, len(state.turns)), 3),
        "turn_counts": turn_counts,
        "avg_words_by_persona": avg_words_by_persona,
        "question_density": question_density,
        "avg_words_per_turn": avg_words,
        "repaired_turns": repaired,
        "repair_rate": round(repaired / n_part, 3),
        # Turns left with a (often warn-level) validation flag after any repair — informational.
        "flagged_turns": flagged,
        # Fraction of participants who voted for or accepted the final option (1.0 = full consensus).
        "final_support_fraction": _final_support_fraction(state, outcome),
        "option_coverage": {opt: {"mentions": c.mentions, "reasons": c.reasons, "objections": c.objections, "acceptances": c.acceptances} for opt, c in state.coverage.items()},
        "outcome_status": outcome.status,
        "final_option": outcome.final_option,
        # Per-run pacing target derived from group size/composition (see dialogue.derive_pacing).
        "min_discussion_turns": state.min_discussion_turns,
    } | token_summary_for(state)


def _final_support_fraction(state: DialogueState, outcome: RunOutcome) -> float:
    if not outcome.final_option:
        return 0.0
    final = outcome.final_option
    backers = sum(
        1 for p in state.personas
        if state.runtimes[p.id].explicit_vote == final or final in state.runtimes[p.id].accepted_options
    )
    return round(backers / max(1, len(state.personas)), 3)


def flat_metrics_for(run_id: str, state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    """One-row-per-run, scalar-only view for the master metrics.csv."""
    m = metrics_for(state, outcome)
    scalar = {k: v for k, v in m.items() if not isinstance(v, dict)}
    return {
        "run_id": run_id,
        "topic": state.scenario.topic,
        "num_participants": len(state.personas),
        "stubborn_participant": any(p.traits.agreeableness == 1 for p in state.personas),
        **scalar,
    }


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj
