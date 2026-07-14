"""Compact human-readable and structured run logging."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

from config_loader import cfg
from models import DialogueState, IssueKind, IssueStatus, RunOutcome, VoteStatus


class DialogueLogger:
    def __init__(self, topic: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic.strip()).strip("_")[:45] or "manual"
        self.run_id = f"{stamp}_{slug}"
        root = Path(cfg.root) / str(cfg.output.log_dir)
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_prompt(self, prompt: str, kind: str) -> str:
        if not bool(cfg.output.get("write_prompts", False)):
            return ""
        path = self.run_dir / str(cfg.output.prompt_file)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, "prompt": prompt}, ensure_ascii=False) + "\n")
        return str(path)

    def write_run(self, state: DialogueState, outcome: RunOutcome, *, seed: int) -> dict[str, str]:
        transcript = self.run_dir / str(cfg.output.transcript_file)
        json_path = self.run_dir / str(cfg.output.json_file)
        metrics_path = Path(cfg.root) / str(cfg.output.log_dir) / str(cfg.output.metrics_csv)
        metrics = metrics_for(state, outcome)
        transcript.write_text("\n".join(self._transcript_lines(state, outcome, seed, metrics)), encoding="utf-8")

        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "seed": seed,
            "scenario": _jsonable(state.scenario),
            "personas": [_jsonable(persona) for persona in state.personas],
            "runtimes": _jsonable(state.runtimes),
            "phase_history": list(state.phase_history),
            "issue_history": _jsonable(state.issue_history),
            "turns": [_turn_payload(turn, include_action=bool(cfg.output.get("write_action_trace", False))) for turn in state.turns],
            "first_round_votes": dict(state.first_round_votes),
            "vote_records": _jsonable(state.vote_records),
            "votes": dict(state.votes),
            "outcome": _jsonable(outcome),
            "metrics": metrics,
            "failed_generation_attempts": _jsonable([
                attempt
                for attempt in state.generation_attempts
                if attempt.final_status in {"dropped", "fallback"}
            ]),
        }
        if bool(cfg.output.get("debug_metrics", False)):
            payload["generation_attempts"] = _jsonable(state.generation_attempts)
            payload["validation_failures"] = dict(state.validation_failures)
            payload["issue_records"] = _jsonable(state.issue_records)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_metrics(metrics_path, metrics, outcome)
        return {
            "dir": str(self.run_dir),
            "transcript": str(transcript),
            "json": str(json_path),
            "metrics_csv": str(metrics_path),
        }

    def _append_metrics(self, path: Path, metrics: dict[str, Any], outcome: RunOutcome) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "run_id": self.run_id,
            "outcome": outcome.status,
            "final_option": outcome.final_option or "",
            "participant_turns": metrics["turns"]["participant"],
            "voluntary_turns": metrics["turns"]["voluntary"],
            "moderator_turns": metrics["turns"]["moderator"],
            "repairs": metrics["generation"]["repairs"],
            "dropped_turns": metrics["generation"]["dropped"],
            "questions_answered": metrics["questions"]["answered"],
            "questions_opened": metrics["questions"]["opened"],
            "issues_resolved": metrics["issues"]["resolved"],
            "issues_stale": metrics["issues"]["stale"],
            "visible_switches": metrics["stances"]["switches"],
            "visible_acceptances": metrics["stances"]["acceptances"],
            "compromise_proposals": metrics["compromise"]["proposals"],
            "revote_skipped": metrics["votes"]["revote_skipped"],
            "semantic_reason_reuse": metrics["generation"]["semantic_reason_reuse"],
            "vote_fallbacks": metrics["generation"]["vote_fallbacks"],
            "mandatory_movement_failures": metrics["generation"]["mandatory_movement_failures"],
            "llm_calls": metrics["tokens"]["llm_calls"],
            "tokens_in": metrics["tokens"]["input"],
            "tokens_out": metrics["tokens"]["output"],
        }
        fields_ = list(row)
        write_header = not path.exists()
        mode = "a"
        if path.exists():
            existing = path.read_text(encoding="utf-8").splitlines()[0].split(",") if path.stat().st_size else []
            if existing != fields_:
                mode = "w"
                write_header = True
        with path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields_)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _transcript_lines(
        self,
        state: DialogueState,
        outcome: RunOutcome,
        seed: int,
        metrics: dict[str, Any],
    ) -> list[str]:
        lines = [
            f"# Dialogue run {self.run_id}",
            "",
            f"Topic: {state.scenario.topic}",
            f"Random seed: {seed}",
            "",
            "## Public option board",
            "",
        ]
        for fact in state.scenario.shared_context:
            lines.append(f"- Shared: {fact}")
        for option in state.scenario.options:
            lines.append(f"- {option.public_line()}")

        lines += ["", "## Participants", "", "| Participant | E/V/D/S | Initial preference | Hard blocker |", "|---|---:|---|---:|"]
        for persona in state.personas:
            traits = persona.sim_params
            lines.append(
                f"| {persona.name} | {traits.engagement}/{traits.verbosity}/{traits.directness}/{traits.stubbornness} "
                f"| {persona.preferred_option} | {'yes' if persona.hard_blocker else 'no'} |"
            )

        lines += ["", "## Transcript", ""]
        for turn in state.turns:
            lines.append(f"**{turn.speaker_name}:** {turn.text}")

        lines += [
            "",
            "## Outcome",
            "",
            f"- Status: {outcome.status}",
            f"- Final option: {outcome.final_option or 'none'}",
            f"- Votes: {outcome.votes}",
            f"- Reason: {outcome.reason}",
            "",
            "## Run summary",
            "",
            f"- Participant turns: {metrics['turns']['participant']}",
            f"- Voluntary turns: {metrics['turns']['voluntary']}",
            f"- Moderator turns: {metrics['turns']['moderator']}",
            f"- Repairs / dropped turns: {metrics['generation']['repairs']} / {metrics['generation']['dropped']}",
            f"- Vote fallbacks / failed movement realizations: {metrics['generation']['vote_fallbacks']} / {metrics['generation']['mandatory_movement_failures']}",
            f"- Questions answered: {metrics['questions']['answered']}/{metrics['questions']['opened']}",
            f"- Issues resolved / stale: {metrics['issues']['resolved']} / {metrics['issues']['stale']}",
            f"- Visible acceptances / switches: {metrics['stances']['acceptances']} / {metrics['stances']['switches']}",
            f"- Compromise proposals: {metrics['compromise']['proposals']}",
            f"- Re-vote skipped for no movement: {'yes' if metrics['votes']['revote_skipped'] else 'no'}",
            f"- Semantic reason reuse: {metrics['generation']['semantic_reason_reuse']}",
            f"- Repair causes: {metrics['generation']['repair_causes'] or {}}",
            f"- LLM calls: {metrics['tokens']['llm_calls']}",
            f"- Input / output tokens: {metrics['tokens']['input']} / {metrics['tokens']['output']}",
            "",
            "### Participant summary",
            "",
            "| Participant | Total | Voluntary | Avg words | Initial → final |",
            "|---|---:|---:|---:|---|",
        ]
        for pid, row in metrics["participants"].items():
            lines.append(
                f"| {row['name']} | {row['turns']} | {row['voluntary']} | {row['avg_words']:.1f} "
                f"| {row['initial_preference']} → {row['final_preference']} |"
            )
        return lines


def metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    participant_turns = [turn for turn in state.turns if not turn.moderator]
    moderator_turns = [turn for turn in state.turns if turn.moderator]
    repairs = sum(turn.repair_count for turn in participant_turns)
    issue_rows = list(state.issue_history) + ([state.active_issue] if state.active_issue else [])
    question_rows = [issue for issue in issue_rows if issue and issue.kind is IssueKind.QUESTION]
    concern_rows = [issue for issue in issue_rows if issue and issue.kind is IssueKind.CONCERN]
    final_records = state.vote_records.get(state.vote_round, {})

    participants: dict[str, Any] = {}
    for persona in state.personas:
        turns = [turn for turn in participant_turns if turn.speaker_id == persona.id]
        runtime = state.runtimes[persona.id]
        participants[persona.id] = {
            "name": persona.name,
            "traits": {
                "engagement": persona.sim_params.engagement,
                "verbosity": persona.sim_params.verbosity,
                "directness": persona.sim_params.directness,
                "stubbornness": persona.sim_params.stubbornness,
            },
            "turns": len(turns),
            "voluntary": sum(turn.voluntary for turn in turns),
            "avg_words": mean([turn.word_count for turn in turns]) if turns else 0.0,
            "initial_preference": persona.preferred_option,
            "final_preference": runtime.public_preference or runtime.preferred_option,
        }

    return {
        "turns": {
            "participant": len(participant_turns),
            "voluntary": sum(turn.voluntary for turn in participant_turns),
            "mandatory": sum(turn.mandatory for turn in participant_turns),
            "moderator": len(moderator_turns),
        },
        "generation": {
            "repairs": repairs,
            "dropped": state.stats.dropped_turns,
            "liveness_forced": state.stats.liveness_forced_turns,
            "semantic_reason_reuse": state.stats.semantic_reason_reuse,
            "vote_fallbacks": state.stats.vote_fallbacks,
            "mandatory_movement_failures": state.stats.mandatory_movement_failures,
            "repair_causes": dict(state.validation_failures),
        },
        "questions": {
            "opened": len(question_rows),
            "answered": sum(issue.response_count > 0 for issue in question_rows),
        },
        "issues": {
            "opened": len(issue_rows),
            "resolved": sum(issue.status is IssueStatus.RESOLVED for issue in issue_rows if issue),
            "stale": sum(issue.status is IssueStatus.STALE for issue in issue_rows if issue),
            "concerns_opened": len(concern_rows),
            "concerns_resolved": sum(issue.status is IssueStatus.RESOLVED for issue in concern_rows),
            "concerns_stale": sum(issue.status is IssueStatus.STALE for issue in concern_rows),
        },
        "stances": {
            "switches": sum(runtime.visible_switches for runtime in state.runtimes.values()),
            "acceptances": sum(len(runtime.public_acceptances) for runtime in state.runtimes.values()),
            "narrowing_movements": state.stats.narrowing_movements,
        },
        "compromise": {
            "proposals": state.stats.compromise_proposals,
            "acceptances": state.stats.compromise_acceptances,
        },
        "coverage": {
            option_id: coverage.substantive_count for option_id, coverage in state.coverage.items()
        },
        "votes": {
            "round": state.vote_round,
            "valid": sum(record.status is VoteStatus.VALID for record in final_records.values()),
            "unclear": sum(record.status is not VoteStatus.VALID for record in final_records.values()),
            "protocol_degraded": state.vote_protocol_degraded,
            "revote_skipped": state.revote_skipped_no_movement,
        },
        "tokens": {
            "llm_calls": state.stats.llm_calls + state.stats.setup_llm_calls,
            "input": state.stats.input_tokens,
            "output": state.stats.output_tokens,
        },
        "participants": participants,
        "outcome": {"status": outcome.status, "final_option": outcome.final_option},
    }


def _turn_payload(turn: Any, *, include_action: bool) -> dict[str, Any]:
    payload = {
        "index": turn.index,
        "phase": turn.phase.value,
        "speaker_id": turn.speaker_id,
        "speaker_name": turn.speaker_name,
        "text": turn.text,
        "moderator": turn.moderator,
        "mandatory": turn.mandatory,
        "voluntary": turn.voluntary,
        "liveness_forced": turn.liveness_forced,
        "priority": int(turn.priority),
        "repair_count": turn.repair_count,
        "issue_event": turn.issue_event,
        "stance_update": _jsonable(turn.stance_update),
        "vote_option": turn.vote_option,
        "word_count": turn.word_count,
        "prompt_tokens": turn.prompt_tokens,
        "output_tokens": turn.output_tokens,
        "intended_word_max": turn.intended_word_max,
    }
    if include_action:
        payload["action"] = _jsonable(turn.action)
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Counter):
        return dict(value)
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset, tuple, list)):
        return [_jsonable(item) for item in value]
    return value
