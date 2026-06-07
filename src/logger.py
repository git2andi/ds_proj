"""Logging for simulator runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from config_loader import PROJECT_ROOT, cfg
from schemas import DialogueRunResult, DialogueState, RunOutcome, Scenario


class DialogueLogger:
    def __init__(self, run_id: str, topic: str) -> None:
        self.run_id = run_id
        self.topic = topic
        log_root = Path(str(cfg.output.log_dir))
        if not log_root.is_absolute():
            log_root = PROJECT_ROOT / log_root
        self.base_dir = log_root / run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_dir = self.base_dir / "prompts"
        if bool(cfg.output.write_prompts):
            self.prompt_dir.mkdir(exist_ok=True)
        self.prompt_counter = 0

    def write_prompt(self, prompt: str, kind: str) -> str:
        if not bool(cfg.output.write_prompts):
            return ""
        self.prompt_counter += 1
        path = self.prompt_dir / f"{self.prompt_counter:04d}_{kind}.txt"
        path.write_text(prompt, encoding="utf-8")
        return str(path)

    def finish(self, state: DialogueState, outcome: RunOutcome, tokens: dict | None = None) -> dict[str, str]:
        tokens = tokens or {}
        transcript_lines = self._transcript_lines(state, outcome, tokens)
        transcript_path = self.base_dir / "transcript.md"
        transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

        json_path = self.base_dir / "run.json"
        payload = self._json_payload(state, outcome, tokens)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"transcript": str(transcript_path), "json": str(json_path), "dir": str(self.base_dir)}

    def _transcript_lines(self, state: DialogueState, outcome: RunOutcome, tokens: dict) -> list[str]:
        lines = [f"# Dialogue run {self.run_id}", "", f"Topic: {state.scenario.topic}", "", "## Options", ""]
        for option in state.scenario.options:
            lines.append(f"- {option.source_text()}")
        lines += ["", "## Participants", ""]
        for persona in state.personas:
            hard = " hard-blocker" if persona.is_hard_blocker else ""
            lines.append(
                f"- {persona.name} ({persona.role}){hard}: prefers Option {persona.preferred_option}; "
                f"acceptable {persona.acceptable_options}; soft rejections {persona.soft_rejections}; hard rejections {persona.hard_rejections}"
            )
        lines += ["", "## Transcript", ""]
        for turn in state.turns:
            lines.append(f"**{turn.speaker_name}:** {turn.text}")
        lines += ["", "## Outcome", "", f"Status: {outcome.status}", f"Final option: {outcome.final_option}", f"Reason: {outcome.reason}"]
        lines += ["", "## Tokens", "", _token_line(tokens)]
        lines += ["", "## Metrics", ""]
        metrics = self._metrics(state, outcome)
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        return lines

    def _json_payload(self, state: DialogueState, outcome: RunOutcome, tokens: dict) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "scenario": _to_jsonable(state.scenario),
            "personas": [_to_jsonable(p) for p in state.personas],
            "turns": [_to_jsonable(t) for t in state.turns],
            "outcome": _to_jsonable(outcome),
            "tokens": tokens,
            "metrics": self._metrics(state, outcome),
        }

    def _metrics(self, state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
        participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
        counts = {p.name: state.runtimes[p.id].turn_count for p in state.personas}
        total = max(1, len(participant_turns))
        question_turns = sum(1 for t in participant_turns if "?" in t.text)
        compromise_turns = sum(1 for t in participant_turns if t.act.proposes_option)
        moderator_turns = sum(1 for t in state.turns if t.speaker_id == "moderator")
        option_coverage = {
            oid: {
                "mentions": c.mentions,
                "reasons": c.reasons,
                "objections": c.objections,
                "acceptances": c.acceptances,
            }
            for oid, c in state.coverage.items()
        }
        return {
            "participant_turns": len(participant_turns),
            "moderator_turns": moderator_turns,
            "moderator_ratio": round(moderator_turns / max(1, len(state.turns)), 3),
            "turn_counts": counts,
            "question_density": round(question_turns / total, 3),
            "sim_compromise_proposals": compromise_turns,
            "readiness_score_final": state.readiness_score,
            "option_coverage": option_coverage,
            "outcome_status": outcome.status,
            "final_option": outcome.final_option,
        }


def _token_line(tokens: dict) -> str:
    setup = tokens.get("setup", [0, 0])
    dialogue = tokens.get("dialogue", [0, 0])
    total = tokens.get("total", [0, 0])
    return (
        f"setup={setup[0]}/{setup[1]}  dialogue={dialogue[0]}/{dialogue[1]}  "
        f"total={total[0]}/{total[1]} (in/out)"
    )


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "value"):
        return obj.value
    return obj
