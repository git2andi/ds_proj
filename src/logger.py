"""Run logging: transcript, JSON payload, metrics CSV, and optional prompts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from config_loader import cfg
from eval import flat_metrics_for, metrics_for, token_summary_for
from models import DialogueState, RunOutcome, STANCE_ACCEPTABLE, STANCE_DISLIKED, STANCE_NEUTRAL, STANCE_PREFERRED, STANCE_REJECTED


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
        fieldnames = list(row.keys())
        write_header = not path.exists()
        mode = "a"
        if path.exists():
            try:
                existing_header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            except IndexError:
                existing_header = []
            if existing_header != fieldnames:
                # Never append a second incompatible header into one CSV.
                mode = "w"
                write_header = True
        with path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return path

    def _transcript_lines(self, state: DialogueState, outcome: RunOutcome) -> list[str]:
        dialogue_provider, dialogue_model = _active_provider_model("dialogue")
        env_mode = str((cfg.get("environment", None) or {}).get("mode", "auto"))
        participants_mode = str((cfg.get("participants", None) or {}).get("mode", "auto"))
        seed = cfg.simulation.get("random_seed", None)
        moderator = _resolved_moderator_config()
        lines = [
            f"# Dialogue run {self.run_id}",
            "",
            f"Topic: {state.scenario.topic}",
            f"Environment: {state.scenario.environment_type}",
            # Provider differences change style, grounding, and failure patterns
            # (issue 9) — a transcript must say which backend wrote it and which
            # backend judged it (the two roles are independently configurable).
            f"Dialogue LLM: {dialogue_provider} ({dialogue_model})",
            "Runtime validation: deterministic critical checks (LLM disabled)",
            f"Environment mode: {env_mode}",
            f"Participants mode: {participants_mode}",
            f"Validation mode: {str((cfg.get('validation', None) or {}).get('mode', 'critical'))}",
            f"Moderator: enabled={moderator['enabled']} opening={moderator['opening']} mid_nudges={moderator['mid_discussion_nudges']} final_vote_call={moderator['final_vote_call']} closing={moderator['closing']}",
            f"Random seed: {seed if seed is not None else 'null'}",
            f"Pacing: min={state.min_discussion_turns} force={state.force_narrow_turns} hard={state.hard_max_turns}",
            "",
            "## Options",
            "",
        ]
        for option in state.scenario.options:
            lines.append(f"- {option.public_line()}")
        lines += ["", "## Simulated users", ""]
        for persona in state.personas:
            lines.append(f"### {persona.name}")
            params = persona.sim_params
            lines.append(
                f"OCEAN: open={persona.traits.openness} consc={persona.traits.conscientiousness} "
                f"extra={persona.traits.extraversion} agree={persona.traits.agreeableness} neuro={persona.traits.neuroticism}"
            )
            lines.append(
                f"sim params: engagement={params.engagement:.2f} verbosity={params.verbosity:.2f} "
                f"directness={params.directness:.2f} stubbornness={params.stubbornness:.2f} "
                f"switch_resistance={params.switch_resistance:.2f}"
            )
            lines.append(f"age/speech_style: {persona.age} — {persona.speech_style}")
            lines.append(f"profile: {persona.background}")
            lines.append(f"goal: {persona.private_goal}")
            lines.append(f"initial preference: {', '.join(persona.preferred_options)}")
            stance_bits = []
            for oid in state.scenario.option_ids:
                stance = (persona.option_stances or {}).get(oid)
                if not stance or stance.rank == STANCE_NEUTRAL:
                    continue
                label = {
                    STANCE_PREFERRED: "preferred",
                    STANCE_ACCEPTABLE: "acceptable",
                    STANCE_DISLIKED: "disliked",
                    STANCE_REJECTED: "rejected",
                }.get(stance.rank, str(stance.rank))
                reason = stance.reason_for if stance.rank >= STANCE_ACCEPTABLE else stance.reason_against
                stance_bits.append(f"{oid}:{label}" + (f" ({reason})" if reason else ""))
            if stance_bits:
                lines.append("initial option ranks: " + "; ".join(stance_bits))
            lines.append("")
        lines += ["", "## Transcript", ""]
        for turn in state.turns:
            lines.append(f"**{turn.speaker_name}:** {turn.text}")
        lines += [
            "",
            "## Outcome",
            "",
            f"Status: {outcome.status}",
            f"Final option: {outcome.final_option}",
            f"Reason: {outcome.reason}",
            "",
            "## Metrics",
            "",
        ]
        metrics = metrics_for(state, outcome)
        for section, values in metrics.items():
            if section == "metric_schema_version":
                lines.append(f"- metric_schema_version: {values}")
                continue
            lines.append(f"- {section}:")
            for key, value in values.items():
                lines.append(f"  - {key}: {value}")
        totals = metrics["token_usage"]["total"]
        lines += [
            "",
            f"--- Tokens: total={totals['input_tokens']}/{totals['output_tokens']} "
            f"across {totals['api_calls']} API calls (in/out) ---",
        ]
        return lines

    def _json_payload(self, state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
        dialogue_provider, dialogue_model = _active_provider_model("dialogue")
        return {
            "run_id": self.run_id,
            # Suite case identifier (eval item 9): present when the eval harness
            # sets output.case_id, empty for ad-hoc runs. Ties this run.json,
            # its transcript, and the summary row to one suite case.
            "case_id": str((cfg.get("output", None) or {}).get("case_id", "")),
            "suite_version": str((cfg.get("output", None) or {}).get("suite_version", "")),
            "case_fingerprint": str((cfg.get("output", None) or {}).get("case_fingerprint", "")),
            "topic": self.topic,
            "llm": {
                "dialogue": {"provider": dialogue_provider, "model": dialogue_model},
                "runtime_validation": {"mode": "critical", "llm_used": False},
            },
            "participants_mode": str((cfg.get("participants", None) or {}).get("mode", "auto")),
            "environment_mode": str((cfg.get("environment", None) or {}).get("mode", "auto")),
            "validation_mode": str((cfg.get("validation", None) or {}).get("mode", "critical")),
            "moderator_config": _resolved_moderator_config(),
            "scenario": _to_jsonable(state.scenario),
            "personas": [_to_jsonable(p) for p in state.personas],
            # Visible runtime state per simulator — switch_events in particular
            # are needed for preference-movement analysis (todo issues 4/5).
            "runtimes": {pid: _to_jsonable(rt) for pid, rt in state.runtimes.items()},
            # Thread-engine state and the active repair state; the thread that
            # routed each turn is in controller_trace ("routed_thread_id").
            "threads": {tid: _to_jsonable(t) for tid, t in state.threads.items()},
            "active_repair": _to_jsonable(state.active_repair) if state.active_repair else None,
            "repair_history": [_to_jsonable(r) for r in state.repair_history],
            "turns": [_to_jsonable(t) for t in state.turns],
            "outcome": _to_jsonable(outcome),
            # Ordered phase trace (pacing / discussion / narrowing / closure).
            # Part of the analyzable structured trace, and the surface where
            # issue 6 is checkable: closure entries appear only once the outcome
            # is actually resolved, never on a run still heading into compromise.
            "phase_history": list(state.phase_history),
            # Immutable controller trace: pre-turn routing snapshot + realized
            # result per participant turn, plus phase transitions and moderator/
            # peer-procedure markers. Explains why every turn was selected.
            "controller_trace": _to_jsonable(state.controller_trace),
            "metrics": metrics_for(state, outcome),
            "tokens": token_summary_for(state),
            "token_usage_by_call_type": _to_jsonable(state.token_usage_by_call_type),
            "validation_stats": _to_jsonable(state.validation_stats),
        }


def _active_provider_model(role: str) -> tuple[str, str]:
    """The LLM backend a role used this run, read from config (issue 9)."""
    provider = str(cfg.llm.get(role, "")).lower()
    model = str(cfg.llm.models.get(provider, "unknown"))
    return provider, model


def _resolved_moderator_config() -> dict[str, bool]:
    """The moderator flags actually in force this run (issue 7), so the trace
    records whether a lower-/no-moderator mode was used. An absent section or a
    disabled master switch resolves the sub-flags accordingly."""
    mod = dict(cfg.get("moderator", None) or {})
    enabled = bool(mod.get("enabled", True))
    return {
        "enabled": enabled,
        "opening": enabled and bool(mod.get("opening", True)),
        "mid_discussion_nudges": enabled and bool(mod.get("mid_discussion_nudges", True)),
        "final_vote_call": enabled and bool(mod.get("final_vote_call", True)),
        "closing": enabled and bool(mod.get("closing", True)),
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
