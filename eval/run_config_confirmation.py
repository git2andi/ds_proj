"""Confirm sweep-selected settings across several held-out topics.

The script reads ``eval2/logs_config_sweep/sweep_selection.json`` produced by
``run_config_sweep.py`` and builds cumulative profiles in the same order:

1. current baseline
2. + selected duplicate-detection values
3. + selected issue-follow-up value
4. + selected consecutive-turn value
5. + selected small-group-closure values

Each profile is run on the same five three-participant topics selected from
``scenarios.txt``. For each topic, all profiles reuse the exact same generated
scenario and personas, so configuration effects are not confounded by setup variation. No manual setting entry is required:

    py eval2/run_config_confirmation.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation_metrics import extract_run_metrics, load_run, resolve_log_dir, write_csv
from experiment_common import (
    EVAL_DIR,
    ScenarioCase,
    cfg,
    config_overrides,
    prepare_dialogue_setup,
    read_scenarios,
    run_dialogue,
)
from llm_client import get_llm_client

LOG_DIR = "eval2/logs_config_confirmation"
OUTPUT_ROOT = EVAL_DIR / "logs_config_confirmation"
DEFAULT_SELECTION = EVAL_DIR / "logs_config_sweep" / "sweep_selection.json"
DEFAULT_TOPIC_COUNT = 5
DEFAULT_SEED_BASE = 2400

ConfigKey = tuple[str, str]


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    overrides: dict[ConfigKey, Any]


def dotted_to_overrides(values: dict[str, Any]) -> dict[ConfigKey, Any]:
    overrides: dict[ConfigKey, Any] = {}
    for dotted, value in values.items():
        if "." not in dotted:
            raise ValueError(f"invalid dotted config key in sweep selection: {dotted!r}")
        section, key = dotted.split(".", 1)
        if section not in cfg._raw or key not in cfg._raw[section]:
            raise KeyError(f"selected config key no longer exists: {dotted}")
        overrides[(section, key)] = value
    return overrides


def load_profiles(path: Path) -> tuple[list[Profile], int]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run 'py eval2/run_config_sweep.py' first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    selections = data.get("selections", {})
    order = data.get("selection_order", [])
    if not isinstance(selections, dict) or not isinstance(order, list) or not order:
        raise ValueError(f"invalid sweep selection file: {path}")

    profiles = [Profile("baseline", "Current config.yaml values.", {})]
    cumulative: dict[ConfigKey, Any] = {}
    for name in order:
        selected = selections.get(name, {})
        selected_values = selected.get("selected_values", {})
        if not isinstance(selected_values, dict):
            raise ValueError(f"invalid selected_values for {name}")
        cumulative.update(dotted_to_overrides(selected_values))
        profiles.append(
            Profile(
                name,
                f"Cumulative profile through {name}; selected variant: {selected.get('selected_variant', '')}.",
                dict(cumulative),
            )
        )
    participants = int(data.get("participants", 3))
    return profiles, participants


def spread_topics(cases: list[ScenarioCase], participants: int, count: int) -> list[ScenarioCase]:
    matching = [case for case in cases if case.participants == participants]
    if len(matching) < count:
        raise ValueError(
            f"scenarios.txt contains only {len(matching)} cases with {participants} participants; need {count}"
        )
    if count == 1:
        return [matching[len(matching) // 2]]
    indices = [round(index * (len(matching) - 1) / (count - 1)) for index in range(count)]
    return [matching[index] for index in dict.fromkeys(indices)]


def _dotted(values: dict[ConfigKey, Any]) -> dict[str, Any]:
    return {f"{section}.{key}": value for (section, key), value in values.items()}


def _write_metadata(run_dir: str, metadata: dict[str, Any]) -> None:
    if not run_dir:
        return
    path = resolve_log_dir(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _enrich(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("outcome") == "error" or not result.get("log_dir"):
        return result
    try:
        run_dir = resolve_log_dir(str(result["log_dir"]))
        extracted = extract_run_metrics(load_run(run_dir), run_dir)
        for key in (
            "structural_pass",
            "outcome_consistent",
            "participant_turns",
            "repair_rate",
            "fallback_rate",
            "repetition_per_10_turns",
            "same_speaker_repeats_per_10_turns",
            "issue_resolution_rate",
            "option_coverage_ratio",
            "question_answer_rate",
            "response_failures",
            "protocol_error_count",
            "tokens_per_participant_turn",
        ):
            result[key] = extracted[key]
        opened = float(extracted.get("issues_opened", 0) or 0)
        result["issue_stale_rate"] = float(extracted.get("issues_stale", 0) or 0) / opened if opened else 0.0
    except Exception as exc:
        result["metric_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.mean(values) if values else None


def _shown(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def write_summary(profiles: list[Profile], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Combined configuration confirmation",
        "",
        "Profiles are cumulative and use matched topics and seeds.",
        "",
        "| Profile | Completed | Structural pass | Outcomes | Avg turns | Option coverage | Repetition/10 turns | Stale issue rate | Same-speaker repeats/10 | Repairs/turn | Tokens/turn |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in profiles:
        group = [row for row in rows if row.get("variant") == profile.name]
        completed = [row for row in group if row.get("outcome") != "error" and not row.get("metric_error")]
        outcomes = Counter(str(row.get("outcome")) for row in group)
        outcome_text = ", ".join(f"{key}: {value}" for key, value in sorted(outcomes.items())) or "–"
        lines.append(
            f"| {profile.name} | {len(completed)}/{len(group)} | "
            f"{sum(bool(row.get('structural_pass')) for row in completed)}/{len(group)} | {outcome_text} "
            f"| {_shown(_mean(completed, 'participant_turns'), 1)} "
            f"| {_shown(_mean(completed, 'option_coverage_ratio'))} "
            f"| {_shown(_mean(completed, 'repetition_per_10_turns'))} "
            f"| {_shown(_mean(completed, 'issue_stale_rate'))} "
            f"| {_shown(_mean(completed, 'same_speaker_repeats_per_10_turns'))} "
            f"| {_shown(_mean(completed, 'repair_rate'))} "
            f"| {_shown(_mean(completed, 'tokens_per_participant_turn'), 0)} |"
        )
    errors = [row for row in rows if row.get("outcome") == "error" or row.get("metric_error")]
    if errors:
        lines.extend(["", "## Errors", ""])
        for row in errors:
            lines.append(
                f"- {row.get('variant')} / {row.get('topic')}: "
                f"{row.get('error') or row.get('metric_error')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--topics", type=int, default=DEFAULT_TOPIC_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        profiles, participants = load_profiles(args.selection.resolve())
        topics = spread_topics(read_scenarios(), participants, args.topics)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.list:
        print(f"Participants: {participants}")
        print("Profiles:")
        for profile in profiles:
            print(f"  {profile.name}: {json.dumps(_dotted(profile.overrides), ensure_ascii=False, sort_keys=True)}")
        print("Topics:")
        for case in topics:
            print(f"  {case.index}: {case.topic}")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "confirmation_runs.csv"
    summary_path = OUTPUT_ROOT / "confirmation_summary.md"
    rows: list[dict[str, Any]] = []
    total = len(profiles) * len(topics)
    position = 0

    llm = get_llm_client()
    for topic_index, case in enumerate(topics, start=1):
        seed = args.seed + topic_index
        print(f"\n=== shared setup topic {topic_index}/{len(topics)} | seed {seed}")
        print(case.topic)
        setup_error = ""
        setup_fingerprint = ""
        scenario = None
        personas = None
        try:
            scenario, personas, setup_fingerprint = prepare_dialogue_setup(
                case.topic,
                participants=participants,
                seed=seed,
                llm=llm,
            )
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

        for profile in profiles:
            position += 1
            print(f"\n=== [{position}/{total}] {profile.name} | topic {topic_index}/{len(topics)} | seed {seed}")
            current_values = {
                f"{section}.{key}": cfg._raw[section][key]
                for section, key in profile.overrides
            }
            tested_values = _dotted(profile.overrides)
            if setup_error:
                result: dict[str, Any] = {
                    "topic": case.topic,
                    "participants": participants,
                    "seed": seed,
                    "outcome": "error",
                    "log_dir": "",
                    "error": f"shared setup failed: {setup_error}",
                }
            else:
                with config_overrides(profile.overrides):
                    result = run_dialogue(
                        case.topic,
                        participants=participants,
                        seed=seed,
                        llm=llm,
                        log_dir=LOG_DIR,
                        scenario=scenario,
                        personas=personas,
                    )
                result = _enrich(result)
            metadata = {
                "experiment": "combined_confirmation",
                "variant": profile.name,
                "description": profile.description,
                "changed_settings": list(tested_values),
                "current_values": current_values,
                "tested_values": tested_values,
                "replicate": topic_index,
                "topic_index": topic_index,
                "scenario_index": case.index,
                "topic": case.topic,
                "seed": seed,
                "participants": participants,
                "paired_setup": True,
                "setup_fingerprint": setup_fingerprint,
            }
            _write_metadata(str(result.get("log_dir", "")), metadata)
            rows.append(
                {
                    "experiment": "combined_confirmation",
                    "variant": profile.name,
                    "description": profile.description,
                    "changed_settings": ",".join(tested_values),
                    "current_values": json.dumps(current_values, ensure_ascii=False, sort_keys=True),
                    "tested_values": json.dumps(tested_values, ensure_ascii=False, sort_keys=True),
                    "topic_index": topic_index,
                    "scenario_index": case.index,
                    "paired_setup": True,
                    "setup_fingerprint": setup_fingerprint,
                    **result,
                }
            )
            write_csv(csv_path, rows)
            write_summary(profiles, rows, summary_path)

    print(f"\nCompleted {len(rows)} runs.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
