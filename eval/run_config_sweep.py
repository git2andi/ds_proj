"""Run a compact low/current/high sweep for the remaining dialogue settings.

Only four configuration areas connected to observed defects are tested:

1. semantic duplicate detection
2. issue follow-up depth
3. consecutive participant turns
4. small-group closure pacing

The current configuration is run once as a shared paired baseline. Every
variant uses the same topic, participant count, and seeds. After the runs, the
script writes ``sweep_selection.json`` with the best deterministic candidate
for each setting. ``run_config_confirmation.py`` reads that file directly.

Normal use requires no arguments:

    py eval2/run_config_sweep.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation_metrics import extract_run_metrics, load_run, resolve_log_dir, write_csv
from experiment_common import (
    EVAL_DIR,
    cfg,
    config_overrides,
    prepare_dialogue_setup,
    run_dialogue,
)
from llm_client import get_llm_client

LOG_DIR = "eval2/logs_config_sweep"
OUTPUT_ROOT = EVAL_DIR / "logs_config_sweep"
DEFAULT_TOPIC = "Choose a coffee machine for a shared office kitchen"
DEFAULT_PARTICIPANTS = 3
DEFAULT_RUNS = 3
DEFAULT_SEED_BASE = 1000

ConfigKey = tuple[str, str]


@dataclass(frozen=True)
class Variant:
    label: str
    description: str
    overrides: dict[ConfigKey, Any]


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    target_metric: str
    variants: tuple[Variant, ...]

    @property
    def config_keys(self) -> tuple[ConfigKey, ...]:
        keys: list[ConfigKey] = []
        for variant in self.variants:
            for key in variant.overrides:
                if key not in keys:
                    keys.append(key)
        return tuple(keys)


def _raw(section: str, key: str) -> Any:
    return cfg._raw[section][key]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dotted_values(values: dict[ConfigKey, Any]) -> dict[str, Any]:
    return {f"{section}.{key}": value for (section, key), value in values.items()}


def _current_values(keys: tuple[ConfigKey, ...]) -> dict[ConfigKey, Any]:
    return {key: _raw(*key) for key in keys}


def build_experiments(participants: int) -> list[Experiment]:
    duplicate_threshold = float(_raw("language", "near_duplicate_similarity_threshold"))
    duplicate_lookback = int(_raw("language", "near_duplicate_recent_turns"))
    issue_cap = int(_raw("conversation", "issue_follow_up_cap"))
    consecutive = int(_raw("conversation", "max_consecutive_turns"))

    experiments = [
        Experiment(
            "duplicate_detection",
            "Tune duplicate sensitivity while changing threshold and lookback together.",
            "repetition_per_10_turns",
            (
                Variant(
                    "sensitive",
                    "Catch moderately similar wording across a longer recent window.",
                    {
                        ("language", "near_duplicate_similarity_threshold"): round(max(0.75, duplicate_threshold - 0.05), 2),
                        ("language", "near_duplicate_recent_turns"): duplicate_lookback + 2,
                    },
                ),
                Variant(
                    "conservative",
                    "Repair only very close repetitions in a shorter recent window.",
                    {
                        ("language", "near_duplicate_similarity_threshold"): round(min(0.99, duplicate_threshold + 0.04), 2),
                        ("language", "near_duplicate_recent_turns"): max(1, duplicate_lookback - 1),
                    },
                ),
            ),
        ),
        Experiment(
            "issue_follow_up",
            "Tune the bounded number of contributions allowed around an active issue.",
            "issue_stale_rate",
            (
                Variant(
                    "lower",
                    "Close issue threads one contribution earlier.",
                    {("conversation", "issue_follow_up_cap"): max(1, issue_cap - 1)},
                ),
                Variant(
                    "higher",
                    "Allow one additional issue contribution.",
                    {("conversation", "issue_follow_up_cap"): issue_cap + 1},
                ),
            ),
        ),
        Experiment(
            "consecutive_turns",
            "Tune the hard boundary for consecutive participant turns.",
            "same_speaker_repeats_per_10_turns",
            (
                Variant(
                    "strict",
                    "Do not allow the same participant twice consecutively unless the runtime obligation overrides it.",
                    {("conversation", "max_consecutive_turns"): max(1, consecutive - 1)},
                ),
                Variant(
                    "permissive",
                    "Allow one additional consecutive participant turn.",
                    {("conversation", "max_consecutive_turns"): consecutive + 1},
                ),
            ),
        ),
    ]

    small_group_max = int(_raw("conversation", "small_group_max_participants"))
    if participants <= small_group_max:
        no_bid = int(_raw("conversation", "small_group_extra_no_bid_rounds"))
        shared_extra = int(_raw("conversation", "small_group_shared_acceptance_extra_turns"))
        minimum_turns = float(_raw("conversation", "unanimous_closure_min_voluntary_turns_per_participant"))
        experiments.append(
            Experiment(
                "small_group_closure",
                "Tune the coupled evidence requirements for early small-group closure.",
                "participant_turns",
                (
                    Variant(
                        "faster",
                        "Close agreement with less empty-floor and acceptance padding.",
                        {
                            ("conversation", "small_group_extra_no_bid_rounds"): max(0, no_bid - 1),
                            ("conversation", "small_group_shared_acceptance_extra_turns"): max(0, shared_extra - 2),
                            ("conversation", "unanimous_closure_min_voluntary_turns_per_participant"): max(0.5, minimum_turns - 0.5),
                        },
                    ),
                    Variant(
                        "slower",
                        "Require more evidence before early closure.",
                        {
                            ("conversation", "small_group_extra_no_bid_rounds"): no_bid + 1,
                            ("conversation", "small_group_shared_acceptance_extra_turns"): shared_extra + 2,
                            ("conversation", "unanimous_closure_min_voluntary_turns_per_participant"): minimum_turns + 0.5,
                        },
                    ),
                ),
            )
        )
    return experiments


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
            "question_answer_rate",
            "response_failures",
            "protocol_error_count",
            "option_coverage_ratio",
            "repair_rate",
            "dropped_rate",
            "fallback_rate",
            "repetition_per_10_turns",
            "same_speaker_repeats_per_10_turns",
            "issue_resolution_rate",
            "tokens_per_participant_turn",
            "participant_turns",
            "hard_blocker_violations",
            "unexplained_movements",
        ):
            result[key] = extracted[key]
        opened = float(extracted.get("issues_opened", 0) or 0)
        result["issue_stale_rate"] = float(extracted.get("issues_stale", 0) or 0) / opened if opened else 0.0
    except Exception as exc:
        result["metric_error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--participants", type=int, default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="replicates per baseline or variant")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def _mean(group: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [float(row[key]) for row in group if row.get(key) not in (None, "")]
    return statistics.mean(values) if values else default


def _score(experiment: Experiment, group: list[dict[str, Any]]) -> tuple[float, ...]:
    completed = [row for row in group if row.get("outcome") != "error" and not row.get("metric_error")]
    errors = len(group) - len(completed)
    structural_failures = errors + sum(not bool(row.get("structural_pass")) for row in completed)
    protocol_violations = sum(
        (not bool(row.get("outcome_consistent")))
        + int(float(row.get("hard_blocker_violations", 0) or 0) > 0)
        + int(float(row.get("unexplained_movements", 0) or 0) > 0)
        + int(float(row.get("response_failures", 0) or 0) > 0)
        + int(float(row.get("protocol_error_count", 0) or 0) > 0)
        for row in completed
    )
    if not completed:
        return (float("inf"),)
    target = _mean(completed, experiment.target_metric)
    generation_cost = (
        _mean(completed, "repair_rate")
        + _mean(completed, "fallback_rate")
        + _mean(completed, "dropped_rate")
    )
    guardrail_deficit = (
        (1.0 - _mean(completed, "question_answer_rate"))
        + (1.0 - _mean(completed, "option_coverage_ratio"))
    )
    tokens = _mean(completed, "tokens_per_participant_turn")
    return (
        structural_failures,
        protocol_violations,
        guardrail_deficit,
        target,
        generation_cost,
        tokens,
    )


def choose_settings(
    experiments: list[Experiment],
    rows: list[dict[str, Any]],
    *,
    topic: str,
    participants: int,
    runs: int,
) -> dict[str, Any]:
    baseline = [row for row in rows if row.get("experiment") == "baseline"]
    selections: dict[str, Any] = {}
    for experiment in experiments:
        candidate_groups: dict[str, list[dict[str, Any]]] = {"current": baseline}
        candidate_values: dict[str, dict[ConfigKey, Any]] = {
            "current": _current_values(experiment.config_keys)
        }
        descriptions = {"current": "Current config.yaml values."}
        for variant in experiment.variants:
            candidate_groups[variant.label] = [
                row
                for row in rows
                if row.get("experiment") == experiment.name and row.get("variant") == variant.label
            ]
            candidate_values[variant.label] = {
                **_current_values(experiment.config_keys),
                **variant.overrides,
            }
            descriptions[variant.label] = variant.description
        scores = {label: _score(experiment, group) for label, group in candidate_groups.items()}
        selected = min(scores, key=scores.get)
        selections[experiment.name] = {
            "description": experiment.description,
            "target_metric": experiment.target_metric,
            "selected_variant": selected,
            "selected_description": descriptions[selected],
            "selected_values": _dotted_values(candidate_values[selected]),
            "current_values": _dotted_values(candidate_values["current"]),
            "candidate_scores": {label: list(score) for label, score in scores.items()},
        }
    return {
        "experiment": "config_sweep",
        "topic": topic,
        "participants": participants,
        "runs_per_candidate": runs,
        "selection_order": [experiment.name for experiment in experiments],
        "selections": selections,
    }


def write_summary(
    experiments: list[Experiment],
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
    path: Path,
) -> None:
    baseline = [row for row in rows if row.get("experiment") == "baseline"]
    lines = [
        "# Config sweep summary",
        "",
        "The current configuration is a shared paired baseline. Selection is deterministic and target-specific; structural failures take precedence over metric improvements.",
        "",
        "| Setting | Candidate | Selected | Target metric | Mean target | Structural passes | Avg repairs/turn | Avg tokens/turn |",
        "|---|---|:---:|---|---:|---:|---:|---:|",
    ]
    for experiment in experiments:
        selected = selection["selections"][experiment.name]["selected_variant"]
        candidates = [("current", baseline)] + [
            (
                variant.label,
                [row for row in rows if row.get("experiment") == experiment.name and row.get("variant") == variant.label],
            )
            for variant in experiment.variants
        ]
        for label, group in candidates:
            completed = [row for row in group if row.get("outcome") != "error" and not row.get("metric_error")]
            target = _mean(completed, experiment.target_metric) if completed else float("nan")
            structural = sum(bool(row.get("structural_pass")) for row in completed)
            lines.append(
                f"| {experiment.name} | {label} | {'yes' if label == selected else ''} | {experiment.target_metric} "
                f"| {target:.3f} | {structural}/{len(group)} | {_mean(completed, 'repair_rate'):.3f} "
                f"| {_mean(completed, 'tokens_per_participant_turn'):.0f} |"
            )
    lines.extend(["", "## Selected values", ""])
    for name in selection["selection_order"]:
        selected = selection["selections"][name]
        lines.append(
            f"- **{name}: {selected['selected_variant']}** — "
            f"`{_json(selected['selected_values'])}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    experiments = build_experiments(args.participants)
    if args.list:
        print(f"Baseline: current config; topic={args.topic!r}; participants={args.participants}; runs={args.runs}")
        for experiment in experiments:
            print(f"\n{experiment.name}: {experiment.description}")
            print(f"  current: {_json(_dotted_values(_current_values(experiment.config_keys)))}")
            for variant in experiment.variants:
                print(f"  {variant.label}: {_json(_dotted_values(variant.overrides))}")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "sweep_runs.csv"
    rows: list[dict[str, Any]] = []
    profiles: list[tuple[str, str, str, dict[ConfigKey, Any]]] = [
        ("baseline", "current", "Current config.yaml values.", {})
    ]
    profiles.extend(
        (experiment.name, variant.label, variant.description, variant.overrides)
        for experiment in experiments
        for variant in experiment.variants
    )
    llm = get_llm_client()

    for replicate in range(1, args.runs + 1):
        seed = args.seed + replicate
        print(f"\n=== shared setup {replicate}/{args.runs} (seed {seed})")
        setup_error = ""
        setup_fingerprint = ""
        scenario = None
        personas = None
        try:
            scenario, personas, setup_fingerprint = prepare_dialogue_setup(
                args.topic,
                participants=args.participants,
                seed=seed,
                llm=llm,
            )
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

        for experiment, variant, description, overrides in profiles:
            current = _dotted_values(_current_values(tuple(overrides))) if overrides else {}
            tested = _dotted_values(overrides)
            print(f"--- {experiment} [{variant}]")
            if setup_error:
                result: dict[str, Any] = {
                    "topic": args.topic,
                    "participants": args.participants,
                    "seed": seed,
                    "outcome": "error",
                    "log_dir": "",
                    "error": f"shared setup failed: {setup_error}",
                }
            else:
                with config_overrides(overrides):
                    result = run_dialogue(
                        args.topic,
                        participants=args.participants,
                        seed=seed,
                        llm=llm,
                        log_dir=LOG_DIR,
                        scenario=scenario,
                        personas=personas,
                    )
                result = _enrich(result)
            metadata = {
                "experiment": experiment,
                "variant": variant,
                "description": description,
                "changed_settings": list(tested),
                "current_values": current,
                "tested_values": tested,
                "replicate": replicate,
                "seed": seed,
                "participants": args.participants,
                "topic": args.topic,
                "paired_setup": True,
                "setup_fingerprint": setup_fingerprint,
            }
            _write_metadata(str(result.get("log_dir", "")), metadata)
            rows.append(
                {
                    "experiment": experiment,
                    "variant": variant,
                    "description": description,
                    "changed_settings": ",".join(tested),
                    "current_values": _json(current),
                    "tested_values": _json(tested),
                    "replicate": replicate,
                    "paired_setup": True,
                    "setup_fingerprint": setup_fingerprint,
                    **result,
                }
            )
            write_csv(csv_path, rows)

    selection = choose_settings(
        experiments,
        rows,
        topic=args.topic,
        participants=args.participants,
        runs=args.runs,
    )
    (OUTPUT_ROOT / "sweep_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(experiments, rows, selection, OUTPUT_ROOT / "sweep_summary.md")

    print(f"\nCompleted {len(rows)} runs.")
    print(f"CSV: {csv_path}")
    print(f"Selected settings: {OUTPUT_ROOT / 'sweep_selection.json'}")
    print(f"Summary: {OUTPUT_ROOT / 'sweep_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
