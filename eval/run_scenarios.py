"""Run the topic/count pairs in scenarios.txt as automatic end-to-end tests.

Format of scenarios.txt:
    participant_count | topic

The script updates only the temporary experimental conditions required by each
case (participant count, seed, automatic setup, and output directory), restores
config.yaml afterward, and writes incremental CSV/JSON/Markdown summaries.

Examples
--------
Run all 100 scenarios:
    py eval/run_scenarios.py

Run the first five as a smoke test:
    py eval/run_scenarios.py --limit 5

Resume after an interrupted run:
    py eval/run_scenarios.py --resume
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .experiment_utils import (
        ROOT,
        ConfigExperimentSession,
        PathKey,
        run_topic_subprocess,
        write_csv,
        zip_directory,
    )
except ImportError:  # Direct execution: py eval/run_scenarios.py
    from experiment_utils import (
        ROOT,
        ConfigExperimentSession,
        PathKey,
        run_topic_subprocess,
        write_csv,
        zip_directory,
    )

DEFAULT_SCENARIOS = ROOT / "scenarios.txt"


def read_scenarios(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    cases: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"{path}:{line_number}: expected 'participant_count | topic'")
        count_text, topic = (part.strip() for part in line.split("|", 1))
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid participant count {count_text!r}") from exc
        if not 2 <= count <= 7:
            raise ValueError(f"{path}:{line_number}: participant count must be 2..7")
        if not topic:
            raise ValueError(f"{path}:{line_number}: topic must not be empty")
        cases.append({"index": len(cases) + 1, "participants": count, "topic": topic})
    if not cases:
        raise ValueError(f"{path} contains no scenarios")
    duplicates = [topic for topic, count in Counter(case["topic"] for case in cases).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate topics in {path}: {duplicates}")
    return cases


def row_from_payload(case: dict[str, Any], payload: dict[str, Any], run_dir: str, seed: int) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    turns = metrics.get("turns") or {}
    generation = metrics.get("generation") or {}
    questions = metrics.get("questions") or {}
    issues = metrics.get("issues") or {}
    stances = metrics.get("stances") or {}
    votes = metrics.get("votes") or {}
    tokens = metrics.get("tokens") or {}
    outcome = payload.get("outcome") or {}
    participant_turns = int(turns.get("participant", 0))
    vote_round = int(votes.get("round", 1))
    vote_records = payload.get("vote_records") or {}
    final_records = vote_records.get(str(vote_round), vote_records.get(vote_round, {})) or {}
    final_votes_valid = (
        len(final_records) == int(case["participants"])
        and all(str(record.get("status", "")) == "valid" for record in final_records.values())
    )
    movement_commit_ok = int(generation.get("selected_movement_actions", 0)) == int(
        generation.get("committed_movement_actions", 0)
    )
    closed = bool(payload.get("phase_history")) and payload["phase_history"][-1] == "CLOSED"
    participant_count_ok = len(payload.get("personas") or []) == int(case["participants"])
    outcome_ok = str(outcome.get("status", "")) in {"successful", "majority", "unresolved"}
    protocol_ok = not bool(votes.get("protocol_degraded", False))
    structural_pass = all((
        closed, participant_count_ok, outcome_ok, final_votes_valid, movement_commit_ok, protocol_ok
    ))
    repair_rate = int(generation.get("repairs", 0)) / max(1, participant_turns)
    quality_warnings: list[str] = []
    if repair_rate > 0.25:
        quality_warnings.append("repair_rate_above_25_percent")
    if int(stances.get("unexplained_movements", 0)) > 0:
        quality_warnings.append("unexplained_movement")
    if int(generation.get("semantic_reason_reuse", 0)) > max(3, participant_turns // 3):
        quality_warnings.append("high_semantic_reason_reuse")
    return {
        "case": case["index"],
        "participants": case["participants"],
        "topic": case["topic"],
        "seed": seed,
        "status": "ok",
        "test_pass": structural_pass,
        "quality_warnings": ";".join(quality_warnings),
        "outcome": outcome.get("status", ""),
        "final_option": outcome.get("final_option") or "",
        "participant_turns": participant_turns,
        "voluntary_turns": int(turns.get("voluntary", 0)),
        "moderator_turns": int(turns.get("moderator", 0)),
        "repairs": int(generation.get("repairs", 0)),
        "repair_rate": round(repair_rate, 4),
        "dropped": int(generation.get("dropped", 0)),
        "liveness_forced": int(generation.get("liveness_forced", 0)),
        "semantic_reason_reuse": int(generation.get("semantic_reason_reuse", 0)),
        "movement_realization_failures": int(generation.get("movement_realization_failures", 0)),
        "questions": f"{int(questions.get('answered', 0))}/{int(questions.get('opened', 0))}",
        "issues_resolved": int(issues.get("resolved", 0)),
        "issues_stale": int(issues.get("stale", 0)),
        "acceptances": int(stances.get("acceptances", 0)),
        "switches": int(stances.get("switches", 0)),
        "tokens_in": int(tokens.get("input", 0)),
        "tokens_out": int(tokens.get("output", 0)),
        "run_dir": run_dir,
        "error": "",
    }


def write_summaries(root: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["case"]))
    write_csv(root / "scenario_batch.csv", ordered)
    successful = [row for row in ordered if row.get("status") == "ok"]
    passed = [row for row in successful if bool(row.get("test_pass"))]
    outcomes = Counter(str(row.get("outcome")) for row in successful)
    participant_turns = [int(row["participant_turns"]) for row in successful]
    payload = {
        "completed": len(successful),
        "process_failures": len(ordered) - len(successful),
        "structural_passes": len(passed),
        "structural_failures": len(successful) - len(passed),
        "outcomes": dict(outcomes),
        "mean_participant_turns": round(statistics.mean(participant_turns), 2) if participant_turns else None,
        "median_participant_turns": statistics.median(participant_turns) if participant_turns else None,
        "cases": ordered,
    }
    (root / "scenario_batch_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 100-scenario batch",
        "",
        f"- Completed processes: {payload['completed']}",
        f"- Process failures: {payload['process_failures']}",
        f"- Structural passes: {payload['structural_passes']}",
        f"- Structural failures: {payload['structural_failures']}",
        f"- Outcomes: {dict(outcomes)}",
        f"- Mean participant turns: {payload['mean_participant_turns']}",
        f"- Median participant turns: {payload['median_participant_turns']}",
        "",
        "| # | Sims | Topic | Outcome | Participant turns | Repairs | Reuse | Test | Warnings |",
        "|---:|---:|---|---|---:|---:|---:|---|---|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['case']} | {row['participants']} | {row['topic']} | {row.get('outcome', '')} | "
            f"{row.get('participant_turns', '')} | {row.get('repairs', '')} | "
            f"{row.get('semantic_reason_reuse', '')} | "
            f"{'pass' if row.get('test_pass') else 'FAIL'} | {row.get('quality_warnings', '')} |"
        )
    (root / "scenario_batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "logs_scenarios_100")
    parser.add_argument("--base-seed", type=int, default=81000)
    parser.add_argument("--start-at", type=int, default=1, help="one-based scenario index")
    parser.add_argument("--limit", type=int, help="maximum number of selected scenarios to run")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--restore-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.restore_config:
        restored = ConfigExperimentSession.restore_stale_backup()
        print("Restored config.yaml." if restored else "No experiment backup was present.")
        return 0
    try:
        cases = read_scenarios(args.scenarios)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.list:
        for case in cases:
            print(f"{case['index']:03d} | {case['participants']} | {case['topic']}")
        return 0

    selected = [case for case in cases if case["index"] >= args.start_at]
    if args.limit is not None:
        if args.limit < 1:
            print("--limit must be positive.", file=sys.stderr)
            return 2
        selected = selected[: args.limit]
    if not selected:
        print("No scenarios selected.", file=sys.stderr)
        return 2

    output_root = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "scenario_batch_summary.json"
    rows_by_case: dict[int, dict[str, Any]] = {}
    if args.resume and summary_json.exists():
        try:
            previous = json.loads(summary_json.read_text(encoding="utf-8"))
            rows_by_case = {int(row["case"]): row for row in previous.get("cases", [])}
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            rows_by_case = {}

    print(f"Scenarios: {args.scenarios}")
    print(f"Selected: {len(selected)} of {len(cases)}")
    print(f"Output: {output_root}")

    relative_output = output_root.relative_to(ROOT) if output_root.is_relative_to(ROOT) else output_root
    with ConfigExperimentSession() as session:
        for position, case in enumerate(selected, start=1):
            if args.resume and rows_by_case.get(case["index"], {}).get("status") == "ok":
                print(f"[{position:03d}/{len(selected):03d}] skip completed case {case['index']:03d}")
                continue
            seed = args.base_seed + case["index"] - 1
            updates: dict[PathKey, Any] = {
                ("simulation", "num_participants"): case["participants"],
                ("simulation", "random_seed"): seed,
                ("environment", "mode"): "auto",
                ("participants", "mode"): "auto",
                ("output", "log_dir"): relative_output.as_posix(),
                ("output", "write_prompts"): False,
                ("output", "debug_metrics"): False,
            }
            session.write(updates)
            print(
                f"[{position:03d}/{len(selected):03d}] case {case['index']:03d}, "
                f"n={case['participants']}: {case['topic']}",
                flush=True,
            )
            result = run_topic_subprocess(case["topic"], timeout_seconds=args.timeout_seconds)
            if result.ok and result.run_json is not None:
                row = row_from_payload(case, result.run_json, result.run_dir, seed)
            else:
                row = {
                    "case": case["index"],
                    "participants": case["participants"],
                    "topic": case["topic"],
                    "seed": seed,
                    "status": "failed",
                    "test_pass": False,
                    "quality_warnings": "process_failure",
                    "outcome": "",
                    "run_dir": result.run_dir,
                    "error": (result.stderr or result.stdout).strip()[-4000:],
                }
                error_path = output_root / f"case_{case['index']:03d}_error.txt"
                error_path.write_text(
                    f"STDOUT\n======\n{result.stdout}\n\nSTDERR\n======\n{result.stderr}",
                    encoding="utf-8",
                )
            rows_by_case[case["index"]] = row
            write_summaries(output_root, list(rows_by_case.values()))

    rows = list(rows_by_case.values())
    write_summaries(output_root, rows)
    archive = zip_directory(output_root, output_root.parent / f"{output_root.name}.zip")
    selected_ids = {case["index"] for case in selected}
    failed = sum(
        row.get("status") != "ok" or not bool(row.get("test_pass"))
        for row in rows
        if int(row["case"]) in selected_ids
    )
    print(f"\nSummary: {output_root / 'scenario_batch_summary.md'}")
    print(f"CSV: {output_root / 'scenario_batch.csv'}")
    print(f"Archive: {archive}")
    print(f"Selected-case failures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
