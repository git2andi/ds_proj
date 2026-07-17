"""Run the ``participant_count | topic`` lines in scenarios.txt end to end.

Each line becomes one complete automatic run: scenario generation, persona
generation, and the full dialogue with the configured LLM. The participant
count is taken from the line; every other setting keeps its config.yaml value.
Results accumulate in ``eval/logs_scenarios/`` with one flat CSV row per run
and a Markdown summary.

Examples
--------
List the parsed cases without contacting the LLM:
    py eval/run_scenarios.py --list

Smoke-test the first three cases:
    py eval/run_scenarios.py --limit 3

Run only the five-participant cases with reproducible seeds:
    py eval/run_scenarios.py --counts 5 --seed 500
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiment_common import (
    ROOT,
    ScenarioCase,
    read_scenarios,
    run_dialogue,
    write_csv,
)

LOG_DIR = "eval/logs_scenarios"
OUTPUT_ROOT = ROOT / LOG_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None, help="alternative scenarios file")
    parser.add_argument("--list", action="store_true", help="list parsed cases and exit")
    parser.add_argument("--start", type=int, default=1, help="first case index to run (1-based)")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of cases to run (0 = all)")
    parser.add_argument(
        "--counts",
        type=str,
        default="",
        help="comma-separated participant counts to include, e.g. 2,3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed; case i runs with seed base+i (omit for nondeterministic runs)",
    )
    return parser.parse_args()


def select_cases(cases: list[ScenarioCase], args: argparse.Namespace) -> list[ScenarioCase]:
    selected = [case for case in cases if case.index >= args.start]
    if args.counts:
        wanted = {int(part) for part in args.counts.split(",") if part.strip()}
        selected = [case for case in selected if case.participants in wanted]
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    completed = [row for row in rows if row.get("outcome") != "error"]
    outcomes = Counter(str(row.get("outcome")) for row in rows)
    lines = [
        "# Scenario batch summary",
        "",
        f"Runs: {len(rows)} ({len(rows) - len(completed)} errors)",
        f"Outcomes: {dict(sorted(outcomes.items()))}",
        "",
        "| Participants | Runs | Successful | Majority | Unresolved | Avg turns | Avg tokens in |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in sorted({int(row["participants"]) for row in completed}):
        group = [row for row in completed if int(row["participants"]) == count]
        lines.append(
            f"| {count} | {len(group)} "
            f"| {sum(row['outcome'] == 'successful' for row in group)} "
            f"| {sum(row['outcome'] == 'majority' for row in group)} "
            f"| {sum(row['outcome'] == 'unresolved' for row in group)} "
            f"| {statistics.mean(int(row['participant_turns']) for row in group):.1f} "
            f"| {statistics.mean(int(row['tokens_in']) for row in group):.0f} |"
        )
    errors = [row for row in rows if row.get("outcome") == "error"]
    if errors:
        lines += ["", "## Errors", ""]
        lines += [f"- case {row['case_index']} ({row['topic']!r}): {row['error']}" for row in errors]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cases = read_scenarios(args.file) if args.file else read_scenarios()
    selected = select_cases(cases, args)
    if args.list:
        for case in selected:
            print(f"{case.index:3d} | {case.participants} | {case.topic}")
        print(f"{len(selected)} case(s) selected.")
        return 0
    if not selected:
        print("No matching cases.", file=sys.stderr)
        return 2

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "scenario_runs.csv"
    summary_path = OUTPUT_ROOT / "scenario_summary.md"
    rows: list[dict[str, Any]] = []
    for position, case in enumerate(selected, start=1):
        seed = args.seed + case.index if args.seed is not None else None
        print(f"\n=== [{position}/{len(selected)}] case {case.index}: n={case.participants} | {case.topic}")
        row = run_dialogue(
            case.topic,
            participants=case.participants,
            seed=seed,
            log_dir=LOG_DIR,
        )
        row = {"case_index": case.index, **row}
        rows.append(row)
        write_csv(csv_path, rows)
        write_summary(rows, summary_path)
        if row.get("outcome") == "error":
            print(f"ERROR: {row['error']}", file=sys.stderr)

    print(f"\nCompleted {len(rows)} run(s).")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
