"""Evaluate completed run logs using deterministic metrics only.

Default input and output are both ``eval/logs_scenarios``. The script can be
run from the project root or from ``eval/`` without changing paths:

    py eval/evaluate_runs.py

It writes:

- ``deterministic_runs.csv``: one row per completed run, plus batch errors
- ``trait_participants.csv``: one row per simulated participant
- ``trait_levels.csv``: level-wise trait realization summaries
- ``evaluation_summary.md``: compact report-ready aggregate results
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation_metrics import (
    EVAL_DIR,
    discover_batch_error_rows,
    extract_run_metrics,
    find_run_dirs,
    load_run,
    participant_trait_rows,
    trait_correlations,
    trait_level_rows,
    write_csv,
)

DEFAULT_LOGS = EVAL_DIR / "logs_scenarios"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS, help="directory containing run.json files")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory; defaults to the input directory",
    )
    parser.add_argument("--limit", type=int, default=0, help="evaluate at most this many completed runs")
    return parser.parse_args()


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return statistics.mean(values) if values else None


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        try:
            total += float(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def _fraction(numerator: float, denominator: float) -> str:
    if denominator <= 0:
        return "–"
    return f"{numerator}/{int(denominator)} ({100.0 * numerator / denominator:.1f}%)"


def _shown(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "–" if value is None else f"{value:.{digits}f}{suffix}"


def write_summary(
    run_rows: list[dict[str, Any]],
    participant_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    completed = [row for row in run_rows if row.get("outcome") != "error"]
    errors = [row for row in run_rows if row.get("outcome") == "error"]
    outcomes = Counter(str(row.get("outcome")) for row in completed)
    participant_turns = _sum(completed, "participant_turns")
    questions_opened = _sum(completed, "questions_opened")
    questions_answered = _sum(completed, "questions_answered")
    selected_movements = _sum(completed, "selected_movement_actions")
    direct_movements = _sum(completed, "directly_realized_movement_actions")
    movement_fallbacks = _sum(completed, "movement_fallbacks")
    committed_movements = _sum(completed, "committed_movement_actions")
    correlations = trait_correlations(participant_rows)

    lines = [
        "# Deterministic evaluation summary",
        "",
        f"Attempted runs: {len(run_rows)}",
        f"Completed runs: {len(completed)}",
        f"Batch errors: {len(errors)}",
        f"Outcomes among completed runs: {dict(sorted(outcomes.items()))}",
        "",
        "## Protocol correctness",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Runs ending in CLOSED | {_fraction(sum(bool(row.get('closed')) for row in completed), len(completed))} |",
        f"| Full structural pass | {_fraction(sum(bool(row.get('structural_pass')) for row in completed), len(completed))} |",
        f"| Questions answered | {_fraction(questions_answered, questions_opened)} |",
        f"| Valid final votes | {_fraction(_sum(completed, 'valid_final_votes'), _sum(completed, 'participants'))} |",
        f"| Outcome consistent with final votes | {_fraction(sum(bool(row.get('outcome_consistent')) for row in completed), len(completed))} |",
        f"| Hard-blocker violations | {int(_sum(completed, 'hard_blocker_violations'))} |",
        f"| Unexplained stance movements | {int(_sum(completed, 'unexplained_movements'))} |",
        "",
        "## Generation reliability",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Repairs per 100 participant turns | {_shown(100.0 * _sum(completed, 'repairs') / participant_turns if participant_turns else None)} |",
        f"| Dropped turns per 100 participant turns | {_shown(100.0 * _sum(completed, 'dropped_turns') / participant_turns if participant_turns else None)} |",
        f"| Fallback turns per 100 participant turns | {_shown(100.0 * _sum(completed, 'fallback_turns') / participant_turns if participant_turns else None)} |",
        f"| Semantic-reuse signals per 10 participant turns | {_shown(10.0 * _sum(completed, 'semantic_reason_reuse') / participant_turns if participant_turns else None)} |",
        f"| Unsupported-fact validator flags | {int(_sum(completed, 'unsupported_fact_flags'))} |",
        f"| Directly realized movement actions | {_fraction(direct_movements, selected_movements)} |",
        f"| Movement fallbacks | {_fraction(movement_fallbacks, selected_movements)} |",
        f"| Committed movement actions | {_fraction(committed_movements, selected_movements)} |",
        "",
        "## Process and efficiency",
        "",
        "| Measure | Mean |",
        "|---|---:|",
        f"| Participant turns per run | {_shown(_mean(completed, 'participant_turns'), 1)} |",
        f"| Voluntary-turn share | {_shown(100.0 * _sum(completed, 'self_selected_turns') / participant_turns if participant_turns else None, 1, '%')} |",
        f"| Issue-resolution rate | {_shown(100.0 * _sum(completed, 'issues_resolved') / _sum(completed, 'issues_opened') if _sum(completed, 'issues_opened') else None, 1, '%')} |",
        f"| Option coverage | {_shown(100.0 * _mean(completed, 'option_coverage_ratio') if completed else None, 1, '%')} |",
        f"| Input tokens per participant turn | {_shown(_sum(completed, 'tokens_in') / participant_turns if participant_turns else None, 0)} |",
        "",
        "## Trait realization",
        "",
        "Spearman correlations are descriptive pooled associations across simulated participants.",
        "Hard blockers are excluded from the stubbornness correlation because their immobility is a categorical constraint.",
        "",
        "| Relationship | Spearman rho |",
        "|---|---:|",
        f"| Engagement vs. normalized voluntary-turn share | {_shown(correlations['engagement_vs_voluntary_share'])} |",
        f"| Verbosity vs. average words per turn | {_shown(correlations['verbosity_vs_words'])} |",
        f"| Stubbornness vs. visible flexibility events | {_shown(correlations['stubbornness_vs_flexibility'])} |",
    ]

    if errors:
        lines.extend(["", "## Errors", ""])
        for row in errors:
            lines.append(f"- {row.get('topic', 'unknown topic')}: {row.get('error', 'unknown error')}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    log_root = args.logs.resolve()
    output = args.output.resolve() if args.output else log_root
    run_dirs = find_run_dirs(log_root)
    if args.limit > 0:
        run_dirs = run_dirs[: args.limit]
    if not run_dirs and not discover_batch_error_rows(log_root):
        print(f"No run.json files or batch error CSV found under {log_root}.", file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    participants: list[dict[str, Any]] = []

    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir.name}")
        try:
            payload = load_run(run_dir)
            run_rows.append(extract_run_metrics(payload, run_dir))
            participants.extend(participant_trait_rows(payload, run_dir))
        except Exception as exc:
            run_rows.append(
                {
                    "run_id": run_dir.name,
                    "outcome": "error",
                    "structural_pass": False,
                    "log_dir": str(run_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    run_rows.extend(discover_batch_error_rows(log_root))
    level_rows = trait_level_rows(participants)

    write_csv(output / "deterministic_runs.csv", run_rows)
    if participants:
        write_csv(output / "trait_participants.csv", participants)
        write_csv(output / "trait_levels.csv", level_rows)
    write_summary(run_rows, participants, output / "evaluation_summary.md")

    print(f"\nRuns: {output / 'deterministic_runs.csv'}")
    if participants:
        print(f"Traits: {output / 'trait_levels.csv'}")
    print(f"Summary: {output / 'evaluation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
