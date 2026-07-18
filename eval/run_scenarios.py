"""Run the ``participant_count | topic`` lines in ``scenarios.txt`` end to end.

Each line creates one complete automatic run: scenario generation, LLM-generated
personas, dialogue, voting, and logging. The participant count comes from the
line; other settings keep their ``config.yaml`` values.

Examples:
    py eval/run_scenarios.py --list
    py eval/run_scenarios.py --limit 3
    py eval/run_scenarios.py --counts 3,4 --seed 500
"""

from __future__ import annotations

import argparse
import copy
import random
import shutil
import statistics
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC), str(EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config_loader import Section, cfg  # noqa: E402
from dialogue import DialogueRunner  # noqa: E402
from eval import flat_metrics_for  # noqa: E402
from summarize_runs import write_csv  # noqa: E402

SCENARIOS_PATH = EVAL_DIR / "scenarios.txt"
LOG_DIR = "eval/logs_scenarios"
OUTPUT_ROOT = ROOT / LOG_DIR


@dataclass(frozen=True)
class ScenarioCase:
    index: int
    participants: int
    topic: str


def read_scenarios(path: Path = SCENARIOS_PATH) -> list[ScenarioCase]:
    """Parse ``participant_count | topic`` lines with strict validation."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    cases: list[ScenarioCase] = []
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
        minimum = int(cfg.simulation.min_participants)
        maximum = int(cfg.simulation.max_participants)
        if not minimum <= count <= maximum:
            raise ValueError(f"{path}:{line_number}: participant count must be {minimum}..{maximum}")
        if not topic:
            raise ValueError(f"{path}:{line_number}: topic must not be empty")
        cases.append(ScenarioCase(len(cases) + 1, count, topic))
    if not cases:
        raise ValueError(f"{path} contains no scenarios")
    duplicates = [topic for topic, count in Counter(case.topic for case in cases).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate topics in {path}: {duplicates}")
    return cases


def set_config_value(section_name: str, key: str, value: Any) -> Any:
    """Override one configuration value in memory and return its old value."""
    section_raw = cfg._raw[section_name]
    if key not in section_raw:
        raise KeyError(f"unknown config key: {section_name}.{key}")
    previous = section_raw[key]
    section_raw[key] = value
    section = getattr(cfg, section_name)
    setattr(section, key, Section(value) if isinstance(value, dict) else value)
    return previous


@contextmanager
def config_overrides(overrides: dict[tuple[str, str], Any]) -> Iterator[None]:
    previous: dict[tuple[str, str], Any] = {}
    try:
        for (section_name, key), value in overrides.items():
            previous[(section_name, key)] = set_config_value(section_name, key, value)
        yield
    finally:
        for (section_name, key), value in reversed(list(previous.items())):
            set_config_value(section_name, key, value)


def run_dialogue(
    topic: str,
    *,
    participants: int | None = None,
    seed: int | None = None,
    llm: Any = None,
    log_dir: str | None = None,
    scenario: Any = None,
    personas: Any = None,
) -> dict[str, Any]:
    """Run one dialogue and return a flat batch row; preserve failures as rows."""
    overrides: dict[tuple[str, str], Any] = {}
    if participants is not None:
        overrides[("simulation", "num_participants")] = int(participants)
    if log_dir is not None:
        overrides[("output", "log_dir")] = log_dir
    actual_seed = int(seed) if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    row: dict[str, Any] = {
        "topic": topic,
        "participants": participants if participants is not None else cfg.participant_count(),
        "seed": actual_seed,
    }
    try:
        if (scenario is None) != (personas is None):
            raise ValueError("scenario and personas must be supplied together")
        with config_overrides(overrides):
            runner = DialogueRunner(
                topic,
                force_auto_scenario=scenario is None,
                scenario=copy.deepcopy(scenario),
                personas=copy.deepcopy(personas),
                llm=llm,
                seed=actual_seed,
            )
            result = runner.run()
        row.update(flat_metrics_for(result.state, result.outcome))
        row["log_dir"] = result.log_paths["dir"]
        row["error"] = ""
    except Exception as exc:
        row.update({"outcome": "error", "log_dir": "", "error": f"{type(exc).__name__}: {exc}"})
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None, help="alternative scenarios file")
    parser.add_argument("--list", action="store_true", help="list parsed cases and exit")
    parser.add_argument("--start", type=int, default=1, help="first case index to run (1-based)")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of cases (0 = all)")
    parser.add_argument("--counts", type=str, default="", help="participant counts, e.g. 2,3")
    parser.add_argument("--seed", type=int, default=None, help="base seed; case i uses base+i")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT, help="batch output directory")
    parser.add_argument("--clean", action="store_true", help="delete a non-empty output directory before running")
    return parser.parse_args()


def select_cases(cases: list[ScenarioCase], args: argparse.Namespace) -> list[ScenarioCase]:
    selected = [case for case in cases if case.index >= args.start]
    if args.counts:
        wanted = {int(part) for part in args.counts.split(",") if part.strip()}
        selected = [case for case in selected if case.participants in wanted]
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def prepare_output_root(path: Path, *, clean: bool) -> Path:
    """Create one uncontaminated batch directory without silent deletion."""

    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        if not clean:
            raise RuntimeError(
                f"output directory is not empty: {path}. "
                "Use --clean to replace it or --output to choose another directory."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    completed = [row for row in rows if row.get("outcome") != "error"]
    outcomes = Counter(str(row.get("outcome")) for row in rows)
    lines = [
        "# Scenario batch summary", "",
        f"Runs: {len(rows)} ({len(rows) - len(completed)} errors)",
        f"Outcomes: {dict(sorted(outcomes.items()))}", "",
        "| Participants | Runs | Successful | Majority | Unresolved | Avg participant turns | Avg input tokens |",
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

    try:
        output_root = prepare_output_root(args.output, clean=bool(args.clean))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    csv_path = output_root / "scenario_runs.csv"
    summary_path = output_root / "scenario_summary.md"
    rows: list[dict[str, Any]] = []
    base_seed = int(args.seed) if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    print(f"Base seed: {base_seed}")
    for position, case in enumerate(selected, start=1):
        seed = base_seed + case.index
        print(f"\n=== [{position}/{len(selected)}] case {case.index}: n={case.participants} | {case.topic}")
        row = run_dialogue(
            case.topic, participants=case.participants, seed=seed,
            log_dir=str(output_root),
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
