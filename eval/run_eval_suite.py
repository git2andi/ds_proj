"""Run a small focused LLM-backed protocol suite.

The suite is intentionally compact. It checks representative preference shapes,
hard blockers, moderator-off operation, direct/group discussion behavior, and
bounded voting. It is a development regression suite, not the main paper dataset.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC), str(EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from models import (  # noqa: E402
    OptionCard,
    OptionStance,
    Persona,
    Scenario,
    SimulatorParameters,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
)
from run_scenarios import config_overrides, run_dialogue  # noqa: E402
from summarize_runs import write_csv  # noqa: E402


LOG_ROOT = EVAL_DIR / "logs_eval_suite"


@dataclass(frozen=True)
class EvalCase:
    id: str
    why: str
    preferences: tuple[str, ...]
    seed: int
    moderator: bool = True
    hard_blocker: int | None = None


CASES = (
    EvalCase("split_n2", "Two opposing participants may remain unresolved.", ("A", "B"), 101),
    EvalCase("split_n3", "A three-way split receives one bounded compromise opportunity.", ("A", "B", "C"), 102),
    EvalCase("narrow_2_1", "A 2-1 majority may receive one outlier-repair opportunity.", ("A", "A", "B"), 103),
    EvalCase("decisive_3_1", "A 3-1 majority proceeds to voting without unanimity pressure.", ("A", "A", "A", "B"), 104),
    EvalCase("hard_blocker", "The hard blocker never moves or votes for another option.", ("C", "A", "A"), 105, hard_blocker=0),
    EvalCase("no_moderator", "Bidding, threads, and voting still work without visible moderator turns.", ("A", "B", "C", "A"), 106, moderator=False),
    EvalCase("large_group", "A seven-person discussion remains bounded.", ("A", "A", "B", "B", "C", "C", "D"), 107),
)


def scenario() -> Scenario:
    return Scenario(
        topic="Choose a shared study workspace",
        shared_context=["The group needs one workspace for a Saturday project session."],
        options=[
            OptionCard(id="A", name="Central Library", short_name="Library", attrs={"cost": "free", "closing time": "20:00", "noise": "low"}, upside="quiet work areas", concern="can become crowded"),
            OptionCard(id="B", name="Riverside Cafe", short_name="Cafe", attrs={"cost": "8 euros", "closing time": "22:00", "noise": "moderate"}, upside="relaxed atmosphere", concern="background noise"),
            OptionCard(id="C", name="Engineering Lab", short_name="Lab", attrs={"cost": "free", "closing time": "19:00", "equipment": "specialist workstations"}, upside="technical equipment", concern="earlier closing time"),
            OptionCard(id="D", name="Online Session", short_name="Online", attrs={"cost": "free", "travel": "none", "access": "from home"}, upside="no travel", concern="less social interaction"),
        ],
    )


def personas(case: EvalCase, board: Scenario) -> list[Persona]:
    names = ("Nora", "Ben", "Mira", "Omar", "Lea", "Tariq", "Sofia")
    result: list[Persona] = []
    for index, preferred in enumerate(case.preferences):
        hard = case.hard_blocker == index
        stances: dict[str, OptionStance] = {}
        for option in board.options:
            if option.id == preferred:
                stances[option.id] = OptionStance(option.id, STANCE_PREFERRED, option.upside, "")
            elif hard:
                stances[option.id] = OptionStance(option.id, STANCE_REJECTED, "", option.concern)
            else:
                stances[option.id] = OptionStance(option.id, STANCE_NEUTRAL, option.upside, option.concern)
        result.append(
            Persona(
                id=f"p{index + 1}",
                name=names[index],
                sim_params=SimulatorParameters(3 + index % 3, 2 + index % 4, 2 + index % 4, 5 if hard else 1 + index % 4).validated(hard_blocker=hard),
                background=f"{names[index]} is working on a university group project.",
                private_goal="wants a practical workspace for focused collaboration",
                preferred_options=[preferred],
                age=24 + index * 4,
                speech_style="plain conversational wording",
                option_stances=stances,
                hard_blocker=hard,
                rejection=None,
                rejection_reason="will not accept an alternative workspace" if hard else "",
            )
        )
    return result


def evaluate_row(row: dict[str, Any], case: EvalCase) -> dict[str, Any]:
    completed = row.get("outcome") in {"successful", "majority", "unresolved"}
    protocol_pass = (
        completed
        and int(row.get("protocol_error_count", 0)) == 0
        and int(row.get("response_failures", 0)) == 0
        and bool(row.get("vote_outcome_consistent", False))
    )
    return {
        "case": case.id,
        "why": case.why,
        **row,
        "protocol_pass": protocol_pass,
    }


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Focused evaluation suite",
        "",
        f"Cases: {len(rows)}",
        f"Protocol pass: {sum(bool(row['protocol_pass']) for row in rows)}/{len(rows)}",
        "",
        "| Case | Outcome | Participant turns | Repairs | Drops | Fallbacks | Pass |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row.get('outcome', '')} | {row.get('participant_turns', 0)} "
            f"| {row.get('repairs', 0)} | {row.get('dropped_turns', 0)} "
            f"| {row.get('fallback_turns', 0)} | {'yes' if row['protocol_pass'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="replace existing suite logs")
    parser.add_argument("--case", action="append", default=[], help="run only named case(s)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [case for case in CASES if not args.case or case.id in set(args.case)]
    if not selected:
        print("No matching cases.", file=sys.stderr)
        return 2
    if LOG_ROOT.exists() and any(LOG_ROOT.iterdir()):
        if not args.clean:
            print(f"{LOG_ROOT} is not empty; use --clean.", file=sys.stderr)
            return 2
        shutil.rmtree(LOG_ROOT)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    board = scenario()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case.id}")
        with config_overrides({("moderator", "enabled"): case.moderator}):
            row = run_dialogue(
                board.topic,
                participants=len(case.preferences),
                seed=case.seed,
                log_dir=str(LOG_ROOT.relative_to(ROOT)),
                scenario=board,
                personas=personas(case, board),
            )
        rows.append(evaluate_row(row, case))
        write_csv(LOG_ROOT / "focused_suite_results.csv", rows)
        write_summary(rows, LOG_ROOT / "focused_suite_summary.md")
    print(f"Results: {LOG_ROOT / 'focused_suite_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
