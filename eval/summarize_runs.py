"""Summarize completed ``run.json`` files into report-sized evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC), str(EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


HEDGES = (
    "maybe", "perhaps", "possibly", "i think", "i guess", "i feel like",
    "kind of", "sort of", "might", "could", "not sure", "probably", "it seems",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("run.json"))


def load_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def hedge_rate_per_100_words(texts: Iterable[str] | str) -> float:
    text = texts if isinstance(texts, str) else " ".join(texts)
    lowered = text.lower()
    words = max(1, len(text.split()))
    count = sum(lowered.count(marker) for marker in HEDGES)
    return 100.0 * count / words


def _vote_outcome_consistent(payload: dict[str, Any]) -> bool:
    outcome = payload.get("outcome") or {}
    votes = payload.get("votes") or {}
    counts = Counter(option for option in votes.values() if option)
    participant_count = len(payload.get("personas") or [])
    status = outcome.get("status")
    final = outcome.get("final_option")
    if status == "successful":
        return bool(final) and counts[final] == participant_count
    if status == "majority":
        return bool(final) and counts[final] > participant_count / 2
    return max(counts.values(), default=0) <= participant_count / 2


def extract_run_metrics(payload: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    turns = payload.get("turns") or []
    participant_turns = [turn for turn in turns if not turn.get("moderator")]
    questions = sum((turn.get("action") or {}).get("act") == "ask" for turn in participant_turns)
    answers = sum((turn.get("action") or {}).get("act") == "answer" for turn in participant_turns)
    votes = payload.get("votes") or {}
    participant_count = len(payload.get("personas") or [])
    valid_option_ids = {str(option.get("id")) for option in (payload.get("scenario") or {}).get("options", [])}
    valid_votes = sum(option in valid_option_ids for option in votes.values())
    protocol_errors = len(payload.get("protocol_errors") or [])
    response_failures = int(metrics.get("response_failures", 0))
    consistent = _vote_outcome_consistent(payload)
    return {
        "run_id": run_dir.name,
        "topic": (payload.get("scenario") or {}).get("topic", ""),
        "participants": participant_count,
        "seed": (payload.get("provenance") or {}).get("seed", ""),
        "outcome": (payload.get("outcome") or {}).get("status", ""),
        "final_option": (payload.get("outcome") or {}).get("final_option") or "",
        "participant_turns": len(participant_turns),
        "voluntary_turns": int(metrics.get("voluntary_turns", 0)),
        "moderator_turns": int(metrics.get("moderator_turns", 0)),
        "moderator_ratio": _num(metrics.get("moderator_ratio")),
        "avg_words": _num(metrics.get("avg_words_per_participant_turn")),
        "questions": questions,
        "answers": answers,
        "visible_preference_changes": int(metrics.get("visible_preference_changes", 0)),
        "repairs": int(metrics.get("repair_turns", 0)),
        "dropped_turns": int(metrics.get("dropped_turns", 0)),
        "fallback_turns": int(metrics.get("fallback_turns", 0)),
        "response_failures": response_failures,
        "protocol_errors": protocol_errors,
        "valid_final_votes": valid_votes,
        "vote_outcome_consistent": consistent,
        "protocol_pass": (
            valid_votes == participant_count
            and consistent
            and protocol_errors == 0
            and response_failures == 0
        ),
        "needs_review": bool(payload.get("needs_review")),
        "input_tokens": int(metrics.get("input_tokens", 0)),
        "output_tokens": int(metrics.get("output_tokens", 0)),
        "llm_calls": int(metrics.get("llm_calls", 0)),
        "log_dir": str(run_dir),
    }


def participant_trait_rows(payload: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    personas = {row["id"]: row for row in payload.get("personas") or []}
    runtime = payload.get("runtime") or {}
    turns = payload.get("turns") or []
    texts: dict[str, list[str]] = defaultdict(list)
    voluntary_counts: Counter[str] = Counter()
    for turn in turns:
        if turn.get("moderator"):
            continue
        pid = str(turn.get("speaker_id"))
        action = turn.get("action") or {}
        # Deterministic vote wording is not a realization of verbosity or directness.
        if action.get("act") != "vote":
            texts[pid].append(str(turn.get("text") or ""))
        if turn.get("voluntary"):
            voluntary_counts[pid] += 1
    total_voluntary = sum(voluntary_counts.values())
    rows: list[dict[str, Any]] = []
    for pid, persona in personas.items():
        traits = persona.get("sim_params") or {}
        words = [len(text.split()) for text in texts.get(pid, [])]
        current = runtime.get(pid) or {}
        rows.append(
            {
                "run_id": run_dir.name,
                "participant_id": pid,
                "name": persona.get("name", ""),
                "engagement": int(traits.get("engagement", 0)),
                "verbosity": int(traits.get("verbosity", 0)),
                "directness": int(traits.get("directness", 0)),
                "stubbornness": int(traits.get("stubbornness", 0)),
                "voluntary_turns": voluntary_counts[pid],
                # Relative to equal participation (1.0 = the equal-share baseline),
                # so participants from different group sizes remain comparable.
                "normalized_voluntary_share": (
                    voluntary_counts[pid] * max(1, len(personas)) / max(1, total_voluntary)
                ),
                "avg_words": statistics.mean(words) if words else 0.0,
                "hedge_rate_per_100_words": hedge_rate_per_100_words(texts.get(pid, [])),
                "inverse_hedge_rate": -hedge_rate_per_100_words(texts.get(pid, [])),
                "visible_switches": int(current.get("visible_switches", 0)),
                "public_acceptances": len(current.get("public_acceptances") or []),
                "showed_flexibility": bool(
                    int(current.get("visible_switches", 0))
                    or current.get("public_acceptances")
                ),
            }
        )
    return rows


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end + 2) / 2
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    left, right = list(x), list(y)
    return pearson(_rankdata(left), _rankdata(right))


def trait_correlations(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        "engagement_vs_voluntary_share": spearman(
            [float(row["engagement"]) for row in rows],
            [float(row["normalized_voluntary_share"]) for row in rows],
        ),
        "verbosity_vs_avg_words": spearman(
            [float(row["verbosity"]) for row in rows],
            [float(row["avg_words"]) for row in rows],
        ),
        "stubbornness_vs_flexibility": spearman(
            [float(row["stubbornness"]) for row in rows],
            [float(bool(row["showed_flexibility"])) for row in rows],
        ),
        "directness_vs_inverse_hedge_rate": spearman(
            [float(row["directness"]) for row in rows],
            [float(row["inverse_hedge_rate"]) for row in rows],
        ),
    }


def trait_level_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    specs = (
        ("engagement", "normalized_voluntary_share"),
        ("verbosity", "avg_words"),
        ("stubbornness", "showed_flexibility"),
        ("directness", "hedge_rate_per_100_words"),
    )
    for trait, measure in specs:
        for level in sorted({int(row[trait]) for row in rows}):
            group = [row for row in rows if int(row[trait]) == level]
            values = [float(row[measure]) for row in group]
            output.append(
                {
                    "trait": trait,
                    "level": level,
                    "participants": len(group),
                    "measure": measure,
                    "mean": round(statistics.mean(values), 4) if values else 0.0,
                }
            )
    return output


def batch_error_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "scenario_runs.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("outcome") == "error"]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows) if rows else 0.0


def write_summary(
    runs: list[dict[str, Any]],
    traits: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    path: Path,
) -> None:
    outcomes = Counter(row["outcome"] for row in runs)
    correlations = trait_correlations(traits)
    total_turns = sum(int(row["participant_turns"]) for row in runs)
    lines = [
        "# Evaluation summary",
        "",
        f"Attempted scenarios: {len(runs) + len(errors)}",
        f"Completed runs: {len(runs)}",
        f"Setup/runtime failures: {len(errors)}",
        f"Outcomes: {dict(sorted(outcomes.items()))}",
        f"Protocol pass: {sum(bool(row['protocol_pass']) for row in runs)}/{len(runs)}",
        f"Runs needing review: {sum(bool(row['needs_review']) for row in runs)}",
        "",
        "## Runtime quality",
        "",
        f"- Mean participant turns: {_mean(runs, 'participant_turns'):.1f}",
        f"- Mean moderator ratio: {_mean(runs, 'moderator_ratio'):.3f}",
        f"- Repairs per 100 participant turns: {100 * sum(int(row['repairs']) for row in runs) / max(1, total_turns):.2f}",
        f"- Drops per 100 participant turns: {100 * sum(int(row['dropped_turns']) for row in runs) / max(1, total_turns):.2f}",
        f"- Fallbacks per 100 participant turns: {100 * sum(int(row['fallback_turns']) for row in runs) / max(1, total_turns):.2f}",
        f"- Total input tokens: {sum(int(row['input_tokens']) for row in runs)}",
        f"- Total output tokens: {sum(int(row['output_tokens']) for row in runs)}",
        "",
        "## Trait realization (Spearman)",
        "",
    ]
    for name, value in correlations.items():
        lines.append(f"- {name}: {'n/a' if value is None else f'{value:.3f}'}")
    if errors:
        lines.extend(["", "## Failed scenarios", ""])
        for row in errors:
            lines.append(f"- {row.get('topic', '')}: {row.get('error', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=EVAL_DIR / "logs_scenarios")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.logs.resolve()
    output = args.output.resolve() if args.output else root
    run_dirs = find_run_dirs(root)
    if not run_dirs:
        print(f"No run.json files found under {root}.", file=sys.stderr)
        return 2
    payloads = [(run_dir, load_run(run_dir)) for run_dir in run_dirs]
    runs = [extract_run_metrics(payload, run_dir) for run_dir, payload in payloads]
    traits = [
        row
        for run_dir, payload in payloads
        for row in participant_trait_rows(payload, run_dir)
    ]
    levels = trait_level_rows(traits)
    errors = batch_error_rows(root)
    write_csv(output / "deterministic_runs.csv", runs)
    write_csv(output / "trait_participants.csv", traits)
    write_csv(output / "trait_levels.csv", levels)
    write_summary(runs, traits, errors, output / "evaluation_summary.md")
    print(f"Runs: {output / 'deterministic_runs.csv'}")
    print(f"Traits: {output / 'trait_participants.csv'}")
    print(f"Summary: {output / 'evaluation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
