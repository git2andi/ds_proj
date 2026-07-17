"""Sensitivity-check the LLM judge with deterministic transcript corruptions.

The script compares valid runs with four targeted corruptions:

- shuffled discussion turns -> naturalness and coherence
- inserted unsupported claim -> groundedness
- swapped persona contents -> persona consistency
- vote-inconsistent outcome -> deliberation quality

It never modifies the original run folders. By default, three runs spread
across ``eval2/logs_scenarios`` are tested and results are written to
``eval2/logs_judge_validation``.

    py eval2/validate_judge.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Callable

from evaluation_metrics import EVAL_DIR, ROOT, find_run_dirs, load_run, write_csv
from judge_transcripts import (
    DIMENSIONS,
    aggregate_assessments,
    judge_payload,
)

SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from llm_client import LLMClient  # noqa: E402

Corruption = Callable[[dict[str, Any], int], dict[str, Any]]


def _seed_for(run_id: str, suffix: str) -> int:
    digest = hashlib.sha256(f"{run_id}:{suffix}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def corrupt_turn_order(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    corrupted = copy.deepcopy(payload)
    turns = corrupted.get("turns", [])
    indices = [
        index
        for index, turn in enumerate(turns)
        if not turn.get("moderator") and str(turn.get("phase")) in {"DISCUSSION", "NARROWING"}
    ]
    if len(indices) < 3:
        indices = [index for index, turn in enumerate(turns) if not turn.get("moderator")]
    if len(indices) >= 2:
        values = [turns[index] for index in indices]
        random.Random(seed).shuffle(values)
        if all(values[index] is turns[indices[index]] for index in range(len(indices))):
            values = values[1:] + values[:1]
        for index, value in zip(indices, values):
            turns[index] = value
    corrupted["run_id"] = f"{payload.get('run_id', 'run')}__turn_order_corrupted"
    return corrupted


def corrupt_grounding(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    del seed
    corrupted = copy.deepcopy(payload)
    turns = corrupted.get("turns", [])
    target = next(
        (
            turn
            for turn in turns
            if not turn.get("moderator") and str(turn.get("phase")) in {"DISCUSSION", "NARROWING"}
        ),
        next((turn for turn in turns if not turn.get("moderator")), None),
    )
    if target is not None:
        target["text"] = (
            str(target.get("text", "")).rstrip()
            + " The option card also guarantees a 42 percent improvement, lifetime warranty, and next-day delivery."
        )
    corrupted["run_id"] = f"{payload.get('run_id', 'run')}__grounding_corrupted"
    return corrupted


def corrupt_personas(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    del seed
    corrupted = copy.deepcopy(payload)
    personas = corrupted.get("personas", [])
    if len(personas) >= 2:
        identity_fields = {"id", "name"}
        first_values = {key: value for key, value in personas[0].items() if key not in identity_fields}
        second_values = {key: value for key, value in personas[1].items() if key not in identity_fields}
        for key in list(personas[0]):
            if key not in identity_fields:
                personas[0].pop(key, None)
        for key in list(personas[1]):
            if key not in identity_fields:
                personas[1].pop(key, None)
        personas[0].update(second_values)
        personas[1].update(first_values)
    corrupted["run_id"] = f"{payload.get('run_id', 'run')}__personas_corrupted"
    return corrupted


def corrupt_outcome(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    del seed
    corrupted = copy.deepcopy(payload)
    option_ids = [str(option.get("id")) for option in corrupted.get("scenario", {}).get("options", [])]
    votes = corrupted.get("votes") or corrupted.get("outcome", {}).get("votes") or {}
    vote_values = [str(value) for value in votes.values()]
    current = str(corrupted.get("outcome", {}).get("final_option") or "")
    wrong = next((option for option in option_ids if option != current and option not in vote_values), None)
    if wrong is None:
        wrong = next((option for option in option_ids if option != current), option_ids[0] if option_ids else "A")
    corrupted.setdefault("outcome", {})["status"] = "successful"
    corrupted["outcome"]["final_option"] = wrong
    corrupted["outcome"]["reason"] = "All participants selected this option."
    corrupted["run_id"] = f"{payload.get('run_id', 'run')}__outcome_corrupted"
    return corrupted


CORRUPTIONS: tuple[tuple[str, tuple[str, ...], Corruption], ...] = (
    ("turn_order", ("naturalness", "coherence"), corrupt_turn_order),
    ("unsupported_claim", ("groundedness",), corrupt_grounding),
    ("persona_swap", ("persona_consistency",), corrupt_personas),
    ("outcome_mismatch", ("deliberation_quality",), corrupt_outcome),
)


def spread_sample(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or count >= len(paths):
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]
    indices = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
    return [paths[index] for index in dict.fromkeys(indices)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=EVAL_DIR / "logs_scenarios")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "logs_judge_validation")
    parser.add_argument("--sample", type=int, default=3, help="number of original runs to test")
    parser.add_argument("--judges", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--provider",
        type=str,
        default="uni",
        choices=("uni", "groq", "gemini", "gpt"),
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="show selected runs without calling the judge")
    return parser.parse_args()


def _details(
    run_id: str,
    version: str,
    corruption: str,
    assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assessment in assessments:
        rows.append(
            {
                "run_id": run_id,
                "version": version,
                "corruption": corruption,
                "judge_order": assessment["judge_order"],
                "judge": assessment["judge"],
                **assessment["scores"],
                "verdict": assessment["verdict"],
                "retries": assessment["retries"],
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], errors: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# LLM judge corruption validation",
        "",
        "A comparison passes when the corrupted transcript receives a lower score on every targeted dimension.",
        "",
        "| Corruption | Valid comparisons | Detected | Detection rate |",
        "|---|---:|---:|---:|",
    ]
    for name, _, _ in CORRUPTIONS:
        group = [row for row in rows if row.get("corruption") == name]
        detected = sum(bool(row.get("detected")) for row in group)
        rate = 100.0 * detected / len(group) if group else 0.0
        lines.append(f"| {name} | {len(group)} | {detected} | {rate:.1f}% |")
    detected_total = sum(bool(row.get("detected")) for row in rows)
    lines.extend(
        [
            "",
            f"Overall targeted corruptions detected: {detected_total}/{len(rows)} "
            f"({100.0 * detected_total / len(rows):.1f}%)" if rows else "Overall targeted corruptions detected: 0/0",
            f"Judge-call failures after retries: {len(errors)}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dirs = spread_sample(find_run_dirs(args.logs.resolve()), args.sample)
    if not run_dirs:
        print(f"No run.json files found under {args.logs.resolve()}.", file=sys.stderr)
        return 2
    if args.list:
        for run_dir in run_dirs:
            print(run_dir)
        return 0

    llm = LLMClient(provider=args.provider, model=args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    total = len(run_dirs) * (1 + len(CORRUPTIONS))
    position = 0
    for run_dir in run_dirs:
        payload = load_run(run_dir)
        original_run_id = str(payload.get("run_id", run_dir.name))
        position += 1
        print(f"[{position}/{total}] original: {run_dir.name}")
        original_assessments, original_errors = judge_payload(
            payload,
            llm,
            judges=args.judges,
            max_retries=args.retries,
            order_key=original_run_id,
        )
        detailed_rows.extend(_details(original_run_id, "original", "", original_assessments))
        error_rows.extend({"run_id": original_run_id, "version": "original", **error} for error in original_errors)
        original_scores = aggregate_assessments(original_assessments)
        if not original_scores:
            continue

        for corruption_name, target_dimensions, corruption_fn in CORRUPTIONS:
            position += 1
            print(f"[{position}/{total}] {corruption_name}: {run_dir.name}")
            corrupted = corruption_fn(payload, _seed_for(original_run_id, corruption_name))
            corrupted_assessments, corrupted_errors = judge_payload(
                corrupted,
                llm,
                judges=args.judges,
                max_retries=args.retries,
                order_key=original_run_id,
            )
            detailed_rows.extend(
                _details(original_run_id, "corrupted", corruption_name, corrupted_assessments)
            )
            error_rows.extend(
                {"run_id": original_run_id, "version": "corrupted", "corruption": corruption_name, **error}
                for error in corrupted_errors
            )
            corrupted_scores = aggregate_assessments(corrupted_assessments)
            if not corrupted_scores:
                continue
            row: dict[str, Any] = {
                "run_id": original_run_id,
                "judge_provider": getattr(llm, "provider", ""),
                "judge_model": getattr(llm, "model_id", ""),
                "topic": payload.get("scenario", {}).get("topic", ""),
                "corruption": corruption_name,
                "target_dimensions": ",".join(target_dimensions),
            }
            detected = True
            for dimension in DIMENSIONS:
                original_value = float(original_scores[dimension])
                corrupted_value = float(corrupted_scores[dimension])
                row[f"original_{dimension}"] = original_value
                row[f"corrupted_{dimension}"] = corrupted_value
                row[f"delta_{dimension}"] = round(corrupted_value - original_value, 2)
                if dimension in target_dimensions and not corrupted_value < original_value:
                    detected = False
            row["detected"] = detected
            pair_rows.append(row)
            write_csv(args.output / "judge_validation_pairs.csv", pair_rows)
            write_csv(args.output / "judge_validation_detailed.csv", detailed_rows)
            if error_rows:
                write_csv(args.output / "judge_validation_errors.csv", error_rows)
            write_summary(pair_rows, error_rows, args.output / "judge_validation_summary.md")

    if pair_rows:
        write_csv(args.output / "judge_validation_pairs.csv", pair_rows)
    if detailed_rows:
        write_csv(args.output / "judge_validation_detailed.csv", detailed_rows)
    if error_rows:
        write_csv(args.output / "judge_validation_errors.csv", error_rows)
    write_summary(pair_rows, error_rows, args.output / "judge_validation_summary.md")

    print(f"\nPairs: {args.output / 'judge_validation_pairs.csv'}")
    print(f"Summary: {args.output / 'judge_validation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
