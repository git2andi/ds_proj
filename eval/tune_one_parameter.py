"""Run one topic repeatedly while varying exactly one numeric config parameter.

The script compares a small, predefined value grid for one scalar under
conversation:, simulator:, or language:. All other behavioral settings remain
at their original values. The participant count and random seed are controlled
experimental conditions, not tuned parameters.

Examples
--------
List supported scalar parameters and their suggested values:
    py eval/tune_one_parameter.py --list-parameters

Run 50 repetitions for every suggested value of one parameter:
    py eval/tune_one_parameter.py ^
        --topic "Choose a storage system for a university research group" ^
        --participants 3 ^
        --parameter conversation.soft_target_voluntary_turns_per_participant

Use a smaller smoke run:
    py eval/tune_one_parameter.py --topic "Choose a project laptop" ^
        --participants 4 --parameter language.near_duplicate_similarity_threshold ^
        --runs-per-value 2 --yes

Persist the best value according to the transparent heuristic score:
    py eval/tune_one_parameter.py ... --apply-best

The default is deliberately conservative: results are written, but config.yaml
is restored. Inspect aggregate.csv/summary.md before using --apply-best.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from .experiment_utils import (
        CONFIG_PATH,
        ROOT,
        ConfigExperimentSession,
        PathKey,
        read_config,
        run_topic_subprocess,
        slugify,
        validate_current_config,
        value_at,
        write_csv,
    )
except ImportError:  # Direct execution: py eval/tune_one_parameter.py
    from experiment_utils import (
        CONFIG_PATH,
        ROOT,
        ConfigExperimentSession,
        PathKey,
        read_config,
        run_topic_subprocess,
        slugify,
        validate_current_config,
        value_at,
        write_csv,
    )

# Each grid contains the current default and modest alternatives. Only one
# scalar path is varied per script invocation.
TUNING_GRIDS: dict[str, tuple[Any, ...]] = {
    # Conversation length and issue pacing.
    "conversation.min_voluntary_turns_per_participant": (1.5, 2.0, 2.5),
    "conversation.soft_target_voluntary_turns_per_participant": (4.0, 5.0, 6.0),
    "conversation.hard_max_voluntary_turns_per_participant": (6.0, 7.0, 8.0),
    "conversation.soft_target_voluntary_turn_cap": (18, 22, 26),
    "conversation.hard_max_voluntary_turn_cap": (26, 30, 36),
    "conversation.issue_follow_up_cap": (2, 3, 4),
    "conversation.direct_question_optional_follow_up_cap": (0, 1, 2),
    "conversation.concern_external_response_cap": (1, 2, 3),
    "conversation.max_concerns_per_participant": (0, 1, 2),
    "conversation.max_concern_reopens": (0, 1, 2),
    "conversation.stagnation_no_bid_rounds": (1, 2, 3),
    "conversation.compromise_window_max_turns": (1, 2, 3),
    "conversation.narrowing_reaction_turn_cap": (1, 2, 3),
    "conversation.small_group_max_participants": (3, 4),
    "conversation.small_group_extra_no_bid_rounds": (0, 1, 2),
    "conversation.small_group_shared_acceptance_extra_turns": (1, 3, 5),
    "conversation.unanimous_closure_min_voluntary_turns_per_participant": (0.5, 1.0, 1.5),
    "conversation.large_group_min_participants": (5, 6, 7),
    "conversation.large_group_optional_reaction_window_cap": (1, 2, 3),
    "conversation.large_group_narrowing_issue_turn_cap": (0, 1, 2),
    "conversation.large_group_narrowing_final_position_cap": (2, 3, 4),
    "conversation.recent_turns_in_prompt": (5, 7, 9),
    "conversation.max_consecutive_turns": (1, 2, 3),
    # Simulator participation and movement. Level 5 stubbornness remains a hard invariant at zero.
    "simulator.unknown_information_question_probability": (0.00, 0.04, 0.08, 0.12),
    "simulator.bid_probability_by_engagement.1": (0.10, 0.20, 0.30),
    "simulator.bid_probability_by_engagement.2": (0.25, 0.35, 0.45),
    "simulator.bid_probability_by_engagement.3": (0.40, 0.50, 0.60),
    "simulator.bid_probability_by_engagement.4": (0.60, 0.70, 0.80),
    "simulator.bid_probability_by_engagement.5": (0.80, 0.90, 1.00),
    "simulator.movement_probability_by_stubbornness.1": (0.70, 0.80, 0.90),
    "simulator.movement_probability_by_stubbornness.2": (0.50, 0.60, 0.70),
    "simulator.movement_probability_by_stubbornness.3": (0.30, 0.40, 0.50),
    "simulator.movement_probability_by_stubbornness.4": (0.10, 0.20, 0.30),
    # Language length and repetition controls.
    "language.max_words_by_verbosity.1": (6, 8, 10),
    "language.max_words_by_verbosity.2": (10, 12, 14),
    "language.max_words_by_verbosity.3": (14, 16, 18),
    "language.max_words_by_verbosity.4": (20, 22, 24),
    "language.max_words_by_verbosity.5": (24, 27, 30),
    "language.action_max_words.acknowledge": (10, 12, 14),
    "language.action_max_words.ask": (16, 18, 20),
    "language.action_max_words.answer": (16, 18, 20),
    "language.action_max_words.final_position": (16, 18, 20),
    "language.action_max_words.vote": (8, 10, 12),
    "language.action_max_words.simple_vote": (6, 8, 10),
    "language.near_duplicate_similarity_threshold": (0.88, 0.92, 0.96),
    "language.near_duplicate_recent_turns": (2, 3, 5),
}


def dotted_path(value: str) -> PathKey:
    parts = tuple(part.strip() for part in value.split(".") if part.strip())
    if not parts or parts[0] not in {"conversation", "simulator", "language"}:
        raise argparse.ArgumentTypeError(
            "parameter must be a scalar below conversation, simulator, or language"
        )
    return parts


def parse_values(raw: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = yaml.safe_load(item)
        if isinstance(value, (dict, list, tuple)) or isinstance(value, bool) or value is None:
            raise argparse.ArgumentTypeError("--values accepts only numeric scalar values")
        if not isinstance(value, (int, float)):
            raise argparse.ArgumentTypeError(f"not a number: {item!r}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--values did not contain any numbers")
    return tuple(values)


def default_target_turns(participants: int) -> int:
    return {2: 14, 3: 24, 4: 28, 5: 32, 6: 36, 7: 40}[participants]


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average_rank
        index = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys) or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry)
    )
    return numerator / denominator if denominator else None


def flatten_run(payload: dict[str, Any], *, value: Any, run_index: int, seed: int) -> dict[str, Any]:
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
    selected_movements = int(generation.get("selected_movement_actions", 0))
    return {
        "value": value,
        "run_index": run_index,
        "seed": seed,
        "run_id": payload.get("run_id", ""),
        "outcome": outcome.get("status", ""),
        "participant_turns": participant_turns,
        "voluntary_turns": int(turns.get("voluntary", 0)),
        "moderator_turns": int(turns.get("moderator", 0)),
        "repairs": int(generation.get("repairs", 0)),
        "repair_rate": int(generation.get("repairs", 0)) / max(1, participant_turns),
        "dropped": int(generation.get("dropped", 0)),
        "drop_rate": int(generation.get("dropped", 0)) / max(1, participant_turns),
        "liveness_forced": int(generation.get("liveness_forced", 0)),
        "semantic_reason_reuse": int(generation.get("semantic_reason_reuse", 0)),
        "reason_reuse_rate": int(generation.get("semantic_reason_reuse", 0)) / max(1, participant_turns),
        "movement_realization_failures": int(generation.get("movement_realization_failures", 0)),
        "selected_movement_actions": selected_movements,
        "movement_failure_rate": int(generation.get("movement_realization_failures", 0)) / max(1, selected_movements),
        "questions_opened": int(questions.get("opened", 0)),
        "questions_answered": int(questions.get("answered", 0)),
        "issues_resolved": int(issues.get("resolved", 0)),
        "issues_stale": int(issues.get("stale", 0)),
        "switches": int(stances.get("switches", 0)),
        "acceptances": int(stances.get("acceptances", 0)),
        "protocol_degraded": bool(votes.get("protocol_degraded", False)),
        "tokens_in": int(tokens.get("input", 0)),
        "tokens_out": int(tokens.get("output", 0)),
    }


def aggregate(value: Any, rows: list[dict[str, Any]], payloads: list[dict[str, Any]], target_turns: int) -> dict[str, Any]:
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    if not successful_rows:
        return {
            "value": value,
            "completed_runs": 0,
            "failed_runs": len(rows),
            "score": -1.0,
        }

    def mean_of(key: str) -> float:
        return statistics.mean(float(row[key]) for row in successful_rows)

    participant_turns = [int(row["participant_turns"]) for row in successful_rows]
    outcomes = Counter(str(row["outcome"]) for row in successful_rows)
    total_questions = sum(int(row["questions_opened"]) for row in successful_rows)
    answered = sum(int(row["questions_answered"]) for row in successful_rows)
    total_selected = sum(int(row["selected_movement_actions"]) for row in successful_rows)
    movement_failures = sum(int(row["movement_realization_failures"]) for row in successful_rows)

    engagement: list[float] = []
    voluntary: list[float] = []
    verbosity: list[float] = []
    avg_words: list[float] = []
    for payload in payloads:
        participants = ((payload.get("metrics") or {}).get("participants") or {})
        for item in participants.values():
            traits = item.get("traits") or {}
            engagement.append(float(traits.get("engagement", 0)))
            voluntary.append(float(item.get("voluntary", 0)))
            verbosity.append(float(traits.get("verbosity", 0)))
            avg_words.append(float(item.get("avg_words", 0)))

    median_turns = statistics.median(participant_turns)
    completion = len(successful_rows) / max(1, len(rows))
    answer_rate = answered / total_questions if total_questions else 1.0
    movement_success = 1.0 - movement_failures / total_selected if total_selected else 1.0
    turn_fit = max(0.0, 1.0 - abs(median_turns - target_turns) / max(1, target_turns))
    repair_quality = max(0.0, 1.0 - mean_of("repair_rate") / 0.20)
    reuse_quality = max(0.0, 1.0 - mean_of("reason_reuse_rate") / 0.30)
    drop_quality = max(0.0, 1.0 - mean_of("drop_rate") / 0.05)
    # Transparent heuristic, not a scientific optimizer. Turn fit has the largest
    # weight because this script was requested for modest pacing calibration.
    score = 100.0 * (
        0.30 * turn_fit
        + 0.20 * repair_quality
        + 0.15 * reuse_quality
        + 0.10 * drop_quality
        + 0.10 * answer_rate
        + 0.10 * movement_success
        + 0.05 * completion
    )

    return {
        "value": value,
        "completed_runs": len(successful_rows),
        "failed_runs": len(rows) - len(successful_rows),
        "score": round(score, 3),
        "target_participant_turns": target_turns,
        "mean_participant_turns": round(statistics.mean(participant_turns), 3),
        "median_participant_turns": round(median_turns, 3),
        "min_participant_turns": min(participant_turns),
        "max_participant_turns": max(participant_turns),
        "mean_voluntary_turns": round(mean_of("voluntary_turns"), 3),
        "mean_repair_rate": round(mean_of("repair_rate"), 4),
        "mean_drop_rate": round(mean_of("drop_rate"), 4),
        "mean_reason_reuse_rate": round(mean_of("reason_reuse_rate"), 4),
        "question_answer_rate": round(answer_rate, 4),
        "movement_realization_success_rate": round(movement_success, 4),
        "engagement_voluntary_spearman": None if (corr := spearman(engagement, voluntary)) is None else round(corr, 4),
        "verbosity_words_spearman": None if (corr := spearman(verbosity, avg_words)) is None else round(corr, 4),
        "successful_outcomes": outcomes.get("successful", 0),
        "majority_outcomes": outcomes.get("majority", 0),
        "unresolved_outcomes": outcomes.get("unresolved", 0),
        "mean_tokens_in": round(mean_of("tokens_in"), 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="single automatic scenario topic used for every run")
    parser.add_argument("--participants", type=int, default=3, choices=range(2, 8))
    parser.add_argument("--parameter", type=dotted_path, help="one scalar path to vary")
    parser.add_argument("--values", type=parse_values, help="comma-separated numeric values overriding the suggested grid")
    parser.add_argument("--runs-per-value", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=74000)
    parser.add_argument("--target-turns", type=int, help="participant-turn target used only by the ranking heuristic")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--apply-best", action="store_true", help="persist the highest-scoring value after all runs")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    parser.add_argument("--list-parameters", action="store_true")
    parser.add_argument("--restore-config", action="store_true", help="restore config.yaml from a stale experiment backup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.restore_config:
        restored = ConfigExperimentSession.restore_stale_backup()
        print("Restored config.yaml." if restored else "No experiment backup was present.")
        return 0
    if args.list_parameters:
        config = read_config()
        for name, values in TUNING_GRIDS.items():
            try:
                current = value_at(config, tuple(name.split(".")))
            except KeyError:
                current = "<missing>"
            print(f"{name}\n  current={current!r}\n  suggested={list(values)!r}")
        return 0
    if not args.topic or not args.parameter:
        print("--topic and --parameter are required unless using --list-parameters.", file=sys.stderr)
        return 2
    if args.runs_per_value < 1:
        print("--runs-per-value must be positive.", file=sys.stderr)
        return 2

    parameter_name = ".".join(args.parameter)
    if parameter_name not in TUNING_GRIDS and args.values is None:
        print(
            f"No suggested grid for {parameter_name}. Use --list-parameters or provide --values.",
            file=sys.stderr,
        )
        return 2

    config = read_config()
    try:
        current_value = value_at(config, args.parameter)
    except KeyError:
        print(f"Config path not found: {parameter_name}", file=sys.stderr)
        return 2
    candidates = list(args.values or TUNING_GRIDS[parameter_name])
    if current_value not in candidates:
        candidates.append(current_value)
    candidates = list(dict.fromkeys(candidates))
    target_turns = args.target_turns or default_target_turns(args.participants)
    total_runs = len(candidates) * args.runs_per_value

    print(f"Topic: {args.topic}")
    print(f"Participants: {args.participants}")
    print(f"Parameter: {parameter_name}")
    print(f"Current value: {current_value!r}")
    print(f"Candidate values: {candidates}")
    print(f"Runs per value: {args.runs_per_value}; total LLM-backed runs: {total_runs}")
    print(f"Heuristic participant-turn target: {target_turns}")
    if not args.yes:
        response = input("Proceed? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = ROOT / "eval" / "tuning_results" / f"{stamp}_{slugify(parameter_name)}"
    log_root = ROOT / "eval" / "tuning_runs" / f"{stamp}_{slugify(parameter_name)}"
    result_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    payloads_by_value: dict[str, list[dict[str, Any]]] = {}

    with ConfigExperimentSession() as session:
        valid_candidates: list[Any] = []
        for candidate in candidates:
            session.write({args.parameter: candidate})
            valid, message = validate_current_config()
            if valid:
                valid_candidates.append(candidate)
            else:
                print(f"Skipping invalid candidate {candidate!r}: {message}", file=sys.stderr)
        if not valid_candidates:
            print("No candidate value produced a valid configuration.", file=sys.stderr)
            return 1

        for candidate_index, candidate in enumerate(valid_candidates, start=1):
            print(f"\n=== {parameter_name} = {candidate!r} ({candidate_index}/{len(valid_candidates)}) ===")
            value_slug = slugify(str(candidate))
            candidate_log_dir = log_root.relative_to(ROOT) / f"value_{value_slug}"
            candidate_rows: list[dict[str, Any]] = []
            candidate_payloads: list[dict[str, Any]] = []
            for run_index in range(args.runs_per_value):
                seed = args.base_seed + run_index
                updates: dict[PathKey, Any] = {
                    args.parameter: candidate,
                    ("simulation", "num_participants"): args.participants,
                    ("simulation", "random_seed"): seed,
                    ("environment", "mode"): "auto",
                    ("participants", "mode"): "auto",
                    ("output", "log_dir"): candidate_log_dir.as_posix(),
                    ("output", "write_prompts"): False,
                    ("output", "debug_metrics"): False,
                }
                session.write(updates)
                result = run_topic_subprocess(args.topic, timeout_seconds=args.timeout_seconds)
                if result.ok and result.run_json is not None:
                    row = flatten_run(result.run_json, value=candidate, run_index=run_index + 1, seed=seed)
                    row.update({"status": "ok", "run_dir": result.run_dir, "error": ""})
                    candidate_payloads.append(result.run_json)
                else:
                    row = {
                        "value": candidate,
                        "run_index": run_index + 1,
                        "seed": seed,
                        "status": "failed",
                        "run_dir": result.run_dir,
                        "error": (result.stderr or result.stdout).strip()[-2000:],
                    }
                candidate_rows.append(row)
                all_rows.append(row)
                print(
                    f"[{run_index + 1:02d}/{args.runs_per_value}] "
                    f"{'ok' if result.ok else 'FAILED'}",
                    flush=True,
                )
                write_csv(result_root / "runs.csv", all_rows)

            payloads_by_value[json.dumps(candidate)] = candidate_payloads
            summary = aggregate(candidate, candidate_rows, candidate_payloads, target_turns)
            aggregates.append(summary)
            write_csv(result_root / "aggregate.csv", aggregates)

        aggregates.sort(key=lambda row: float(row.get("score", -1)), reverse=True)
        best = aggregates[0]
        if args.apply_best and int(best.get("completed_runs", 0)) > 0:
            session.apply_to_original({args.parameter: best["value"]})
            applied = True
        else:
            applied = False

        summary_payload = {
            "topic": args.topic,
            "participants": args.participants,
            "parameter": parameter_name,
            "current_value": current_value,
            "runs_per_value": args.runs_per_value,
            "target_participant_turns": target_turns,
            "score_note": (
                "Heuristic only: 30% turn-target fit, 20% repair quality, 15% semantic-reuse quality, "
                "10% drop quality, 10% question completion, 10% movement realization, 5% run completion."
            ),
            "best_value": best.get("value"),
            "best_score": best.get("score"),
            "applied_to_config": applied,
            "candidates": aggregates,
        }
        (result_root / "summary.json").write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        lines = [
            "# One-parameter tuning result",
            "",
            f"- Topic: {args.topic}",
            f"- Participants: {args.participants}",
            f"- Parameter: `{parameter_name}`",
            f"- Original value: `{current_value}`",
            f"- Runs per value: {args.runs_per_value}",
            f"- Turn target used by heuristic: {target_turns}",
            f"- Best heuristic value: `{best.get('value')}` (score {best.get('score')})",
            f"- Applied to config.yaml: {'yes' if applied else 'no'}",
            "",
            "The score is a comparison aid, not proof of a globally optimal value. A single topic can overfit.",
            "",
            "| Value | Score | Runs | Median turns | Repair rate | Reuse rate | Movement success | Tokens in |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in aggregates:
            lines.append(
                f"| {row.get('value')} | {row.get('score')} | {row.get('completed_runs')} | "
                f"{row.get('median_participant_turns', '')} | {row.get('mean_repair_rate', '')} | "
                f"{row.get('mean_reason_reuse_rate', '')} | "
                f"{row.get('movement_realization_success_rate', '')} | {row.get('mean_tokens_in', '')} |"
            )
        (result_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nResults: {result_root}")
    print(f"Best heuristic value: {best.get('value')!r} (score {best.get('score')})")
    if applied:
        print(f"Applied {parameter_name} = {best.get('value')!r} to {CONFIG_PATH}")
    else:
        print("config.yaml was restored. Re-run with --apply-best only after reviewing the summaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
