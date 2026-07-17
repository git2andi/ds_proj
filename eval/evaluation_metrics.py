"""Deterministic metrics shared by the evaluation scripts.

The module only reads completed ``run.json`` files. It has no dependency on the
runtime, configuration loader, or an LLM, which keeps post-hoc evaluation cheap
and reproducible.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite a CSV using the union of keys in insertion order."""
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def find_run_dirs(root: Path) -> list[Path]:
    """Return all directories containing a ``run.json`` below *root*."""
    if not root.exists():
        return []
    return sorted(path.parent for path in root.rglob("run.json"))


def load_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def load_experiment_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "experiment_metadata.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def resolve_log_dir(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _number(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    return int(round(_number(value, default)))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _compact(value: Any) -> str:
    if value in (None, "", {}):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def metadata_columns(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": metadata.get("experiment", ""),
        "variant": metadata.get("variant", ""),
        "changed_settings": _compact(metadata.get("changed_settings")),
        "current_values": _compact(metadata.get("current_values")),
        "tested_values": _compact(metadata.get("tested_values")),
        "replicate": metadata.get("replicate", ""),
        "seed": metadata.get("seed", ""),
    }


def expected_outcome_from_votes(votes: dict[str, str], participant_count: int) -> tuple[str, str | None]:
    """Derive the protocol outcome implied by the final valid votes."""
    if participant_count <= 0 or len(votes) != participant_count:
        return "unresolved", None
    counts = Counter(votes.values())
    if not counts:
        return "unresolved", None
    option, count = counts.most_common(1)[0]
    if count == participant_count:
        return "successful", option
    if count > participant_count / 2:
        return "majority", option
    return "unresolved", None


def outcome_is_consistent(payload: dict[str, Any]) -> bool:
    option_ids = {str(option.get("id")) for option in payload.get("scenario", {}).get("options", [])}
    participants = payload.get("personas", [])
    raw_votes = payload.get("votes") or payload.get("outcome", {}).get("votes") or {}
    votes = {
        str(participant_id): str(option_id)
        for participant_id, option_id in raw_votes.items()
        if str(option_id) in option_ids
    }
    expected_status, expected_option = expected_outcome_from_votes(votes, len(participants))
    outcome = payload.get("outcome", {})
    actual_status = str(outcome.get("status", ""))
    actual_option = outcome.get("final_option")
    return actual_status == expected_status and actual_option == expected_option


def hard_blocker_violations(payload: dict[str, Any]) -> int:
    """Count visible state or vote violations for hard-blocker personas."""
    runtimes = payload.get("runtimes", {})
    votes = payload.get("votes") or payload.get("outcome", {}).get("votes") or {}
    violations = 0
    for persona in payload.get("personas", []):
        if not persona.get("hard_blocker"):
            continue
        participant_id = str(persona.get("id", ""))
        preferred = {str(value) for value in persona.get("preferred_options", [])}
        runtime = runtimes.get(participant_id, {}) if isinstance(runtimes, dict) else {}
        vote = votes.get(participant_id)
        if vote is not None and str(vote) not in preferred:
            violations += 1
        public_preference = runtime.get("public_preference")
        if public_preference is not None and str(public_preference) not in preferred:
            violations += 1
        accepted = {str(value) for value in runtime.get("public_acceptances", [])}
        violations += len(accepted - preferred)
        if _integer(runtime.get("visible_switches")) > 0:
            violations += _integer(runtime.get("visible_switches"))
    return violations


def same_speaker_statistics(turns: list[dict[str, Any]]) -> tuple[int, int]:
    """Return adjacent same-participant events and longest direct streak."""
    repeats = 0
    longest = 0
    current_speaker: str | None = None
    current_length = 0
    for turn in turns:
        if turn.get("moderator"):
            current_speaker = None
            current_length = 0
            continue
        speaker = str(turn.get("speaker_id", ""))
        if speaker and speaker == current_speaker:
            repeats += 1
            current_length += 1
        else:
            current_speaker = speaker
            current_length = 1 if speaker else 0
        longest = max(longest, current_length)
    return repeats, longest


def unsupported_fact_flags(payload: dict[str, Any]) -> int:
    """Count validator flags whose labels explicitly indicate grounding issues.

    This is intentionally a count of validator signals, not a claim that every
    unsupported fact in the transcript is detected.
    """
    causes = payload.get("metrics", {}).get("generation", {}).get("repair_causes", {})
    if not isinstance(causes, dict):
        return 0
    terms = (
        "unsupported",
        "ungrounded",
        "not_grounded",
        "not_in_the_option",
        "not_in_shared_context",
        "invented",
        "fabricated",
        "factual_claim",
    )
    total = 0
    for key, value in causes.items():
        normalized = str(key).lower().replace(" ", "_")
        if any(term in normalized for term in terms):
            total += _integer(value)
    return total


def fallback_turns(payload: dict[str, Any]) -> int:
    attempts = payload.get("failed_generation_attempts", [])
    if isinstance(attempts, list):
        return sum(1 for attempt in attempts if str(attempt.get("final_status", "")) == "fallback")
    generation = payload.get("metrics", {}).get("generation", {})
    return _integer(generation.get("vote_fallbacks")) + _integer(generation.get("movement_fallbacks"))


def extract_run_metrics(payload: dict[str, Any], run_dir: Path | None = None) -> dict[str, Any]:
    """Extract the compact deterministic metrics used by the project report."""
    metrics = payload.get("metrics", {})
    turns_metrics = metrics.get("turns", {})
    generation = metrics.get("generation", {})
    questions = metrics.get("questions", {})
    issues = metrics.get("issues", {})
    stances = metrics.get("stances", {})
    coverage = metrics.get("coverage", {})
    vote_metrics = metrics.get("votes", {})
    token_metrics = metrics.get("tokens", {})

    personas = payload.get("personas", [])
    participant_count = len(personas)
    participant_turns = _integer(turns_metrics.get("participant"))
    self_selected_turns = _integer(turns_metrics.get("self_selected", turns_metrics.get("voluntary")))
    questions_opened = _integer(questions.get("opened"))
    questions_answered = _integer(questions.get("answered"))
    issues_opened = _integer(issues.get("opened"))
    issues_resolved = _integer(issues.get("resolved"))
    issues_stale = _integer(issues.get("stale"))
    valid_votes = _integer(vote_metrics.get("valid"))
    unclear_votes = _integer(vote_metrics.get("unclear"))
    repairs = _integer(generation.get("repairs"))
    dropped = _integer(generation.get("dropped"))
    response_failures = _integer(generation.get("response_failures"))
    fallbacks = fallback_turns(payload)
    semantic_reuse = _integer(generation.get("semantic_reason_reuse"))
    selected_movement = _integer(generation.get("selected_movement_actions"))
    movement_failures = _integer(generation.get("movement_realization_failures"))
    movement_fallbacks = _integer(generation.get("movement_fallbacks"))
    committed_movement = _integer(generation.get("committed_movement_actions"))
    direct_movement = max(0, selected_movement - movement_failures)
    unexplained_movement = _integer(stances.get("unexplained_movements"))
    tokens_in = _integer(token_metrics.get("input"))
    option_count = len(payload.get("scenario", {}).get("options", []))
    covered_options = sum(1 for value in coverage.values() if _number(value) > 0) if isinstance(coverage, dict) else 0
    same_speaker_repeats, longest_streak = same_speaker_statistics(payload.get("turns", []))
    blocker_violations = hard_blocker_violations(payload)
    consistent_outcome = outcome_is_consistent(payload)
    phase_history = payload.get("phase_history", [])
    closed = bool(phase_history) and str(phase_history[-1]) == "CLOSED"
    vote_protocol_degraded = bool(vote_metrics.get("protocol_degraded", False))
    protocol_errors = vote_metrics.get("protocol_errors", [])
    protocol_error_count = len(protocol_errors) if isinstance(protocol_errors, list) else 0

    structural_pass = (
        closed
        and valid_votes == participant_count
        and questions_answered == questions_opened
        and unclear_votes == 0
        and not vote_protocol_degraded
        and response_failures == 0
        and protocol_error_count == 0
        and consistent_outcome
        and blocker_violations == 0
        and unexplained_movement == 0
    )

    metadata = load_experiment_metadata(run_dir) if run_dir else {}
    outcome = payload.get("outcome", {})
    return {
        "run_id": payload.get("run_id", run_dir.name if run_dir else ""),
        **metadata_columns(metadata),
        "topic": payload.get("scenario", {}).get("topic", ""),
        "participants": participant_count,
        "outcome": outcome.get("status", ""),
        "final_option": outcome.get("final_option") or "",
        "closed": closed,
        "structural_pass": structural_pass,
        "outcome_consistent": consistent_outcome,
        "valid_final_votes": valid_votes,
        "valid_vote_rate": _ratio(valid_votes, participant_count),
        "unclear_final_votes": unclear_votes,
        "question_answer_rate": _ratio(questions_answered, questions_opened) if questions_opened else 1.0,
        "questions_opened": questions_opened,
        "questions_answered": questions_answered,
        "response_failures": response_failures,
        "protocol_error_count": protocol_error_count,
        "hard_blocker_violations": blocker_violations,
        "unexplained_movements": unexplained_movement,
        "participant_turns": participant_turns,
        "self_selected_turns": self_selected_turns,
        "voluntary_turn_share": _ratio(self_selected_turns, participant_turns),
        "issues_opened": issues_opened,
        "issues_resolved": issues_resolved,
        "issues_stale": issues_stale,
        "issue_resolution_rate": _ratio(issues_resolved, issues_opened) if issues_opened else 1.0,
        "option_coverage_ratio": _ratio(covered_options, option_count) if option_count else 0.0,
        "repairs": repairs,
        "repair_rate": _ratio(repairs, participant_turns),
        "dropped_turns": dropped,
        "dropped_rate": _ratio(dropped, participant_turns),
        "fallback_turns": fallbacks,
        "fallback_rate": _ratio(fallbacks, participant_turns),
        "semantic_reason_reuse": semantic_reuse,
        "repetition_per_10_turns": 10.0 * _ratio(semantic_reuse, participant_turns),
        "same_speaker_repeat_events": same_speaker_repeats,
        "same_speaker_repeats_per_10_turns": 10.0 * _ratio(same_speaker_repeats, participant_turns),
        "longest_participant_streak": longest_streak,
        "unsupported_fact_flags": unsupported_fact_flags(payload),
        "selected_movement_actions": selected_movement,
        "directly_realized_movement_actions": direct_movement,
        "movement_fallbacks": movement_fallbacks,
        "movement_realization_failures": movement_failures,
        "committed_movement_actions": committed_movement,
        "direct_movement_realization_rate": _ratio(direct_movement, selected_movement) if selected_movement else 1.0,
        "movement_commit_rate": _ratio(committed_movement, selected_movement) if selected_movement else 1.0,
        "tokens_in": tokens_in,
        "tokens_per_participant_turn": _ratio(tokens_in, participant_turns),
        "log_dir": str(run_dir) if run_dir else "",
        "error": "",
    }


def participant_trait_rows(payload: dict[str, Any], run_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return one observable-behaviour row per simulated participant."""
    run_id = payload.get("run_id", run_dir.name if run_dir else "")
    participant_metrics = payload.get("metrics", {}).get("participants", {})
    runtimes = payload.get("runtimes", {})
    total_voluntary = sum(
        _integer(values.get("voluntary"))
        for values in participant_metrics.values()
        if isinstance(values, dict)
    )
    rows: list[dict[str, Any]] = []
    for persona in payload.get("personas", []):
        participant_id = str(persona.get("id", ""))
        observed = participant_metrics.get(participant_id, {}) if isinstance(participant_metrics, dict) else {}
        runtime = runtimes.get(participant_id, {}) if isinstance(runtimes, dict) else {}
        traits = persona.get("sim_params", {})
        voluntary = _integer(observed.get("voluntary", runtime.get("voluntary_turns")))
        initial_preference = observed.get("initial_preference", runtime.get("preferred_option", ""))
        final_preference = observed.get("final_preference", runtime.get("public_preference", ""))
        acceptances = len(runtime.get("public_acceptances", []))
        switches = _integer(runtime.get("visible_switches"))
        flexibility_events = acceptances + switches
        rows.append(
            {
                "run_id": run_id,
                "topic": payload.get("scenario", {}).get("topic", ""),
                "participant_id": participant_id,
                "name": persona.get("name", ""),
                "engagement": _integer(traits.get("engagement")),
                "verbosity": _integer(traits.get("verbosity")),
                "directness": _integer(traits.get("directness")),
                "stubbornness": _integer(traits.get("stubbornness")),
                "turns": _integer(observed.get("turns")),
                "voluntary_turns": voluntary,
                "normalized_voluntary_share": _ratio(voluntary, total_voluntary),
                "avg_words_per_turn": _number(observed.get("avg_words")),
                "initial_preference": initial_preference,
                "final_preference": final_preference,
                "preference_changed": bool(initial_preference and final_preference and initial_preference != final_preference),
                "public_acceptances": acceptances,
                "visible_switches": switches,
                "flexibility_events": flexibility_events,
                "showed_flexibility": flexibility_events > 0 or bool(initial_preference and final_preference and initial_preference != final_preference),
                "hard_blocker": bool(persona.get("hard_blocker")),
                "log_dir": str(run_dir) if run_dir else "",
            }
        )
    return rows


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in range(position, end):
            ranks[indexed[index][0]] = average_rank
        position = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_values = [float(value) for value in x]
    y_values = [float(value) for value in y]
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return pearson(_rankdata(x_values), _rankdata(y_values))


def trait_correlations(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    non_blockers = [row for row in rows if not row.get("hard_blocker")]
    return {
        "engagement_vs_voluntary_share": spearman(
            [row["engagement"] for row in rows],
            [row["normalized_voluntary_share"] for row in rows],
        ),
        "verbosity_vs_words": spearman(
            [row["verbosity"] for row in rows],
            [row["avg_words_per_turn"] for row in rows],
        ),
        "stubbornness_vs_flexibility": spearman(
            [row["stubbornness"] for row in non_blockers],
            [float(row["flexibility_events"]) for row in non_blockers],
        ),
    }


def trait_level_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specifications = (
        ("engagement", "normalized_voluntary_share", "normalized voluntary-turn share"),
        ("verbosity", "avg_words_per_turn", "average words per turn"),
        ("stubbornness", "showed_flexibility", "fraction showing acceptance or movement"),
    )
    output: list[dict[str, Any]] = []
    for trait, observed_key, label in specifications:
        source = rows if trait != "stubbornness" else [row for row in rows if not row.get("hard_blocker")]
        for level in range(1, 6):
            group = [row for row in source if _integer(row.get(trait)) == level]
            values = [float(row[observed_key]) for row in group]
            output.append(
                {
                    "trait": trait,
                    "level": level,
                    "n": len(group),
                    "observed_metric": label,
                    "mean": statistics.mean(values) if values else "",
                    "median": statistics.median(values) if values else "",
                }
            )
    return output


def discover_batch_error_rows(log_root: Path) -> list[dict[str, Any]]:
    """Read failed batch entries that have no ``run.json`` directory."""
    candidates = (
        log_root / "scenario_runs.csv",
        log_root / "sweep_runs.csv",
        log_root / "confirmation_runs.csv",
    )
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        errors: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            if str(row.get("outcome", "")).lower() != "error":
                continue
            errors.append(
                {
                    "run_id": f"batch_error_{index}",
                    "experiment": row.get("experiment", ""),
                    "variant": row.get("variant", ""),
                    "topic": row.get("topic", ""),
                    "participants": row.get("participants", ""),
                    "outcome": "error",
                    "structural_pass": False,
                    "log_dir": row.get("log_dir", ""),
                    "error": row.get("error", "unknown batch error"),
                }
            )
        return errors
    return []
