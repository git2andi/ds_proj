"""Independent multi-perspective LLM judging of completed transcripts.

The evaluator uses up to three diverse referee personas. Every referee receives
the same public scenario, complete persona cards, visible transcript, votes, and
outcome, but never another referee's assessment. Judge order is rotated
deterministically per run only to distribute API call ordering.

Default paths are ready for the scenario batch:

    py eval/judge_transcripts.py

Input:  ``eval/logs_scenarios``
Output: ``eval/logs_judge_scenarios``

The output is resumable. Existing complete judge panels are preserved and
skipped; incomplete, failed, or newly discovered runs are processed on the next
invocation. Results are persisted after every run. Different runs may be judged
in parallel, while each run's referee calls remain sequential.

The five dimensions use the same 1-5 range as the simulator traits:
``naturalness``, ``coherence``, ``groundedness``, ``persona_consistency``, and
``deliberation_quality``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from summarize_runs import (
    EVAL_DIR,
    ROOT,
    find_run_dirs,
    load_run,
)

SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from llm_client import LLMClient  # noqa: E402

JUDGE_PROMPT_VERSION = "strict-naturalness-coherence-v3"

DIMENSIONS = (
    "naturalness",
    "coherence",
    "groundedness",
    "persona_consistency",
    "deliberation_quality",
)

SCORE_ANCHORS = {
    1: "fundamentally broken, contradictory, or implausible",
    2: "major or recurring defects substantially harm the dialogue",
    3: "usable, but with clear and noticeable defects",
    4: "strong, with no more than one isolated minor defect",
    5: "fully convincing, with no observable defect; use rarely",
}

JUDGE_PERSONAS: tuple[tuple[str, str], ...] = (
    (
        "Conversation analyst",
        "Focus especially on interactional flow, response relevance, turn-taking, repetition, and whether the exchange reads like a plausible group chat rather than a script.",
    ),
    (
        "Behavioral scientist",
        "Focus especially on whether each simulated participant follows the complete assigned persona card: goals, preferences, stances, engagement, verbosity, directness, stubbornness, and speech style. Do not evaluate the moderator as a persona.",
    ),
    (
        "Grounding and decision auditor",
        "Focus especially on unsupported claims, vote and outcome consistency, visible stance development, and whether the publicly visible deliberation earns the recorded result.",
    ),
)


def default_output_for(log_root: Path) -> Path:
    name = log_root.name
    suffix = name[5:] if name.startswith("logs_") else name
    return EVAL_DIR / f"logs_judge_{suffix}"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _parse_number(value: Any) -> Any:
    text = str(value).strip()
    if not text:
        return value
    try:
        number = float(text)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read existing judge output so interrupted runs can resume safely."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    numeric_fields = {
        "judges_requested",
        "judges_completed",
        "participants",
        "seed",
        "judge_order",
        "retries",
        "overall",
        *(DIMENSIONS),
        *(f"{dimension}_sd" for dimension in DIMENSIONS),
        *(f"{dimension}_range" for dimension in DIMENSIONS),
    }
    for row in rows:
        if "panel_complete" in row:
            row["panel_complete"] = _parse_bool(row["panel_complete"])
        for field in numeric_fields:
            if field in row:
                row[field] = _parse_number(row[field])
    return rows


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace one CSV atomically while preserving all accumulated rows."""

    if not rows:
        path.unlink(missing_ok=True)
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def panel_is_complete_for(
    row: dict[str, Any] | None,
    *,
    judges: int,
    provider: str,
    model: str,
    prompt_version: str,
) -> bool:
    if not row or not _parse_bool(row.get("panel_complete", False)):
        return False
    return (
        int(row.get("judges_requested", 0) or 0) == judges
        and str(row.get("judge_provider", "")) == provider
        and str(row.get("judge_model", "")) == model
        and str(row.get("judge_prompt_version", "")) == prompt_version
    )


def run_context(payload: dict[str, Any]) -> str:
    scenario = payload.get("scenario", {})
    raw_context = scenario.get("shared_context", "")
    shared_context = (
        " ".join(str(item).strip() for item in raw_context if str(item).strip())
        if isinstance(raw_context, list)
        else str(raw_context).strip()
    )
    option_lines: list[str] = []
    for option in scenario.get("options", []):
        attributes = "; ".join(f"{key}: {value}" for key, value in option.get("attrs", {}).items())
        option_lines.append(
            f"- {option.get('id')}) {option.get('name')} — {attributes} "
            f"(+ {option.get('upside', '')}; - {option.get('concern', '')})"
        )
    personas = "\n\n".join(
        f"Persona {index}:\n{json.dumps(persona, ensure_ascii=False, indent=2, sort_keys=True)}"
        for index, persona in enumerate(payload.get("personas", []), start=1)
    )
    transcript = "\n".join(
        f"{turn.get('speaker_name', turn.get('speaker_id', 'Unknown'))}: {turn.get('text', '')}"
        for turn in payload.get("turns", [])
    )
    votes = payload.get("votes") or payload.get("outcome", {}).get("votes") or {}
    outcome = payload.get("outcome", {})
    return (
        f"Topic: {scenario.get('topic', '')}\n"
        f"Shared context: {shared_context}\n"
        f"Options:\n{chr(10).join(option_lines)}\n\n"
        f"Complete persona cards (ground truth for persona consistency):\n{personas}\n\n"
        f"Visible transcript, including moderator turns:\n{transcript}\n\n"
        f"Final votes: {json.dumps(votes, ensure_ascii=False, sort_keys=True)}\n"
        f"Recorded outcome: {outcome.get('status', '')} ({outcome.get('final_option') or 'no option'})"
    )


def rotated_judges(order_key: str, judges: int) -> list[tuple[str, str]]:
    digest = hashlib.sha256(order_key.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % len(JUDGE_PERSONAS)
    ordered = list(JUDGE_PERSONAS[offset:] + JUDGE_PERSONAS[:offset])
    return ordered[:judges]


def judge_prompt(
    role_name: str,
    role_description: str,
    context: str,
    validation_feedback: str = "",
) -> str:
    anchors = "\n".join(
        f"- {score}: {description}" for score, description in SCORE_ANCHORS.items()
    )
    correction = (
        f"\n\nYour previous output was invalid: {validation_feedback}"
        "\nReturn a corrected object."
        if validation_feedback
        else ""
    )
    score_template = ",\n".join(
        f'    "{dimension}": 3' for dimension in DIMENSIONS
    )
    return f"""You are {role_name}, one referee evaluating a simulated multi-user decision dialogue.
{role_description}

Score all five dimensions independently. Do not reward consensus merely because it occurred, and do not penalize an unresolved result merely because agreement was not reached. Your assessment is independent; you receive no other referee scores.

The moderator has no persona. Include moderator turns in naturalness, coherence, groundedness, and deliberation quality, but exclude the moderator from persona_consistency. Private persona fields are reference information and need not be stated aloud.

{context}

Dimensions:
- naturalness: Judge whether the complete exchange reads like an actual group discussion rather than a generated script. Penalize repeated option facts, formulaic sentence openings, serial restatement of preferences, excessive agreement phrases, unnatural use of names, abrupt or overly polished turns, and moderator language that feels mechanical.
- coherence: Judge whether turns respond to the active point, direct questions are answered or explicitly deferred, the moderator allows responses before changing phase, and public stances, narrowing, votes, and outcome remain consistent. Penalize repeated concerns without progress, irrelevant responses, abrupt phase changes, ignored questions, and unexplained movement or votes.
- groundedness: Are factual and qualitative claims supported by the shared context and option cards? Penalize invented or altered numbers, unsupported properties, transferred attributes, guarantees, exaggerations, and strengthened claims.
- persona_consistency: Evaluate only simulated participants against their complete persona cards, including preferences, goals, stances, traits, and speech style.
- deliberation_quality: Does the visible support, concern handling, movement, narrowing, and voting provide a plausible basis for the final votes and outcome?

Strict calibration:
- Use the full scale. Start naturalness and coherence at 3, and raise them only when the complete transcript clearly earns a higher anchor.
- Fluent grammar alone does not imply naturalness. A fluent but scripted, repetitive, or weakly interactive exchange should remain at 3 or below.
- A 5 is rare and requires no observable defect in that dimension.
- Naturalness or coherence may receive 4 only when there is at most one isolated minor defect.
- Recurring templating, repeated factual restatement, or a sequence of largely independent statements should cap naturalness at 3.
- An ignored direct question, moderator phase change without a response opportunity, or unexplained stance/vote inconsistency should cap coherence at 3.
- When uncertain between two scores, choose the lower score.
- The verdict must identify a concrete transcript-specific strength or defect.

Use integer scores with these anchors:
{anchors}

Return JSON only, with every dimension present and no additional prose:
{{
  "scores": {{
{score_template}
  }},
  "verdict": "one or two specific sentences naming the main strength or defect"
}}{correction}
"""


def validate_response(data: Any) -> tuple[dict[str, int], str]:
    if not isinstance(data, dict):
        raise ValueError("top-level value is not an object")
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("missing scores object")
    scores: dict[str, int] = {}
    for dimension in DIMENSIONS:
        if dimension not in raw_scores:
            raise ValueError(f"missing score: {dimension}")
        value = raw_scores[dimension]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{dimension} must be an integer from 1 to 5")
        if int(value) != value or not 1 <= int(value) <= 5:
            raise ValueError(f"{dimension} must be an integer from 1 to 5")
        scores[dimension] = int(value)
    verdict = str(data.get("verdict", "")).strip()
    if not verdict:
        raise ValueError("missing verdict")
    return scores, verdict


def call_judge(
    llm: Any,
    *,
    role_name: str,
    role_description: str,
    context: str,
    max_retries: int,
) -> tuple[dict[str, int], str, int]:
    feedback = ""
    errors: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            data = llm.generate_json(
                judge_prompt(role_name, role_description, context, feedback),
                profile="setup",
            )
            scores, verdict = validate_response(data)
            return scores, verdict, attempt
        except Exception as exc:
            feedback = f"{type(exc).__name__}: {exc}"
            errors.append(feedback)
    raise RuntimeError("; ".join(errors))


def judge_payload(
    payload: dict[str, Any],
    llm: Any,
    *,
    judges: int = 3,
    max_retries: int = 2,
    order_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = run_context(payload)
    run_id = str(payload.get("run_id", "run"))
    roles = rotated_judges(order_key or run_id, judges)
    assessments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for order, (role_name, role_description) in enumerate(roles, start=1):
        try:
            scores, verdict, retries = call_judge(
                llm,
                role_name=role_name,
                role_description=role_description,
                context=context,
                max_retries=max_retries,
            )
            assessment = {
                "judge_order": order,
                "judge": role_name,
                "scores": scores,
                "verdict": verdict,
                "retries": retries,
            }
            assessments.append(assessment)
        except Exception as exc:
            errors.append(
                {
                    "judge_order": order,
                    "judge": role_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return assessments, errors


def aggregate_assessments(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    if not assessments:
        return {}
    row: dict[str, Any] = {"judges_completed": len(assessments)}
    dimension_means: list[float] = []
    for dimension in DIMENSIONS:
        values = [float(assessment["scores"][dimension]) for assessment in assessments]
        mean = statistics.mean(values)
        row[dimension] = round(mean, 2)
        row[f"{dimension}_sd"] = round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0
        row[f"{dimension}_range"] = max(values) - min(values)
        dimension_means.append(mean)
    row["overall"] = round(statistics.mean(dimension_means), 2)
    row["verdicts"] = " | ".join(f"{item['judge']}: {item['verdict']}" for item in assessments)
    return row


def judge_run(
    run_dir: Path,
    llm: Any,
    *,
    judges: int,
    max_retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_run(run_dir)
    run_id = str(payload.get("run_id", run_dir.name))
    assessments, errors = judge_payload(
        payload,
        llm,
        judges=judges,
        max_retries=max_retries,
        order_key=run_id,
    )
    common = {
        "run_id": run_id,
        "judge_provider": getattr(llm, "provider", ""),
        "judge_model": getattr(llm, "model_id", ""),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "seed": (payload.get("provenance") or {}).get("seed", ""),
        "dialogue_provider": (payload.get("provenance") or {}).get("dialogue_provider", ""),
        "dialogue_model": (payload.get("provenance") or {}).get("dialogue_model", ""),
        "topic": payload.get("scenario", {}).get("topic", ""),
        "participants": len(payload.get("personas", [])),
        "outcome": payload.get("outcome", {}).get("status", ""),
        "log_dir": str(run_dir),
    }
    detailed: list[dict[str, Any]] = []
    for assessment in assessments:
        detailed.append(
            {
                **common,
                "judge_order": assessment["judge_order"],
                "judge": assessment["judge"],
                **assessment["scores"],
                "verdict": assessment["verdict"],
                "retries": assessment["retries"],
            }
        )
    error_rows = [{**common, **error} for error in errors]
    aggregate = {
        **common,
        "judges_requested": judges,
        "panel_complete": len(assessments) == judges,
        **aggregate_assessments(assessments),
    }
    return aggregate, detailed, error_rows


def judge_run_with_client(
    run_dir: Path,
    *,
    provider: str,
    model: str,
    judges: int,
    max_retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Judge one complete run with a worker-local client."""

    llm = LLMClient(provider=provider, model=model)
    return judge_run(
        run_dir,
        llm,
        judges=judges,
        max_retries=max_retries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=EVAL_DIR / "logs_scenarios", help="root scanned for run.json")
    parser.add_argument("--runs", nargs="*", type=Path, default=None, help="explicit run directories")
    parser.add_argument("--judges", type=int, default=3, choices=(1, 2, 3), help="number of referee personas")
    parser.add_argument("--retries", type=int, default=2, help="retries per malformed judge response")
    parser.add_argument(
        "--provider",
        type=str,
        default="uni",
        choices=("uni", "groq", "gemini", "gpt"),
        help="judge provider; defaults to a provider separate from the normal dialogue runtime",
    )
    parser.add_argument("--model", type=str, default=None, help="judge model id")
    parser.add_argument("--limit", type=int, default=0, help="judge at most this many runs")
    parser.add_argument("--workers", type=int, default=2, help="parallel run-level workers; default: 2")
    parser.add_argument("--output", type=Path, default=None, help="output directory inferred from --logs when omitted")
    return parser.parse_args()


def write_summary(rows: list[dict[str, Any]], errors: list[dict[str, Any]], path: Path) -> None:
    current = [
        row for row in rows
        if str(row.get("judge_prompt_version", "")) == JUDGE_PROMPT_VERSION
    ]
    scored = [row for row in current if row.get("judges_completed", 0)]
    complete = [row for row in scored if row.get("panel_complete")]
    current_errors = [
        row for row in errors
        if str(row.get("judge_prompt_version", "")) == JUDGE_PROMPT_VERSION
    ]
    provider = scored[0].get("judge_provider", "") if scored else ""
    model = scored[0].get("judge_model", "") if scored else ""
    lines = [
        "# Transcript judge summary",
        "",
        f"Judge: {provider}/{model}" if provider or model else "Judge: unavailable",
        f"Runs scored: {len(scored)}",
        f"Complete judge panels: {len(complete)}/{len(current)}",
        f"Judge-call failures after retries: {len(current_errors)}",
        "",
        "| Dimension | Mean | Mean inter-judge SD |",
        "|---|---:|---:|",
    ]
    for dimension in (*DIMENSIONS, "overall"):
        mean = statistics.mean(float(row[dimension]) for row in scored) if scored else float("nan")
        if dimension == "overall":
            lines.append(f"| {dimension} | {mean:.2f} | – |")
        else:
            sd = statistics.mean(float(row[f"{dimension}_sd"]) for row in scored)
            lines.append(f"| {dimension} | {mean:.2f} | {sd:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    log_root = args.logs.resolve()
    output = args.output.resolve() if args.output else default_output_for(log_root)
    run_dirs = [path.resolve() for path in args.runs] if args.runs else find_run_dirs(log_root)
    if not run_dirs:
        print(f"No run.json files found under {log_root}.", file=sys.stderr)
        return 2

    llm = LLMClient(provider=args.provider, model=args.model)
    output.mkdir(parents=True, exist_ok=True)

    aggregate_path = output / "judge_scores.csv"
    detailed_path = output / "judge_scores_detailed.csv"
    error_path = output / "judge_errors.csv"
    summary_path = output / "judge_summary.md"

    aggregate_rows = read_csv_rows(aggregate_path)
    detailed_rows = read_csv_rows(detailed_path)
    error_rows = read_csv_rows(error_path)
    aggregate_by_run = {str(row.get("run_id", "")): row for row in aggregate_rows}

    pending: list[tuple[Path, str]] = []
    skipped = 0
    for run_dir in run_dirs:
        try:
            payload = load_run(run_dir)
            run_id = str(payload.get("run_id", run_dir.name))
        except Exception:
            run_id = run_dir.name
        if panel_is_complete_for(
            aggregate_by_run.get(run_id),
            judges=args.judges,
            provider=str(llm.provider),
            model=str(llm.model_id),
            prompt_version=JUDGE_PROMPT_VERSION,
        ):
            skipped += 1
            continue
        pending.append((run_dir, run_id))

    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"Found {len(run_dirs)} completed run(s); skipping {skipped} already judged "
        f"panel(s). Judging {len(pending)} pending run(s) with {args.judges} "
        f"referees via {llm.provider}/{llm.model_id} using {args.workers} worker(s)."
    )

    if not pending:
        write_summary(aggregate_rows, error_rows, summary_path)
        print("No new or incomplete runs require judging.")
        print(f"Aggregate scores: {aggregate_path}")
        print(f"Detailed scores: {detailed_path}")
        print(f"Summary: {summary_path}")
        return 0

    provider = str(llm.provider)
    model = str(llm.model_id)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                judge_run_with_client,
                run_dir,
                provider=provider,
                model=model,
                judges=args.judges,
                max_retries=args.retries,
            ): (run_dir, known_run_id)
            for run_dir, known_run_id in pending
        }

        for index, future in enumerate(as_completed(futures), start=1):
            run_dir, known_run_id = futures[future]
            print(f"[{index}/{len(pending)}] {run_dir.name}")

            # Replace only rows for the completed run. All writes remain in the
            # parent thread, so concurrent workers cannot corrupt the CSV files.
            aggregate_rows = [
                row for row in aggregate_rows
                if str(row.get("run_id", "")) != known_run_id
            ]
            detailed_rows = [
                row for row in detailed_rows
                if str(row.get("run_id", "")) != known_run_id
            ]
            error_rows = [
                row for row in error_rows
                if str(row.get("run_id", "")) != known_run_id
            ]

            try:
                aggregate, detailed, errors = future.result()
                aggregate_rows.append(aggregate)
                detailed_rows.extend(detailed)
                error_rows.extend(errors)
            except Exception as exc:
                error_rows.append(
                    {
                        "run_id": known_run_id,
                        "judge_provider": provider,
                        "judge_model": model,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                        "judge": "run_loader",
                        "error": f"{type(exc).__name__}: {exc}",
                        "log_dir": str(run_dir),
                    }
                )
                aggregate_rows.append(
                    {
                        "run_id": known_run_id,
                        "judge_provider": provider,
                        "judge_model": model,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                        "panel_complete": False,
                        "judges_requested": args.judges,
                        "judges_completed": 0,
                        "log_dir": str(run_dir),
                    }
                )

            write_csv_atomic(aggregate_path, aggregate_rows)
            write_csv_atomic(detailed_path, detailed_rows)
            write_csv_atomic(error_path, error_rows)
            temporary_summary = summary_path.with_name(f".{summary_path.name}.tmp")
            write_summary(aggregate_rows, error_rows, temporary_summary)
            temporary_summary.replace(summary_path)

    print(f"\nAggregate scores: {aggregate_path}")
    print(f"Detailed scores: {detailed_path}")
    print(f"Summary: {summary_path}")
    if error_rows:
        print(f"Errors: {error_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
