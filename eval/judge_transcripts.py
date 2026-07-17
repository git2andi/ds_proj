"""ChatEval-inspired LLM judging of completed decision transcripts.

The evaluator uses three diverse referee personas in a one-by-one sequence.
Each later referee sees the earlier assessments, matching the communication
strategy used in ChatEval. Judge order is rotated deterministically per run to
avoid assigning the same role permanent first-mover influence.

Default paths are ready for the scenario batch:

    py eval/judge_transcripts.py

Input:  ``eval/logs_scenarios``
Output: ``eval/logs_judge_scenarios``

The five dimensions use the same 1-5 range as the simulator traits:
``naturalness``, ``coherence``, ``groundedness``, ``persona_consistency``, and
``deliberation_quality``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from evaluation_metrics import (
    EVAL_DIR,
    ROOT,
    find_run_dirs,
    load_experiment_metadata,
    load_run,
    metadata_columns,
    write_csv,
)

SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from llm_client import LLMClient  # noqa: E402

DIMENSIONS = (
    "naturalness",
    "coherence",
    "groundedness",
    "persona_consistency",
    "deliberation_quality",
)

SCORE_ANCHORS = {
    1: "fundamentally invalid, contradictory, or highly unnatural",
    2: "major and repeated defects",
    3: "usable, but with clear and noticeable defects",
    4: "good, with only minor defects",
    5: "no meaningful defect found",
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


def run_context(payload: dict[str, Any]) -> str:
    scenario = payload.get("scenario", {})
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
        f"Shared context: {scenario.get('shared_context', '')}\n"
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
    prior_assessments: list[str],
    validation_feedback: str = "",
) -> str:
    prior = ""
    if prior_assessments:
        prior = (
            "\n\nAssessments from referees who spoke earlier. Treat them as arguments, not ground truth; "
            "correct them when necessary:\n" + "\n".join(prior_assessments)
        )
    anchors = "\n".join(f"- {score}: {description}" for score, description in SCORE_ANCHORS.items())
    correction = f"\n\nYour previous output was invalid: {validation_feedback}\nReturn a corrected object." if validation_feedback else ""
    score_template = ",\n".join(f'    "{dimension}": 3' for dimension in DIMENSIONS)
    return f"""You are {role_name}, one referee evaluating a simulated multi-user decision dialogue.
{role_description}

You must still score all five dimensions independently. Do not reward consensus merely because it occurred; an unresolved or majority result can be appropriate when the visible preferences justify it.{prior}

{context}

Dimensions:
- naturalness: Does the complete visible exchange sound like a plausible human group discussion, without excessive templating, repetition, unnatural addressing, or awkward moderator behavior?
- coherence: Do turns respond to the active discussion, are direct questions answered before moving on, and does the conversation progress logically across turns and phases?
- groundedness: Are factual claims by participants and moderator supported by the shared context and option cards? Penalize invented numbers, properties, guarantees, or strengthened claims.
- persona_consistency: Evaluate only simulated participants against their own complete persona cards. Consider preferences, private goals, stances, traits, and lexical style. Do not score the moderator as a persona.
- deliberation_quality: Does the publicly visible support, concern handling, movement, narrowing, and voting provide a plausible and sufficient basis for the recorded votes and outcome?

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
    prior_assessments: list[str],
    max_retries: int,
) -> tuple[dict[str, int], str, int]:
    feedback = ""
    errors: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            data = llm.generate_json(
                judge_prompt(role_name, role_description, context, prior_assessments, feedback),
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
    prior_assessments: list[str] = []
    assessments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for order, (role_name, role_description) in enumerate(roles, start=1):
        try:
            scores, verdict, retries = call_judge(
                llm,
                role_name=role_name,
                role_description=role_description,
                context=context,
                prior_assessments=prior_assessments,
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
            prior_assessments.append(f"{role_name}: scores={scores}; verdict={verdict}")
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
    metadata = load_experiment_metadata(run_dir)
    common = {
        "run_id": run_id,
        "judge_provider": getattr(llm, "provider", ""),
        "judge_model": getattr(llm, "model_id", ""),
        **metadata_columns(metadata),
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
    parser.add_argument("--output", type=Path, default=None, help="output directory inferred from --logs when omitted")
    return parser.parse_args()


def write_summary(rows: list[dict[str, Any]], errors: list[dict[str, Any]], path: Path) -> None:
    scored = [row for row in rows if row.get("judges_completed", 0)]
    complete = [row for row in scored if row.get("panel_complete")]
    provider = scored[0].get("judge_provider", "") if scored else ""
    model = scored[0].get("judge_model", "") if scored else ""
    lines = [
        "# Transcript judge summary",
        "",
        f"Judge: {provider}/{model}" if provider or model else "Judge: unavailable",
        f"Runs scored: {len(scored)}",
        f"Complete judge panels: {len(complete)}/{len(rows)}",
        f"Judge-call failures after retries: {len(errors)}",
        "",
        "| Dimension | Mean | Mean judge SD |",
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
    log_root = args.logs.resolve()
    output = args.output.resolve() if args.output else default_output_for(log_root)
    run_dirs = [path.resolve() for path in args.runs] if args.runs else find_run_dirs(log_root)
    if args.limit > 0:
        run_dirs = run_dirs[: args.limit]
    if not run_dirs:
        print(f"No run.json files found under {log_root}.", file=sys.stderr)
        return 2

    llm = LLMClient(provider=args.provider, model=args.model)
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"Judging {len(run_dirs)} run(s) with {args.judges} referees via "
        f"{llm.provider}/{llm.model_id}"
    )

    aggregate_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir.name}")
        try:
            aggregate, detailed, errors = judge_run(
                run_dir,
                llm,
                judges=args.judges,
                max_retries=args.retries,
            )
            aggregate_rows.append(aggregate)
            detailed_rows.extend(detailed)
            error_rows.extend(errors)
        except Exception as exc:
            error_rows.append(
                {
                    "run_id": run_dir.name,
                    "judge": "run_loader",
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_dir": str(run_dir),
                }
            )
            aggregate_rows.append(
                {
                    "run_id": run_dir.name,
                    "panel_complete": False,
                    "judges_requested": args.judges,
                    "judges_completed": 0,
                    "log_dir": str(run_dir),
                }
            )
        write_csv(output / "judge_scores.csv", aggregate_rows)
        if detailed_rows:
            write_csv(output / "judge_scores_detailed.csv", detailed_rows)
        if error_rows:
            write_csv(output / "judge_errors.csv", error_rows)
        write_summary(aggregate_rows, error_rows, output / "judge_summary.md")

    print(f"\nAggregate scores: {output / 'judge_scores.csv'}")
    print(f"Detailed scores: {output / 'judge_scores_detailed.csv'}")
    print(f"Summary: {output / 'judge_summary.md'}")
    if error_rows:
        print(f"Errors: {output / 'judge_errors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
