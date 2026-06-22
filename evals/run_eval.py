"""Post-run evaluation: drive runs from scenarios.yaml and check for regressions.

Usage:
    # Analyse the most recent run logs (no new runs)
    py evals/run_eval.py --check-latest 3

    # Drive new runs from the scenario spread then analyse them
    py evals/run_eval.py --run

    # Drive new runs for one specific topic/size
    py evals/run_eval.py --run --topic "Plan a group holiday trip" --size 3

All runs require a live LLM provider (uni endpoint + VPN). There is no
offline/mock mode — if the endpoint is unreachable the run raises immediately.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
SCENARIOS = Path(__file__).resolve().parent / "scenarios.yaml"
PYTHON = ROOT / "dspro" / "Scripts" / "python.exe"
CONFIG = ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MAX_FIRST_PERSON_OPENER_RATIO = 0.50
MAX_MID_DISCUSSION_ACCEPTS = 0
MAX_DUPLICATE_MODERATOR_LINES = 0
MIN_CONSENSUS_SUPPORT = 1.0
MIN_FALLBACK_SUPPORT = 0.66

_STANCE_OPENER = re.compile(
    r"\s*(?:i(?:[''](?:m|d|ve|ll))?|we(?:[''](?:re|ve|d|ll))?|our)\b", re.I
)


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------

def load_scenarios() -> list[dict]:
    with SCENARIOS.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("scenarios", [])


def set_num_participants(n: int) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    text = re.sub(
        r"(num_participants:\s*)\d+",
        rf"\g<1>{n}",
        text,
        count=1,
    )
    CONFIG.write_text(text, encoding="utf-8")


def run_topic(topic: str, size: int) -> Path | None:
    set_num_participants(size)
    print(f"\n{'='*60}")
    print(f"  Running: {topic!r}  (n={size})")
    print(f"{'='*60}")
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "main.py")],
        input=topic,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(ROOT),
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        raise RuntimeError(
            f"Run failed for {topic!r} n={size}: {result.stderr[:300]}"
        )
    # Find the run dir from stdout
    for line in result.stdout.splitlines():
        if "Logs" in line and "logs" in line:
            match = re.search(r"logs[\\/]\d{8}_\d{6}_\d+", line)
            if match:
                return ROOT / match.group(0).replace("\\", "/")
    # Fall back to newest log dir
    return latest_run_dirs(1)[0] if latest_run_dirs(1) else None


def drive_runs(topic_filter: str | None = None, size_filter: int | None = None) -> list[Path]:
    scenarios = load_scenarios()
    run_dirs: list[Path] = []
    for scenario in scenarios:
        topic = scenario["topic"]
        if topic_filter and topic_filter.lower() not in topic.lower():
            continue
        for size in scenario.get("sizes", [3]):
            if size_filter and size != size_filter:
                continue
            d = run_topic(topic, size)
            if d:
                run_dirs.append(d)
    return run_dirs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def latest_run_dirs(n: int) -> list[Path]:
    dirs = sorted(
        (d for d in LOGS.iterdir() if d.is_dir() and (d / "run.json").exists()),
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs[:n]


def load_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def check_run(run_dir: Path) -> list[str]:
    data = load_run(run_dir)
    run_id = data["run_id"]
    topic = data["topic"]
    turns = data["turns"]
    personas = {p["id"]: p for p in data["personas"]}
    metrics = data.get("metrics", {})
    outcome = data.get("outcome", {})
    failures: list[str] = []

    header = f"[{run_id}] {topic} (n={len(personas)})"

    # --- Opener variety ---
    participant_turns = [t for t in turns if t["speaker_id"] != "moderator"]
    if participant_turns:
        fp_openers = sum(1 for t in participant_turns if _STANCE_OPENER.match(t["text"]))
        ratio = fp_openers / len(participant_turns)
        if ratio > MAX_FIRST_PERSON_OPENER_RATIO:
            failures.append(f"  FAIL opener_variety: {ratio:.0%} first-person openers (max {MAX_FIRST_PERSON_OPENER_RATIO:.0%})")

    # --- Mid-discussion binding accepts ---
    mid_accepts = 0
    for t in turns:
        if t["speaker_id"] == "moderator":
            continue
        phase = t.get("phase", "")
        act = t.get("act", {})
        if phase in ("opening", "discussion") and act.get("act_type") == "accept" and act.get("accepts"):
            mid_accepts += 1
    if mid_accepts > MAX_MID_DISCUSSION_ACCEPTS:
        failures.append(f"  FAIL mid_discussion_accepts: {mid_accepts} (max {MAX_MID_DISCUSSION_ACCEPTS})")

    # --- Hard-blocker integrity ---
    for pid, persona in personas.items():
        if not persona.get("is_hard_blocker"):
            continue
        preferred = persona.get("preferred_option")
        for t in turns:
            if t["speaker_id"] != pid:
                continue
            act = t.get("act", {})
            vote = act.get("explicit_vote")
            accepts = act.get("accepts", [])
            if vote and vote != preferred:
                failures.append(f"  FAIL hard_blocker_integrity: {persona['name']} voted {vote} (preferred {preferred})")
            for a in accepts:
                if a != preferred:
                    failures.append(f"  FAIL hard_blocker_integrity: {persona['name']} accepted {a} (preferred {preferred})")

    # --- Duplicate moderator lines ---
    mod_texts = [t["text"] for t in turns if t["speaker_id"] == "moderator"]
    seen: set[str] = set()
    dupes = 0
    for text in mod_texts:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        if normalized in seen:
            dupes += 1
        seen.add(normalized)
    if dupes > MAX_DUPLICATE_MODERATOR_LINES:
        failures.append(f"  FAIL duplicate_moderator_lines: {dupes}")

    # --- Robotic template / possessive subject in final issues ---
    robotic_count = 0
    for t in participant_turns:
        for issue in t.get("validation_issues", []):
            if issue in ("ROBOTIC_TEMPLATE", "POSSESSIVE_SUBJECT"):
                robotic_count += 1
    if robotic_count > 0:
        failures.append(f"  WARN robotic_templates_remaining: {robotic_count}")

    # --- Outcome sanity ---
    status = outcome.get("status", "")
    support = metrics.get("final_support_fraction", 0)
    if status == "consensus" and support < MIN_CONSENSUS_SUPPORT:
        failures.append(f"  FAIL outcome_sanity: consensus but support={support} (need {MIN_CONSENSUS_SUPPORT})")
    if status == "fallback" and support < MIN_FALLBACK_SUPPORT:
        failures.append(f"  FAIL outcome_sanity: fallback but support={support} (need {MIN_FALLBACK_SUPPORT})")

    # --- Repair rate (informational) ---
    repair_rate = metrics.get("repair_rate", 0)
    flagged = metrics.get("flagged_turns", 0)

    # --- Report ---
    if failures:
        print(f"\n{header}")
        for f in failures:
            print(f)
    else:
        print(f"  OK   {header}  (repair_rate={repair_rate:.2f}, flagged={flagged})")

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate group-discussion simulator runs.")
    parser.add_argument("--run", action="store_true", help="Drive new runs from scenarios.yaml before checking.")
    parser.add_argument("--topic", type=str, default=None, help="Filter to scenarios matching this substring.")
    parser.add_argument("--size", type=int, default=None, help="Filter to a specific group size.")
    parser.add_argument("--check-latest", type=int, default=0, metavar="N",
                        help="Check the N most recent existing run dirs (no new runs).")
    args = parser.parse_args()

    run_dirs: list[Path] = []

    if args.run:
        run_dirs = drive_runs(topic_filter=args.topic, size_filter=args.size)
        if not run_dirs:
            print("No runs completed.")
            sys.exit(1)
    elif args.check_latest > 0:
        run_dirs = latest_run_dirs(args.check_latest)
        if not run_dirs:
            print("No run logs found.")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    print(f"\nChecking {len(run_dirs)} run(s)...\n")
    all_failures: list[str] = []
    for d in run_dirs:
        all_failures.extend(check_run(d))

    total_fail = sum(1 for f in all_failures if "FAIL" in f)
    total_warn = sum(1 for f in all_failures if "WARN" in f)
    print(f"\n{'='*60}")
    print(f"  {len(run_dirs)} runs checked: {total_fail} failures, {total_warn} warnings")
    print(f"{'='*60}")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
