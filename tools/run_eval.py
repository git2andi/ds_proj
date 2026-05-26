"""
tools/run_eval.py
-----------------
Evaluation harness — Stage 12 (MASTERPLAN).

Runs all scenarios from eval/eval_scenarios.txt at a fixed random seed and
emits a metrics report by delegating to tools/analyze_logs.py.

Usage:
  python tools/run_eval.py                     # run all 50 topics
  python tools/run_eval.py --limit 10          # first N topics only (quick check)
  python tools/run_eval.py --json              # output metrics as JSON
  python tools/run_eval.py --no-run            # analyse existing logs without re-running

Comparison:
  Run once to establish a baseline.  After each stage, re-run and compare
  the new metrics against the saved baseline.

  python tools/run_eval.py --save-baseline     # write eval/baseline_metrics.json
  python tools/run_eval.py --compare           # compare against baseline after run

Config key used: cfg.evaluation.seed, cfg.evaluation.eval_scenarios_path
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Resolve project root (one level above tools/)
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config_loader import cfg

# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def _load_topics(path: Path, limit: int | None = None) -> list[str]:
    topics: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            topics.append(line)
            if limit and len(topics) >= limit:
                break
    return topics


# ---------------------------------------------------------------------------
# Single-dialogue runner
# ---------------------------------------------------------------------------

def _run_one(topic: str, seed: int) -> dict:
    """Run one dialogue and return basic per-dialogue stats."""
    import random as _rng
    _rng.seed(seed)

    from llm_client import get_llm_client
    from orchestrator import Orchestrator
    from persona import PersonaBuilder
    from simulator import Simulator

    names_pool: list[str] = list(cfg.simulation.name_pool)
    _rng.shuffle(names_pool)

    n = cfg.simulation.num_participants
    names = names_pool[:n]

    builder = PersonaBuilder(topic)
    personas = builder.build_all(names)

    orch = Orchestrator(topic, moderator_style=cfg.simulation.moderator_style)
    builder.dialogue_id = orch.dialogue_id
    builder.assign_beliefs(personas, orch.options)

    setup_in = get_llm_client().session_tokens_in
    setup_out = get_llm_client().session_tokens_out
    get_llm_client()._reset_session_tokens()

    for p in personas:
        orch.add_sim(Simulator(p, topic, orch.options))

    orch.run_simulation(setup_tokens_in=setup_in, setup_tokens_out=setup_out)
    return {
        "topic": topic,
        "outcome": orch.outcome,
        "dialogue_id": orch.dialogue_id,
    }


# ---------------------------------------------------------------------------
# Metrics comparison
# ---------------------------------------------------------------------------

def _compare(current: dict, baseline: dict) -> None:
    skip_keys = {"dialogues_analysed", "consensus_tier_distribution", "avg_phase_turn_counts"}
    print("\n  Delta vs baseline:")
    for key, val in current.items():
        if key in skip_keys or not isinstance(val, (int, float)):
            continue
        base_val = baseline.get(key)
        if base_val is None:
            continue
        delta = val - base_val
        sign = "+" if delta >= 0 else ""
        unit = "%" if "rate" in key else ""
        if "rate" in key:
            print(f"    {key:<40} {val:.1%}  ({sign}{delta:.1%})")
        else:
            print(f"    {key:<40} {val:.1f}  ({sign}{delta:.1f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation harness (Stage 12)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of topics to run (default: all)")
    parser.add_argument("--json", action="store_true",
                        help="Print metrics as JSON instead of text")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip dialogue runs; only analyse existing logs")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save metrics to eval/baseline_metrics.json after analysis")
    parser.add_argument("--compare", action="store_true",
                        help="Compare current metrics against baseline after analysis")
    args = parser.parse_args()

    scenarios_path = _PROJECT_ROOT / cfg.evaluation.eval_scenarios_path
    if not scenarios_path.exists():
        print(f"Error: eval scenarios file not found: {scenarios_path}", file=sys.stderr)
        sys.exit(1)

    topics = _load_topics(scenarios_path, limit=args.limit)
    seed = int(cfg.evaluation.seed)

    # ── Run dialogues ──────────────────────────────────────────────────────
    if not args.no_run:
        print(f"\nRunning {len(topics)} topic(s) at seed={seed} ...")
        t0 = time.time()
        results: list[dict] = []
        for i, topic in enumerate(topics, 1):
            topic_seed = seed + i
            print(f"\n[{i}/{len(topics)}] {topic}")
            try:
                result = _run_one(topic, topic_seed)
                results.append(result)
            except Exception as exc:
                print(f"!! Dialogue failed: {exc}")
                results.append({"topic": topic, "outcome": "error"})

        elapsed = time.time() - t0
        print(f"\nAll {len(topics)} dialogues completed in {elapsed:.0f}s.")

        success = sum(1 for r in results if r.get("outcome") == "success")
        force_closed = sum(1 for r in results
                           if r.get("outcome") in ("force_close", "best_available_decision"))
        errors = sum(1 for r in results if r.get("outcome") == "error")
        print(f"  success={success}  force_close={force_closed}  errors={errors}")

    # ── Analyse logs ──────────────────────────────────────────────────────
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))
    from analyze_logs import analyse, _print_report

    log_dir = _PROJECT_ROOT / cfg.output.log_dir
    metrics = analyse(log_dir)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        _print_report(metrics)

    # ── Save baseline ─────────────────────────────────────────────────────
    if args.save_baseline:
        baseline_path = _PROJECT_ROOT / "eval" / "baseline_metrics.json"
        baseline_path.parent.mkdir(exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Baseline saved → {baseline_path}")

    # ── Compare with baseline ─────────────────────────────────────────────
    if args.compare:
        baseline_path = _PROJECT_ROOT / "eval" / "baseline_metrics.json"
        if not baseline_path.exists():
            print("\n  No baseline found. Run with --save-baseline first.")
        else:
            with open(baseline_path, encoding="utf-8") as f:
                baseline = json.load(f)
            _compare(metrics, baseline)


if __name__ == "__main__":
    main()
