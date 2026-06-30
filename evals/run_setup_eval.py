"""Run setup-only live validation without generating any dialogue turns."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from builders import SetupBuilder  # noqa: E402
from config_loader import cfg  # noqa: E402
from llm_client import get_llm_client  # noqa: E402


CASES = [
    {"n": 2, "topic": "Choose a name for our new dog"},
    {"n": 3, "topic": "Pick a hiking trail for Saturday"},
    {"n": 4, "topic": "Decide on a gift for a departing colleague"},
    {"n": 5, "topic": "Plan a volunteer day activity for the team"},
    {"n": 6, "topic": "Pick a board game for family game night"},
    {"n": 7, "topic": "Decide which city to visit for spring break"},
    {"n": 2, "topic": "Choose a dessert for Sunday dinner"},
    {"n": 3, "topic": "Choose a research question for a term paper"},
    {"n": 4, "topic": "Decide which feature to build next sprint"},
    {"n": 5, "topic": "Choose a new coffee machine for the shared kitchen"},
    {"n": 6, "topic": "Pick a theme for a charity fundraiser gala"},
    {"n": 7, "topic": "Plan a weekend team offsite"},
    {"n": 3, "topic": "Choose a lunch spot for a client meeting", "forced_shape": [2, 1]},
    {"n": 4, "topic": "Choose a biology group-presentation topic", "forced_shape": [2, 2]},
]


class CountingClient:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.scenario_calls = 0
        self.persona_calls = 0

    def generate_json(self, prompt: str, *, profile: str = "setup") -> dict[str, Any]:
        if prompt.startswith("Create a fictional group-decision scenario"):
            self.scenario_calls += 1
        else:
            self.persona_calls += 1
        return self.client.generate_json(prompt, profile=profile)


def _shape(personas) -> list[int]:
    return sorted(Counter(persona.preferred_option for persona in personas).values(), reverse=True)


def _shape_allowed(n: int, shape: list[int], forced_shape: list[int] | None) -> bool:
    if forced_shape is not None:
        return shape == forced_shape
    weights = cfg.personas.preference_distribution.shape_weights.get(n)
    key = "-".join(str(part) for part in shape)
    return key in weights


def run_case(index: int, case: dict[str, Any], client: Any) -> dict[str, Any]:
    n = int(case["n"])
    topic = str(case["topic"])
    forced_shape = case.get("forced_shape")
    distribution = cfg.personas.preference_distribution
    previous_forced = distribution._raw.get("forced_shape")
    distribution._raw["forced_shape"] = forced_shape
    client.reset_session()
    counter = CountingClient(client)
    result: dict[str, Any] = {
        "case": index,
        "n": n,
        "topic": topic,
        "forced_shape": forced_shape,
        "provider": client.provider,
        "model": client.model_id,
    }
    try:
        builder = SetupBuilder(topic)
        builder._llm = counter
        scenario, personas = builder.build(n)
        realized_shape = _shape(personas)
        result.update({
            "success": True,
            "scenario_calls": counter.scenario_calls,
            "persona_calls": counter.persona_calls,
            "realized_shape": realized_shape,
            "shape_allowed": _shape_allowed(n, realized_shape, forced_shape),
            "tokens_in": client.session_tokens_in,
            "tokens_out": client.session_tokens_out,
            "options": [{"id": option.id, "name": option.name} for option in scenario.options],
            "personas": [
                {
                    "id": persona.id,
                    "name": persona.name,
                    "primary": persona.preferred_option,
                    "preferred_options": persona.preferred_options,
                    "background": persona.background,
                    "private_goal": persona.private_goal,
                    "rejection": persona.rejection,
                }
                for persona in personas
            ],
        })
        if not result["shape_allowed"]:
            result["success"] = False
            result["error"] = "realized preference shape was not configured"
    except Exception as exc:  # noqa: BLE001 - evaluation records the exact setup failure
        result.update({
            "success": False,
            "scenario_calls": counter.scenario_calls,
            "persona_calls": counter.persona_calls,
            "tokens_in": client.session_tokens_in,
            "tokens_out": client.session_tokens_out,
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        distribution._raw["forced_shape"] = previous_forced
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evals" / "setup_eval_results.json")
    args = parser.parse_args()
    client = get_llm_client()
    results = [run_case(index, case, client) for index, case in enumerate(CASES, start=1)]
    summary = {
        "provider": client.provider,
        "model": client.model_id,
        "total": len(results),
        "successful": sum(1 for result in results if result["success"]),
        "failed": sum(1 for result in results if not result["success"]),
        "first_attempt": sum(
            1 for result in results
            if result["success"] and result["scenario_calls"] == 1 and result["persona_calls"] == 1
        ),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    for result in results:
        status = "PASS" if result["success"] else "FAIL"
        print(
            f"{status} {result['case']:02d} n={result['n']} "
            f"shape={result.get('realized_shape', '-')} "
            f"calls={result['scenario_calls']}+{result['persona_calls']} "
            f"topic={result['topic']}"
        )
        if not result["success"]:
            print(f"  {result['error']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
