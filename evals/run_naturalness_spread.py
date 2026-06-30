"""Run the controlled 2/4/5/6/7 GPT spread for the P0 naturalness upgrade."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_loader import cfg  # noqa: E402
from main import run_dialogue  # noqa: E402


CASES = [
    (2, [1, 1], "Choose a name for our new dog"),
    (4, [2, 1, 1], "Decide on a gift for our departing colleague"),
    (5, [2, 2, 1], "Plan a volunteer day activity for the team"),
    (6, [2, 2, 1, 1], "Pick a board game for family game night"),
    (7, [3, 2, 1, 1], "Decide which city to visit for spring break"),
]


def main() -> int:
    distribution = cfg.personas.preference_distribution
    original_n = cfg.simulation.num_participants
    original_shape = distribution._raw.get("forced_shape")
    succeeded = True
    try:
        for n, shape, topic in CASES:
            cfg.simulation.num_participants = n
            distribution._raw["forced_shape"] = shape
            print(f"\n### VALIDATION n={n} shape={'-'.join(str(part) for part in shape)} ###")
            succeeded = run_dialogue(topic) and succeeded
    finally:
        cfg.simulation.num_participants = original_n
        distribution._raw["forced_shape"] = original_shape
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
