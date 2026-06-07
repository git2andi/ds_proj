"""CLI entry point for the discussion simulator.

Recommended project layout:
  root/
    main.py
    config.yaml
    logs/
    src/
      *.py

Usage:
  py main.py
  py main.py scenarios.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orchestrator import Orchestrator  # noqa: E402


def run_dialogue(topic: str) -> None:
    result = Orchestrator(topic).run()
    tokens = result.token_summary
    print("\n" + "=" * 70)
    print(f"Topic       : {result.scenario.topic}")
    print(f"Outcome     : {result.outcome.status}")
    print(f"Final option: {result.outcome.final_option}")
    print(f"Reason      : {result.outcome.reason}")
    print(
        "Tokens      : "
        f"setup={tokens.get('setup_tokens_in', 0)}/{tokens.get('setup_tokens_out', 0)}  "
        f"dialogue={tokens.get('dialogue_tokens_in', 0)}/{tokens.get('dialogue_tokens_out', 0)}  "
        f"total={tokens.get('total_tokens_in', 0)}/{tokens.get('total_tokens_out', 0)} (in/out)"
    )
    print(f"Logs        : {result.log_paths.get('dir', '')}")
    print("=" * 70 + "\n")
    for line in result.transcript:
        print(line)


def run_batch(path: str) -> None:
    topics = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for topic in topics:
        run_dialogue(topic)


def main() -> None:
    if len(sys.argv) > 1:
        run_batch(sys.argv[1])
        return
    topic = input("Topic: ").strip()
    if not topic:
        raise SystemExit("No topic provided.")
    run_dialogue(topic)


if __name__ == "__main__":
    main()
