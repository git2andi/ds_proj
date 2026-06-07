"""CLI entry point for the dialogue simulator.

Usage:
  python main.py
  python main.py scenarios.txt

The runnable entry point, config.yaml, and logs/ live in the project root.
All implementation modules live in src/, which is added to the import path here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from orchestrator import Orchestrator  # noqa: E402


def _format_tokens(tokens: dict) -> str:
    setup = tokens.get("setup", [0, 0])
    dialogue = tokens.get("dialogue", [0, 0])
    total = tokens.get("total", [0, 0])
    return (
        f"Tokens      : setup={setup[0]}/{setup[1]}  "
        f"dialogue={dialogue[0]}/{dialogue[1]}  "
        f"total={total[0]}/{total[1]} (in/out)"
    )


def run_dialogue(topic: str) -> None:
    result = Orchestrator(topic).run()
    print("\n" + "=" * 70)
    print(f"Topic       : {result.scenario.topic}")
    print(f"Outcome     : {result.outcome.status}")
    print(f"Final option: {result.outcome.final_option}")
    print(f"Reason      : {result.outcome.reason}")
    print(_format_tokens(result.tokens))
    print(f"Logs        : {result.log_paths.get('dir', '')}")
    print("=" * 70 + "\n")
    for line in result.transcript:
        print(line)


def run_batch(path: str) -> None:
    topics = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]
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
