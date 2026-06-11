"""CLI entry point for the group-discussion simulator.

Project layout:
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

# Stream unicode (em-dashes, umlauts) to the console regardless of the OS codepage.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialogue import Orchestrator  # noqa: E402


def run_dialogue(topic: str) -> None:
    # The orchestrator streams the header and every turn to stdout as they happen,
    # so here we only print the closing summary.
    try:
        result = Orchestrator(topic).run()
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
        print(f"\n[error] Could not complete dialogue for {topic!r}: {type(exc).__name__}: {exc}\n")
        return
    tokens = result.token_summary
    print("\n" + "-" * 72)
    print(f"Outcome     : {result.outcome.status}")
    print(f"Final option: {result.outcome.final_option or '-'}")
    print(f"Reason      : {result.outcome.reason}")
    print(
        "Tokens      : "
        f"setup={tokens['setup_tokens_in']}/{tokens['setup_tokens_out']}  "
        f"dialogue={tokens['dialogue_tokens_in']}/{tokens['dialogue_tokens_out']}  "
        f"total={tokens['total_tokens_in']}/{tokens['total_tokens_out']} (in/out)"
    )
    print(f"Logs        : {result.log_paths.get('dir', '')}")
    print("-" * 72 + "\n")


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
