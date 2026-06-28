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

# Read/stream unicode (em-dashes, umlauts) regardless of the OS codepage. Reading stdin as
# UTF-8 means a piped topic's leading byte-order mark decodes to a single ﻿ that
# _clean_topic strips, instead of the cp1252 mojibake "ï»¿" that used to leak into the topic.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialogue import Orchestrator  # noqa: E402


def run_dialogue(topic: str) -> bool:
    # The orchestrator streams the header and every turn to stdout as they happen,
    # so here we only print the closing summary.
    try:
        result = Orchestrator(topic).run()
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
        print(f"\n[error] Could not complete dialogue for {topic!r}: {type(exc).__name__}: {exc}\n")
        return False
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
    return True


def _clean_topic(text: str) -> str:
    # Strip a UTF-8 BOM, whether it arrived as the real char (U+FEFF) or as the cp1252
    # mojibake "ï»¿" (UTF-8 BOM bytes misread under a legacy codepage).
    return text.replace("﻿", "").replace("ï»¿", "").strip()


def run_batch(path: str) -> bool:
    topics = [
        cleaned
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if (cleaned := _clean_topic(line)) and not cleaned.startswith("#")
    ]
    succeeded = True
    for topic in topics:
        succeeded = run_dialogue(topic) and succeeded
    return succeeded


def main() -> None:
    if len(sys.argv) > 1:
        if not run_batch(sys.argv[1]):
            raise SystemExit(1)
        return
    topic = _clean_topic(input("Topic: "))
    if not topic:
        raise SystemExit("No topic provided.")
    if not run_dialogue(topic):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
