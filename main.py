from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_loader import cfg  # noqa: E402
from dialogue import DialogueRunner  # noqa: E402


def _topics_from_args(argv: list[str]) -> list[str]:
    if len(argv) >= 2:
        path = Path(argv[1])
        if path.exists():
            topics = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    topics.append(line)
            return topics
        return [" ".join(argv[1:]).strip()]
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return [line.strip() for line in piped.splitlines() if line.strip()]
    topic = input("Topic: ").strip()
    return [topic] if topic else []


def main() -> int:
    topics = _topics_from_args(sys.argv)
    if not topics:
        print("No topic provided.", file=sys.stderr)
        return 2
    for topic in topics:
        try:
            result = DialogueRunner(topic).run()
            print(f"\nOutcome: {result.outcome.status} ({result.outcome.final_option})")
            tokens = result.token_summary
            print(
                "Tokens: "
                f"setup={tokens['setup_tokens_in']}/{tokens['setup_tokens_out']} "
                f"dialogue={tokens['dialogue_tokens_in']}/{tokens['dialogue_tokens_out']} "
                f"total={tokens['total_tokens_in']}/{tokens['total_tokens_out']} (in/out)"
            )
            print(f"Logs: {result.log_paths['dir']}")
        except Exception as exc:
            print(f"ERROR for topic {topic!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
