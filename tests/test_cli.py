from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import main


def test_explicit_topic_is_read_from_command_line():
    assert main._explicit_topics(["main.py", "Choose", "a", "trip"]) == ["Choose a trip"]


def test_topic_file_ignores_blank_lines_and_comments(tmp_path):
    path = tmp_path / "topics.txt"
    path.write_text("# cases\nChoose A\n\nChoose B\n", encoding="utf-8")
    assert main._explicit_topics(["main.py", str(path)]) == ["Choose A", "Choose B"]


def test_main_runs_injected_dialogue_runner_without_validator_fields(monkeypatch, capsys):
    class FakeRunner:
        def __init__(self, topic: str, *, force_auto_scenario: bool = False) -> None:
            self.topic = topic
            self.force = force_auto_scenario

        def run(self):
            return SimpleNamespace(
                outcome=SimpleNamespace(status="majority", final_option="A"),
                token_summary={
                    "setup_tokens_in": 10, "setup_tokens_out": 5,
                    "dialogue_tokens_in": 20, "dialogue_tokens_out": 8,
                    "total_tokens_in": 30, "total_tokens_out": 13,
                    "llm_calls": 4,
                },
                log_paths={"dir": "logs/test"},
            )

    monkeypatch.setattr(main, "DialogueRunner", FakeRunner)
    monkeypatch.setattr(sys, "argv", ["main.py", "Choose a study location"])
    assert main.main() == 0
    output = capsys.readouterr().out
    assert "Outcome: majority (A)" in output
    assert "calls=4" in output
    assert "validator" not in output.casefold()
