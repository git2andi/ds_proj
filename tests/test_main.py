"""CLI process-status behavior."""

from __future__ import annotations

import sys

import pytest

import main


class _FailingOrchestrator:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    def run(self):
        raise RuntimeError("setup failed")


def test_run_dialogue_reports_failure(monkeypatch):
    monkeypatch.setattr(main, "Orchestrator", _FailingOrchestrator)

    assert main.run_dialogue("Test topic") is False


def test_single_topic_failure_exits_nonzero(monkeypatch):
    monkeypatch.setattr(main, "Orchestrator", _FailingOrchestrator)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "Test topic")

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
