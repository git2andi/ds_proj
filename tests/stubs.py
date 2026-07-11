"""Deterministic stand-ins for the LLM client and logger.

These let tests drive DialogueRunner's generation pipeline
(_generate_and_append and friends) without network access.
"""

from __future__ import annotations

from typing import Any

from dialogue import DialogueRunner
from models import DialogueState
from parsing import OptionResolver


class FakeLLM:
    """Returns scripted responses in order; falls back to a neutral line."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.prompts: list[str] = []
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0

    def reset_session(self) -> None:
        self.session_tokens_in = 0
        self.session_tokens_out = 0

    def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else "Fair point, that seems workable."
        self.last_tokens_in = max(1, len(prompt.split()))
        self.last_tokens_out = max(1, len(text.split()))
        self.session_tokens_in += self.last_tokens_in
        self.session_tokens_out += self.last_tokens_out
        return text

    def generate_json(self, prompt: str, *, profile: str = "setup") -> dict[str, Any]:
        self.prompts.append(prompt)
        self.last_tokens_in = max(1, len(prompt.split()))
        self.last_tokens_out = 1
        return {"unsupported": False}


class NullLogger:
    def write_prompt(self, prompt: str, kind: str) -> str:
        return ""


def make_runner(state: DialogueState, responses: list[str] | None = None) -> DialogueRunner:
    """A DialogueRunner wired to fakes, without running setup/LLM __init__."""
    runner = DialogueRunner.__new__(DialogueRunner)
    runner.topic = state.scenario.topic
    runner._llm = FakeLLM(responses)
    runner._resolver = OptionResolver(state.scenario.options)
    runner._intervention_count = 0
    runner._last_intervention_turn = -999
    runner.logger = NullLogger()
    return runner
