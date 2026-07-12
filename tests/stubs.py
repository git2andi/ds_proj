"""Deterministic stand-ins for the LLM client and logger.

These let tests drive DialogueRunner's generation pipeline
(_generate_and_append and friends) without network access.
"""

from __future__ import annotations

from typing import Any

from dialogue import DialogueRunner
from interpreter import InterpretationResult
from models import DialogueState
from parsing import OptionResolver

from tests.evidence_adapter import derive_evidence


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


class StubInterpreter:
    """Offline validator stand-in: derives evidence deterministically through
    the test adapter (tests/evidence_adapter.py), so scripted-response tests
    exercise the real candidate pipeline without a validator endpoint. Its
    recall equals the old conservative parser's — natural-language variants
    need a scripted validator payload (see tests/test_interpreter.py)."""

    def __init__(self, resolver: OptionResolver, participant_names: dict[str, str]) -> None:
        self._resolver = resolver
        self._names = dict(participant_names)

    def interpret(self, *, text: str, speaker_id: str, intent=None, **_context) -> InterpretationResult:
        return InterpretationResult(evidence=derive_evidence(
            text, self._resolver,
            speaker_id=speaker_id, participant_names=self._names, intent=intent,
        ))


def make_runner(state: DialogueState, responses: list[str] | None = None) -> DialogueRunner:
    """A DialogueRunner wired to fakes, without running setup/LLM __init__."""
    runner = DialogueRunner.__new__(DialogueRunner)
    runner.topic = state.scenario.topic
    runner._llm = FakeLLM(responses)
    # Separate fake for the validator role, mirroring the real runner's
    # role-split wiring (independent instance and token counters).
    runner._validator_llm = FakeLLM()
    runner._resolver = OptionResolver(state.scenario.options)
    runner._interpreter = StubInterpreter(
        runner._resolver, {p.id: p.name for p in state.personas}
    )
    runner._intervention_count = 0
    runner._last_intervention_turn = -999
    runner.logger = NullLogger()
    return runner
