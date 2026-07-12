"""Deterministic stand-ins for the LLM client and logger.

These let tests drive DialogueRunner's generation pipeline
(_generate_and_append and friends) without network access.
"""

from __future__ import annotations

from typing import Any

from dialogue import DialogueRunner
from interpreter import InterpretationResult
from models import DialogueState, VisibleEvidence
from parsing import OptionResolver

from tests.fixtures import critical_evidence


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
    """Offline stand-in that emits deterministic critical evidence only.

    Tests needing soft semantic evidence use explicit validator payloads or
    construct ``VisibleEvidence`` directly.
    """

    def __init__(
        self, resolver: OptionResolver, participant_names: dict[str, str],
        evidences: list[VisibleEvidence] | None = None,
    ) -> None:
        self._resolver = resolver
        self._names = dict(participant_names)
        self._evidences = list(evidences or [])

    def interpret(self, *, text: str, speaker_id: str, intent=None, **_context) -> InterpretationResult:
        if self._evidences:
            evidence = self._evidences.pop(0)
            evidence.utterance = text
            return InterpretationResult(evidence=evidence)
        return InterpretationResult(evidence=critical_evidence(
            text, self._resolver, speaker_id=speaker_id,
            participant_names=self._names,
            sanctioned_switch=bool(intent and intent.allow_vote_change),
        ))


def make_runner(
    state: DialogueState, responses: list[str] | None = None,
    evidences: list[VisibleEvidence] | None = None,
) -> DialogueRunner:
    """A DialogueRunner wired to fakes, without running setup/LLM __init__."""
    runner = DialogueRunner.__new__(DialogueRunner)
    runner.topic = state.scenario.topic
    runner._llm = FakeLLM(responses)
    # Separate fake for the validator role, mirroring the real runner's
    # role-split wiring (independent instance and token counters).
    runner._validator_llm = FakeLLM()
    runner._resolver = OptionResolver(state.scenario.options)
    runner._interpreter = StubInterpreter(
        runner._resolver, {p.id: p.name for p in state.personas}, evidences
    )
    runner._intervention_count = 0
    runner._last_intervention_turn = -999
    runner.logger = NullLogger()
    return runner
