"""Provider-level tests for the shared LLM client interface."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from config_loader import cfg
from llm_client import LLMClient


def test_gpt_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg.llm, "provider", "gpt")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="Missing OPENAI_API_KEY"):
        LLMClient()


def test_gpt_generation_and_token_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-openai-key"
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        @staticmethod
        def _create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
            )

    monkeypatch.setattr(cfg.llm, "provider", "gpt")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    client = LLMClient()
    result = client.generate_json("Return JSON.")

    assert result == {"ok": True}
    assert calls[0]["model"] == "gpt-4.1-mini"
    assert calls[0]["messages"] == [{"role": "user", "content": "Return JSON."}]
    assert client.last_tokens_in == 11
    assert client.last_tokens_out == 4
    assert client.session_tokens_in == 11
    assert client.session_tokens_out == 4
