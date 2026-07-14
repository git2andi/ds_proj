"""Thin provider abstraction for the single dialogue/setup LLM."""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

from config_loader import cfg
from utils import extract_json_object

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self.provider = str(cfg.llm.dialogue).lower()
        self.model_id = str(cfg.llm.models[self.provider])
        self._client: Any = self._build_client()
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        self.session_calls = 0

    def _build_client(self) -> Any:
        if self.provider == "gemini":
            from google import genai  # type: ignore
            key = os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise EnvironmentError("Missing GOOGLE_API_KEY")
            return genai.Client(api_key=key)
        if self.provider == "groq":
            from openai import OpenAI  # type: ignore
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise EnvironmentError("Missing GROQ_API_KEY")
            return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        if self.provider == "gpt":
            from openai import OpenAI  # type: ignore
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise EnvironmentError("Missing OPENAI_API_KEY")
            return OpenAI(api_key=key)
        if self.provider == "uni":
            return None
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def reset_session(self) -> None:
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        self.session_calls = 0

    def _record(self, prompt: str, text: str, tokens_in: int | None, tokens_out: int | None) -> None:
        self.last_tokens_in = int(tokens_in if tokens_in is not None else max(1, len(prompt.split())))
        self.last_tokens_out = int(tokens_out if tokens_out is not None else max(1, len(text.split())))
        self.session_tokens_in += self.last_tokens_in
        self.session_tokens_out += self.last_tokens_out
        self.session_calls += 1

    def _sampling(self, profile: str) -> dict[str, Any]:
        section = getattr(cfg.llm.sampling, profile)
        return {
            "temperature": float(section.temperature),
            "top_k": int(section.top_k),
            "top_p": float(section.top_p),
            "max_output_tokens": int(section.get("max_output_tokens", 0)),
        }

    def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
        sampling = self._sampling(profile)
        self.last_tokens_in = self.last_tokens_out = 0
        if self.provider == "gemini":
            time.sleep(float(cfg.llm.gemini_rpm_delay_seconds))
            response = self._client.models.generate_content(model=self.model_id, contents=prompt)
            text = (response.text or "").strip()
            usage = getattr(response, "usage_metadata", None)
            self._record(
                prompt, text,
                getattr(usage, "prompt_token_count", None) if usage else None,
                getattr(usage, "candidates_token_count", None) if usage else None,
            )
            return text
        if self.provider in {"gpt", "groq"}:
            request: dict[str, Any] = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
            }
            if sampling["max_output_tokens"]:
                request["max_tokens"] = sampling["max_output_tokens"]
            response = self._client.chat.completions.create(**request)
            text = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            self._record(
                prompt, text,
                getattr(usage, "prompt_tokens", None) if usage else None,
                getattr(usage, "completion_tokens", None) if usage else None,
            )
            return text
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": sampling["temperature"],
                "top_k": sampling["top_k"],
                "top_p": sampling["top_p"],
                **({"num_predict": sampling["max_output_tokens"]} if sampling["max_output_tokens"] else {}),
            },
        }
        response = requests.post(
            str(cfg.llm.endpoints.uni),
            json=payload,
            timeout=float(cfg.llm.timeouts.request_seconds),
        )
        response.raise_for_status()
        data = response.json()
        text = str(data.get("response", "")).strip()
        self._record(prompt, text, data.get("prompt_eval_count"), data.get("eval_count"))
        return text

    def generate_json(self, prompt: str, *, profile: str = "setup") -> dict[str, Any]:
        if self.provider in {"gpt", "groq"}:
            sampling = self._sampling(profile)
            request: dict[str, Any] = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "response_format": {"type": "json_object"},
            }
            if sampling["max_output_tokens"]:
                request["max_tokens"] = sampling["max_output_tokens"]
            response = self._client.chat.completions.create(**request)
            text = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            self._record(
                prompt, text,
                getattr(usage, "prompt_tokens", None) if usage else None,
                getattr(usage, "completion_tokens", None) if usage else None,
            )
            return extract_json_object(text)
        return extract_json_object(self.generate(prompt, profile=profile))


_CLIENT: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
