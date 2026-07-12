"""Thin provider abstraction over OpenAI, Gemini, Groq, and an Ollama-style endpoint."""

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
    """One backend bound to an LLM role.

    Roles are independently configurable for compatibility. The live runner
    constructs only the dialogue role, which owns setup, utterance, moderator,
    and repair generation. Critical interpretation is deterministic and does
    not instantiate a validator client.
    """

    def __init__(self, role: str = "dialogue") -> None:
        self.role = str(role)
        provider = cfg.llm.get(self.role)
        if not provider:
            raise ValueError(f"No provider configured for LLM role {self.role!r} (set llm.{self.role}).")
        self.provider = str(provider).lower()
        self.model_id = getattr(cfg.llm.models, self.provider)
        self._client: Any = self._build_client()
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0

    def _build_client(self) -> Any:
        if self.provider == "gemini":
            from google import genai  # type: ignore
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GOOGLE_API_KEY.")
            return genai.Client(api_key=api_key)
        if self.provider == "groq":
            from openai import OpenAI  # type: ignore
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GROQ_API_KEY.")
            return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        if self.provider == "gpt":
            from openai import OpenAI  # type: ignore
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing OPENAI_API_KEY.")
            return OpenAI(api_key=api_key)
        if self.provider == "uni":
            return None
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def reset_session(self) -> None:
        self.session_tokens_in = 0
        self.session_tokens_out = 0

    def _record_tokens(self, prompt: str, text: str, tokens_in: int | None = None, tokens_out: int | None = None) -> None:
        # Telemetry only: when a provider omits usage metadata, estimate from word counts.
        # This is a token *count*, never a stand-in for model content.
        in_count = int(tokens_in if tokens_in is not None else max(1, len(prompt.split())))
        out_count = int(tokens_out if tokens_out is not None else max(1, len(text.split())))
        self.last_tokens_in = in_count
        self.last_tokens_out = out_count
        self.session_tokens_in += in_count
        self.session_tokens_out += out_count

    def _sampling(self, profile: str) -> dict[str, Any]:
        section = getattr(cfg.llm.sampling, profile, cfg.llm.sampling.dialogue)
        return {
            "temperature": float(section.temperature),
            "top_k": int(section.top_k),
            "top_p": float(section.top_p),
            # 0 = provider default; the validator profile caps output tightly
            # (structured JSON only), independent of provider (item 9).
            "max_output_tokens": int(section.get("max_output_tokens", 0)),
        }

    def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        sampling = self._sampling(profile)

        if self.provider == "gemini":
            time.sleep(float(cfg.llm.gemini_rpm_delay_seconds))
            response = self._client.models.generate_content(model=self.model_id, contents=prompt)
            text = (response.text or "").strip()
            meta = getattr(response, "usage_metadata", None)
            self._record_tokens(
                prompt,
                text,
                getattr(meta, "prompt_token_count", None) if meta else None,
                getattr(meta, "candidates_token_count", None) if meta else None,
            )
            return text

        if self.provider in {"groq", "gpt"}:
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
            self._record_tokens(
                prompt,
                text,
                getattr(usage, "prompt_tokens", None) if usage else None,
                getattr(usage, "completion_tokens", None) if usage else None,
            )
            return text

        if self.provider == "uni":
            options = {k: v for k, v in sampling.items() if k != "max_output_tokens"}
            if sampling["max_output_tokens"]:
                options["num_predict"] = sampling["max_output_tokens"]
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "temperature": sampling["temperature"],
                "top_k": sampling["top_k"],
                "top_p": sampling["top_p"],
                "options": options,
            }
            response = requests.post(
                str(cfg.llm.endpoints.uni),
                json=payload,
                timeout=float(cfg.llm.timeouts.request_seconds),
            )
            response.raise_for_status()
            data = response.json()
            text = str(data.get("response", "")).strip()
            self._record_tokens(
                prompt,
                text,
                data.get("prompt_eval_count"),
                data.get("eval_count"),
            )
            return text

        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def generate_json(self, prompt: str, *, profile: str = "setup") -> dict[str, Any]:
        """Generate one JSON object.

        OpenAI-compatible providers use native JSON mode so structured
        validation does not spend a second API call recovering from prose or
        fenced output. Other providers keep the common text parser.
        """
        if self.provider in {"gpt", "groq"}:
            self.last_tokens_in = 0
            self.last_tokens_out = 0
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
            self._record_tokens(
                prompt,
                text,
                getattr(usage, "prompt_tokens", None) if usage else None,
                getattr(usage, "completion_tokens", None) if usage else None,
            )
            return extract_json_object(text)
        return extract_json_object(self.generate(prompt, profile=profile))


_CLIENTS: dict[str, LLMClient] = {}


def get_llm_client(role: str = "dialogue") -> LLMClient:
    """Role-aware client lookup: one cached instance (and token counter) per role."""
    client = _CLIENTS.get(role)
    if client is None:
        client = _CLIENTS[role] = LLMClient(role)
    return client
