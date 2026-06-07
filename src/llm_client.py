"""Thin provider abstraction over Gemini, Groq, and a local Ollama endpoint."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from config_loader import cfg

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self.provider: str = str(cfg.llm.provider).lower()
        self.model_id: str = getattr(cfg.llm.models, self.provider)
        self._client: Any = self._build_client()
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        # Per-profile token accumulation, e.g. {"setup": [in, out], ...}.
        self._profile_tokens: dict[str, list[int]] = {}

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
        if self.provider == "uni":
            return None
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def reset_session(self) -> None:
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        self._profile_tokens = {}

    def token_summary(self) -> dict[str, list[int]]:
        """Tokens grouped into setup vs dialogue buckets, plus total.

        ``setup`` covers scenario/persona/belief generation; ``dialogue`` covers
        utterance generation and repairs.  Values are ``[tokens_in, tokens_out]``.
        """
        setup = self._profile_tokens.get("setup", [0, 0])
        dialogue_in = sum(v[0] for k, v in self._profile_tokens.items() if k != "setup")
        dialogue_out = sum(v[1] for k, v in self._profile_tokens.items() if k != "setup")
        total_in = sum(v[0] for v in self._profile_tokens.values())
        total_out = sum(v[1] for v in self._profile_tokens.values())
        return {
            "setup": [setup[0], setup[1]],
            "dialogue": [dialogue_in, dialogue_out],
            "total": [total_in, total_out],
        }

    def _record_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.last_tokens_in = int(tokens_in or 0)
        self.last_tokens_out = int(tokens_out or 0)
        self.session_tokens_in += self.last_tokens_in
        self.session_tokens_out += self.last_tokens_out

    def _sampling(self, profile: str) -> dict[str, Any]:
        section = getattr(cfg.llm.sampling, profile, cfg.llm.sampling.dialogue)
        return {
            "temperature": float(section.temperature),
            "top_k": int(section.top_k),
            "top_p": float(section.top_p),
        }

    def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
        text = self._generate_raw(prompt, profile)
        bucket = self._profile_tokens.setdefault(profile, [0, 0])
        bucket[0] += self.last_tokens_in
        bucket[1] += self.last_tokens_out
        return text

    def _generate_raw(self, prompt: str, profile: str) -> str:
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        sampling = self._sampling(profile)

        if self.provider == "gemini":
            time.sleep(float(cfg.llm.gemini_rpm_delay))
            response = self._client.models.generate_content(model=self.model_id, contents=prompt)
            meta = getattr(response, "usage_metadata", None)
            self._record_tokens(
                getattr(meta, "prompt_token_count", 0) or 0,
                getattr(meta, "candidates_token_count", 0) or 0,
            )
            return (response.text or "").strip()

        if self.provider == "groq":
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
            )
            usage = getattr(response, "usage", None)
            self._record_tokens(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )
            return (response.choices[0].message.content or "").strip()

        if self.provider == "uni":
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                # The target endpoint is Ollama-like.  Some deployments accept
                # both root-level sampling keys and an options dict; send both.
                "temperature": sampling["temperature"],
                "top_k": sampling["top_k"],
                "top_p": sampling["top_p"],
                "options": sampling,
            }
            resp = requests.post(
                cfg.llm.endpoints.uni,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=float(cfg.llm.timeouts.request_seconds),
            )
            resp.raise_for_status()
            result = resp.json()
            if "response" not in result:
                raise ValueError(f"Unexpected uni API response: {result}")
            self._record_tokens(result.get("prompt_eval_count", 0), result.get("eval_count", 0))
            return str(result["response"]).strip()

        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_json(self, prompt: str, *, profile: str = "setup") -> dict[str, Any]:
        text = self.generate(prompt, profile=profile)
        text = _strip_markdown_fence(text)
        candidate = _extract_json_object(text)
        if candidate is None:
            raise ValueError(f"Model did not return a JSON object:\n{text}")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair_json(candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Model returned invalid JSON after repair: {exc}\n{candidate}") from exc


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _repair_json(text: str) -> str:
    # Conservative repairs only: trailing commas and smart quotes.
    repaired = text.replace("“", '"').replace("”", '"').replace("’", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _instance
    if _instance is None:
        _instance = LLMClient()
    return _instance
