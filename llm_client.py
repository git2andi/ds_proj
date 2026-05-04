"""
llm_client.py
-------------
Thin provider abstraction over Gemini, Groq, and a local Ollama endpoint.
All provider details come from config. Exposes two methods: generate() and generate_json().
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from config_loader import cfg

load_dotenv()


class LLMClient:

    def __init__(self) -> None:
        self.provider: str = cfg.llm.provider.lower()
        self.model_id: str = getattr(cfg.llm.models, self.provider)
        self._client: Any = self._build_client()

    def _build_client(self) -> Any:
        if self.provider == "gemini":
            import os
            from google import genai  # type: ignore
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GOOGLE_API_KEY.")
            return genai.Client(api_key=api_key)

        if self.provider == "groq":
            import os
            from openai import OpenAI  # type: ignore
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GROQ_API_KEY.")
            return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

        if self.provider == "uni":
            return None  # requests-based; no client object needed

        raise ValueError(f"Unsupported LLM provider: '{self.provider}'")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        if self.provider == "gemini":
            time.sleep(cfg.llm.gemini_rpm_delay)
            response = self._client.models.generate_content(
                model=self.model_id, contents=prompt
            )
            return (response.text or "").strip()

        if self.provider == "groq":
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            return (response.choices[0].message.content or "").strip()

        if self.provider == "uni":
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "temperature": cfg.llm.sampling.temperature,
                "top_k": cfg.llm.sampling.top_k,
                "top_p": cfg.llm.sampling.top_p,
            }
            resp = requests.post(
                cfg.llm.endpoints.uni,
                data=json.dumps(payload),
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            if "response" not in result:
                raise ValueError(f"Unexpected uni API response: {result}")
            return result["response"].strip()

        raise ValueError(f"Unsupported provider: '{self.provider}'")

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """
        Send a prompt and parse the response as JSON.
        Strips markdown fences and applies a best-effort repair pass before failing.
        """
        text = self.generate(prompt)

        # Strip markdown code fences.
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)

        # Extract the outermost JSON object.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start: end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(_repair_json(candidate))
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Model did not return valid JSON:\n{text}")


# ---------------------------------------------------------------------------
# JSON repair — fixes the most common LLM mistakes
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str:
    """
    Two-pass repair:
    1. Swap ] or ) that close an object context → }
    2. Remove trailing commas before } or ]
    """
    result: list[str] = []
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string

        if not in_string:
            if ch in ("{", "[", "("):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch in ("]", ")"):
                if stack and stack[-1] == "{":
                    ch = "}"
                    stack.pop()
                elif stack and stack[-1] in ("[", "("):
                    stack.pop()

        result.append(ch)

    repaired = "".join(result)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _instance
    if _instance is None:
        _instance = LLMClient()
    return _instance
