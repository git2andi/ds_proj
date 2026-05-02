"""
modules/llm_client.py
---------------------
Thin provider abstraction over Gemini, Groq, and a local Ollama endpoint.
All provider-specific details (URLs, model IDs, rate limits) come from config.
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
                raise EnvironmentError("Missing GOOGLE_API_KEY in environment.")
            return genai.Client(api_key=api_key)

        if self.provider == "groq":
            import os
            from openai import OpenAI  # type: ignore

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GROQ_API_KEY in environment.")
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
            url = cfg.llm.endpoints.uni
            response = requests.post(url, data=json.dumps(payload), timeout=120)
            response.raise_for_status()
            result = response.json()
            if "response" not in result:
                raise ValueError(f"Unexpected uni API response shape: {result}")
            return result["response"].strip()

        raise ValueError(f"Unsupported LLM provider: '{self.provider}'")

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """Send a prompt and parse the response as JSON.

        Applies a repair pass before giving up, which fixes the most common
        model mistake: using ] instead of } to close an object block.
        """
        text = self.generate(prompt)

        # Strip markdown code fences if present.
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)

        # Extract the outermost JSON object.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = _repair_json(candidate)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Model did not return valid JSON:\n{text}")



# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str:
    """
    Best-effort repair of common LLM JSON mistakes:

    1. Wrong closers used instead of `}` inside an object context.
       The model occasionally closes an object block with ] or ) rather than }.
       We track brace/bracket/paren depth and swap ] or ) → } when we're
       inside an object (last opener was {).

    2. Trailing commas before } or ] (strict JSON forbids them).
    """
    # Pass 1: fix mismatched ] or ) used to close an object.
    result: list[str] = []
    depth_stack: list[str] = []  # '{', '[', or '('
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
                depth_stack.append(ch)
            elif ch == "}":
                if depth_stack and depth_stack[-1] == "{":
                    depth_stack.pop()
            elif ch in ("]", ")"):
                if depth_stack and depth_stack[-1] == "{":
                    # Model closed an object with ] or ) — swap to }
                    ch = "}"
                    depth_stack.pop()
                elif depth_stack and depth_stack[-1] in ("[", "("):
                    depth_stack.pop()

        result.append(ch)

    repaired = "".join(result)

    # Pass 2: remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    return repaired


# ---------------------------------------------------------------------------
# Module-level singleton — import and call get_llm_client() anywhere.
# ---------------------------------------------------------------------------

_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _instance
    if _instance is None:
        _instance = LLMClient()
    return _instance