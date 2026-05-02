import json
import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()

LLM = "uni"  # "groq", "gemini", or "uni"

UNI_API_URL = "http://ursa.ds.uni-bamberg.de:11434/api/generate"
UNI_DEFAULT_MODEL = "llama3.3:latest"


class LLMClient:
    def __init__(self, provider: str = LLM):
        self.provider = provider.lower()
        self.client: Any = self._setup_client()
        self.model_id: str = self._select_model_id()

    def _setup_client(self) -> Any:
        if self.provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing GOOGLE_API_KEY in environment.")
            return genai.Client(api_key=api_key)

        if self.provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Missing GROQ_API_KEY in environment.")
            return OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )

        if self.provider == "uni":
            # No client object needed — we use requests directly in generate().
            return None

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _select_model_id(self) -> str:
        if self.provider == "gemini":
            return "gemini-2.0-flash-lite"
        if self.provider == "groq":
            return "llama-3.3-70b-versatile"
        if self.provider == "uni":
            # Override via UNI_MODEL env var, e.g. "deepseek-r1:70b"
            return os.environ.get("UNI_MODEL", UNI_DEFAULT_MODEL)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(self, prompt: str) -> str:
        """Call the LLM and return the response text."""
        if self.provider == "gemini":
            # Gemini free tier allows ~5 RPM; sleep to stay within limit.
            time.sleep(12)
            response = self.client.models.generate_content(  # type: ignore[attr-defined]
                model=self.model_id,
                contents=prompt,
            )
            return (response.text or "").strip()

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            return (response.choices[0].message.content or "").strip()

        if self.provider == "uni":
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                # Hyperparameters — adjust as needed for your simulations.
                "temperature": 0.85,
                "top_k": 50,
                "top_p": 0.95,
            }
            response = requests.post(UNI_API_URL, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            if "response" not in result:
                raise ValueError(f"Unexpected uni API response: {result}")
            return result["response"].strip()

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_json(self, prompt: str) -> dict:
        """Call the LLM and parse the response as JSON."""
        text = self.generate(prompt)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract a JSON object if the model added surrounding text.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Model did not return valid JSON:\n{text}")


_llm_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance