"""Thin provider abstraction over Gemini, Groq, Ollama-style local endpoint, and mock."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv

from config_loader import cfg
from utils import extract_json_object

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self.provider = str(cfg.llm.provider).lower()
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
        if self.provider in {"uni", "mock"}:
            return None
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def reset_session(self) -> None:
        self.session_tokens_in = 0
        self.session_tokens_out = 0

    def _record_tokens(self, prompt: str, text: str, tokens_in: int | None = None, tokens_out: int | None = None) -> None:
        # Fallback estimate is intentionally simple; provider usage metadata replaces it when available.
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
        }

    def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        sampling = self._sampling(profile)

        if self.provider == "mock":
            text = _mock_setup(prompt) if profile == "setup" else _mock_dialogue(prompt)
            self._record_tokens(prompt, text)
            return text

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

        if self.provider == "groq":
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
            )
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
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "temperature": sampling["temperature"],
                "top_k": sampling["top_k"],
                "top_p": sampling["top_p"],
                "options": sampling,
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
        return extract_json_object(self.generate(prompt, profile=profile))


_CLIENT: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


# ---------------------------------------------------------------------------
# Mock provider (provider: "mock") — deterministic, offline plumbing only.
# It mirrors the prompt's requested move so the full pipeline can run without an
# LLM endpoint. It is NOT a fallback for failed real calls.
# ---------------------------------------------------------------------------

_MOCK_NAMES = ["Ava", "Liam", "Mia", "Noah", "Lena", "Jonas", "Sofia"]
_STANCE_BY_ACT = {"vote": "vote", "accept": "accept", "reject": "object", "propose_compromise": "propose"}


def _mock_setup(prompt: str) -> str:
    labels = [str(x) for x in cfg.scenario.option_labels]
    n = len(re.findall(r'"id":\s*"p\d+"', prompt)) or int(cfg.simulation.num_participants)
    names = ["Budget", "Comfort", "Balanced", "Flexible", "Premium", "Local"]
    options = []
    for idx, label in enumerate(labels):
        options.append({
            "id": label,
            "name": f"{names[idx % len(names)]} Option {label}",
            "attrs": {"cost": f"${100 + idx * 60}", "effort": ["Low", "Medium", "High", "Low"][idx % 4], "rating": f"{3 + idx % 3}/5"},
            "upside": "clear strength on its main priority",
            "tradeoff": "a real trade-off others may dislike",
            "concern": "one stable downside to weigh",
            "best_for": ["cost", "comfort", "balance", "flexibility"][idx % 4],
        })
    participants = []
    for idx in range(n):
        pref = labels[idx % len(labels)]
        accept2 = labels[(idx + 2) % len(labels)]
        participants.append({
            "id": f"p{idx + 1}",
            "name": _MOCK_NAMES[idx % len(_MOCK_NAMES)],
            "role": "group member",
            "speech_style": "plain",
            "private_goal": f"land on a choice close to Option {pref}",
            "backstory": "has made similar small decisions before",
            "main_concern": ["cost", "comfort", "balance", "flexibility"][idx % 4],
            "preferred_option": pref,
            "acceptable_options": [pref, accept2],
            "soft_rejections": [],
            "hard_rejections": [],
            "reasons": {pref: ["fits my main priority"], accept2: ["an acceptable backup"]},
            "reservation": "one trade-off still matters",
            "reconsider_if": "if the group values another priority more",
        })
    return json.dumps({"scenario": {"decision_kind": "generic_decision", "opening_question": "What matters most before we pick?", "options": options}, "participants": participants})


def _mock_dialogue(prompt: str) -> str:
    if "facilitator" in prompt.lower():
        return "Okay, where are we all at on this — anything shifting, or shall we move toward a decision?"
    act_match = re.search(r"move=([a-z_]+)", prompt)
    act = act_match.group(1) if act_match else "react"
    focus_match = re.search(r"focus=([A-D])", prompt)
    opt = focus_match.group(1) if focus_match else "-"
    stance = _STANCE_BY_ACT.get(act, "neutral")
    line = {
        "vote": f"My pick is Option {opt}, it fits what I care about most.",
        "accept": f"I still had my own lean, but Option {opt} works for me.",
        "propose_compromise": f"Could Option {opt} be the middle ground we settle on?",
    }.get(act, f"Fair point — Option {opt} has something going for it, though I'd weigh the trade-off.")
    return f"{line}\n[act={act}; opt={opt}; stance={stance}]"
