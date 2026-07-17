"""Small setup/client utilities used by the simplified runtime."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from typing import Any


def sample_int_range(
    values: Sequence[int],
    *,
    rng: random.Random | None = None,
) -> int:
    lo, hi = int(values[0]), int(values[1])
    return (rng or random).randint(lo, hi)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from a setup-model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start:end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("No JSON object found in model response.")
