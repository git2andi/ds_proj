"""Deterministic utilities: normalization, sampling, token overlap, JSON helpers."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalise_ws(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.!?,;:?!])", r"\1", text)
    return text


def normalise_lines(text: str) -> str:
    """Collapse spaces within each line but keep line breaks (for multi-line
    moderator messages like the option board)."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def strip_speaker_prefix(text: str, speaker_name: str) -> str:
    return re.sub(rf"^\s*{re.escape(speaker_name)}\s*:\s*", "", text, flags=re.I).strip()


def tokenize(text: str, min_len: int = 3) -> list[str]:
    out: list[str] = []
    for token in text.split():
        clean = re.sub(r"[^\wäöüÄÖÜß'-]", "", token).lower()
        if len(clean) >= min_len:
            out.append(clean)
    return out


def jaccard_text(a: str, b: str, min_len: int = 3) -> float:
    aa = set(tokenize(a, min_len=min_len))
    bb = set(tokenize(b, min_len=min_len))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def weighted_choice(items: Sequence[T], weights: Sequence[float]) -> T:
    if not items:
        raise ValueError("weighted_choice requires at least one item")
    clean = [max(0.0, float(w)) for w in weights]
    total = sum(clean)
    if total <= 0.0:
        return random.choice(list(items))
    threshold = random.random() * total
    acc = 0.0
    for item, weight in zip(items, clean):
        acc += weight
        if acc >= threshold:
            return item
    return items[-1]


def sample_range(rng: Sequence[float]) -> float:
    lo, hi = float(rng[0]), float(rng[1])
    return random.uniform(lo, hi)


def sample_int_range(rng: Sequence[int]) -> int:
    lo, hi = int(rng[0]), int(rng[1])
    return random.randint(lo, hi)


def extract_json_object(text: str) -> dict[str, Any]:
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


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"(?:[$€£]\s*)?\d+(?:[.,:]\d+)?(?:\s*(?:kg|km|h|hr|hrs|hours?|min|minutes?|%|/\s*5))?", text, flags=re.I)


_DANGLING_TRAIL = {
    "to", "of", "for", "and", "but", "or", "with", "the", "a", "an", "than",
    "because", "so", "that", "is", "are", "in", "on", "at", "as", "if", "from",
}


def compact_words(text: str, max_words: int) -> str:
    words = normalise_ws(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    trimmed = words[:max_words]
    while trimmed and trimmed[-1].lower().rstrip(".,;:") in _DANGLING_TRAIL:
        trimmed.pop()
    if not trimmed:
        trimmed = words[:max_words]
    return " ".join(trimmed).rstrip(" ,;:") + "."
