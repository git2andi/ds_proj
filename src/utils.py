"""Deterministic helper functions.

No LLM-facing prompt prose belongs here.  This module handles normalization,
option aliasing, token overlap, and weighted sampling.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from typing import Optional, TypeVar

from schemas import OptionCard

T = TypeVar("T")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_speaker_prefix(text: str, speaker_name: str) -> str:
    pattern = rf"^\s*{re.escape(speaker_name)}\s*:\s*"
    return re.sub(pattern, "", text, flags=re.I).strip()


def tokenize(text: str, min_len: int = 3) -> list[str]:
    return [
        re.sub(r"[^\wäöüÄÖÜß'-]", "", token).lower()
        for token in text.split()
        if len(re.sub(r"[^\wäöüÄÖÜß'-]", "", token)) >= min_len
    ]


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
    if total <= 0:
        return random.choice(list(items))
    threshold = random.random() * total
    acc = 0.0
    for item, weight in zip(items, clean):
        acc += weight
        if acc >= threshold:
            return item
    return items[-1]


def sample_length_hint(probabilities: dict[str, float]) -> str:
    keys = ["short", "medium", "long"]
    return weighted_choice(keys, [float(probabilities.get(k, 0.0)) for k in keys])


_WEAK_ALIAS_WORDS = {
    "the", "and", "or", "with", "without", "for", "from", "into", "onto",
    "option", "place", "choice", "topic", "plan", "trip", "flight", "hotel",
    "restaurant", "budget", "direct", "short", "long", "fast", "slow",
    "central", "local", "standard", "premium", "basic", "comfort", "convenient",
}


def option_aliases(option: OptionCard) -> set[str]:
    aliases = {f"option {option.id.lower()}", option.name.lower()}
    clean = re.sub(r"[^\wäöüÄÖÜß\s'-]", " ", option.name.lower())
    words = [w for w in clean.split() if len(w) > 1]
    # Keep the full name and meaningful multi-token chunks. Avoid weak single
    # words such as "with" or "budget" that caused false Option C references in
    # phrases like "okay with Option A".
    if len(words) >= 2:
        aliases.add(" ".join(words[:2]))
        aliases.add(" ".join(words[-2:]))
    for word in words:
        if len(word) >= 4 and word not in _WEAK_ALIAS_WORDS:
            aliases.add(word)
        elif word.isupper() and len(word) >= 2:
            aliases.add(word.lower())
    return {a.strip() for a in aliases if a.strip()}


class OptionResolver:
    def __init__(self, options: Iterable[OptionCard]) -> None:
        self.options = list(options)
        self.valid_ids = {o.id for o in self.options}
        self._alias_to_id: dict[str, str] = {}
        for option in self.options:
            for alias in option_aliases(option):
                self._alias_to_id.setdefault(alias, option.id)

    def ids_in_text(self, text: str) -> list[str]:
        lower = text.lower()
        found: list[str] = []
        for option_id in sorted(self.valid_ids):
            if re.search(rf"\boption\s+{re.escape(option_id)}\b", lower, flags=re.I):
                found.append(option_id)
        for alias, option_id in self._alias_to_id.items():
            if option_id in found:
                continue
            if len(alias) == 1:
                continue
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lower):
                found.append(option_id)
        return found

    def first_id_in_text(self, text: str) -> Optional[str]:
        ids = self.ids_in_text(text)
        return ids[0] if ids else None

    def invalid_option_refs(self, text: str) -> list[str]:
        refs = re.findall(r"\bOption\s+([A-Z])\b", text, flags=re.I)
        return sorted({ref.upper() for ref in refs if ref.upper() not in self.valid_ids})

    def option_text(self, option_id: str) -> str:
        for option in self.options:
            if option.id == option_id:
                return option.source_text()
        return ""


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"(?<!\w)(?:€\s*)?\d+(?:[.,]\d+)?\s*(?:%|eur|€|min|h|hours?|days?|km|m²|m2)?", text, flags=re.I)
