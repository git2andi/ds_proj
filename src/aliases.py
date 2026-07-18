"""Validated public option references.

Each option is recognized by its full name, one primary short name, optional
setup-generated aliases, and ``Option <ID>``. Generated aliases must be short,
unique across options, and composed from words in the corresponding full name.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from models import Scenario


def normalize_option_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    words = re.findall(r"[a-z0-9]+", folded.lower())
    if words and words[0] in {"the", "a", "an"}:
        words = words[1:]
    return " ".join(words)


def _tokens(text: str) -> list[str]:
    return normalize_option_text(text).split()


def _is_subsequence(parts: list[str], whole: list[str]) -> bool:
    if not parts:
        return False
    cursor = iter(whole)
    return all(any(token == candidate for candidate in cursor) for token in parts)


def validated_alias(option_name: str, proposed: str, *, max_words: int = 4) -> str:
    alias = " ".join(str(proposed or "").strip().split())
    if not alias:
        raise ValueError(f"Option {option_name!r} requires a non-empty alias")
    parts = _tokens(alias)
    if len(parts) > max_words:
        raise ValueError(f"Alias {alias!r} is too long")
    if not _is_subsequence(parts, _tokens(option_name)):
        raise ValueError(f"Alias {alias!r} is not derived from {option_name!r}")
    return alias


def unique_generated_aliases(
    option_names: dict[str, str],
    proposed: dict[str, Iterable[str]],
    *,
    max_words: int = 4,
) -> dict[str, tuple[str, ...]]:
    """Keep valid aliases that identify exactly one option."""

    candidates: dict[str, list[str]] = {option_id: [] for option_id in option_names}
    owners: Counter[str] = Counter()
    full_name_owners = {
        normalize_option_text(name): option_id for option_id, name in option_names.items()
    }
    for option_id, name in option_names.items():
        seen: set[str] = set()
        for raw in proposed.get(option_id, ()):
            try:
                alias = validated_alias(name, str(raw), max_words=max_words)
            except ValueError:
                continue
            normalized = normalize_option_text(alias)
            if not normalized or normalized == normalize_option_text(name) or normalized in seen:
                continue
            full_owner = full_name_owners.get(normalized)
            if full_owner is not None and full_owner != option_id:
                continue
            seen.add(normalized)
            candidates[option_id].append(alias)
            owners[normalized] += 1

    return {
        option_id: tuple(
            alias
            for alias in values
            if owners[normalize_option_text(alias)] == 1
        )
        for option_id, values in candidates.items()
    }


def option_aliases(scenario: "Scenario", option_id: str) -> tuple[str, ...]:
    option = scenario.option(option_id)
    aliases = [
        option.name,
        option.short_name,
        *option.aliases,
        f"Option {option.id}",
    ]
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        normalized = normalize_option_text(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(alias)
    return tuple(result)


def validate_unique_aliases(scenario: "Scenario") -> None:
    seen: dict[str, str] = {}
    for option in scenario.options:
        for alias in option_aliases(scenario, option.id):
            normalized = normalize_option_text(alias)
            if normalized == normalize_option_text(f"Option {option.id}"):
                continue
            other = seen.get(normalized)
            if other is not None and other != option.id:
                raise ValueError(
                    f"Options {other} and {option.id} share the alias {alias!r}"
                )
            seen[normalized] = option.id


def _patterns(scenario: "Scenario") -> list[tuple[re.Pattern[str], str]]:
    result: list[tuple[re.Pattern[str], str]] = []
    for option in scenario.options:
        for alias in option_aliases(scenario, option.id):
            words = re.findall(r"[A-Za-z0-9]+", alias)
            if not words:
                continue
            body = r"[\W_]+".join(re.escape(word) for word in words)
            result.append((re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE), option.id))
    return result


def resolve_option_mentions(text: str, scenario: "Scenario") -> set[str]:
    return {option_id for pattern, option_id in _patterns(scenario) if pattern.search(text)}


def resolve_visible_vote(text: str, scenario: "Scenario") -> str | None:
    lowered = text.lower()
    if not any(
        marker in lowered
        for marker in ("vote", "choose", "pick", "going with", "final choice")
    ):
        return None
    mentions = resolve_option_mentions(text, scenario)
    return next(iter(mentions)) if len(mentions) == 1 else None
