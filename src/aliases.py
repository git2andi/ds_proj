"""One option-alias contract shared by setup, prompts, validation, and parsing."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from config_loader import cfg


_STOPWORDS = frozenset({"and", "or", "of", "to", "a", "an", "in", "on", "at", "for", "by", "with", "the"})
_GENERIC = frozenset({"option", "choice", "plan", "method", "activity", "approach", "idea"})


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text, re.UNICODE)


def _inflection_match(word: str, name_words: set[str]) -> bool:
    """Exact name word, or the same word under trivial singular/plural
    inflection ("Board Games" naturally refers to "Board Game Night"). The
    alias stays the name's own vocabulary — never clipped or invented."""
    if word in name_words:
        return True
    shorter = {word[:-2]} if word.endswith("es") else set()
    if word.endswith("s"):
        shorter.add(word[:-1])
    if any(c and c in name_words for c in shorter):
        return True
    return word + "s" in name_words or word + "es" in name_words


def validated_short_alias(option_name: str, proposed: str) -> str:
    """Return a recognizable proposed alias, or empty when it is unsafe to expose."""
    alias = " ".join(proposed.strip().strip('"\'').split())
    words = _words(alias)
    if not words:
        return ""
    if len(alias) < int(cfg.scenario.short_alias_min_chars):
        return ""
    if len(words) > int(cfg.scenario.short_alias_max_words):
        return ""
    lowered = [word.lower() for word in words]
    if lowered[-1] in _STOPWORDS or all(word in _GENERIC for word in lowered):
        return ""
    name_words = {word.lower() for word in _words(option_name)}
    if not all(_inflection_match(word, name_words) for word in lowered):
        return ""
    return alias


def short_alias_map(options: Iterable[Any]) -> dict[str, str]:
    """Return unique display aliases for a complete option set.

    Setup guarantees every option carries a valid, unique short_name (invalid
    generated ones reject the scenario attempt; invalid manual ones are a
    config error), so no clipped fallback aliases are ever derived here. The
    full-name fallback only protects against objects built outside setup.
    """
    option_list = list(options)
    aliases = {
        option.id: (getattr(option, "short_name", "") or option.name).strip()
        for option in option_list
    }
    owners: dict[str, list[str]] = {}
    for option_id, alias in aliases.items():
        owners.setdefault(alias.casefold(), []).append(option_id)
    for ids in owners.values():
        if len(ids) > 1:
            for option in option_list:
                if option.id in ids:
                    aliases[option.id] = option.name.strip()
    return aliases
