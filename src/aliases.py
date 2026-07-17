"""One canonical option-alias contract shared by setup, prompts, and validation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from models import Scenario


_STOPWORDS = frozenset(
    {"and", "or", "of", "to", "a", "an", "in", "on", "at", "for", "by", "with", "the"}
)
_GENERIC = frozenset({"option", "choice", "plan", "method", "activity", "approach", "idea", "budget"})
_ARTICLES = frozenset({"the", "a", "an"})


def normalize_option_text(text: str) -> str:
    """Return accent-insensitive, punctuation-normalized option text.

    NFKD decomposition plus combining-mark removal makes ``café`` and ``Cafe``
    equivalent. Punctuation (including hyphens and apostrophes) becomes spaces
    so natural variants such as ``one-stop`` and ``one stop`` share one form.
    """
    decomposed = unicodedata.normalize("NFKD", str(text))
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    folded = unaccented.casefold()
    folded = re.sub(r"[^\w]+", " ", folded, flags=re.UNICODE)
    return " ".join(folded.split())


def _without_leading_article(text: str) -> str:
    words = text.split()
    if words and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text, re.UNICODE)


def _inflection_match(word: str, name_words: set[str]) -> bool:
    """Exact name word, or the same word under trivial singular/plural inflection."""
    if word in name_words:
        return True
    shorter = {word[:-2]} if word.endswith("es") else set()
    if word.endswith("s"):
        shorter.add(word[:-1])
    if any(candidate and candidate in name_words for candidate in shorter):
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


def _base_aliases(option: Any) -> set[str]:
    short = normalize_option_text(getattr(option, "short_name", ""))
    aliases = {
        normalize_option_text(f"Option {option.id}"),
        normalize_option_text(option.name),
    }
    if short and short not in _GENERIC:
        aliases.add(short)
    manual = getattr(option, "aliases", ()) or ()
    if isinstance(manual, str):
        manual = (manual,)
    aliases.update(normalize_option_text(alias) for alias in manual)
    aliases = {_without_leading_article(alias) for alias in aliases if alias}
    if len(str(option.id)) > 1:
        aliases.add(normalize_option_text(str(option.id)))
    return {alias for alias in aliases if alias}


def _natural_alias_candidates(option: Any) -> set[str]:
    """Return conservative natural aliases before scenario-level uniqueness checks."""
    name = normalize_option_text(option.name)
    short = normalize_option_text(getattr(option, "short_name", ""))
    words = [word for word in name.split() if word not in _STOPWORDS]
    candidates: set[str] = set()
    if words:
        head = words[-1]
        if head not in _GENERIC:
            candidates.add(head)
            for modifier in words[:-1]:
                if modifier not in _GENERIC:
                    candidates.add(f"{modifier} {head}")
    if short and not all(word in _GENERIC for word in short.split()):
        candidates.add(f"{short} option")
        candidates.add(f"{short} choice")
    return candidates


def option_aliases(scenario: "Scenario", option_id: str) -> tuple[str, ...]:
    """Return all safe normalized aliases for one option in this scenario.

    Natural head aliases are included only when they identify exactly one option
    in the current scenario. This allows ``the library`` and ``the direct
    flight`` while rejecting bare ``the flight`` when multiple flights exist.
    """
    base_by_id = {option.id: _base_aliases(option) for option in scenario.options}
    natural_by_id = {option.id: _natural_alias_candidates(option) for option in scenario.options}
    natural_owners = Counter(
        alias for aliases in natural_by_id.values() for alias in aliases if alias
    )
    aliases = set(base_by_id[option_id])
    aliases.update(
        alias for alias in natural_by_id[option_id]
        if natural_owners[alias] == 1
    )
    return tuple(sorted(aliases, key=lambda value: (-len(value.split()), -len(value), value)))


def _normalized_alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?:^|\s){re.escape(alias)}(?:$|\s)")


def option_mention_spans(text: str, scenario: "Scenario") -> list[tuple[int, int, str]]:
    """Return normalized-text spans for all safely resolved option mentions."""
    normalized = normalize_option_text(text)
    mentions: list[tuple[int, int, str]] = []
    for option in scenario.options:
        for alias in option_aliases(scenario, option.id):
            for match in _normalized_alias_pattern(alias).finditer(normalized):
                start = match.start() + (1 if match.group(0).startswith(" ") else 0)
                end = start + len(alias)
                mentions.append((start, end, option.id))

    # Compact one-character labels remain accepted only in explicit list or
    # movement/preference contexts, never as the English article "a".
    raw = str(text)
    for option in scenario.options:
        option_id = str(option.id)
        if len(option_id) != 1:
            continue
        contextual_patterns = (
            rf"\boptions?\s+[^.!?;:]{{0,70}}(?<!\w){re.escape(option_id)}(?!\w)",
            rf"\b(?:prefer(?:red|s|ring)?|lean(?:ed|ing)?\s+(?:toward|towards)|from|to|choose|choosing|choice\s+was|moving\s+to|switch(?:ed|ing)?\s+to|between|over|vote(?:d|s|ing)?(?:\s+for)?)\s+(?:option\s+)?{re.escape(option_id)}\b",
        )
        if any(re.search(pattern, raw, re.I) for pattern in contextual_patterns):
            mentions.append((-1, -1, option_id))

    # Keep one entry per concrete span/option. Alias overlaps for the same option
    # are harmless but unnecessary for callers.
    return sorted(set(mentions), key=lambda item: (item[0], item[1], item[2]))


def resolve_option_mentions(text: str, scenario: "Scenario") -> set[str]:
    """Resolve every unambiguous public option reference in ``text``."""
    return {option_id for _start, _end, option_id in option_mention_spans(text, scenario)}


def _resolve_vote_fragment(fragment: str, scenario: "Scenario") -> set[str]:
    """Resolve options inside an explicit vote-target fragment.

    Bare one-character IDs are safe here because the caller has already found a
    vote/choice commitment marker.  Outside this narrow context they remain
    protected from collisions with ordinary articles such as ``a``.
    """
    resolved = resolve_option_mentions(fragment, scenario)
    normalized = normalize_option_text(fragment)
    for option in scenario.options:
        option_id = normalize_option_text(str(option.id))
        if option_id and re.search(rf"(?<!\w){re.escape(option_id)}(?!\w)", normalized):
            resolved.add(option.id)
    return resolved


def resolve_visible_vote(text: str, scenario: "Scenario") -> str | None:
    """Return the one option explicitly committed to in a clear vote.

    Only a vote/choose/select commitment clause contributes a target.  Reasons
    may mention other options without becoming additional votes.  Old-to-new
    constructions resolve to the destination.  ``None`` represents either no
    visible commitment or an ambiguous commitment.
    """
    raw = str(text)
    normalized = normalize_option_text(raw)

    # Reject a visibly uncertain vote choice before extracting one side of it.
    if re.search(
        r"\b(?:vot(?:e|ed|es|ing)|choos(?:e|es|ing)|select(?:ed|s|ing))\b"
        r"[^.!?;]{0,90}\b(?:or|between)\b",
        normalized,
    ):
        visible = _resolve_vote_fragment(normalized, scenario)
        if len(visible) > 1:
            return None

    targets: set[str] = set()

    # Explicit old-to-new bridges. The source is context; only the destination
    # is the current vote target.
    bridge = re.compile(
        r"\b(?:switch(?:ing|ed)?|shift(?:ing|ed)?|chang(?:e|ed|ing)|mov(?:e|ed|ing))"
        r"(?:\s+(?:my\s+)?(?:vote|choice|preference))?\s+from\s+(.{1,80}?)\s+to\s+(.{1,80})",
        re.I,
    )
    for match in bridge.finditer(normalized):
        destination = re.split(
            r"\b(?:because|since|as|given|while|whereas|but|although|though)\b",
            match.group(2), maxsplit=1,
        )[0]
        resolved = _resolve_vote_fragment(destination, scenario)
        if len(resolved) != 1:
            return None
        targets.update(resolved)

    # Prefix commitments. Process sentence-by-sentence so a later reason or
    # separate sentence cannot become a second target.
    prefix = re.compile(
        r"\b(?:"
        r"vot(?:e|ed|es|ing)(?:\s+for)?|"
        r"choos(?:e|es|ing)|select(?:ed|s|ing)|"
        r"go(?:ing)?\s+with|"
        r"mov(?:e|ed|ing)\s+to|"
        r"switch(?:ed|ing)?\s+to|"
        r"stick(?:ing)?\s+with|"
        r"settle(?:d|ing)?\s+on|"
        r"my\s+vote(?:\s+(?:is|goes\s+to))?(?:\s+now)?(?:\s+for)?|"
        r"my\s+choice(?:\s+is)?(?:\s+now)?"
        r")\b",
        re.I,
    )
    for sentence in re.split(r"[.!?;]+", raw):
        clause = normalize_option_text(sentence)
        for match in prefix.finditer(clause):
            fragment = clause[match.end():]
            # In ``switching my vote from B to A`` the inner ``my vote`` is
            # not a second commitment clause; the bridge above owns it.
            if re.match(r"\s*from\b", fragment):
                continue
            fragment = re.split(
                r"\b(?:because|since|as|given|which|who|while|whereas|but|although|though|"
                r"instead\s+of|over|rather\s+than|not|"
                r"(?:after\s+)?(?:switch(?:ing|ed)?|shift(?:ing|ed)?|chang(?:e|ed|ing)|mov(?:e|ed|ing)))\b",
                fragment, maxsplit=1,
            )[0]
            resolved = _resolve_vote_fragment(fragment, scenario)
            if len(resolved) != 1:
                if resolved:
                    return None
                continue
            targets.update(resolved)


    # Natural group-chat form: ``For me, it is A``.
    for_me = re.compile(r"\bfor\s+me\s*,?\s*(?:it(?:'s|\s+is)|that(?:'s|\s+is))\s+", re.I)
    for sentence in re.split(r"[.!?;]+", raw):
        clause = normalize_option_text(sentence)
        for match in for_me.finditer(clause):
            fragment = re.split(
                r"\b(?:because|since|as|but|although|though)\b",
                clause[match.end():],
                maxsplit=1,
            )[0]
            resolved = _resolve_vote_fragment(fragment, scenario)
            if len(resolved) != 1:
                if resolved:
                    return None
                continue
            targets.update(resolved)

    # Suffix form: ``A gets my vote``.
    suffix = re.compile(r"\b(?:gets?|has|is)\s+my\s+vote\b", re.I)
    for sentence in re.split(r"[.!?;]+", raw):
        clause = normalize_option_text(sentence)
        for match in suffix.finditer(clause):
            fragment = clause[max(0, match.start() - 80):match.start()]
            fragment = re.split(r"\b(?:but|although|though|whereas)\b", fragment)[-1]
            resolved = _resolve_vote_fragment(fragment, scenario)
            if len(resolved) != 1:
                if resolved:
                    return None
                continue
            targets.update(resolved)

    return next(iter(targets)) if len(targets) == 1 else None
