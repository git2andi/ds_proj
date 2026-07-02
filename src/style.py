"""Local surface-style tracking for participant turns.

These are deterministic, LLM-free helpers used by the controller to keep the
dialogue from becoming templated: too many ``Name, ...`` openings (issue #2) or
the same concession-plus-objection sentence shape repeating turn after turn
(issue #3). They operate only on visible text, so they are easy to unit-test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Leading "Name," or "Name:" address marker at the very start of a turn.
_NAME_PREFIX = re.compile(r"^\s*([A-Za-zÀ-ÖØ-öø-ÿ][\wÀ-ÖØ-öø-ÿ'-]+)\s*[,:]\s+")
# A leading article/determiner that often precedes an option name opening.
_LEAD_ARTICLE = re.compile(r"^\s*(?:the|a|an|that|this|option)\s+", re.I)
_OPTION_LETTER = re.compile(r"^\s*option\s+[a-z]\b", re.I)

_AGREE_OPENER = re.compile(
    r"^\s*(?:[A-Za-zÀ-ÿ'-]+\s*[,:]\s*)?"  # optional leading name
    r"(?:i\s+get|i\s+hear|i\s+see|that'?s\s+fair|that'?s\s+a\s+fair|fair\s+(?:point|enough)|"
    r"true|good\s+point|agreed|i\s+agree|sure|right|yeah|yes|ok|okay)\b",
    re.I,
)
_CONTRAST = re.compile(r"\b(?:but|however|though|although|still|yet|that\s+said)\b", re.I)
_WORRY = re.compile(r"\b(?:worry|worried|concern(?:ed)?|afraid|nervous|hesitant|not\s+sure)\b", re.I)
_VOTE = re.compile(r"\b(?:i\s+vote|my\s+vote|i\s+choose|i'?d\s+choose|let'?s\s+go\s+with|i\s+pick)\b", re.I)


def leading_name(text: str, names: Iterable[str]) -> str | None:
    """Return the participant name a turn opens with, if any."""
    match = _NAME_PREFIX.match(text or "")
    if not match:
        return None
    candidate = match.group(1).casefold()
    for name in names:
        if name.casefold() == candidate:
            return name
    return None


def strip_leading_name(text: str, names: Iterable[str]) -> str:
    """Remove an opening ``Name,`` address marker and recapitalize the remainder."""
    if not leading_name(text, names):
        return text
    stripped = _NAME_PREFIX.sub("", text, count=1).lstrip()
    if not stripped:
        return text
    return stripped[0].upper() + stripped[1:]


def name_prefix_fraction(texts: list[str], names: Iterable[str]) -> float:
    """Share of recent turns that open with a participant's name."""
    names = list(names)
    if not texts:
        return 0.0
    hits = sum(1 for t in texts if leading_name(t, names))
    return hits / len(texts)


# Short words that must not count as an option-opener even if they start a name.
_OPENER_STOP = frozenset({
    "the", "a", "an", "and", "for", "with", "our", "new", "one", "two",
    "option", "plan", "standard", "premium", "basic", "pro", "free", "cloud",
})


def option_opener_terms(aliases: Iterable[str]) -> set[str]:
    """Tokens/phrases that signal a turn is opening on an option name.

    Includes the full alias and its distinctive first word (>= 3 chars), so both
    'The rooftop lounge…' and single brand words like 'Trello…'/'ClickUp…' count.
    """
    terms: set[str] = set()
    for alias in aliases:
        al = str(alias).lower().strip()
        if not al:
            continue
        terms.add(al)
        first = al.split()[0]
        if len(first) >= 3 and first not in _OPENER_STOP:
            terms.add(first)
    return terms


def leading_option(text: str, aliases: Iterable[str]) -> bool:
    """True if a turn opens with an option name/alias (e.g. 'The rooftop lounge…', 'Trello…')."""
    if not text:
        return False
    body = _NAME_PREFIX.sub("", text).strip()
    body_l = body.lower()
    if _OPTION_LETTER.match(body_l):
        return True
    terms = option_opener_terms(aliases)
    if opening_signature(text) in terms:
        return True
    stripped = _LEAD_ARTICLE.sub("", body_l)
    return any(" " in t and (stripped.startswith(t) or body_l.startswith(t)) for t in terms)


def option_opening_fraction(texts: list[str], aliases: Iterable[str]) -> float:
    """Share of recent turns that open with an option name/alias."""
    aliases = list(aliases)
    if not texts:
        return 0.0
    return sum(1 for t in texts if leading_option(t, aliases)) / len(texts)


_I_OPENER = re.compile(r"^i(?:'m|'d|'ll|'ve)?\b", re.I)
_WE_OPENER = re.compile(r"^we(?:'re|'d|'ll|'ve)?\b", re.I)


def leading_first_person(text: str) -> bool:
    """True if a turn's sentence (after any name prefix) opens with 'I …'."""
    body = _NAME_PREFIX.sub("", text or "").strip()
    return bool(_I_OPENER.match(body))


def first_person_opening_fraction(texts: list[str]) -> float:
    """Share of recent turns that open with first-person 'I/I'm/I'd/I'll/I've'."""
    if not texts:
        return 0.0
    return sum(1 for t in texts if leading_first_person(t)) / len(texts)


def leading_we(text: str) -> bool:
    """True if a turn's sentence (after any name prefix) opens with 'We …'."""
    body = _NAME_PREFIX.sub("", text or "").strip()
    return bool(_WE_OPENER.match(body))


def we_opening_fraction(texts: list[str]) -> float:
    """Share of recent turns that open with 'We/We're/We'd/We'll/We've'."""
    if not texts:
        return 0.0
    return sum(1 for t in texts if leading_we(t)) / len(texts)


def opening_signature(text: str) -> str:
    """First meaningful word of a turn, ignoring a leading name/article."""
    body = _NAME_PREFIX.sub("", text or "").strip().lower()
    body = _LEAD_ARTICLE.sub("", body)
    match = re.match(r"[a-zà-ÿ']+", body)
    return match.group(0) if match else ""


def repeated_opening_token(texts: list[str], window: int) -> str | None:
    """Return the opening word shared by the last ``window`` turns, if any."""
    if window <= 1 or len(texts) < window:
        return None
    sigs = [opening_signature(t) for t in texts[-window:]]
    if sigs[0] and all(s == sigs[0] for s in sigs):
        return sigs[0]
    return None


def surface_pattern(text: str) -> str:
    """Classify a turn's rhetorical shape into a coarse, stable bucket."""
    stripped = (text or "").strip()
    if not stripped:
        return "empty"
    if _VOTE.search(stripped):
        return "vote"
    has_contrast = bool(_CONTRAST.search(stripped))
    if _AGREE_OPENER.match(stripped) and has_contrast:
        return "concede_but"
    if _WORRY.search(stripped) and has_contrast:
        return "worry_but"
    if stripped.endswith("?"):
        return "question"
    if has_contrast:
        return "tradeoff_but"
    if _AGREE_OPENER.match(stripped):
        return "agree"
    return "statement"


_TEMPLATED = {"concede_but", "worry_but", "tradeoff_but"}


def repeated_pattern(texts: list[str], window: int) -> str | None:
    """Return a templated surface pattern dominating the last ``window`` turns.

    Fires when every one of the last ``window`` turns uses a concession/worry/
    trade-off-plus-objection shape (the structures that read as samey), even if
    they mix among those three. Plain statements, votes and questions are not
    penalized for repeating.
    """
    if window <= 1 or len(texts) < window:
        return None
    recent = [surface_pattern(t) for t in texts[-window:]]
    if all(p in _TEMPLATED for p in recent):
        # Report the most frequent templated shape so the prompt can name it.
        return max(set(recent), key=recent.count)
    return None
