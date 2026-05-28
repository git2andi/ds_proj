"""
utils.py
--------
Deterministic helpers: option resolution (letter + venue-name aliasing),
vote extraction, and history scanning.

The OptionResolver is the bridge between natural chat ("let's do la brisa")
and the structured state layer (Option A). It is built once per dialogue from
the option texts and threaded into StateTracker, the simulator, and the
orchestrator so every layer reads the same references.
"""

from __future__ import annotations

import re
from typing import Optional

from config_loader import cfg


# ---------------------------------------------------------------------------
# Option resolution (letter + alias)
# ---------------------------------------------------------------------------

# Generic words that must never stand alone as an alias (they collide across
# options: "Boutique Hotel", "Budget Hostel", "Eco-Friendly Hotel" all share
# "hotel"). Multi-word aliases are always kept; single tokens must pass these.
_GENERIC_TOKENS = {
    "the", "a", "an", "de", "la", "le", "el", "and", "in", "of", "go",
    "hotel", "hostel", "cafe", "café", "restaurant", "bar", "bistro", "eatery",
    "house", "haven", "nest", "spot", "place", "option", "budget", "boutique",
    "central", "grand", "green", "downtown", "city", "room", "rooms",
}

# Trailing/leading category words stripped to expose the proper name
# ("La Brisa Cafe" -> "la brisa", "Budget Hostel" handled by parenthetical).
_CATEGORY_WORDS = {
    "cafe", "café", "restaurant", "hotel", "hostel", "bar", "bistro", "eatery",
    "house", "inn", "lodge", "resort", "kitchen", "grill", "bistro",
}

_FIRST_PERSON_VOTE = re.compile(
    r"\b("
    r"i\s+(?:prefer|pick|choose|want|vote\s+for|like|say|favou?r)|"
    r"i(?:'m|\s+am)\s+(?:going\s+with|for|leaning(?:\s+toward(?:s)?)?|sold\s+on|set\s+on)|"
    r"i(?:'d|\s+would)\s+(?:go\s+with|choose|prefer|say|pick)|"
    r"let'?s\s+(?:go\s+with|do|pick|book|try)|"
    r"going\s+with|"
    r"my\s+(?:pick|choice|vote|preference)\s+(?:is|would\s+be)"
    r")\b",
    re.I,
)

_NEGATION_BEFORE = re.compile(
    r"\b(not|never|rather\s+not|anything\s+but|except|avoid|skip|hate|"
    r"dislike|reject|refuse|no\s+to|nope\s+to|don'?t\s+(?:want|like))\s*$",
    re.I,
)

# Committal cues that appear AFTER the option name ("la brisa works for me").
_VOTE_AFTER = re.compile(
    r"^\W*(?:"
    r"works?(?:\s+for\s+me|\s+best)?|sounds?\s+good|for\s+me|it\s+is|"
    r"all\s+the\s+way|i'?m\s+in|gets?\s+my\s+vote|is\s+my\s+pick|wins?"
    r")\b",
    re.I,
)

# Soft suggestions that look like a reference but are not a commitment.
_SOFT_SUGGEST = re.compile(r"\b(what\s+about|how\s+about|what\s+if|maybe|or\s+we\s+could)\b", re.I)

_AGREE_LEAD = {"yeah", "yep", "yes", "ok", "okay", "sure", "agreed", "same", "fine"}


def _option_aliases(label: str) -> list[str]:
    """Extract proper-name aliases from an option's label (text before first ':')."""
    aliases: set[str] = set()

    # Parenthetical names are the strongest aliases ("(The Grand Kaiser)").
    for paren in re.findall(r"\(([^)]+)\)", label):
        aliases.add(paren)
        aliases.add(re.sub(r"^the\s+", "", paren, flags=re.I))

    # Base = label minus parentheticals minus a trailing " in <Location>".
    base = re.sub(r"\([^)]*\)", "", label)
    base = re.sub(r"\bin\s+[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*\s*$", "", base)
    base = base.strip(" -,–")
    if base:
        aliases.add(base)
        aliases.add(re.sub(r"^the\s+", "", base, flags=re.I))
        words = base.split()
        if len(words) >= 2 and words[-1].lower().strip(".,") in _CATEGORY_WORDS:
            aliases.add(" ".join(words[:-1]))
        if len(words) >= 2 and words[0].lower() in _CATEGORY_WORDS:
            aliases.add(" ".join(words[1:]))

    cleaned: list[str] = []
    seen: set[str] = set()
    for a in aliases:
        a = re.sub(r"\s+", " ", a).strip().lower()
        if len(a) < 4 or a in seen:
            continue
        toks = a.split()
        if len(toks) == 1 and a in _GENERIC_TOKENS:
            continue
        seen.add(a)
        cleaned.append(a)
    # Longest first so the most specific alias wins on overlap.
    cleaned.sort(key=len, reverse=True)
    return cleaned


class OptionResolver:
    """Maps both 'Option X' and venue/proper-name mentions to option letters."""

    def __init__(self, options: list[str]) -> None:
        self.letters: list[str] = []
        self.aliases: dict[str, list[str]] = {}
        self._regex: dict[str, re.Pattern] = {}

        for opt in options:
            m = re.match(r"^\s*option\s+([a-d])\b", opt, re.I)
            if not m:
                continue
            letter = m.group(1).upper()
            body = opt.split("-", 1)[1] if "-" in opt else opt
            label = body.split(":", 1)[0].strip()
            als = _option_aliases(label)
            self.letters.append(letter)
            self.aliases[letter] = als
            # Trailing boundary allows a possessive ("la brisa's") but not more
            # word characters; leading boundary avoids matching mid-word.
            patterns = [rf"\boption\s+{letter.lower()}\b"]
            patterns += [rf"(?<!\w){re.escape(a)}(?!\w)" for a in als]
            self._regex[letter] = re.compile("|".join(patterns), re.I)

    def _match(self, letter: str, text: str) -> Optional[re.Match]:
        return self._regex[letter].search(text)

    def options_in(self, text: str) -> list[str]:
        """All options referenced in the text, by letter or alias."""
        hits = [(m.start(), l) for l in self.letters if (m := self._match(l, text))]
        hits.sort(key=lambda x: x[0])
        return [l for _, l in hits]

    def vote_in(self, text: str) -> Optional[str]:
        """The single option this turn commits to, or None.

        A commitment needs an explicit signal — a first-person preference frame
        ("I'd go with X"), a committal cue after the name ("X works for me"), or
        an agreeing lead ("yeah, X") — not a bare mention or a soft suggestion
        ("what about X?"), which are discussion, not votes.
        """
        t = text.strip().lower()
        if not t:
            return None
        words = t.split()
        lead = words[0].strip(",.!'") if words else ""

        refs = [(m.start(), m.end(), l) for l in self.letters if (m := self._match(l, t))]
        if not refs:
            return None
        refs.sort(key=lambda x: x[0])

        pref = _FIRST_PERSON_VOTE.search(t)
        soft = _SOFT_SUGGEST.search(t)

        # Pick the candidate reference.
        if pref:
            after = [r for r in refs if r[0] >= pref.start()]
            start, end, letter = (after or refs)[0]
        elif len(refs) > 1:
            return None  # comparison without a preference frame -> discussion
        else:
            start, end, letter = refs[0]
            after_cue = _VOTE_AFTER.match(t[end:].lstrip())
            agreeing = lead in _AGREE_LEAD
            if not (after_cue or agreeing):
                return None
            if soft and not after_cue:
                return None

        # Negation guards.
        if _NEGATION_BEFORE.search(t[:start]):
            return None
        tail = t[start:start + 60]
        if re.search(r"\b(isn'?t|aren'?t|won'?t|can'?t|too\s+\w+|sucks|terrible|"
                     r"awful|not\s+sure)\b", tail) and not pref:
            return None
        return letter


def options_referenced(text: str, resolver: Optional[OptionResolver]) -> list[str]:
    return resolver.options_in(text) if resolver else []


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def participant_turn_count(history: list[str]) -> int:
    return sum(
        1 for line in history
        if ":" in line
        and line.split(":", 1)[0].strip() not in cfg.EXCLUDED_SPEAKERS
    )


def last_n_turns_for(name: str, history: list[str], n: int) -> list[str]:
    turns: list[str] = []
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        if speaker.strip() == name:
            turns.append(msg.strip().lower())
            if len(turns) >= n:
                break
    return turns


def current_votes(history: list[str], resolver: Optional[OptionResolver]) -> dict[str, str]:
    """Most recent explicit option vote per participant."""
    votes: dict[str, str] = {}
    if resolver is None:
        return votes
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        speaker = speaker.strip()
        if speaker in cfg.EXCLUDED_SPEAKERS or speaker in votes:
            continue
        vote = resolver.vote_in(msg)
        if vote:
            votes[speaker] = vote
    return votes
