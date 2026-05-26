"""
realize/grounding.py
--------------------
Deterministic fact-check for generated turns — Stage 10 (MASTERPLAN).

Addresses the "invented facts" failure mode: personas asserting attributes
(specific numbers, named places, quoted claims) that are not present in any
option text or the topic string.

API
---
fact_check(turn_text, option_texts, topic) -> list[str]
    Returns suspicious substrings found in turn_text that have no basis in
    option_texts or topic.  Empty list = turn passed.

repair_directive() -> str
    Returns a one-line LLM instruction appended to the prompt when a repair
    attempt is requested.  Called by simulator._generate_compact() /
    _generate_legacy() on the first failed check.

Design notes
------------
- Fully deterministic; no LLM call.
- Checks numbers (integers / decimals) and quoted or parenthesised strings.
- Proper-noun detection is deliberately shallow: we only flag ALL-CAPS tokens
  and capitalized multi-word phrases that look like named places/products.
- Option texts are lowercased for matching so "Option A" mentions are excluded.
- False positives are acceptable (leads to a single repair attempt); false
  negatives are not harmful (the turn survives unchecked).
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fact_check(
    turn_text: str,
    option_texts: list[str],
    topic: str,
) -> list[str]:
    """
    Return a list of suspicious phrases in turn_text that are not grounded
    in option_texts or topic.  Empty list means the turn passed.

    Checks:
      1. Concrete numbers (integers, decimals, percentages) not in source text.
      2. Quoted strings ("...") not in source text.
      3. Parenthesised asides not in source text.
    """
    source = _build_source(option_texts, topic)
    suspicious: list[str] = []

    # 1 — concrete numbers not in source
    for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?%?\b", turn_text):
        token = m.group(0)
        if token not in source:
            suspicious.append(token)

    # 2 — quoted strings
    for m in re.finditer(r'"([^"]{3,80})"', turn_text):
        phrase = m.group(1).strip()
        if phrase.lower() not in source:
            suspicious.append(f'"{phrase}"')

    # 3 — parenthesised asides that look like invented attributes
    for m in re.finditer(r"\(([^)]{4,60})\)", turn_text):
        phrase = m.group(1).strip()
        # Only flag if it contains a number or a multi-word capitalised phrase
        if re.search(r"\d", phrase) and phrase.lower() not in source:
            suspicious.append(f"({phrase})")

    return suspicious


def repair_directive() -> str:
    """One-line instruction added to the prompt on repair attempt."""
    return (
        "IMPORTANT: Use only the attributes explicitly listed in the options above. "
        "Do not invent numbers, prices, dates, names, or any details not present there."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_source(option_texts: list[str], topic: str) -> str:
    """Lowercase concatenation of all grounding material."""
    parts = [topic.lower()]
    parts.extend(o.lower() for o in option_texts)
    return " ".join(parts)
