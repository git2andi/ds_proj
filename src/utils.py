"""Deterministic utilities: normalization, sampling, token overlap, JSON helpers."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")


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
    return re.sub(rf"^\s*(?:{re.escape(speaker_name)}\s*:\s*)+", "", text, flags=re.I).strip()


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


def sample_int_range(rng: Sequence[int]) -> int:
    lo, hi = int(rng[0]), int(rng[1])
    return random.randint(lo, hi)


def preset_dominance_weight(
    base: float,
    is_dominant: bool,
    turn_count: int,
    total_turns: int,
    n: int,
    preset: dict,
    quiet_boost: float,
) -> float:
    """Corpus-preset speaker weighting.

    Instead of strict turn-count equalization, keep the designated dominant
    speaker near the preset's expected top share (never above the dominance
    band) and rebalance the others only once their turn share drifts past the
    imbalance tolerance around a fair 1/n split.
    """
    fair = 1.0 / max(1, n)
    share = (turn_count / total_turns) if total_turns > 0 else fair
    tol = float(preset.get("imbalance_tolerance", 0.15))
    if is_dominant:
        dom_lo, dom_hi = preset.get("dominance_range", (fair, 1.0))
        target = min(float(preset.get("top_speaker_share", fair)), float(dom_hi))
        # Structural turns (opening/vote rounds) are evenly distributed, so the
        # free discussion turns must overshoot to reach the corpus-level share.
        if share < float(dom_lo):
            return base * (2.0 + 4.0 * (target - share))
        if share < target:
            return base * (1.0 + 3.0 * (target - share))
        if share > float(dom_hi):
            return base * 0.35
        return base
    if share < fair - tol:
        return base + quiet_boost
    if share > fair + tol:
        return base * 0.5
    return base


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


_DANGLING_TRAIL = {
    "to", "of", "for", "and", "but", "or", "with", "the", "a", "an", "than",
    "because", "so", "that", "is", "are", "in", "on", "at", "as", "if", "from",
}

def clause_fragment(text: str, proper_context: str = "") -> str:
    """Turn a card sentence into a mid-clause fragment (item 9): trim the
    trailing period and lowercase a sentence-initial capital, unless the word
    is capitalized inside ``proper_context`` too (a proper noun/brand)."""
    t = " ".join(str(text).split()).rstrip(".")
    words = t.split()
    if words:
        first = words[0]
        if len(first) > 1 and first[0].isupper() and first[1:].islower() and first not in proper_context.split():
            words[0] = first[0].lower() + first[1:]
            t = " ".join(words)
    return t


# Wording that voids a decision line when embedded as its reason: hedges,
# conditionals, and questions read as non-commitment to the parser.
_REASON_NOISE = re.compile(
    r"[?]|\b(?:maybe|might|could|unless|only\s+if|depends|not\s+sure|i\s+guess|would\s+need|worr\w+|concern\w*)\b",
    re.I,
)


def usable_reason_fragment(text: str, proper_context: str = "", max_words: int = 14) -> str:
    """Clause-ready reason fragment, or "" when the stored reason is too long or
    hedged to embed in a decision line. Stored per-option reasons may be whole
    earlier utterances; embedding those verbatim can void a vote's parse."""
    t = clause_fragment(text or "", proper_context)
    if not t or len(t.split()) > max_words or _REASON_NOISE.search(t):
        return ""
    return t


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


# Matched surrounding quote pairs cleanup may unwrap (exactly one pair).
_QUOTE_PAIRS = {('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’"), ("„", "“")}


def _unwrap_quote_pair(text: str) -> str:
    """Remove ONE matching surrounding quote pair; inner quotes stay intact."""
    if len(text) >= 2 and (text[0], text[-1]) in _QUOTE_PAIRS:
        return text[1:-1].strip()
    return text


def clean_generated(text: str, speaker_name: str) -> str:
    """Structural normalization for non-envelope outputs (moderator lines):
    drop speaker prefixes, unwrap one surrounding quote pair, collapse
    whitespace, and cut a bracketed metadata trailer.

    Never deletes semantic content: no filler-tail removal, no word-budget
    clipping — a complete sentence over target stays complete.
    """
    text = strip_speaker_prefix(text, speaker_name)
    text = normalise_ws(text.replace("\n", " "))
    text = _unwrap_quote_pair(text)
    return re.sub(r"\s*\[\s*(?:act|opt|stance)\s*=.*$", "", text, flags=re.I).strip()


# Explicit output envelope requested from participant generation and repair.
_ENVELOPE = re.compile(r"<utterance>(.*?)</utterance>", re.I | re.S)
_STRAY_ENVELOPE_TAG = re.compile(r"</?utterance\b[^>]*>", re.I)

# Structural flags extract_utterance can report (raw-output stage findings).
FLAG_MISSING_ENVELOPE = "MISSING_ENVELOPE"        # telemetry: full response kept
FLAG_MALFORMED_ENVELOPE = "MALFORMED_ENVELOPE"    # stray/unbalanced tags
FLAG_EMPTY_ENVELOPE = "EMPTY_ENVELOPE"            # envelope present, no content
FLAG_MULTI_TURN_OUTPUT = "MULTI_TURN_OUTPUT"      # several envelopes returned
FLAG_LEAKED_METADATA = "LEAKED_METADATA"          # bracketed metadata inside the utterance


def extract_utterance(raw: str, speaker_name: str) -> tuple[str, list[str]]:
    """Extract exactly one visible utterance from a raw model response.

    Conservative and non-destructive (todo_validation item 4). May only:
    take the text inside the explicit ``<utterance>`` envelope, remove an
    accidental speaker prefix, unwrap one surrounding quote pair, and
    normalize whitespace. It must never remove natural tails ("What do you
    think?"), clauses after dashes/semicolons/colons, or anything resembling
    filler, and never clips to a word budget.

    When the envelope is missing, the complete response is preserved after
    minimal normalization and flagged — validation assesses it; cleanup never
    guesses where the utterance ends. Returned flags are raw-output structure
    findings for validation/trace.
    """
    flags: list[str] = []
    text = str(raw or "").strip()
    matches = [m.strip() for m in _ENVELOPE.findall(text)]
    if matches:
        if len(matches) > 1:
            flags.append(FLAG_MULTI_TURN_OUTPUT)
        body = matches[0]
        if not body:
            flags.append(FLAG_EMPTY_ENVELOPE)
    elif _STRAY_ENVELOPE_TAG.search(text):
        flags.append(FLAG_MALFORMED_ENVELOPE)
        body = _STRAY_ENVELOPE_TAG.sub(" ", text)
    else:
        flags.append(FLAG_MISSING_ENVELOPE)
        body = text
    body = strip_speaker_prefix(body, speaker_name)
    body = normalise_ws(body.replace("\n", " "))
    body = _unwrap_quote_pair(body)
    if re.search(r"\[\s*(?:act|opt|stance)\s*=", body, re.I):
        flags.append(FLAG_LEAKED_METADATA)
    return body, flags


