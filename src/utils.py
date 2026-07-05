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


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"(?:[$€£]\s*)?\d+(?:[.,:]\d+)?(?:\s*(?:kg|km|h|hr|hrs|hours?|min|minutes?|%|/\s*5))?", text, flags=re.I)


_DANGLING_TRAIL = {
    "to", "of", "for", "and", "but", "or", "with", "the", "a", "an", "than",
    "because", "so", "that", "is", "are", "in", "on", "at", "as", "if", "from",
}

# Only used on chopped text (where the ending is known to be cut): a trailing
# wh-word, pronoun, modal, or "about" is a fragment there, even though some of
# these can end a full sentence.
_CHOP_TRAIL = _DANGLING_TRAIL | {
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "about",
    "should", "could", "would", "can", "will", "might", "must", "may",
    "we", "you", "they", "he", "she", "it", "i", "our", "your", "their", "my",
    "this", "these", "those", "there", "be", "been", "was", "were",
    "has", "have", "had", "do", "does", "did", "just", "really", "very",
    "though", "while", "unless", "until", "whether",
    # Transitive/linking verbs that obviously hang without their complement
    # ("…the family-style menu means.", "…and I wonder.").
    "means", "wonder", "wonders", "makes", "gets", "keeps", "feels", "sounds",
    "seems", "brings", "gives", "helps", "lets", "needs", "wants", "suits",
    "offers", "offer", "make", "get", "keep", "feel", "sound", "seem",
}

# A chopped stub that still reads as a question ("does the slower setup bother
# you more") keeps its "?"; a stub whose interrogative clause was cut away
# ("We get fresh seafood and a chic vibe at Sushi Bar") must not.
_INTERROGATIVE_STUB = re.compile(
    r"\b(?:what|who|whom|whose|which|when|where|why|how|anyone|anybody)\b"
    r"|\b(?:do|does|did|are|is|was|were|can|could|should|would|will|shall|am)\s+"
    r"(?:i|we|you|they|it|he|she|that|this|there|anyone|everyone|any|the|our|a|an)\b",
    re.I,
)

# A coordinated interrogative tail cut mid-clause: "..., but what about those who".
_BROKEN_QUESTION_TAIL = re.compile(
    r"[,;]?\s*(?:and|but|or|so)?\s*(?:what|how)\s+about\s+"
    r"(?:the|a|an|those|these|that|this|them|it|us|anyone|everyone)?\s*(?:who|which|that)?\s*$",
    re.I,
)

# A chopped subordinate/coordinated clause opener left with at most a few
# words ("…as our base since the clean") — drop the whole stub.
_TRAILING_SUBCLAUSE_STUB = re.compile(
    r"[,;]?\s*\b(?:since|because|although|though|while|unless|if|when|as|but|and|or|so)\b"
    r"(?:\s+\S+){0,3}$",
    re.I,
)

# Terminal punctuation that ends a sentence (not a decimal point: "." inside
# "$4.50" is followed by a digit, so the lookahead excludes it).
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


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


def clean_generated(text: str, speaker_name: str, max_words: int) -> str:
    """Normalize a raw model utterance: drop speaker prefix, metadata trailer,
    generic filler, and enforce a word cap without leaving a dangling fragment."""
    text = strip_speaker_prefix(text, speaker_name)
    text = normalise_ws(text.replace("\n", " "))
    text = text.strip('"“”')
    text = re.sub(r"\s*\[\s*(?:act|opt|stance)\s*=.*$", "", text, flags=re.I).strip()
    text = _remove_generic_filler_tail(text)
    words = text.split()
    if len(words) <= max_words:
        return text
    # The word budget is a style target, not a correctness bound: a complete
    # sentence slightly over budget reads far better than a chopped stub (I14).
    soft_cap = max_words + max(8, round(max_words * 0.4))
    if len(words) <= soft_cap and text[-1] in ".!?":
        return text
    was_question = text.rstrip().endswith("?")
    # Cut at the last full sentence inside the soft window if one exists.
    window = " ".join(words[:soft_cap]).rstrip(" ,;:")
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    if ends and ends[-1] >= 10:
        return window[: ends[-1]].strip()
    # Next best: the last clause boundary inside the window. Ending a long
    # sentence at "…solid vegetarian choices" reads complete; a mid-clause word
    # chop ("…and the Rustic") never does.
    clause_cut = None
    min_keep = max(4, round(max_words * 0.5))
    for m in re.finditer(r"[,;]|\s[—–-]\s|\s--\s", window):
        prefix = window[: m.start()].rstrip(" ,;:")
        if len(prefix.split()) >= min_keep:
            clause_cut = prefix
    if clause_cut:
        chopped = clause_cut
    elif len(words) <= soft_cap + max_words and text[-1] in ".!?":
        # One unbroken clause, moderately over budget: a complete sentence over
        # target reads far better than any chopped stub.
        return text
    else:
        # Last resort: a runaway unbroken clause. Chop at the budget and
        # salvage; the stub must not end mid-thought.
        chopped = " ".join(words[:max_words]).rstrip(" ,;:")
    text = _remove_dangling_fragment(_remove_generic_filler_tail(chopped))
    text = _BROKEN_QUESTION_TAIL.sub("", text).rstrip(" ,;:")
    text = _TRAILING_SUBCLAUSE_STUB.sub("", text).rstrip(" ,;:")
    # A chop can still end on a bare function word ("... what and",
    # "... more than the") or a cut wh-word; strip those.
    tail = text.split()
    while tail and tail[-1].lower().rstrip(".,;:") in _CHOP_TRAIL:
        tail.pop()
    if tail:
        text = " ".join(tail).rstrip(" ,;:")
    if text and text[-1] not in ".!?":
        text += "?" if was_question and _INTERROGATIVE_STUB.search(text) else "."
    return text


def _remove_generic_filler_tail(text: str) -> str:
    patterns = [
        r"\s*(?:what do you think|thoughts|any thoughts)\??$",
        r"\s*(?:what about you|does that help|does that work)\??$",
        r"\s*(?:right|yeah)\??$",
    ]
    out = text
    for pattern in patterns:
        out = re.sub(pattern, "", out, flags=re.I).rstrip(" ,;:")
    return out.strip()


def _remove_dangling_fragment(text: str) -> str:
    patterns = [
        r"\s+(?:even if|even though|although|though|because|since|but|while|whereas|if|when|with|without)$",
        r"\s+(?:though|although|but|because|since|if)\s+(?:it|that|this|we|they|there|the)\s+(?:might|could|would|is|are|was|were|has|have)?\s*$",
        r"\s+(?:what|what do|what do you|can|can you|does|does that),?\.?$",
    ]
    out = text
    for pattern in patterns:
        out = re.sub(pattern, "", out, flags=re.I).rstrip(" ,;:")
    return out
