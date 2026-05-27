"""
utils.py
--------
Shared utilities used across orchestrator, consensus, and moderation modules.
Consolidates helpers that were previously duplicated in those three files.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from simulator import Simulator


# ---------------------------------------------------------------------------
# Vote / option extraction
# ---------------------------------------------------------------------------

def extract_preference_vote(msg: str) -> Optional[str]:
    """
    Return the option letter (A-D) this message votes for, or None if ambiguous.

    Counts:  first-person preference phrases ("I prefer Option B"),
             option letters in the first 5 words of a non-question message.
    Ignores: question-starting messages, comparisons ("Option A or B"),
             option mentions that are references to others' positions.
    """
    text = msg.strip().lower()
    words = text.split()

    fp_patterns = [
        r"\bi\s+prefer\s+option\s+([a-d])\b",
        r"\bi(?:'m|\s+am)\s+(?:going\s+with|for|set\s+on|sold\s+on|leaning)\s+(?:option\s+)?([a-d])\b",
        r"\bi(?:'d|\s+would)\s+(?:go\s+with|choose|prefer)\s+(?:option\s+)?([a-d])\b",
        r"\bi\s+(?:choose|pick|want|vote\s+for)\s+option\s+([a-d])\b",
        r"\bmy\s+(?:choice|pick|preference)\s+is\s+(?:option\s+)?([a-d])\b",
    ]
    for pat in fp_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()

    first5 = " ".join(words[:5])
    early = re.search(r"\boption\s+([a-d])\b", first5)
    if early:
        lead = words[0].rstrip(",'") if words else ""
        if lead in {"what", "how", "why", "when", "where", "is", "are", "does", "can", "could"}:
            return None
        nearby = " ".join(words[:10])
        if len(re.findall(r"\boption\s+[a-d]\b", nearby)) > 1:
            return None
        letter = early.group(1)
        pos = text.find(f"option {letter}")

        # Check what comes BEFORE -- "not", "never", "rather not", "anything but"
        before = text[:pos].rstrip()
        if re.search(
            r"\b(not|never|rather\s+not|anything\s+but|except|avoid|skip|hate|dislike|reject|refuse|nope\s+to)\s*$",
            before,
        ):
            return None

        char_after = text[pos + len(f"option {letter}"):].lstrip()[:1]
        if char_after == "'":
            return None
        text_after = text[pos + len(f"option {letter}"):].lstrip()
        if re.search(
            r"^(is\s+(?:only|too|just|not|barely|merely)|isn'?t|aren'?t|wasn'?t|"
            r"might\s+be|could\s+be|won't|can't|seems|sounds\s+(?:bad|terrible|awful)|sucks|nope)",
            text_after,
        ):
            return None
        # Also reject if the surrounding clause is non-committal ("not bad either",
        # "either", "maybe", "perhaps" near the option mention).
        nearby_after = text[pos:pos + len(f"option {letter}") + 50].lower()
        if re.search(r"\b(either|maybe|perhaps|possibly|might\s+work)\b", nearby_after):
            return None
        return letter.upper()

    return None


def extract_option_letters(text: str) -> list[str]:
    return [m.upper() for m in re.findall(r"\boption\s+([a-d])\b", text.lower())]


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


def recent_participant_lines(history: list[str], limit: int) -> list[str]:
    """Return up to `limit` most-recent participant lines, newest-first."""
    lines: list[str] = []
    for line in reversed(history):
        if ":" not in line:
            continue
        if line.split(":", 1)[0].strip() in cfg.EXCLUDED_SPEAKERS:
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def current_votes(history: list[str], sims: list) -> dict[str, str]:
    """Most recent explicit option vote per participant."""
    votes: dict[str, str] = {}
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        speaker = speaker.strip()
        if speaker in cfg.EXCLUDED_SPEAKERS or speaker in votes:
            continue
        vote = extract_preference_vote(msg)
        if vote:
            votes[speaker] = vote
        if len(votes) == len(sims):
            break
    return votes


def latest_turn_per_speaker(history: list[str], sims: list) -> dict[str, str]:
    """Most recent message text (lowercased) per participant."""
    latest: dict[str, str] = {}
    for line in reversed(history):
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        speaker = speaker.strip()
        if speaker in cfg.EXCLUDED_SPEAKERS:
            continue
        if speaker not in latest:
            latest[speaker] = msg.lower()
        if len(latest) == len(sims):
            break
    return latest
