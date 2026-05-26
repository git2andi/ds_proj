"""
realize/prompt_context.py
--------------------------
Builds the six compact sections passed into prompts.sim_turn_compact().

Speaker-card pattern (Stage 4):
  1. speaker_card      — name, role, register, private stance (4 lines)
  2. relevant_options  — only the 2-3 options this persona cares about
  3. group_state       — candidate, current votes, unresolved rejections
  4. local_context     — last 4 turns + optional deterministic summary
  5. move_instruction  — phase + interaction + position guidance (compact)
  6. output_contract   — hard word-budget and formatting rules

Saves ~50-60% of per-turn input tokens vs the legacy monolithic template.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from persona import Persona
    from orchestrator import DialogueState
    from policy.act_planner import TurnPlan


# ---------------------------------------------------------------------------
# 1. Speaker card
# ---------------------------------------------------------------------------

def build_speaker_card(persona: "Persona") -> str:
    """Compact 3-4 line persona block replacing backstory + personality + style prose."""
    role_tag = " (primary)" if persona.is_primary else ""
    lines = [
        f"Name: {persona.name}  Role: {persona.role}{role_tag}",
        f"Register: {persona.style_rule()}",
        f"Personality: {persona.personality_summary()}",
    ]
    if persona.beliefs:
        b = persona.beliefs
        accept_others = [x for x in b.acceptable if x != b.preferred]
        accepts_str = f", can accept {', '.join(accept_others)}" if accept_others else ""
        concession_str = f" (if: {b.concession})" if b.concession else ""
        rejects_str = f"  Opposed to: {', '.join(b.rejected)}." if b.rejected else ""
        lines.append(
            f"Stance: prefers Option {b.preferred}{accepts_str}{concession_str}. "
            f"Key concern: {b.key_concern}.{rejects_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Relevant options
# ---------------------------------------------------------------------------

def build_relevant_options(
    options: list[str],
    persona: "Persona",
    candidate: Optional[str] = None,
) -> str:
    """Return only the 2-3 options this persona's stance and the current candidate require."""
    relevant: set[str] = set()

    if persona.beliefs:
        relevant.add(persona.beliefs.preferred)
        relevant.update(persona.beliefs.acceptable or [])

    if candidate:
        relevant.add(candidate)

    # Ensure at least 2 options for context
    if len(relevant) < 2:
        relevant.update(["A", "B"])

    result: list[str] = []
    for opt in options:
        m = re.match(r"^Option\s+([A-D])\b", opt, re.IGNORECASE)
        if m and m.group(1).upper() in relevant:
            result.append(f"  {opt}")

    return "\n".join(result) if result else "\n".join(f"  {o}" for o in options)


# ---------------------------------------------------------------------------
# 3. Group state
# ---------------------------------------------------------------------------

def build_group_state(state: "DialogueState") -> str:
    """
    Compact paragraph from structured orchestrator state — no history re-scan.
    Reads state.last_known_vote, candidate_option, rejected_options_by_speaker.
    """
    candidate = state.candidate_option or state.current_leading_option
    votes: dict[str, str] = state.last_known_vote or {}

    lines: list[str] = []

    if candidate:
        supporters = sorted(n for n, v in votes.items() if v == candidate)
        others = sorted(n for n, v in votes.items() if v and v != candidate)
        sup_str = f"{', '.join(supporters)} support" if supporters else "no confirmed support yet"
        lines.append(f"Candidate: Option {candidate} ({sup_str})")
        if others:
            others_votes = ", ".join(f"{n}→{v}" for n, v in votes.items() if v and v != candidate)
            lines.append(f"Dissenting votes: {others_votes}")
    elif votes:
        vote_summary = ", ".join(f"{n}→{v}" for n, v in sorted(votes.items()) if v)
        lines.append(f"Current votes: {vote_summary}")
    else:
        lines.append("No votes cast yet.")

    for speaker, opt in (state.rejected_options_by_speaker or {}).items():
        lines.append(f"{speaker} has rejected Option {opt} (unresolved).")

    if state.pending_question_target:
        lines.append(f"Pending question directed at: {state.pending_question_target}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Rolling summary + local context
# ---------------------------------------------------------------------------

def build_rolling_summary(history: list[str], older_than_n: int = 4) -> str:
    """
    Deterministic 1-sentence summary of turns older than `older_than_n` recent turns.
    Covers which options were discussed by whom without re-reading all turns.
    """
    cutoff = len(history) - older_than_n
    if cutoff <= 0:
        return ""

    option_speakers: dict[str, set[str]] = {}

    for line in history[:cutoff]:
        if ":" not in line:
            continue
        speaker, msg = line.split(":", 1)
        speaker = speaker.strip()
        if speaker in cfg.EXCLUDED_SPEAKERS:
            continue
        letters = [m.upper() for m in re.findall(r"\boption\s+([a-d])\b", msg.lower())]
        for letter in letters:
            option_speakers.setdefault(letter, set()).add(speaker)

    if not option_speakers:
        return ""

    parts = [
        f"Option {opt}: {', '.join(sorted(speakers))}"
        for opt, speakers in sorted(option_speakers.items())
    ]
    return "Earlier discussion — " + "; ".join(parts) + "."


def build_local_context(
    history: list[str],
    n_recent: int = 4,
    rolling_summary: str = "",
) -> str:
    """Last n_recent turns, optionally preceded by a rolling summary of older content."""
    recent = history[-n_recent:] if len(history) >= n_recent else history
    recent_text = "\n".join(recent)
    if rolling_summary:
        return rolling_summary + "\n\n" + recent_text
    return recent_text


# ---------------------------------------------------------------------------
# 5. Move instruction
# ---------------------------------------------------------------------------

def build_move_instruction(
    phase_instruction: str,
    interaction_instruction: str = "",
    position_discipline: str = "",
    forced_adaptation: bool = False,
    forbidden_openers: str = "",
    turn_plan: Optional["TurnPlan"] = None,
) -> str:
    """
    Compact directive block replacing the old 14-bullet voice/style section.
    All prose arguments come from prompts.py; this function only assembles them.

    When turn_plan is provided (Stage 7 act planner enabled), the planned act
    is prepended so the LLM knows exactly what move to realize.
    """
    parts: list[str] = []

    # Stage 7: act plan block — prepend before phase instruction
    if turn_plan is not None:
        parts.append(turn_plan.to_prompt_str())

    parts.append(phase_instruction)

    if interaction_instruction:
        parts.append(interaction_instruction.strip())

    if position_discipline:
        parts.append(position_discipline.strip())

    if forced_adaptation:
        parts.append(
            "The moderator just pushed you forward. Do not repeat your last point — "
            "bring a fresh condition, concrete concern, or direct answer."
        )

    if forbidden_openers:
        parts.append(f"Do not open with: {forbidden_openers}.")

    return "\n".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# 6. Output contract
# ---------------------------------------------------------------------------

def build_output_contract(max_words: int, name: str) -> str:
    """Hard formatting rule — appended last so the LLM sees it immediately before generating."""
    return (
        f"Write only {name}'s next chat message. "
        f"≤ {max_words} words. "
        "Casual and direct. No name prefix. No markdown. No em dashes."
    )
