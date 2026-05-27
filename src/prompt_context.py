"""
prompt_context.py
-----------------
Builders for the six sections of the compact speaker-card prompt.
All prose templates live in prompts.py; this module only assembles them
from structured orchestrator state.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from persona import Persona
    from orchestrator import DialogueState
    from policy import TurnPlan


def build_speaker_card(persona: "Persona") -> str:
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


def build_relevant_options(
    options: list[str],
    persona: "Persona",
    candidate: Optional[str] = None,
) -> str:
    relevant: set[str] = set()
    if persona.beliefs:
        relevant.add(persona.beliefs.preferred)
        relevant.update(persona.beliefs.acceptable or [])
    if candidate:
        relevant.add(candidate)
    if len(relevant) < 2:
        relevant.update(["A", "B"])

    result: list[str] = []
    for opt in options:
        m = re.match(r"^Option\s+([A-D])\b", opt, re.IGNORECASE)
        if m and m.group(1).upper() in relevant:
            result.append(f"  {opt}")
    return "\n".join(result) if result else "\n".join(f"  {o}" for o in options)


def build_group_state(state: "DialogueState") -> str:
    candidate = state.candidate_option or state.current_leading_option
    votes: dict[str, str] = state.last_known_vote or {}
    lines: list[str] = []

    if candidate:
        supporters = sorted(n for n, v in votes.items() if v == candidate)
        others = sorted(n for n, v in votes.items() if v and v != candidate)
        sup_str = f"{', '.join(supporters)} support" if supporters else "no confirmed support yet"
        lines.append(f"Candidate: Option {candidate} ({sup_str})")
        if others:
            others_votes = ", ".join(f"{n}->{v}" for n, v in votes.items() if v and v != candidate)
            lines.append(f"Dissenting votes: {others_votes}")
    elif votes:
        vote_summary = ", ".join(f"{n}->{v}" for n, v in sorted(votes.items()) if v)
        lines.append(f"Current votes: {vote_summary}")
    else:
        lines.append("No votes cast yet.")

    for speaker, opt in (state.rejected_options_by_speaker or {}).items():
        lines.append(f"{speaker} has rejected Option {opt} (unresolved).")

    if state.pending_question_target:
        lines.append(f"Pending question directed at: {state.pending_question_target}.")

    return "\n".join(lines)


def build_local_context(history: list[str], n_recent: int = 4) -> str:
    return "\n".join(history[-n_recent:] if len(history) >= n_recent else history)


def build_move_instruction(
    phase_instruction: str,
    interaction_instruction: str = "",
    position_discipline: str = "",
    forced_adaptation: bool = False,
    forbidden_openers: str = "",
    turn_plan: Optional["TurnPlan"] = None,
) -> str:
    parts: list[str] = []
    if turn_plan is not None:
        parts.append(turn_plan.to_prompt_str())
    parts.append(phase_instruction)
    if interaction_instruction:
        parts.append(interaction_instruction.strip())
    if position_discipline:
        parts.append(position_discipline.strip())
    if forced_adaptation:
        parts.append(
            "Moderator just pushed you. Don't repeat -- bring one fresh concern, "
            "a direct answer, or a yes/no with a condition."
        )
    if forbidden_openers:
        parts.append(f"Don't open with: {forbidden_openers}.")
    return "\n".join(p for p in parts if p.strip())


def build_output_contract(max_words: int, name: str) -> str:
    return (
        f"Write only {name}'s next chat message. "
        f"Hard cap: {max_words} words. Most turns should be shorter -- "
        "use the full budget only when you're explaining something real. "
        "No name prefix. No markdown. No em dashes. "
        "Output ONLY the raw dialogue line -- no brackets, no parentheses, "
        "no reasoning annotations, no meta-commentary of any kind."
    )
