"""
prompt_context.py
-----------------
Builds structured sections for LLM prompts. Prompt prose stays in prompts.py;
this module only formats current state, memory, and output contracts.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from persona import Persona
    from orchestrator import DialogueState
    from state import DialogueMemory


def build_speaker_card(persona: "Persona") -> str:
    role_tag = " (primary)" if persona.is_primary else ""
    lines = [
        f"Name: {persona.name}  Role: {persona.role}{role_tag}",
        f"Register: {persona.style_rule()}",
        f"Style: {persona.derived_controls_descriptor()}",
    ]
    if persona.backstory:
        lines.append(
            "Background shaping your view, not a script to recite: "
            f"{persona.backstory}"
        )
    if persona.beliefs:
        beliefs = persona.beliefs
        accept_others = [x for x in beliefs.acceptable if x != beliefs.preferred]
        accepts = f", can accept {', '.join(accept_others)}" if accept_others else ""
        rejects = f" Opposed to: {', '.join(beliefs.rejected)}." if beliefs.rejected else ""
        lines.append(f"Stance: prefers Option {beliefs.preferred}{accepts}.{rejects}")
        lines.append(
            "Underlying priority, translated into concrete option arguments: "
            f"{beliefs.key_concern}"
        )
        if beliefs.reasons:
            lines.append("Reasons you can draw on: " + " | ".join(beliefs.reasons))
        if beliefs.reservation:
            lines.append(f"Honest concern about a rival option: {beliefs.reservation}")
        if beliefs.would_reconsider_if:
            lines.append(f"What would change your mind: {beliefs.would_reconsider_if}")
    return "\n".join(lines)


def build_relevant_options(
    options: list[str],
    persona: "Persona",
    candidate: Optional[str] = None,
) -> str:
    del persona, candidate
    return "\n".join(f"  {option}" for option in options)


def build_group_state(state: "DialogueState") -> str:
    candidate = state.candidate_option or state.current_leading_option
    votes = state.last_known_vote or {}
    lines: list[str] = []

    if candidate:
        supporters = sorted(name for name, vote in votes.items() if vote == candidate)
        support_text = ", ".join(supporters) if supporters else "no confirmed votes yet"
        lines.append(f"Candidate: Option {candidate} ({support_text})")
        other_votes = ", ".join(
            f"{name}->{vote}" for name, vote in sorted(votes.items()) if vote != candidate
        )
        if other_votes:
            lines.append(f"Other votes: {other_votes}")
    elif votes:
        lines.append(
            "Current votes: "
            + ", ".join(f"{name}->{vote}" for name, vote in sorted(votes.items()))
        )
    else:
        lines.append("No votes cast yet.")

    for speaker, option in sorted((state.rejected_options_by_speaker or {}).items()):
        lines.append(f"{speaker} has rejected Option {option}.")
    if state.pending_question_target:
        lines.append(f"Pending question directed at: {state.pending_question_target}.")
    return "\n".join(lines)


def build_memory_block(
    speaker_name: str,
    memory: Optional["DialogueMemory"],
) -> str:
    if memory is None:
        return ""

    ps = memory.participants.get(speaker_name)
    parts: list[str] = []

    last_own = _last_own_turn_text(speaker_name, memory)
    if last_own:
        parts.append(f"Your last turn, do not rephrase it: \"{last_own}\"")

    if ps and len(ps.points_made) > 1:
        earlier = ps.points_made[-int(cfg.memory.points_made_max):][:-1]
        if earlier:
            parts.append("Earlier points from you, do not repeat: " + "; ".join(earlier))

    others = _recent_others_arguments(speaker_name, memory)
    if others:
        parts.append("Recent useful points from others: " + " | ".join(others))

    if getattr(cfg.memory, "show_perceived_priorities", False):
        priorities = _perceived_priorities(speaker_name, memory)
        if priorities:
            parts.append("What others seem to care about: " + "; ".join(priorities))

    return "\n".join(parts)


def build_local_context(history: list[str], n_recent: int) -> str:
    return "\n".join(history[-n_recent:] if len(history) >= n_recent else history)


def build_move_instruction(
    phase_instruction: str,
    interaction_instruction: str = "",
    position_discipline: str = "",
) -> str:
    parts = [phase_instruction]
    if interaction_instruction:
        parts.append(interaction_instruction.strip())
    if position_discipline:
        parts.append(position_discipline.strip())
    return "\n".join(part for part in parts if part.strip())


def build_output_contract(max_words: int, name: str) -> str:
    return (
        f"Write only {name}'s next chat message. "
        f"Hard cap: {max_words} words. Most turns should be shorter; "
        "use the full budget only when explaining something real. "
        "No name prefix. No markdown. No em dashes. "
        "Output only the raw dialogue line."
    )


def _last_own_turn_text(speaker_name: str, memory: "DialogueMemory") -> str:
    for turn in reversed(memory.turns):
        if not turn.is_moderator and turn.speaker == speaker_name:
            return _trim(turn.text, int(cfg.memory.last_turn_chars))
    return ""


def _recent_others_arguments(speaker_name: str, memory: "DialogueMemory") -> list[str]:
    cap = int(cfg.memory.others_arguments_max)
    if cap <= 0:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for turn in reversed(memory.turns):
        if turn.is_moderator or turn.speaker == speaker_name or turn.speaker in seen:
            continue
        if len(turn.text.split()) < int(cfg.memory.other_argument_min_words):
            continue
        seen.add(turn.speaker)
        result.append(f"{turn.speaker}: \"{_trim(turn.text, int(cfg.memory.other_argument_chars))}\"")
        if len(result) >= cap:
            break
    return list(reversed(result))


def _perceived_priorities(speaker_name: str, memory: "DialogueMemory") -> list[str]:
    result: list[str] = []
    for name, ps in memory.participants.items():
        if name != speaker_name and ps.stated_priority:
            result.append(f"{name} -> {ps.stated_priority}")
    return result


def _trim(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."
