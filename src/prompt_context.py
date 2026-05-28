"""
prompt_context.py
-----------------
Builders for the sections of the compact speaker-card prompt.

All prose templates live in prompts.py; this module only assembles them from
structured orchestrator state.

Sections built here:
  - speaker_card        : identity + register + Toulmin argument kit + speech sig
  - relevant_options    : 2-3 options that are actually in play for this sim
  - group_state         : current candidate, votes, unresolved rejections
  - memory_block        : per-sim memory (points made, open challenges, others' args)
  - local_context       : raw last-N transcript lines (still kept for grounding)
  - move_instruction    : phase + interaction + position discipline (+ optional plan)
  - output_contract     : word budget + formatting rules
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from persona import Persona
    from orchestrator import DialogueState
    from state import ParticipantState, StructuredState


# ---------------------------------------------------------------------------
# Speaker card -- identity, register, Toulmin argument kit, speech signature
# ---------------------------------------------------------------------------

def build_speaker_card(persona: "Persona") -> str:
    role_tag = " (primary)" if persona.is_primary else ""
    style_desc = persona.derived_controls_descriptor()
    lines = [
        f"Name: {persona.name}  Role: {persona.role}{role_tag}",
        f"Register: {persona.style_rule()}",
        f"Style: {style_desc}",
    ]
    if persona.backstory:
        lines.append(
            f"Your background (let it shape what you care about, don't recite it): {persona.backstory}"
        )
    if persona.beliefs:
        b = persona.beliefs
        accept_others = [x for x in b.acceptable if x != b.preferred]
        accepts_str = f", can accept {', '.join(accept_others)}" if accept_others else ""
        rejects_str = f"  Opposed to: {', '.join(b.rejected)}." if b.rejected else ""
        lines.append(
            f"Stance: prefers Option {b.preferred}{accepts_str}.{rejects_str}"
        )
        # Key concern is given as guidance, NOT as a phrase to recite. The
        # explicit warning matters -- otherwise sims latch onto the concern
        # words ('teamwork', 'flexibility') and trade them like tokens.
        lines.append(
            f"Underlying priority (translate this into specific-OPTION arguments; "
            f"never recite the phrase): {b.key_concern}"
        )
        if b.reasons:
            joined = " | ".join(b.reasons)
            lines.append(f"Your reasons (use these as warrants, don't quote them): {joined}")
        if b.reservation:
            lines.append(f"Your honest concern about a rival option: {b.reservation}")
        if b.would_reconsider_if:
            lines.append(f"What would change your mind: {b.would_reconsider_if}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# All options -- every participant always sees all four options.
# Filtering caused "option X doesn't exist" failures; full visibility fixes it.
# ---------------------------------------------------------------------------

def build_relevant_options(
    options: list[str],
    persona: "Persona",
    candidate: Optional[str] = None,
) -> str:
    """Return all options for this participant. No filtering.

    Filtering was removed because it caused participants to deny valid options
    that weren't in their personal 'relevant' set. Full visibility ensures no
    participant can claim an option is unavailable.
    """
    del persona, candidate  # kept in signature for call-site compatibility
    return "\n".join(f"  {o}" for o in options)


# ---------------------------------------------------------------------------
# Group state
# ---------------------------------------------------------------------------

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
            others_votes = ", ".join(
                f"{n}->{v}" for n, v in votes.items() if v and v != candidate
            )
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


# ---------------------------------------------------------------------------
# Memory block (Stage 1c + Stage 6)
# ---------------------------------------------------------------------------

def build_memory_block(
    speaker_name: str,
    structured: Optional["StructuredState"],
) -> str:
    """Relevance-filtered per-sim memory. Replaces a raw transcript dump.

    Sections (each optional, capped per cfg.memory):
      - Your last turn (literal)      : strongest anti-repeat signal
      - What you've already said      : compact running anti-repetition
      - Pushback aimed at you         : forces engagement
      - Live arguments from others    : enables build-on
      - What others care about        : perceived priorities for theory-of-mind
    """
    if structured is None:
        return ""

    ps = structured.participants.get(speaker_name)
    parts: list[str] = []

    # 1a) Your literal last turn -- the strongest anti-repeat signal. Summary
    # forms are easy to paraphrase; seeing your own exact prior line forces a
    # genuine pivot.
    last_own = _last_own_turn_text(speaker_name, structured)
    if last_own:
        parts.append(
            f"Your last turn (do NOT rephrase, do NOT restate this thought): \"{last_own}\""
        )

    # 1b) Your other recent points (longer-window anti-repeat). Skip the most
    # recent point -- it's already shown literally as "Your last turn", so
    # surfacing it again here just creates redundant noise.
    if ps and len(ps.points_made) > 1:
        recent = ps.points_made[-cfg.memory.points_made_max:][:-1]
        if recent:
            joined = "; ".join(recent)
            parts.append(f"Earlier you said: {joined}. Don't repeat these either.")

    # 2) Open challenges aimed at this sim (Stage 5).
    open_against = [
        c for c in structured.discourse.challenges
        if c.target == speaker_name and c.answered_turn_id is None
    ][-cfg.memory.open_challenges_max:]
    if open_against:
        ch_lines = []
        for c in open_against:
            ch_text = _challenge_text(structured, c.challenge_turn_id)
            opt_tag = f" (re Option {c.target_option})" if c.target_option else ""
            ch_lines.append(f"{c.challenger}{opt_tag}: \"{ch_text}\"")
        parts.append("Pushback aimed at you (engage with one): " + " | ".join(ch_lines))

    # 3) Live arguments from other sims (Stage 6 build-on).
    others_args = _recent_others_arguments(speaker_name, structured)
    if others_args:
        parts.append("Others' live arguments: " + " | ".join(others_args))

    # 4) Perceived priorities -- what others said in opening (Stage 1c).
    priorities = _perceived_priorities(speaker_name, structured)
    if priorities:
        parts.append("What others care about: " + "; ".join(priorities))

    return "\n".join(parts)


def _challenge_text(structured: "StructuredState", turn_id: int) -> str:
    for t in structured.turns:
        if t.turn_id == turn_id:
            # Truncate hard -- the speaker card should stay compact.
            return _trim(t.text, 80)
    return ""


def _recent_others_arguments(speaker_name: str, structured: "StructuredState") -> list[str]:
    cap = cfg.memory.others_arguments_max
    seen: set[str] = set()
    result: list[str] = []
    for t in reversed(structured.turns):
        if t.is_moderator or t.speaker == speaker_name:
            continue
        if t.speaker in seen:
            continue
        words = len(t.text.split())
        if words < 6:
            continue
        # Skip pure goodbyes / one-word confirms.
        from state import DialogueAct  # local to avoid module-load cycles
        if t.dialogue_act in (DialogueAct.GOODBYE, DialogueAct.CONFIRM):
            continue
        seen.add(t.speaker)
        result.append(f"{t.speaker}: \"{_trim(t.text, 90)}\"")
        if len(result) >= cap:
            break
    # Reverse so output reads oldest-first.
    return list(reversed(result))


def _last_own_turn_text(speaker_name: str, structured: "StructuredState") -> str:
    """The literal text of the speaker's most recent participant turn, trimmed."""
    for t in reversed(structured.turns):
        if not t.is_moderator and t.speaker == speaker_name:
            return _trim(t.text, 140)
    return ""


def _perceived_priorities(speaker_name: str, structured: "StructuredState") -> list[str]:
    result: list[str] = []
    for name, ps in structured.participants.items():
        if name == speaker_name or not ps.stated_priority:
            continue
        result.append(f"{name} -> {ps.stated_priority}")
    return result


def _trim(text: str, n: int) -> str:
    t = text.strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Raw recent turns (kept -- the model still benefits from seeing actual prose)
# ---------------------------------------------------------------------------

def build_local_context(history: list[str], n_recent: int) -> str:
    return "\n".join(history[-n_recent:] if len(history) >= n_recent else history)


# ---------------------------------------------------------------------------
# Move instruction
# ---------------------------------------------------------------------------

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
