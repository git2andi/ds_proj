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

import random
from typing import Optional, TYPE_CHECKING

import prompts
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

    Update.md §5.2 trim: drop summary-heavy blocks that fed the chat back
    abstract concern words ("cost", "reliability", "flexibility") and pushed
    sims to recite them. Kept (functional, anti-repeat, anti-drop):
      - Your literal last turn         : strongest anti-rephrase signal
      - Your recent point signatures   : longer-window anti-repeat
      - Pushback aimed at you          : prevents dropped engagement
    Trimmed:
      - Others' live arguments cap reduced (cfg.memory.others_arguments_max)
    Toggleable (default OFF after update):
      - Perceived priorities (cfg.memory.show_perceived_priorities)

    Build-on and theory-of-mind still happen through the raw RECENT TURNS
    block in the prompt -- we just stopped pre-digesting them.
    """
    if structured is None:
        return ""

    ps = structured.participants.get(speaker_name)
    parts: list[str] = []

    # 1a) Your literal last turn -- strongest anti-rephrase signal.
    last_own = _last_own_turn_text(speaker_name, structured)
    if last_own:
        parts.append(
            f"Your last turn (do NOT rephrase, do NOT restate this thought): \"{last_own}\""
        )

    # 1b) Your other recent points (longer-window anti-repeat). Skip the most
    # recent point -- it's already shown literally as "Your last turn".
    if ps and len(ps.points_made) > 1:
        recent = ps.points_made[-cfg.memory.points_made_max:][:-1]
        if recent:
            joined = "; ".join(recent)
            parts.append(f"Earlier you said: {joined}. Don't repeat these either.")

    # 2) Open challenges aimed at this sim (kept -- functional, not summary).
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

    # 3) Live arguments from other sims -- kept but trimmed to a tighter cap.
    if cfg.memory.others_arguments_max > 0:
        others_args = _recent_others_arguments(speaker_name, structured)
        if others_args:
            parts.append("Others' live arguments: " + " | ".join(others_args))

    # 4) Perceived priorities -- now opt-in. Off by default after update.md §5.2.
    if getattr(cfg.memory, "show_perceived_priorities", False):
        priorities = _perceived_priorities(speaker_name, structured)
        if priorities:
            parts.append("What others care about: " + "; ".join(priorities))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Surface-move sampler (update.md §4.2)
# ---------------------------------------------------------------------------

def _surface_move_weights(repetition_high: bool) -> dict[str, float]:
    sm = getattr(cfg, "surface_moves", None)
    if sm is None:
        return {}
    weights_obj = sm.weights_high_repetition if repetition_high else sm.weights_normal
    if weights_obj is None:
        return {}
    raw = getattr(weights_obj, "_raw", None)
    if isinstance(raw, dict):
        return {k: float(v) for k, v in raw.items()}
    # fallback: walk attributes
    out: dict[str, float] = {}
    for k in ("ack_only", "short_no", "question", "compromise",
              "decision_move", "new_reason"):
        v = getattr(weights_obj, k, None)
        if v is not None:
            out[k] = float(v)
    return out


def pick_surface_move_kind(
    phase: str,
    repetition_high: bool,
    has_open_challenge: bool,
    has_open_question: bool,
) -> Optional[str]:
    """Stochastic nudge that picks ONE surface-move kind, or None for "no hint".

    Returns the KIND only; the prose lives in prompts.surface_move_hint().

    Suppressed when the simulator already has a hard obligation (an open
    challenge or a pending question) -- those instructions own the turn shape.
    """
    sm = getattr(cfg, "surface_moves", None)
    if sm is None or not getattr(sm, "enable", True):
        return None
    if phase not in ("negotiation", "narrowing", "emergence"):
        return None
    if has_open_challenge or has_open_question:
        return None

    prob = float(sm.hint_prob_high_repetition if repetition_high else sm.hint_prob_normal)
    if random.random() >= prob:
        return None

    weights = _surface_move_weights(repetition_high)
    if not weights:
        return None
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0:
        return None
    r = random.uniform(0, total)
    upto = 0.0
    for kind, w in weights.items():
        upto += max(0.0, w)
        if upto >= r:
            return kind
    return next(iter(weights), None)


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
    surface_move_kind: Optional[str] = None,
) -> str:
    """Assemble the YOUR MOVE block.

    Optional `surface_move_kind` injects a short hint nudging the turn toward
    a specific natural shape (update.md §4.2). The hint sits AFTER the
    position-discipline block so the model reads it last and is most likely
    to follow it. Prose lives in prompts.surface_move_hint().
    """
    parts = [phase_instruction]
    if interaction_instruction:
        parts.append(interaction_instruction.strip())
    if position_discipline:
        parts.append(position_discipline.strip())
    if surface_move_kind:
        hint = prompts.surface_move_hint(surface_move_kind).strip()
        if hint:
            parts.append(f"Suggested move for this turn: {hint}")
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
