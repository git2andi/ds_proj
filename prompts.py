"""
prompts.py
----------
Single registry for every prompt template in the system.
All LLM-facing text lives here — nothing is hardcoded in other modules.

Organised into four sections:
  1. Setup prompts       — run once per dialogue (options, roles, personas)
  2. Turn prompt         — called every time a sim speaks
  3. Consensus prompt    — LLM fallback for agreement detection
  4. Moderator prompts   — interventions, narrowing, closure
"""

from __future__ import annotations

from typing import Optional


# =============================================================================
# 1. Setup prompts
# =============================================================================

def option_generation(topic: str) -> str:
    """Generate 4 concrete options and an opening question for the topic."""
    return f"""You are preparing a facilitated group decision discussion.

Topic: {topic}

Tasks:
1. Generate exactly 4 concrete, comparable decision options for this topic.
2. Write a short opening question the moderator will use to start discussion.

Option requirements:
- Each option must include 2–3 concrete attributes participants can compare.
- Infer sensible values from the topic — do NOT use placeholders like "TBD".
- Keep each option to one concise line.
- All 4 options must represent genuinely different trade-offs.

Opening question requirements:
- One short conversational sentence tailored to this specific topic.
- Should prompt participants to share what matters most to them personally.

Return valid JSON only — no markdown, no explanation:
{{
  "options": [
    "Option A - [label]: [attr1], [attr2], [attr3]",
    "Option B - [label]: [attr1], [attr2], [attr3]",
    "Option C - [label]: [attr1], [attr2], [attr3]",
    "Option D - [label]: [attr1], [attr2], [attr3]"
  ],
  "opening_question": "..."
}}"""


def role_assignment(topic: str, names: list[str]) -> str:
    """Assign one topic-aligned role to each participant in a single LLM call."""
    names_str = ", ".join(names)
    first = names[0]
    return f"""You are assigning discussion roles for a group simulation.

Topic: {topic}
Participants: {names_str}

Assign one role to each participant so the roles fit the topic naturally.
Exactly one participant must be the primary person most directly affected by the decision.

Return valid JSON only — no markdown, no explanation:
{{
  "roles": {{
    "{first}": {{"role": "short_role_label", "is_primary": true}},
    "OTHER_NAME": {{"role": "short_role_label", "is_primary": false}}
  }}
}}

Rules:
- Every listed participant must appear exactly once.
- Roles must be topic-aligned (e.g. "budget_traveler", "birthday_person", "team_lead").
- Use short labels with underscores, no spaces.
- Exactly one participant has "is_primary": true."""


def persona_concept(
    topic: str,
    name: str,
    role: str,
    is_primary: bool,
    trait_description_block: str,
) -> str:
    """
    Generate backstory and goal for one participant.
    Traits are pre-sampled and passed in as plain-English descriptions so the
    LLM writes a character that genuinely fits them — not the other way around.
    """
    primary_note = (
        f"{name} is the central person — the decision affects them most directly."
        if is_primary
        else f"{name} is a supporting participant helping reach a good decision."
    )
    return f"""You are creating a participant profile for a group discussion simulation.

Topic: {topic}
Participant: {name}
Role: {role}
{primary_note}

This participant has the following personality traits — these are fixed.
Write the backstory and goal so they reflect these traits naturally:
{trait_description_block}

Return valid JSON only — no markdown, no explanation:
{{
  "backstory": "2–3 sentences grounded in the topic. Must be consistent with the traits above. Include one relevant personal preference or past experience.",
  "goal": "One sentence in third person. What {name} hopes for or values. Must be consistent with the traits above. Use language like 'hopes to find' or 'cares about', not 'will argue for'."
}}

Rules:
- Backstory and goal must clearly reflect the personality traits — a warmth-5 person should sound warm, a contrarian-5 person should sound sceptical.
- Backstory must be specific to the topic domain, not generic.
- Goal must NOT copy trait names or use filler words like "efficiently" or "seamlessly".
- Do not reference simulation mechanics or numeric scores.
- Do not return a "personality" field — traits are already fixed."""


# =============================================================================
# 2. Turn prompt
# =============================================================================

def sim_turn(
    name: str,
    role: str,
    is_primary: bool,
    topic: str,
    options_text: str,
    goal: str,
    backstory: str,
    personality_summary: str,
    style_rule: str,
    phase: str,
    phase_instruction: str,
    state_summary: str,
    recent_history: str,
    forbidden_openers: str,
    forbidden_frames: list[str],
    contrarian_nudge: str = "",
    forced_adaptation: bool = False,
) -> str:
    """Prompt for a single participant turn."""

    forbidden_block = ""
    if forbidden_frames:
        listed = "\n".join(f'  - "{f}"' for f in forbidden_frames)
        forbidden_block = f"\nDo NOT use these overused phrases:\n{listed}"

    opener_block = (
        f"\nDo NOT start your reply with any of these recently overused words: {forbidden_openers}."
        if forbidden_openers else ""
    )

    forced_block = ""
    if forced_adaptation:
        forced_block = """
=== THIS TURN: You have been repeating the same position. You MUST do one of: ===
  (a) Acknowledge a specific point someone else raised and say whether it changes your view.
  (b) Raise a concrete concern about the leading option you have NOT mentioned before.
  (c) Propose a genuine compromise or ask a question to break the deadlock.
Restating your preference without new reasoning is not acceptable.
================================================================================"""

    return f"""=== SPEAKING STYLE — HARD RULE ===
{style_rule}
Do not exceed this limit. If you have more than one point, pick the most important one.
==================================={forced_block}

You are {name}.
Role: {role}. Primary participant: {is_primary}.
Backstory: {backstory}

Scenario: {topic}

Options — the ONLY facts that exist in this discussion:
{options_text}

CRITICAL: Do NOT speculate about features not listed above. If an option does not mention something, it does not have it.

Your internal profile:
- Goal: {goal}
- Personality: {personality_summary}

Current state: {state_summary}

Recent conversation:
{recent_history}

Instructions:
- Reply with your next utterance only — no speaker label, no stage directions.
- Stay in character at all times.
- React to what was just said before expressing your own view.
- If you were directly addressed or asked a question, respond to that first.
- Do NOT summarise what others said — just make your point.
- Do NOT open with "As X mentioned..." or "Building on what X said...".\
{forbidden_block}{contrarian_nudge}{opener_block}

Current phase — {phase}:
{phase_instruction}

Final reminder: obey the SPEAKING STYLE rule above. Sound like a real person. Do not say goodbye unless the phase is closure."""

def sim_turn_open(
    name: str,
    role: str,
    is_primary: bool,
    topic: str,
    goal: str,
    backstory: str,
    personality_summary: str,
    style_rule: str,
    phase: str,
    state_summary: str,
    recent_history: str,
    forbidden_openers: str,
    forbidden_frames: list[str],
    dynamic_forbidden_phrases: list[str],
    forced_adaptation: bool = False,
) -> str:
    """
    Turn prompt for open-ended topics (no options, no voting).
    Used when the scenario is flagged as 'open' mode.
    Sims exchange views freely; the moderator ends on time or natural conclusion.
    """
    all_forbidden = list(forbidden_frames) + list(dynamic_forbidden_phrases)
    forbidden_block = ""
    if all_forbidden:
        listed = "\n".join(f'  - "{f}"' for f in all_forbidden)
        forbidden_block = f"\nDo NOT use these overused phrases:\n{listed}"

    opener_block = (
        f"\nDo NOT start your reply with any of these recently overused words: {forbidden_openers}."
        if forbidden_openers else ""
    )

    forced_block = ""
    if forced_adaptation:
        forced_block = """
=== THIS TURN: You have been repeating the same point. You MUST do one of: ===
  (a) Introduce a genuinely new angle or consideration you have not raised before.
  (b) Ask a specific question to another participant that could change the direction.
  (c) Acknowledge someone else's point and say clearly whether it shifts your view.
Simply restating your position is not acceptable.
============================================================================="""

    phase_instructions = {
        "opening": "Say hello briefly and introduce your initial take on the topic in your own words. Keep it natural — you are just arriving at a conversation.",
        "discussion": "React to what was just said and add your own perspective. Take a clear stance.",
        "deepening": "Push deeper — challenge an assumption, add nuance, or ask a pointed question.",
        "closing": "Wrap up naturally — say where you landed or what you are taking away from this. One sentence, like you are stepping away from a conversation, not delivering a verdict.",
    }
    phase_instruction = phase_instructions.get(phase, "React naturally and honestly to the conversation.")

    return f"""=== SPEAKING STYLE — HARD RULE ===
{style_rule}
Do not exceed this limit. If you have more than one point, pick the most important one.
==================================={forced_block}

You are {name}.
Role: {role}. Primary participant: {is_primary}.
Backstory: {backstory}

Topic of discussion: {topic}

There are no predefined options. This is an open exchange of views.

Your internal profile:
- Goal: {goal}
- Personality: {personality_summary}

Current state: {state_summary}

Recent conversation:
{recent_history}

Instructions:
- Reply with your next utterance only — no speaker label, no stage directions.
- Stay in character at all times.
- React to what was just said before expressing your own view.
- Take a clear personal stance — vague non-answers are not acceptable.
- Do NOT open with "As X mentioned..." or "Building on what X said...".\
{forbidden_block}{opener_block}

Current phase — {phase}:
{phase_instruction}

Final reminder: obey the SPEAKING STYLE rule above. Sound like a real person having an actual opinion."""


# =============================================================================
# =============================================================================

def consensus_check(
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    min_agreeing: int,
    total: int,
) -> str:
    """Ask the LLM whether a clear majority has agreed on one option."""
    return f"""Participants: {", ".join(participant_names)}
Options:
{chr(10).join(options)}

Recent dialogue:
{recent_dialogue}

Has a clear majority (at least {min_agreeing} out of {total} participants) agreed on one option?

Rules:
- A participant "agrees" only if they clearly expressed support for one specific option.
- Asking a question about an option does NOT count as agreement.
- Do not invent votes not present in the dialogue.

Return valid JSON only:
{{
  "consensus_reached": true or false,
  "preferred_option": "A" or "B" or "C" or "D" or null,
  "backup_option": "A" or "B" or "C" or "D" or null
}}"""


# =============================================================================
# 4. Moderator prompts
# =============================================================================

def moderator_intervention(
    topic: str,
    participant_names: list[str],
    recent_dialogue: str,
    reason: str,
    target_participant: Optional[str] = None,
) -> str:
    """
    General moderator intervention. Used for stalls, silent participants, and outliers.
    The `reason` string describes what is happening; `target_participant` is optional.
    """
    target_note = (
        f"\nFocus your line on drawing {target_participant} into the conversation."
        if target_participant else ""
    )
    return f"""You are a neutral moderator facilitating a group discussion.

Topic: {topic}
Participants: {", ".join(participant_names)}
Situation: {reason}{target_note}

Recent dialogue:
{recent_dialogue}

Write a single short moderator line that:
- Addresses the situation described above.
- Is neutral and does not favour any option.
- Moves the conversation forward constructively.
- Sounds natural and conversational, not formal.
- Is one sentence only.

Return only the moderator's line — no label, no markdown."""


def moderator_clarification(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    looping_topic: str,
) -> str:
    """Moderator clarifies what the options do or do not include, to stop speculative loops."""
    options_text = "\n".join(f"  {o}" for o in options)
    return f"""You are a neutral moderator facilitating a group discussion.

Topic: {topic}
Participants: {", ".join(participant_names)}

The available options (these are the ONLY facts):
{options_text}

The group has been speculating about: "{looping_topic}"

Recent dialogue:
{recent_dialogue}

Write a single short moderator line that clarifies what the options include or exclude
regarding "{looping_topic}", based strictly on the option descriptions above.

Rules:
- Only reference attributes explicitly listed in the options.
- Do NOT invent details not in the option descriptions.
- If none of the options mention "{looping_topic}", say so clearly so the group can move on.
- One or two sentences maximum. Sound helpful, not robotic.

Return only the moderator's line — no label, no markdown."""