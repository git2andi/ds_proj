"""
prompts.py
----------
Single registry for every prompt template in the system.
All LLM-facing text lives here — nothing is hardcoded in modules.

Prompts are plain functions that accept typed arguments and return strings.
This makes them easy to read, test, and iterate without touching logic code.
"""

from __future__ import annotations

from typing import Optional


# =============================================================================
# Setup prompts (run once per dialogue)
# =============================================================================

def option_generation(topic: str) -> str:
    return f"""You are preparing a facilitated group decision discussion.

Topic:
{topic}

Task:
1. Generate exactly 4 concrete, comparable decision options for this topic.
2. Write a short opening question the moderator will ask to start the discussion.

Requirements for options:
- Each option must include 2-3 concrete attributes participants can actually compare.
- Infer sensible values from the topic context — do NOT use placeholders like "TBD".
- Keep each option to one concise line.
- All 4 options must represent genuinely different trade-offs.

Requirements for opening_question:
- One short, conversational sentence tailored to this specific topic.
- Should prompt participants to share what matters most to them personally.

Return valid JSON only:
{{
  "options": [
    "Option A - [label]: [attr1], [attr2], [attr3]",
    "Option B - [label]: [attr1], [attr2], [attr3]",
    "Option C - [label]: [attr1], [attr2], [attr3]",
    "Option D - [label]: [attr1], [attr2], [attr3]"
  ],
  "opening_question": "..."
}}
Do not include markdown or explanations outside the JSON."""


def role_planning(topic: str, names: list[str]) -> str:
    names_str = ", ".join(names)
    first = names[0]
    return f"""You are assigning discussion roles for a multi-person simulation.

Topic: {topic}
Participants: {names_str}

Assign one role to each participant so the roles fit the topic naturally.

Requirements:
- Return valid JSON only, using exactly this schema:
{{
  "roles": {{
    "{first}": {{"role": "short_role_name", "is_primary": true}},
    "OTHER_NAME": {{"role": "short_role_name", "is_primary": false}}
  }}
}}
- Every listed participant must appear exactly once.
- Roles must be topic-aligned (e.g. "budget_conscious_traveler", "birthday_person").
- Use short role names with underscores, no spaces.
- Exactly one participant must have "is_primary": true.
- The primary participant is the person most directly affected by the decision.
- Do not invent new participants or add explanations outside the JSON."""


def character_concept(
    topic: str,
    name: str,
    role: str,
    is_primary: bool,
) -> str:
    primary_note = (
        f"{name} is the central person in this scenario — the decision affects them most directly."
        if is_primary
        else f"{name} is a supporting participant helping reach a decision."
    )
    return f"""You are creating a participant profile for a group discussion simulation.

Topic: {topic}
Participant name: {name}
Role: {role}
{primary_note}

Generate a character concept for {name} that fits naturally into this topic.

Requirements:
- Write a 2-3 sentence backstory grounded in the topic domain.
- Include one relevant personal preference or past experience.
- For each personality dimension below, give a qualitative level (low / medium / high)
  AND a brief topic-specific note explaining what that level means in THIS context.
  Avoid generic statements — anchor every note to the topic.
- Do not reference trait numbers or simulation mechanics.

Return valid JSON only:
{{
  "backstory": "...",
  "personality_hints": {{
    "assertiveness": {{"level": "low|medium|high", "note": "..."}},
    "friendliness": {{"level": "low|medium|high", "note": "..."}},
    "talkativeness": {{"level": "low|medium|high", "note": "..."}},
    "agreeableness": {{"level": "low|medium|high", "note": "..."}},
    "patience": {{"level": "low|medium|high", "note": "..."}},
    "contrarian_pressure": {{"level": "low|medium|high", "note": "..."}},
    "initiative": {{"level": "low|medium|high", "note": "..."}}
  }},
  "focus_notes": {{
    "cost": "what cost focus means for {name} in this topic context",
    "comfort": "what comfort focus means for {name} in this topic context",
    "time": "what time focus means for {name} in this topic context",
    "safety": "what safety focus means for {name} in this topic context",
    "flexibility_focus": "what flexibility means for {name} in this topic context"
  }}
}}
Do not include markdown or explanations outside the JSON."""


def goal_generation(
    topic: str,
    name: str,
    role: str,
    is_primary: bool,
    backstory: str,
    traits_summary: str,
    focus_summary: str,
) -> str:
    primary_note = (
        "This participant is the primary person the decision affects."
        if is_primary
        else "This participant is supporting the group in reaching a good decision."
    )
    return f"""Scenario: {topic}
Participant: {name}
Role: {role}
{primary_note}
Backstory: {backstory}
Traits: {traits_summary}
Focus: {focus_summary}

Write exactly one short internal goal sentence for {name}.

Rules:
- The goal must be specific to the scenario domain.
- It should reflect the backstory and the role naturally.
- Express what the participant *hopes for* or *values*, NOT what they will argue for.
  Use language like "hopes to find", "would love", "cares about", not "will insist on" or "aims to secure".
  The goal shapes their priorities, not their stubbornness.
- Do NOT copy trait names or focus dimension labels into the goal text.
- Do NOT use filler phrases like "efficiently" or "seamlessly".
- Write in third person. One sentence only. No bullet points, markdown, or quotes."""


# =============================================================================
# Per-turn prompt (called every time a sim speaks)
# =============================================================================

def sim_turn(
    name: str,
    role: str,
    is_primary: bool,
    topic: str,
    options_text: str,
    goal: str,
    backstory: str,
    behavior_text: str,
    focus_text: str,
    style_instruction: str,
    state_summary: str,
    recent_points: str,
    recent_history: str,
    phase: str,
    contrarian_nudge: str,
    question_nudge: str,
    forbidden_openers: str,
    forbidden_frames: list[str],
    dynamic_forbidden_phrases: list[str],
    forced_adaptation: bool = False,
) -> str:
    # Build combined forbidden phrase block.
    # Static config frames prevent known bad patterns; dynamic phrases are
    # extracted from this specific conversation and grow as repetition occurs.
    all_forbidden = list(forbidden_frames) + list(dynamic_forbidden_phrases)
    forbidden_block = ""
    if all_forbidden:
        listed = "\n".join(f'  - "{f}"' for f in all_forbidden)
        forbidden_block = (
            f"\nDo NOT use these overused phrases "
            f"(they have already appeared too often in this conversation):\n{listed}"
        )

    forbidden_openers_line = (
        f"\nDo NOT start with any of these recently overused opener words: {forbidden_openers}."
        if forbidden_openers else ""
    )

    phase_instructions = {
        "opening": "Briefly introduce your main concern or priority in relation to the topic.",
        "preference_expression": "State which option you lean toward and the one specific reason that matters most to you.",
        "negotiation": "Compare trade-offs, react directly to what was just said, and adjust your position only if genuinely persuaded.",
        "narrowing": "Commit to a preferred option. A backup is fine if you are genuinely unsure.",
        "confirmation": "Clearly confirm or reject the emerging agreement — a plain yes or no is fine.",
        "closure": "One short, natural sign-off that fits your personality. One sentence only. Do not repeat the option name.",
    }
    phase_note = phase_instructions.get(phase, "React naturally to the conversation.")

    # Style instruction is placed FIRST — models weight earlier instructions more heavily
    # in large prompts. The hard rule framing reinforces that length is non-negotiable.
    style_block = f"""=== SPEAKING STYLE — HARD RULE ===
{style_instruction}
This is a strict constraint. Do not exceed the length limit regardless of how much you want to say.
If you have more than one point, pick only the single most important one and drop the rest.
==================================="""

    forced_block = ""
    if forced_adaptation:
        forced_block = """
=== FORCED ADAPTATION — THIS TURN ONLY ===
The moderator has noted that you have been repeating the same position without adding new reasoning.
You MUST do ONE of the following this turn — simply restating your preference is not acceptable:
  (a) Acknowledge a specific point someone else raised and explain whether it changes your view.
  (b) Raise a concrete concern about the leading option that you have NOT mentioned before.
  (c) Propose a genuine compromise or ask a question that could break the deadlock.
Failure to engage substantively will stall the discussion. Choose one and act on it.
=========================================="""

    return f"""{style_block}{forced_block}

You are {name}.
Role: {role}. Primary participant: {is_primary}.
Backstory: {backstory}

Scenario: {topic}

Options — the ONLY facts that exist in this discussion:
{options_text}

CRITICAL: Do NOT speculate about features, services, or amenities not listed above.
If an option does not mention something, it does not have it — full stop.
Do not ask "could they offer X?" or "maybe they have Y?" — if it is not listed, assume it does not exist.

Your internal profile:
- Goal: {goal}
- Personality: {behavior_text}
- Focus priorities: {focus_text}

Dialogue state: {state_summary}

Recent points made:
{recent_points}

Recent conversation:
{recent_history}

Instructions:
- Reply with your next utterance only — no speaker label, no stage directions.
- Stay in character at all times.
- React to what was just said before expressing your own view.
- If you were directly addressed or asked a question, respond to that first.
- Do NOT summarise what others said before making your point — just make it.
- Do NOT open with "As X mentioned..." or "Building on what X said...".{forbidden_block}{contrarian_nudge}{question_nudge}{forbidden_openers_line}

Current phase — {phase}:
{phase_note}

Final reminder: obey the SPEAKING STYLE hard rule at the very top of this prompt.
Sound like a real person in a group chat. Do not say goodbye unless the phase is closure."""


# =============================================================================
# Consensus detection prompt (LLM fallback)
# =============================================================================

def consensus_check(
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    min_agreeing: int,
    total: int,
) -> str:
    return f"""Participants: {", ".join(participant_names)}
Options available:
{chr(10).join(options)}

Recent dialogue:
{recent_dialogue}

Has a clear majority (at least {min_agreeing} out of {total} participants) agreed on one option?

Rules:
- A participant "agrees" only if they have clearly expressed support for one specific option.
- Asking a question about an option does NOT count as agreement.
- Do not invent votes that are not present in the dialogue.

Return valid JSON only:
{{
  "consensus_reached": true or false,
  "preferred_option": "A" or "B" or "C" or "D" or null,
  "backup_option": "A" or "B" or "C" or "D" or null
}}"""


# =============================================================================
# Moderator prompts
# =============================================================================

def moderator_intervention(
    topic: str,
    participant_names: list[str],
    recent_dialogue: str,
    intervention_reason: str,
    quiet_participant: Optional[str] = None,
) -> str:
    quiet_note = (
        f"\nNote: {quiet_participant} has not spoken recently — consider drawing them in."
        if quiet_participant else ""
    )
    return f"""You are a neutral moderator facilitating a group discussion.

Topic: {topic}
Participants: {", ".join(participant_names)}
Reason for intervention: {intervention_reason}{quiet_note}

Recent dialogue:
{recent_dialogue}

Write a single short moderator line that:
- Addresses the specific reason for intervention.
- Is neutral and does not favour any option.
- Moves the conversation forward constructively.
- Sounds natural and conversational, not formal.

Return only the moderator's line — no labels, no markdown."""


def moderator_outlier_nudge(
    topic: str,
    participant_names: list[str],
    recent_dialogue: str,
    outlier_name: str,
    primary_context: str = "",
) -> str:
    primary_note = ("\n" + primary_context) if primary_context else ""
    return f"""You are a neutral moderator facilitating a group discussion.

Topic: {topic}
Participants: {", ".join(participant_names)}
{outlier_name} has been repeating the same position without adding new reasoning.{primary_note}

Recent dialogue:
{recent_dialogue}

Write a single short moderator line directed at {outlier_name} that:
- Acknowledges their position respectfully.
- Asks whether there is a specific concern about the leading option that hasn't been addressed.
- Or asks whether they could accept the group's emerging direction as a compromise.
- Is warm and neutral, not pressuring or dismissive.
- One sentence only.

Return only the moderator's line — no labels, no markdown."""


def moderator_clarification(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    looping_topic: str,
) -> str:
    options_text = "\n".join(f"  {o}" for o in options)
    return f"""You are a neutral moderator facilitating a group discussion.

Topic: {topic}
Participants: {", ".join(participant_names)}

The full list of options — these are the ONLY facts available:
{options_text}

The group has been going in circles speculating about: "{looping_topic}"

Recent dialogue:
{recent_dialogue}

Your task:
Write a single short moderator line that clarifies what the options actually include or exclude
regarding "{looping_topic}", based strictly on the option descriptions above.

Rules:
- Only reference attributes explicitly listed in the options.
- Do NOT invent or imply details that are not in the option descriptions.
- If none of the options mention "{looping_topic}", say so clearly so the group can move on.
- If one or more options do mention something relevant, point it out factually.
- One or two sentences maximum.
- Sound like a helpful moderator, not a robot reading a spec sheet.

Return only the moderator's line — no labels, no markdown."""