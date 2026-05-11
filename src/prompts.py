"""
prompts.py
----------
Single registry for every prompt template in the system.
All LLM-facing text lives here.

Sections:
  1. Setup prompts       — options, role assignment, persona concept, agent beliefs
  2. Turn prompt         — called every time a sim speaks
  3. Consensus prompt    — LLM fallback for agreement detection
  4. Moderator prompts   — interventions, narrowing, closure, force-close
"""

from __future__ import annotations

from collections import Counter
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
    """Assign one topic-aligned role to each participant."""
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
    "{first}": {{"role": "brief natural phrase", "is_primary": true}},
    "OTHER_NAME": {{"role": "brief natural phrase", "is_primary": false}}
  }}
}}

Rules:
- Every listed participant must appear exactly once.
- Use 1–3 word natural phrases describing the person, not a job title.
- No underscores, no camelCase, no title case.
- Exactly one participant has "is_primary": true.
- Role and primary status must be believable and aligned with the topic.
- Primary and Role must be consistent — the primary person should have a role that makes it natural for them to care deeply about the decision, while others should have supporting roles.
"""


def persona_concept(
    topic: str,
    name: str,
    role: str,
    is_primary: bool,
    trait_description_block: str,
) -> str:
    """
    Generate backstory and goal for one participant.
    Traits are pre-sampled and passed in so the LLM writes a character
    that genuinely fits them.
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
  "goal": "One sentence in third person. What {name} hopes for or values. Must be consistent with the traits above."
}}

Rules:
- Backstory and goal must clearly reflect the personality traits.
- Backstory must be specific to the topic domain, not generic.
- Goal must NOT copy trait names or use filler words like "efficiently" or "seamlessly".
- Do not reference simulation mechanics or numeric scores.
- Do not return a "personality" field — traits are already fixed."""


def agent_beliefs(
    name: str,
    role: str,
    goal: str,
    backstory: str,
    personality_summary: str,
    options_text: str,
) -> str:
    """
    Generate a stable internal belief state for one participant.
    Called once per participant after options are known, before the dialogue starts.
    """
    return f"""You are building the internal preference model for a group discussion participant.

Participant: {name}
Role: {role}
Goal: {goal}
Background: {backstory}
Personality: {personality_summary}

Options available:
{options_text}

Based on this person's background, goal, and personality — not on what would be
"objectively best" — decide which option they would most naturally lean toward,
what they could genuinely live with as a compromise, and what they'd resist.

Rules:
- Ground every answer in the backstory and goal. Do not invent new facts.
- "acceptable" must include the preferred option, plus a genuine compromise (or multiple).
- "rejected" can be empty if the person is genuinely flexible.
- key_concern: a short phrase (not a full sentence) — the one thing that matters most.
- concession: a concrete condition, not vague ("if others strongly want it" is too vague;
  "if the group is clearly prioritizing budget over convenience" is concrete).

Return valid JSON only — no markdown, no explanation:
{{
  "preferred": "A" or "B" or "C" or "D",
  "acceptable": ["A"] or ["B", "C"] etc,
  "rejected": [] or ["D"] etc,
  "key_concern": "short phrase",
  "concession": "concrete condition under which they'd accept a compromise"
}}"""


# =============================================================================
# 2. Turn prompt
# =============================================================================

def sim_turn(
    name: str,
    topic: str,
    options_text: str,
    goal: str,
    backstory: str,
    personality_summary: str,
    style_rule: str,
    phase: str,
    phase_instruction: str,
    recent_history: str,
    forbidden_openers: str,
    forbidden_frames: list[str],
    beliefs_block: str = "",
    last_speaker_line: str = "",
    position_discipline: str = "",
    contrarian_nudge: str = "",
    forced_adaptation: bool = False,
) -> str:
    """Prompt for a single participant turn."""

    forbidden_block = ""
    if forbidden_frames:
        listed = "\n".join(f'  - "{f}"' for f in forbidden_frames)
        forbidden_block = f"\nNever say:\n{listed}"

    opener_block = (
        f"\nDon't open with: {forbidden_openers}."
        if forbidden_openers else ""
    )

    forced_block = (
        "\n\nMODERATOR CALLED YOU OUT: Don't repeat your last point. "
        "Bring one new angle — a trade-off you haven't raised, a concession, or a genuine question."
        if forced_adaptation else ""
    )

    last_said_block = (
        f"\nJust said — {last_speaker_line}\n"
        if last_speaker_line else ""
    )

    beliefs_section = f"\n{beliefs_block}\n" if beliefs_block else ""

    return f"""STYLE (non-negotiable): {style_rule}
React to what was just said, then make your point. "Yeah/nah/true/fair/wait" are fine. No formal summaries, no em dashes.
Avoid hollow filler ("great point", "absolutely"). No AI buzzwords.{forced_block}

You are {name}. {backstory} Goal: {goal}. {personality_summary}{beliefs_section}
Deciding: {topic}
Options — these are the only facts, never invent attributes not listed:
{options_text}

Your backstory shapes your priorities and the anecdotes you draw on.
It does NOT give you hidden knowledge about the options — stick to what is listed above.

Recent conversation:
{recent_history}
{last_said_block}
[{phase}] {phase_instruction}{position_discipline}{contrarian_nudge}{forbidden_block}{opener_block}

Write {name}'s next message. One voice. No name prefix. No stage directions."""


# =============================================================================
# 3. Consensus prompt
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
- An option might also be supported by implication or described rather than named directly.
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
    escalation_level: int = 0,
) -> str:
    """
    General moderator intervention for outliers and silent participants.
    escalation_level (0–3) controls directness.
    """
    target_note = (
        f"\nFocus your line on drawing {target_participant} into the conversation."
        if target_participant else ""
    )

    escalation_notes = {
        0: (
            f"Ask {target_participant or 'them'} one short, specific question — "
            "something that gets them to explain WHY their position matters to them, not just restate it."
        ),
        1: (
            f"Ask {target_participant or 'them'} directly: what one specific thing would they need "
            "from the majority option to consider it? One sentence, no open-ended phrasing."
        ),
        2: "Be firm but respectful — tell them the group needs movement and ask for one thing that could change their mind.",
        3: "Be direct — acknowledge the impasse and ask for a final position.",
    }
    escalation_note = escalation_notes.get(escalation_level, escalation_notes[0])

    return f"""You are a neutral moderator facilitating a group discussion.

Topic: {topic}
Participants: {", ".join(participant_names)}
Situation: {reason}{target_note}
Moderator approach: {escalation_note}

Recent dialogue:
{recent_dialogue}

Write a single short moderator line that:
- Addresses the situation using the approach above.
- Is neutral and does not favour any option.
- Sounds natural and conversational, not formal.
- Is one sentence only.

Return only the moderator's line — no label, no markdown."""


def moderator_deadlock(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    current_votes: dict,
    escalation_level: int = 1,
) -> str:
    """
    Moderator addresses a genuine deadlock where everyone has voted but no majority exists.
    """
    options_text = "\n".join(f"  {o}" for o in options)
    votes_text = ", ".join(f"{name} → Option {opt}" for name, opt in current_votes.items())

    # Describe the split honestly
    counts = Counter(current_votes.values()) if current_votes else Counter()
    n = len(participant_names)
    top_count = counts.most_common(1)[0][1] if counts else 0
    has_majority = top_count > n / 2

    if has_majority and counts:
        top_opt = counts.most_common(1)[0][0]
        minority_names = [nm for nm, o in current_votes.items() if o != top_opt]
        split_desc = f"Most prefer Option {top_opt}, but {' and '.join(minority_names)} have not yet moved."
    else:
        split_desc = "The group is split with no clear majority — everyone has voted differently."

    escalation_instructions = {
        1: (
            f"{split_desc} "
            "Name the participant(s) who are holding a different position. Ask that person directly: "
            "what one specific thing about their current choice matters so much "
            "they can't accept the others? One direct question."
        ),
        2: (
            f"{split_desc} "
            "Acknowledge the split by name. Tell the group that unless someone moves, "
            "you will have to make a call. Ask for one final round of genuine compromise — not restatement."
        ),
        3: (
            "Announce that the group has been unable to reach agreement and that you "
            "are going to make a final call based on the discussion so far. "
            "Do not ask another question."
        ),
    }
    instruction = escalation_instructions.get(escalation_level, escalation_instructions[1])

    return f"""You are a neutral moderator facilitating a group discussion.

Topic: {topic}
Participants: {", ".join(participant_names)}

Current situation: Everyone has stated a position but there is no consensus.
Current votes: {votes_text}

Available options:
{options_text}

Recent dialogue:
{recent_dialogue}

Your task: {instruction}

Rules:
- One or two sentences maximum.
- Do not favour any option.
- Sound like a real person, not a formal chair.
- Do not repeat what was already said.

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

Write a single short moderator line that redirects the group toward what IS listed in the options,
rather than the speculation. Do not open with "None of the options mention..." — steer toward
relevant attributes instead.

Rules:
- Only reference attributes explicitly listed in the options.
- Do NOT invent details not in the option descriptions.
- One or two sentences maximum. Sound like a helpful facilitator.

Return only the moderator's line — no label, no markdown."""


def moderator_emergence(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    candidate_option: str,
) -> str:
    """
    Moderator facilitates emergence: acknowledges that the discussion is moving,
    invites conditional openness without pressuring for explicit commitment.
    Fisher Phase 3 — dissent dissolves via ambiguity, not direct agreement.
    """
    options_text = "\n".join(f"  {o}" for o in options)
    return f"""You are a neutral moderator facilitating the natural emergence of consensus.

Topic: {topic}
Participants: {", ".join(participant_names)}
Option gaining traction: Option {candidate_option}

Available options:
{options_text}

Recent dialogue:
{recent_dialogue}

The group has made their main arguments. Some movement toward Option {candidate_option} is visible,
but no one has fully committed yet. This is the moment for gradual convergence — not forced agreement.

Your task: Acknowledge the direction the discussion is heading and invite participants to express
what conditions or concerns, if addressed, would help them move forward.
Do NOT ask for a final vote. Do NOT pressure for explicit agreement. Sound like a facilitator
who sees progress and wants to help it land naturally.

Rules:
- One or two sentences only.
- Do not declare a winner.
- Sound natural and conversational — not formal.

Return only the moderator's line — no label, no markdown."""


def moderator_force_close(
    topic: str,
    participant_names: list[str],
    final_option: str,
    reason: str,
    recent_dialogue: str,
) -> str:
    """
    Moderator ends a discussion that could not reach natural consensus.
    Honest about the lack of agreement; names a moderator-selected outcome.
    """
    return f"""You are a neutral moderator who needs to close a group discussion that did not reach consensus.

Topic: {topic}
Participants: {", ".join(participant_names)}
Selected option: Option {final_option}
Why selected: {reason}

Recent dialogue:
{recent_dialogue}

Write a single moderator line that:
- Honestly acknowledges that the group did not reach agreement.
- States clearly that you are making the call as moderator.
- Names Option {final_option} and gives a brief, honest reason (use the "Why selected" above).
- Sounds natural — not a formal announcement.
- Is one to two sentences maximum.

Return only the moderator's line — no label, no markdown."""