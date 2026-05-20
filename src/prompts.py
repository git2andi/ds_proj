"""
prompts.py
----------
Single registry for every prompt template in the system.
All LLM-facing text lives here.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional


# =============================================================================
# 1. Setup prompts
# =============================================================================

def option_generation(topic: str) -> str:
    """Generate 4 concrete choice frames and an opening question for the topic."""
    return f"""You are preparing a realistic small-group chat about a decision.

Topic: {topic}

Create four distinct options that give the group enough concrete material to discuss.
The options should be useful conversation anchors, not rigid debate positions.

Option requirements:
- Exactly 4 options, A-D.
- Each option should be one compact line with: a label, what it offers, one trade-off or risk, and who it best fits.
- Infer sensible details from the topic, but do not use placeholders or unknown facts.
- Make the options meaningfully different; avoid four versions of the same idea.
- Include at least one discussable practical detail per option, such as time, effort, cost, difficulty, risk, tone, or fit.

Opening question requirements:
- One short, natural moderator question.
- It should invite personal priorities, not a vote yet.

Return valid JSON only - no markdown, no explanation:
{{
  "options": [
    "Option A - [label]: [main upside]; trade-off: [risk/cost]; best for: [priority/person]",
    "Option B - [label]: [main upside]; trade-off: [risk/cost]; best for: [priority/person]",
    "Option C - [label]: [main upside]; trade-off: [risk/cost]; best for: [priority/person]",
    "Option D - [label]: [main upside]; trade-off: [risk/cost]; best for: [priority/person]"
  ],
  "opening_question": "..."
}}"""


def role_assignment(topic: str, names: list[str]) -> str:
    """Assign one topic-aligned role to each participant."""
    names_str = ", ".join(names)
    first = names[0]
    return f"""You are assigning roles for a realistic group chat simulation.

Topic: {topic}
Participants: {names_str}

Assign one natural role to each participant. Exactly one participant must be the primary person most directly affected by the decision.

Return valid JSON only - no markdown, no explanation:
{{
  "roles": {{
    "{first}": {{"role": "brief natural phrase", "is_primary": true}},
    "OTHER_NAME": {{"role": "brief natural phrase", "is_primary": false}}
  }}
}}

Rules:
- Every listed participant must appear exactly once.
- Use 1-4 words that sound like a real relationship to the decision, not a corporate job title unless the topic requires it.
- Exactly one participant has "is_primary": true.
- Primary and role must be consistent."""


def persona_concept(
    topic: str,
    name: str,
    role: str,
    is_primary: bool,
    trait_description_block: str,
) -> str:
    """Generate backstory and goal for one participant."""
    primary_note = (
        f"{name} is the central person - the decision affects them most directly."
        if is_primary
        else f"{name} is involved, but not the central person."
    )
    return f"""You are creating a participant profile for a realistic group chat simulation.

Topic: {topic}
Participant: {name}
Role: {role}
{primary_note}

The participant has these Big Five personality traits. Treat them as behavioral tendencies, not stereotypes:
{trait_description_block}

Return valid JSON only - no markdown, no explanation:
{{
  "backstory": "2-3 grounded sentences. Include one specific preference, past experience, worry, or habit relevant to the topic.",
  "goal": "One sentence in third person describing what {name} wants from the discussion."
}}

Rules:
- Make the profile specific to the topic.
- Reflect the traits subtly through priorities and behavior.
- Do not mention trait names or numeric scores in the output.
- Give them one concrete reason they might resist at least one plausible option.
- Do not write a caricature."""


def agent_beliefs(
    name: str,
    role: str,
    goal: str,
    backstory: str,
    personality_summary: str,
    options_text: str,
) -> str:
    """Generate a stable internal belief state for one participant."""
    return f"""You are building a private preference model for a group chat participant.

Participant: {name}
Role: {role}
Goal: {goal}
Background: {backstory}
Personality tendencies: {personality_summary}

Options available:
{options_text}

Decide what this person would most naturally prefer before the chat starts.
Base this on their background, goal, and personality - not on what is objectively best.

Rules:
- "preferred": the single option that best fits this person's goal and key concern.
- "acceptable": options this person could genuinely live with. Must include the preferred option. Should also include any option that meaningfully addresses the key_concern — for example, if the concern is cost, include cheaper options even if not the top pick; if the concern is atmosphere, include options with a similar feel. Aim for 2-3 acceptable options to represent realistic flexibility. Only exclude an option if it actively conflicts with the key_concern or goal.
- "rejected": only options the person would actively resist and could not live with even as a compromise.
- key_concern: short phrase naming what matters most to them. This must be consistent with the preferred option — do not say "budget" if the preferred option is the most expensive one.
- concession: concrete condition under which they could accept a non-preferred option.
- Do not invent facts beyond the option lines and profile.

Return valid JSON only - no markdown, no explanation:
{{
  "preferred": "A" or "B" or "C" or "D",
  "acceptable": ["A"] or ["B", "C"] etc,
  "rejected": [] or ["D"] etc,
  "key_concern": "short phrase",
  "concession": "concrete condition"
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
    max_words: int,
    phase: str,
    phase_instruction: str,
    interaction_instruction: str,
    own_recent_points: str,
    recent_history: str,
    forbidden_openers: str,
    forbidden_frames: list[str],
    beliefs_block: str = "",
    last_speaker_line: str = "",
    position_discipline: str = "",
    skepticism_nudge: str = "",
    forced_adaptation: bool = False,
) -> str:
    """Prompt for a single participant turn."""

    forbidden_block = ""
    if forbidden_frames:
        listed = "; ".join(f'"{f}"' for f in forbidden_frames[:4])
        forbidden_block = f"\nAvoid canned phrases like {listed}."

    opener_block = (
        f"\nAvoid reusing these exact openers: {forbidden_openers}."
        if forbidden_openers else ""
    )

    forced_block = (
        "\n\nThe moderator just pushed you to move the conversation forward. "
        "Do not repeat your last point. Bring a fresh condition, concern, trade-off, or direct answer."
        if forced_adaptation else ""
    )

    beliefs_section = f"\n{beliefs_block}\n" if beliefs_block else ""

    return f"""Write the next chat message for {name}.

Voice and style:
- {style_rule}
- HARD LIMIT: maximum {max_words} words. Shorter is almost always better.
- Sound like someone texting their friends — casual, direct, and personal. Not a panelist, analyst, or formal debater.
- Write in informal, spoken language. Use contractions, drop formality, let your personality come through. Do NOT repeat the same filler opener ("honestly", "hmm yeah", "but what if") across multiple turns.
- React to the last thing said before adding your own point. A brief reaction is natural, but vary how you open each turn.
- Vary your move: not every turn is [assessment + reason]. Some turns should be a blunt reaction, a short personal detail, a single direct pushback, or a concession with a condition. Avoid hypothetical "but what if..." questions as a default — make a claim instead.
- Do not stack questions. Answer something first; you can ask one thing at most.
- Do not restate your same option-plus-reason. Add a new angle, name a specific drawback, make a concession, or respond to something specific someone just said.
- Skip hollow validation: no "great point", "absolutely", "I completely agree", "that's a good observation".
- No name prefix, no stage directions, no markdown, no em dashes.{forced_block}

Participant:
You are {name}. {backstory}
Goal: {goal}
Personality: {personality_summary}{beliefs_section}

Decision topic: {topic}
Options are the only shared facts. You may discuss implications, but do not invent extra attributes:
{options_text}

Recent conversation:
{recent_history}
{own_recent_points}
Current phase: {phase}
Instruction for this turn: {phase_instruction}{interaction_instruction}{position_discipline}{skepticism_nudge}{forbidden_block}{opener_block}

Write only {name}'s next message."""


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
- A participant counts as agreeing only if their latest position clearly supports one specific option.
- Conditional acceptance can count if the condition is minor or already addressed.
- A question, vague openness, or polite acknowledgement does not count as agreement.
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
    """General moderator intervention for outliers and silent participants."""
    target_note = (
        f"\nFocus your line on drawing {target_participant} into the conversation."
        if target_participant else ""
    )

    escalation_notes = {
        0: f"Point to one unresolved issue and ask {target_participant or 'them'} for an answer or concrete stance.",
        1: f"Ask {target_participant or 'them'} what one concrete condition would make a compromise workable.",
        2: "Be firm but still conversational: the group needs a decision-relevant answer, not another loose question.",
        3: "Ask for a final position, directly and briefly.",
    }
    escalation_note = escalation_notes.get(escalation_level, escalation_notes[0])

    return f"""You are a neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}
Situation: {reason}{target_note}
Moderator approach: {escalation_note}

Recent dialogue:
{recent_dialogue}

Write one short moderator message.
Rules:
- Sound natural and conversational.
- Do not favor an option.
- Do not recap the whole discussion.
- If there is an unanswered question, ask someone to answer it or turn it into a concrete trade-off.
- One sentence only.

Return only the moderator's line."""


def moderator_deadlock(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    current_votes: dict,
    escalation_level: int = 1,
) -> str:
    """Moderator addresses a genuine deadlock where everyone has voted but no majority exists."""
    options_text = "\n".join(f"  {o}" for o in options)
    votes_text = ", ".join(f"{name} -> Option {opt}" for name, opt in current_votes.items())

    counts = Counter(current_votes.values()) if current_votes else Counter()
    n = len(participant_names)
    top_count = counts.most_common(1)[0][1] if counts else 0
    has_majority = top_count > n / 2

    if has_majority and counts:
        top_opt = counts.most_common(1)[0][0]
        minority_names = [nm for nm, o in current_votes.items() if o != top_opt]
        split_desc = f"Most prefer Option {top_opt}, but {' and '.join(minority_names)} have not moved."
    else:
        split_desc = "The group is split with no clear majority."

    escalation_instructions = {
        1: (
            f"{split_desc} Ask the person or people outside the leading option for one concrete condition or objection, "
            "and tell the others to respond to it."
        ),
        2: (
            f"{split_desc} Tell the group you need one final compromise attempt before making a call."
        ),
        3: (
            "Say the group has not reached agreement and you are going to make a final call. Do not ask a question."
        ),
    }
    instruction = escalation_instructions.get(escalation_level, escalation_instructions[1])

    return f"""You are a neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}
Current votes: {votes_text}

Available options:
{options_text}

Recent dialogue:
{recent_dialogue}

Your task: {instruction}

Rules:
- One or two sentences maximum.
- Do not favor an option unless escalation level 3 explicitly requires a final call.
- Sound like a real person, not a formal chair.
- Do not repeat what participants already said.

Return only the moderator's line."""


def moderator_clarification(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    looping_topic: str,
) -> str:
    """Moderator clarifies what the options do or do not include, to stop speculative loops."""
    options_text = "\n".join(f"  {o}" for o in options)
    return f"""You are a neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}

The available options are the only shared facts:
{options_text}

The group has been speculating about: "{looping_topic}"

Recent dialogue:
{recent_dialogue}

Write one short moderator line that redirects them back to listed option details without sounding scolding.

Rules:
- Only reference attributes explicitly listed in the options.
- Do not invent details.
- One or two sentences maximum.

Return only the moderator's line."""


def moderator_emergence(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    candidate_option: str,
) -> str:
    """Moderator facilitates gradual convergence without demanding a vote."""
    options_text = "\n".join(f"  {o}" for o in options)
    return f"""You are a neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}
Option gaining traction: Option {candidate_option}

Available options:
{options_text}

Recent dialogue:
{recent_dialogue}

The group seems close, but not fully settled. Invite people to answer the strongest remaining concern or name the condition that would let them move forward.

Rules:
- One or two sentences only.
- Do not declare a winner.
- Do not ask for a final vote.
- Sound natural and conversational.

Return only the moderator's line."""


def moderator_compromise_test(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    compromise_option: str,
    holdout_names: list[str],
) -> str:
    """Moderator tests whether a compromise can be visibly accepted."""
    options_text = "\n".join(f"  {o}" for o in options)
    holdout_text = ", ".join(holdout_names) if holdout_names else "the group"
    return f"""You are a neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}
Possible compromise: Option {compromise_option}
People who need to react clearly: {holdout_text}

Available options:
{options_text}

Recent dialogue:
{recent_dialogue}

Write one natural moderator message that tests Option {compromise_option} as a compromise.

Rules:
- Do not declare it final.
- Ask whether Option {compromise_option} could work if specific concerns are addressed.
- Name one concern from the recent dialogue if possible.
- The concern must be relevant to Option {compromise_option}; do not mix in a condition that only belongs to another option.
- Ask for a clear yes-with-condition or no-with-objection.
- One or two sentences maximum.

Return only the moderator's line."""


def moderator_force_close(
    topic: str,
    participant_names: list[str],
    final_option: str,
    reason: str,
    recent_dialogue: str,
) -> str:
    """Moderator ends a discussion that could not reach natural consensus."""
    return f"""You are a neutral moderator closing a group chat that did not reach consensus.

Topic: {topic}
Participants: {", ".join(participant_names)}
Selected option: Option {final_option}
Why selected: {reason}

Recent dialogue:
{recent_dialogue}

Write one moderator message that:
- Honestly acknowledges the group did not fully agree.
- Says you are making the call.
- Names Option {final_option} and gives the brief reason above.
- Sounds natural, not ceremonial.
- Is one to two sentences maximum.

Return only the moderator's line."""
