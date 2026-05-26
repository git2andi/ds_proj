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
# 0. Simulator prose helpers — text used inside sim_turn
# =============================================================================

def narrowing_base_text() -> str:
    return (
        "Name your current preferred option clearly (e.g. 'I prefer Option A'). "
        "You may mention a backup if it is genuinely plausible. "
        "Once you have stated a preference, change it at most once, and only because a specific new point "
        "actually changes your mind. Do not switch just because the group repeats itself."
    )


def phase_instruction_text(phase: str, add_brevity: bool, has_voted: bool = False) -> str:
    """Return the phase-level instruction injected into every sim_turn prompt."""
    base = narrowing_base_text()
    if phase == "narrowing":
        if not has_voted:
            text = (
                "You have not yet stated a preferred option. "
                "Before anything else this turn, name your preferred option explicitly. " + base
            )
        else:
            text = (
                "You already named your preferred option. Do not repeat 'I prefer Option X'. "
                "Now give one condition, concern, or possible compromise."
            )
    else:
        instructions = {
            "greeting": (
                "Say a quick, casual hello. One short line only. "
                "Do not introduce your role and do not discuss the topic yet."
            ),
            "opening": (
                "Give your first real reaction. Name what matters to you and why, briefly. "
                "A short concern or question is fine, but do not make the whole turn only a vague question."
            ),
            "negotiation": (
                "Discuss the actual point on the table. Answer if someone asked something, or respond to a claim. "
                "You may disagree, build on it, give a reason, or make a concrete trade-off. "
                "Do not simply say your preferred option is better for the same reason again. "
                "Only use facts listed in the options; do not invent hidden attributes."
            ),
            "narrowing": base,
            "emergence": (
                "The group is trying to land somewhere. "
                "Say what would make the leading option workable, or name the concrete concern that still blocks you. "
                "Do not repeat your original reason; respond to someone else's objection or offer a compromise condition."
            ),
            "confirmation": (
                "The moderator is asking for final confirmation. "
                "Reply with an explicit yes or no. A question, hedge, or silence is treated as no. "
                "Only say no if you have a specific objection you have not yet raised."
            ),
            "closure": (
                "The discussion is over. Write one short goodbye only. "
                "Nothing about the topic. No opinions. No questions."
            ),
        }
        text = instructions.get(phase, "React naturally to the conversation.")

    if add_brevity:
        text += " Add only one new useful thing; do not re-explain your full case."
    return text


def interaction_instruction_block(
    last_has_question: bool,
    question_count: int,
    compromise_option: Optional[str],
    compromise_in_acceptable: bool,
    rejected_option: Optional[str],
    rejecting_self: bool,
    turns_since_rejection: int,
    escalation_threshold: int,
    own_last_was_question: bool,
    speculative_count: int,
    repeated_kws: list[str],
    self_repeated: bool,
) -> str:
    """Prose guidance for conversational obligations, computed from dialogue state flags."""
    parts: list[str] = []

    if last_has_question:
        parts.append(
            " The last participant asked a question. Answer it directly if you can; "
            "if the options do not contain enough information, say what you can infer and then give your view."
        )

    if question_count >= 3:
        parts.append(
            " The chat has too many unanswered questions. Do not ask another one; make a claim, answer, or choose a trade-off."
        )

    if compromise_option:
        if compromise_in_acceptable:
            parts.append(
                f" The moderator is testing Option {compromise_option} as a compromise. "
                f"Say whether you could live with Option {compromise_option}; name a condition that applies to that option."
            )
        else:
            parts.append(
                f" The moderator is testing Option {compromise_option} as a compromise. "
                f"If Option {compromise_option} does not work for you, say no and give the specific objection to that option."
            )

    if rejected_option and rejecting_self:
        if turns_since_rejection >= escalation_threshold:
            parts.append(
                f" You have raised multiple concerns about Option {rejected_option} and the group has tried to address them."
                f" Name your single concrete dealbreaker for Option {rejected_option}, or say whether you can accept it with one specific condition."
                " Do not raise a new objection — commit to a position."
            )
        else:
            parts.append(
                f" You just rejected Option {rejected_option}. Explain the main blocker clearly, "
                "and say what would need to change for you to consider it."
            )

    if own_last_was_question:
        parts.append(
            " Your previous turn was a question, so this turn should not be another bare question."
        )

    if speculative_count >= 2:
        parts.append(
            " You've asked 'what if' hypotheticals multiple times. Stop — make a direct claim or statement instead."
        )

    if len(repeated_kws) >= 2:
        kw_str = ", ".join(repeated_kws[:3])
        parts.append(
            f" You keep returning to the same theme ({kw_str}). That point is on the table — don't repeat it."
            " Either name a concrete dealbreaker, propose a condition, or move to a completely different angle."
        )
    elif self_repeated:
        parts.append(
            " Your recent turns are repeating the same point. Change move: answer someone, name a downside of your own option, or offer a compromise."
        )

    return "\n" + "".join(parts) if parts else ""


def position_discipline_block(
    phase: str,
    prefix: str,
    anchor: str,
    flips: int,
    candidate: Optional[str],
    candidate_in_acceptable: bool,
    candidate_in_rejected: bool,
    candidate_is_anchor: bool,
    can_soften: bool,
    concession_text: str,
    high_agreeableness: bool,
    low_agreeableness_or_high_neuroticism: bool,
) -> str:
    """Coherence anchor prose injected into the turn prompt, based on pre-computed flags."""
    if phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
        return ""

    coherence = f" Stay consistent with Option {anchor}."

    if phase == "emergence":
        cond = f" Concession condition: {concession_text}." if concession_text else ""
        if candidate and candidate_in_acceptable and not candidate_is_anchor:
            if high_agreeableness:
                return f"\n{prefix} Option {candidate} is acceptable.{cond} Give one short condition."
            if low_agreeableness_or_high_neuroticism:
                return f"\n{prefix} Option {candidate} is acceptable.{cond} Name one concern before softening."
            return f"\n{prefix} Option {candidate} is acceptable.{cond} Soften briefly."
        if candidate and candidate_in_rejected:
            if can_soften:
                return (
                    f"\n{prefix} Option {candidate} is where the group is heading and you can find common ground when needed."
                    f"{cond} You've raised concerns already — decide now: name one specific condition that would let you accept it,"
                    " or say it is a genuine dealbreaker and why."
                )
            return (
                f"\n{prefix} Option {candidate} is gaining ground but you genuinely oppose it. "
                "Acknowledge the direction briefly, then state your specific remaining objection."
            )
        if candidate and candidate_is_anchor:
            return (
                f"\n{prefix} Option {candidate} is gaining ground - it is your preferred choice. "
                "Help it land without being heavy-handed."
            )
        return (
            f"\n{prefix} The group is moving toward resolution. "
            "Let your position soften if it is genuinely softening."
        )

    if phase == "negotiation":
        return f"\n{prefix}{coherence}"

    # narrowing / confirmation
    if flips == 0:
        return f"\n{prefix}{coherence} Only change for a genuinely new reason."
    if flips == 1:
        return f"\n{prefix} Already switched once.{coherence} Hold this; no repeat."
    return f"\n{prefix} Switched {flips} times. Commit now - no more switching.{coherence}"


def closure_templates(option: str) -> list[str]:
    """Fixed goodbye lines referencing the agreed option."""
    return [
        f"Okay, Option {option} works for me. Bye!",
        f"Option {option} it is then. See you!",
        f"Sounds good, Option {option}. Bye!",
        f"Alright, let's go with Option {option}. See you.",
    ]


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


def persona_group_generation(
    topic: str,
    names_roles_traits: list[dict],
) -> str:
    """
    Stage 11: Generate all N personas in one LLM call.
    Replaces N individual persona_concept() calls.

    names_roles_traits: list of dicts with keys:
      name, role, is_primary (bool), trait_description_block (str)
    Returns JSON: {"personas": {"<name>": {"backstory": "...", "goal": "..."}, ...}}
    """
    participants_block = ""
    for entry in names_roles_traits:
        primary_note = (
            f"(primary — this decision affects {entry['name']} most directly)"
            if entry["is_primary"] else "(not the central person)"
        )
        participants_block += (
            f"\n{entry['name']} — {entry['role']} {primary_note}\n"
            f"Personality traits:\n{entry['trait_description_block']}\n"
        )

    names_list = ", ".join(e["name"] for e in names_roles_traits)
    return f"""You are creating participant profiles for a realistic group chat simulation.

Topic: {topic}
Participants: {names_list}
{participants_block}
For each participant, write a backstory and goal that:
- Are specific to the topic.
- Reflect the personality traits through priorities and behavior (subtly — do not name traits or scores).
- Include one concrete reason the person might resist at least one plausible option.
- Are not caricatures.

Return valid JSON only — no markdown, no explanation:
{{
  "personas": {{
    "{names_roles_traits[0]['name']}": {{
      "backstory": "2-3 grounded sentences with one specific preference, experience, worry, or habit.",
      "goal": "One sentence in third person describing what this person wants from the discussion."
    }}
  }}
}}

Include all {len(names_roles_traits)} participants in the "personas" object."""


def agent_beliefs_group(
    topic: str,
    personas_text: str,
    options_text: str,
) -> str:
    """
    Stage 11: Generate all N belief states in one LLM call.
    Replaces N individual agent_beliefs() calls.

    personas_text: compact multi-persona summary (name, role, goal, backstory, personality).
    Returns JSON: {"beliefs": {"<name>": {preferred, acceptable, rejected, key_concern, concession}, ...}}
    """
    return f"""You are building private preference models for group chat participants.

Topic: {topic}

Participants:
{personas_text}

Options available:
{options_text}

For each participant decide what they would most naturally prefer before the chat starts.
Base this on their background, goal, and personality — not on what is objectively best.

Rules for every participant:
- "preferred": the single option that best fits their goal and key concern.
- "acceptable": options they could genuinely live with. Must include the preferred option. Include any option that meaningfully addresses the key_concern — aim for 2-3 acceptable options. Only exclude an option if it actively conflicts with the key_concern or goal.
- "rejected": only options they would actively resist and could not live with even as a compromise.
- "key_concern": short phrase naming what matters most. Must be consistent with preferred — do not say "budget" if they prefer the most expensive option.
- "concession": concrete condition under which they could accept a non-preferred option.
- Do not invent facts beyond the option lines and profiles.

Return valid JSON only — no markdown, no explanation:
{{
  "beliefs": {{
    "NAME": {{
      "preferred": "A" or "B" or "C" or "D",
      "acceptable": ["A"] or ["B","C"] etc,
      "rejected": [] or ["D"] etc,
      "key_concern": "short phrase",
      "concession": "concrete condition"
    }}
  }}
}}

Include every participant listed above in the "beliefs" object."""


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
    beliefs_block: str = "",
    last_speaker_line: str = "",
    position_discipline: str = "",
    forced_adaptation: bool = False,
) -> str:
    """Prompt for a single participant turn."""

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
- Write in informal, spoken language. Use contractions, drop formality, let your personality come through.
- React to the last thing said before adding your own point. A brief reaction is natural, but vary how you open each turn.
- Vary your move: not every turn is [assessment + reason]. Some turns should be a blunt reaction, a short personal detail, a single direct pushback, or a concession with a condition. Make claims rather than asking hypothetical questions.
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
Instruction for this turn: {phase_instruction}{interaction_instruction}{position_discipline}{opener_block}

Write only {name}'s next message."""


# =============================================================================
# 2b. Compact turn prompt — speaker-card pattern (Stage 4)
# =============================================================================

def sim_turn_compact(
    speaker_card: str,
    relevant_options: str,
    group_state: str,
    local_context: str,
    move_instruction: str,
    output_contract: str,
) -> str:
    """
    Compact speaker-card turn prompt.  Target: 400-600 input tokens vs ~1100 for legacy.

    Section layout mirrors the MASTERPLAN §Stage 4 example:
      SPEAKER CARD → SHARED FACTS → GROUP STATE → YOUR MOVE → RECENT TURNS → OUTPUT
    """
    group_section = f"\nGROUP STATE\n{group_state}\n" if group_state.strip() else ""

    return f"""SPEAKER CARD
{speaker_card}

SHARED FACTS — cite only attributes listed here; do not invent others
{relevant_options}
{group_section}
YOUR MOVE
{move_instruction}

RECENT TURNS
{local_context}

OUTPUT
{output_contract}"""


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
