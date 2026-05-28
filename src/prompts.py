"""
prompts.py
----------
Single registry of every LLM-facing template. All prose lives here.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional


# =============================================================================
# Phase instructions (injected into sim_turn_compact)
# =============================================================================

def phase_instruction_text(phase: str, has_voted: bool = False,
                           final_option: Optional[str] = None) -> str:
    if phase == "narrowing":
        if has_voted:
            return ("You already named your pick. Don't repeat it. Either react to "
                    "someone else's pick, raise one new consideration, or note a condition.")
        return ("Now say which option you actually lean toward, and one honest reason. "
                "Name it (its name or letter) so it's clear. A sentence is plenty.")
    if phase == "closure":
        if final_option:
            return (f"The group landed on Option {final_option}. Sign off in one natural line — "
                    f"warm if it was your pick, gracious if it wasn't. Don't re-argue anything.")
        return "Wrap up with a short, natural goodbye. One line."
    return {
        "opening": (
            "The discussion is just starting. Say a quick, natural hello and the ONE thing that "
            "matters most to YOU for this decision — your priority, worry, or what you're hoping for. "
            "Talk about that, not the specific options yet. A sentence or two, in your own voice."
        ),
        "negotiation": (
            "This is open discussion, not a vote. React to what someone actually said: build on it, "
            "question it, weigh it against what you care about, or push back on one point. Bring options "
            "in only when they're relevant to the point being made. Don't recite your position or repeat "
            "a point already made. Use only details from the option text."
        ),
        "emergence": (
            "The group is closing in on one option. Say whether it works for you, what would make it "
            "work, or the one thing still bothering you. Add something new, don't restate."
        ),
        "confirmation": (
            "Give a clear yes or no on the option on the table, in your own words. "
            "Only say no if you have a real, specific objection."
        ),
    }.get(phase, "React naturally and briefly.")


# =============================================================================
# Interaction instruction (computed from dialogue state)
# =============================================================================

def interaction_instruction_block(
    last_has_question: bool,
    last_claim_speaker: Optional[str],
    repetition_high: bool,
) -> str:
    parts: list[str] = []

    if last_has_question:
        parts.append(" There's an open question — answer it directly first. If the options "
                     "don't say enough to answer, say that plainly.")
    elif last_claim_speaker:
        parts.append(
            f" {last_claim_speaker} just made a point. Engage with it specifically: agree with a "
            f"detail, push back on one claim, or name a trade-off they glossed over. Don't pivot "
            f"back to your own position without addressing theirs."
        )

    if repetition_high:
        parts.append(
            " This point has been made already — don't repeat it. Either move toward a decision, "
            "concede something, or introduce a genuinely new angle."
        )

    return "\n" + "".join(parts) if parts else ""


# =============================================================================
# Position discipline (collapsed to 3 templates)
# =============================================================================

def position_discipline_block(
    phase: str,
    anchor: str,
    candidate: Optional[str],
    candidate_in_acceptable: bool,
    candidate_in_rejected: bool,
    candidate_is_anchor: bool,
    concession_text: str,
) -> str:
    if phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
        return ""

    cond = f" The condition that matters to you: {concession_text}." if concession_text else ""

    # Decision phases — honour beliefs about the candidate on the table.
    if phase in ("emergence", "confirmation", "narrowing") and candidate:
        if candidate_is_anchor or candidate_in_acceptable:
            return (f"\nYou lean Option {anchor}, and Option {candidate} works for you. "
                    f"Say so briefly and concretely.{cond}")
        if candidate_in_rejected:
            return (f"\nYou lean Option {anchor}. Option {candidate} is one you can't accept — "
                    f"say no and give the one specific reason.{cond}")
        return (f"\nYou lean Option {anchor}. Option {candidate} isn't your pick and isn't a clear "
                f"yes for you. Don't just block it: say what one change would make it acceptable.")

    # Negotiation without a settled candidate yet.
    return (f"\nYou lean Option {anchor}. You've made that point — don't restate it. Engage with "
            f"what others are arguing: challenge a specific claim, concede a real trade-off, or "
            f"acknowledge an honest weakness of Option {anchor} itself.")


def narrowing_lines() -> list[str]:
    return [
        "Okay, where is everyone landing — which option do you each lean toward?",
        "Let's go around: name your pick and why, briefly.",
        "What's everyone leaning toward at this point?",
        "Time to narrow it down — which option works best for each of you?",
        "Let's hear it: one option each, and a backup if you're unsure.",
    ]


# =============================================================================
# Setup prompts (options, names+roles, persona concepts, beliefs)
# =============================================================================

def option_generation(topic: str) -> str:
    return f"""You are preparing a small-group chat about a decision.

Topic: {topic}

Create four distinct options for the group to discuss.

Each option line is the ONLY information the group will have — they cannot look
anything up. So every line must be self-contained and decision-ready.

Option requirements:
- Exactly 4 options, A-D.
- Each on one compact line: a concrete name, its main upside, one trade-off, and
  the kind of priority it serves.
- Give each a real-sounding proper name where it fits the topic (a venue, book,
  product, airline, destination). Example: "The Marriott Downtown" not just
  "Downtown Hotel". Make up plausible names; never use placeholders.
- Describe "best for" by the PRIORITY it suits ("best for: travelers who value
  cost"), never by inventing a person's name.
- State trade-offs in relative, qualitative terms ("higher cost", "longer
  layover", "slower paced"). Do NOT invent specific numbers, prices, dates, or
  times — the group must decide on these qualitative descriptions.
- Make the four meaningfully different; each upside should be a distinct priority.

Opening question:
- One short, natural moderator question that invites priorities, not a vote.

Return JSON only:
{{
  "options": [
    "Option A - [Proper Name]: [main upside]; trade-off: [qualitative cost/risk]; best for: [the priority it serves]",
    "Option B - ...",
    "Option C - ...",
    "Option D - ..."
  ],
  "opening_question": "..."
}}"""


def names_and_roles(topic: str, n: int) -> str:
    return f"""You are casting participants for a small-group chat simulation.

Topic: {topic}
Number of participants: {n}

Pick {n} natural first names that fit the topic's vibe (workplace topics lean
professional; friend/social topics lean casual; family topics may include
nicknames). Don't pick the same name twice. Avoid stereotypes.

Assign each one a natural role tied to the decision. Exactly one participant
is the primary -- the person most directly affected.

Return JSON only:
{{
  "participants": [
    {{"name": "Firstname", "role": "1-4 word relationship", "is_primary": true}},
    {{"name": "Firstname", "role": "1-4 word relationship", "is_primary": false}}
  ]
}}

Rules:
- Exactly {n} entries.
- Roles sound like real relationships, not job titles unless the topic requires it.
- Exactly one "is_primary": true.
- Distinct first names, no duplicates.
- CRITICAL: if the topic names a specific person as the SUBJECT of the decision
  (e.g. "plan a party for Steve", "birthday trip for Mia", "gift for dad"),
  do NOT include that person as a participant -- they are the subject, not a
  planner. The participants are the people MAKING the decision."""


def persona_group_generation(topic: str, names_roles_traits: list[dict]) -> str:
    participants_block = ""
    for entry in names_roles_traits:
        primary_note = ("(primary)" if entry["is_primary"] else "(not central)")
        participants_block += (
            f"\n{entry['name']} -- {entry['role']} {primary_note}\n"
            f"Traits:\n{entry['trait_description_block']}\n"
        )
    names_list = ", ".join(e["name"] for e in names_roles_traits)
    return f"""You are creating participant profiles for a group chat.

Topic: {topic}
Participants: {names_list}
{participants_block}
For each participant write:
- A 2-sentence backstory with one concrete detail tied to the topic.
- One sentence describing what they want from the chat (third person).

Rules:
- Specific to the topic; reflect traits subtly; no caricature.
- One concrete reason they could resist at least one option.

Return JSON only:
{{
  "personas": {{
    "{names_roles_traits[0]['name']}": {{
      "backstory": "...",
      "goal": "..."
    }}
  }}
}}

Include all {len(names_roles_traits)} participants."""


def agent_beliefs_group(topic: str, personas_text: str, options_text: str) -> str:
    return f"""You are building private preference models for chat participants.

Topic: {topic}

Participants:
{personas_text}

Options:
{options_text}

Real people are flexible: they have a top pick but can live with most of the
others. For each participant:
- preferred: best fit for their goal + concern.
- acceptable: a wide list. Include preferred PLUS every option that doesn't
  clearly conflict with the concern. Aim for 3 of the 4 options.
- rejected: typically empty. Only fill when an option genuinely violates a
  hard line for that person.
- key_concern: short phrase, consistent with preferred.
- concession: concrete condition for accepting a non-preferred option.

Return JSON only:
{{
  "beliefs": {{
    "NAME": {{
      "preferred": "A|B|C|D",
      "acceptable": ["A"],
      "rejected": [],
      "key_concern": "short phrase",
      "concession": "concrete condition"
    }}
  }}
}}

Include every participant listed."""


# =============================================================================
# Turn prompt — compact speaker-card
# =============================================================================

def sim_turn_compact(
    speaker_card: str,
    relevant_options: str,
    group_state: str,
    local_context: str,
    move_instruction: str,
    output_contract: str,
) -> str:
    """Compact turn prompt. ~300-500 input tokens."""
    group_section = f"\nGROUP STATE\n{group_state}\n" if group_state.strip() else ""

    return f"""You are writing the next message in a relaxed but real group chat where a few people are deciding something together.

Voice rules (apply every turn):
- Talk like a real person in a group chat — relaxed, but you can be articulate. Not a formal panel, not exaggerated slang.
- Use normal punctuation and capitalization. Full sentences are fine; so is the occasional fragment. Write the way a thoughtful adult texts.
- Vary your length naturally. Most turns are a sentence or two; a quick "yeah, that works for me" is fine when you agree; go to three sentences only when you're actually explaining something.
- Contractions are natural ("I'd", "don't", "it's"). A light filler now and then is okay, but don't lean on slang.
- If you agree, say so briefly and move on — don't manufacture reasons.
- If you disagree, say it plainly with one real reason. One good reason beats three weak ones.
- Don't restate a point you (or someone else) already made. Each turn should add something: an answer, a reaction to a specific claim, a new consideration, or a decision.
- When you mean a specific option, name it (its name or letter) so others know which one.
- No corporate-speak ("great point", "absolutely", "I completely agree"). No name prefix. No markdown. No em dashes.
- Only use facts that appear in the option text below. Do not invent prices, names, or details.

SPEAKER CARD
{speaker_card}

OPTIONS -- use only what's listed
{relevant_options}
{group_section}
YOUR MOVE
{move_instruction}

RECENT TURNS
{local_context}

OUTPUT
{output_contract}"""


# =============================================================================
# Moderator prompts (single style — direct, conversational)
# =============================================================================

def moderator_stall(
    topic: str,
    participant_names: list[str],
    recent_dialogue: str,
    current_votes: dict,
    escalation_level: int = 1,
) -> str:
    counts = Counter(current_votes.values()) if current_votes else Counter()
    n = len(participant_names)
    top_count = counts.most_common(1)[0][1] if counts else 0
    has_majority = top_count > n / 2

    if has_majority and counts:
        top_opt = counts.most_common(1)[0][0]
        minority = [nm for nm, o in current_votes.items() if o != top_opt]
        split = f"Most lean Option {top_opt}; {' and '.join(minority)} hasn't moved."
    else:
        split = "Group is split."

    task = {
        1: f"{split} Ask the holdout(s) for one concrete condition or objection.",
        2: f"{split} Say one final compromise attempt is needed before you call it.",
        3: "Say the group hasn't agreed, you're making the call. Don't ask a question.",
    }.get(escalation_level, f"{split} Ask the holdout(s) for one concrete condition.")

    return f"""Neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}

Recent:
{recent_dialogue}

Your task: {task}

One sentence. <=22 words. Sound real, not formal.

Do NOT write any participant attribution like "Name: ..." -- you are the moderator, write only your own line.
Return only the line."""


def moderator_clarify_info(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
) -> str:
    options_block = "\n".join(options)
    return f"""You are the neutral moderator of a small group chat. The group is
fishing for a detail (a price, a number, a feature) to decide.

Topic: {topic}
Participants: {", ".join(participant_names)}

The options, with everything that is known about them:
{options_block}

Recent:
{recent_dialogue}

Answer the open question using ONLY the option descriptions above. If the detail
they want is in there, restate it plainly. If it is NOT in there, say clearly
that it isn't specified and that they should decide based on what's listed.
Never invent a number, price, date, or feature that isn't written above.

One or two sentences. <=28 words.

Do NOT write any participant attribution like "Name: ..." -- you are the moderator, write only your own line.
Return only the line."""


def moderator_emergence(
    topic: str,
    participant_names: list[str],
    recent_dialogue: str,
    candidate_option: str,
) -> str:
    return f"""Neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}
Option gaining traction: Option {candidate_option}

Recent:
{recent_dialogue}

Group is close but not settled. Invite anyone to name what would let them move forward, or what still bothers them.

One sentence. <=22 words. Don't declare a winner. Don't ask for a vote.

Do NOT write any participant attribution like "Name: ..." -- you are the moderator, write only your own line.
Return only the line."""


def moderator_force_close(
    topic: str,
    participant_names: list[str],
    final_option: str,
    recent_dialogue: str,
) -> str:
    return f"""Neutral moderator closing a chat that didn't reach consensus.

Topic: {topic}
Participants: {", ".join(participant_names)}
Final option: Option {final_option}

Recent:
{recent_dialogue}

Write ONE moderator line that:
- Acknowledges the group didn't fully agree.
- Makes the call: Option {final_option}.
- Sounds real, not ceremonial.

<=22 words.

Do NOT write any participant attribution like "Name: ..." -- you are the moderator, write only your own line.
Return only the line."""
