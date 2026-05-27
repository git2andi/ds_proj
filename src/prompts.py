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

def phase_instruction_text(phase: str, add_brevity: bool, has_voted: bool = False) -> str:
    if phase == "narrowing":
        text = (
            "Say which option you lean to. One short line. "
            "If someone already said the same option, just back them with a few words."
            if not has_voted else
            "You already said your pick. Don't repeat 'I prefer Option X'. "
            "Add one quick condition, or just react to what someone else said."
        )
    else:
        text = {
            "greeting": (
                "Casual hello. One short line. Don't introduce yourself, don't bring up the topic yet."
            ),
            "opening": (
                "Quick first reaction. What jumps out, briefly. "
                "No essay, no full case -- one or two short sentences max."
            ),
            "negotiation": (
                "Real chat. If someone asked something, answer it briefly. "
                "If you agree, a short 'yeah' or 'fair' is fine -- don't justify just to fill space. "
                "If you disagree, say it plainly in a line. "
                "Don't restate your case. Use only what's in the option text."
            ),
            "emergence": (
                "Group's trying to land. One short reaction: "
                "is this workable for you, what would make it workable, or what still bothers you. "
                "Pick one and keep it short."
            ),
            "confirmation": (
                "Quick yes or no. A bare 'yes' or 'yeah ok' is fine. "
                "Only say no if you have a real objection."
            ),
            "closure": "Short goodbye. Nothing else.",
        }.get(phase, "React naturally and briefly.")

    if add_brevity:
        text += " Add only one new thing -- don't re-explain."
    return text


# =============================================================================
# Interaction instruction (computed from dialogue state)
# =============================================================================

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
    last_claim_speaker: Optional[str] = None,
) -> str:
    parts: list[str] = []

    if last_has_question:
        parts.append(" Answer the question first, briefly. If the options don't say enough, say so.")
    elif last_claim_speaker:
        # No pending question -- the obligation is to react to the last claim
        parts.append(
            f" {last_claim_speaker} just made a point. React to it specifically -- "
            f"agree with one detail, push back on one claim, or name a trade-off they're glossing over. "
            f"Don't just restate your own position."
        )

    if question_count >= 3:
        parts.append(" Too many questions floating. Don't ask another -- make a claim or pick a side.")

    if compromise_option:
        if compromise_in_acceptable:
            parts.append(
                f" Moderator is testing Option {compromise_option}. "
                f"Say if you could live with it. One short line, optional condition."
            )
        else:
            parts.append(
                f" Moderator is testing Option {compromise_option}. "
                f"If it doesn't work, say no plus the one specific reason."
            )

    if rejected_option and rejecting_self:
        if turns_since_rejection >= escalation_threshold:
            parts.append(
                f" You've kept blocking Option {rejected_option}. "
                f"Now: name the single dealbreaker, or say what one condition would unblock it. "
                f"Don't open new objections."
            )
        else:
            parts.append(
                f" You just rejected Option {rejected_option}. One line on the main blocker."
            )

    if own_last_was_question:
        parts.append(" Your last turn was a question -- don't ask another. Make a claim instead.")

    if speculative_count >= 2:
        parts.append(" Stop the 'what if' hypotheticals. Direct claim only.")

    if len(repeated_kws) >= 2:
        kw_str = ", ".join(repeated_kws[:3])
        parts.append(
            f" You've said the same thing too many times ({kw_str}). Stop. "
            f"Do exactly one of: challenge a specific claim the other person made, "
            f"name a real trade-off of YOUR OWN option, or concede one point."
        )
    elif self_repeated:
        parts.append(
            " You just repeated yourself. Change move entirely: "
            "react to a specific claim someone else made, or concede one point."
        )

    return "\n" + "".join(parts) if parts else ""


# =============================================================================
# Position discipline
# =============================================================================

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
    if phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
        return ""

    if phase == "emergence":
        cond = f" Condition you care about: {concession_text}." if concession_text else ""
        if candidate and candidate_in_acceptable and not candidate_is_anchor:
            if high_agreeableness:
                return f"\n{prefix} Option {candidate} works for you.{cond} One short line."
            if low_agreeableness_or_high_neuroticism:
                return f"\n{prefix} Option {candidate} could work but you have one concern first.{cond}"
            return f"\n{prefix} Option {candidate} is acceptable.{cond} Soften briefly."
        if candidate and candidate_in_rejected:
            if can_soften:
                return (
                    f"\n{prefix} The group is leaning Option {candidate}.{cond} "
                    f"Decide now: name a condition that would let you accept it, or say it's a real dealbreaker."
                )
            return (
                f"\n{prefix} Option {candidate} is gaining ground but you don't like it. "
                f"Quick acknowledgment, then your one specific objection."
            )
        if candidate and candidate_is_anchor:
            return f"\n{prefix} Option {candidate} is gaining ground -- your pick. Help it land lightly."
        return f"\n{prefix} Group is moving toward resolution. Let your position soften if it really is."

    if phase == "negotiation":
        return (
            f"\n{prefix} You've said this already -- don't repeat it. "
            f"Engage with what the others are actually arguing: challenge one specific claim they made, "
            f"concede a real trade-off, or name one genuine weakness of Option {anchor} itself."
        )

    # confirmation — the candidate is the option the moderator is testing.
    # Honor the sim's actual beliefs: if it's in their acceptable list, they
    # should confirm; if rejected, refuse with reason; otherwise hold the lean.
    if phase == "confirmation":
        if candidate and candidate_in_rejected:
            return (
                f"\n{prefix} The moderator is asking about Option {candidate}, which is on your "
                f"rejected list. Say no and name the one specific reason."
            )
        if candidate and (candidate_is_anchor or candidate_in_acceptable):
            return (
                f"\n{prefix} The moderator is asking about Option {candidate}, which works for you. "
                f"Just say yes (a short 'yeah' / 'works' / 'ok' is fine)."
            )
        # candidate exists but is neither acceptable nor rejected for this sim
        if candidate:
            return (
                f"\n{prefix} The moderator is testing Option {candidate}, which is not your pick "
                f"and not a clear yes for you. Hedge briefly or name what you'd need."
            )
        if flips == 0:
            return f"\n{prefix} Hold this unless a new point genuinely changes it."
        return f"\n{prefix} Already switched. Commit."

    # narrowing — candidate-aware soft cooperation when it works for them
    if candidate and candidate_in_acceptable and not candidate_is_anchor:
        return (
            f"\n{prefix} Option {candidate} is gaining traction and is also acceptable to you. "
            f"If you shift to it, say ONE brief reason why -- don't just say yes with no explanation."
        )
    if flips == 0:
        return f"\n{prefix} Hold this unless a new point genuinely changes it."
    if flips == 1:
        return f"\n{prefix} Already switched once. Hold -- or if shifting again name the specific reason."
    return f"\n{prefix} You've switched {flips} times. Commit and stay."


def narrowing_lines() -> list[str]:
    """Natural moderator lines to nudge the group toward stating preferences."""
    return [
        "ok so where's everyone landing?",
        "alright, what are we thinking?",
        "let's go around -- which one for each of you?",
        "what's everyone leaning toward at this point?",
        "ok let's see where we stand -- which option?",
        "where's everyone at right now?",
        "let's hear it -- one option each, backup if you're unsure",
    ]


def closure_templates(option: str) -> list[str]:
    return [
        f"Option {option} works. Bye!",
        f"Option {option} it is. See you!",
        f"Sounds good. Option {option}. Bye!",
        f"Alright, Option {option}. Cheers.",
    ]


# =============================================================================
# Setup prompts (options, roles, personas, beliefs)
# =============================================================================

def option_generation(topic: str) -> str:
    return f"""You are preparing a small-group chat about a decision.

Topic: {topic}

Create four distinct options for the group to discuss.

Option requirements:
- Exactly 4 options, A-D.
- Each on one compact line: label, main upside, one trade-off, who it best fits.
- Where applicable, include a concrete example name in the label (e.g. a real-
  sounding venue/restaurant/book/destination/product name), not just a generic
  category. Example: "Downtown Hotel (The Marriott)" rather than just
  "Downtown Hotel". Make up plausible names; do not use placeholders.
- For abstract topics where naming doesn't apply, skip the example name.
- Infer sensible details; do not invent specific numbers.
- Make the four options meaningfully different.
- Include one discussable practical detail per option (time, cost, effort, risk, fit, tone).

Opening question:
- One short, natural moderator question.
- Invite priorities, not a vote.

Return JSON only:
{{
  "options": [
    "Option A - [label, with example name where applicable]: [upside]; trade-off: [risk/cost]; best for: [priority/person]",
    "Option B - ...",
    "Option C - ...",
    "Option D - ..."
  ],
  "opening_question": "..."
}}"""


def names_and_roles(topic: str, n: int) -> str:
    """LLM generates N participant names AND roles together, tuned to the topic."""
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


def role_assignment(topic: str, names: list[str]) -> str:
    names_str = ", ".join(names)
    first = names[0]
    return f"""You are assigning roles for a group chat simulation.

Topic: {topic}
Participants: {names_str}

Give each participant one natural role. Exactly one is the primary person (decision affects them most).

Return JSON only:
{{
  "roles": {{
    "{first}": {{"role": "1-4 word phrase", "is_primary": true}},
    "OTHER": {{"role": "1-4 word phrase", "is_primary": false}}
  }}
}}

Rules:
- Every participant appears exactly once.
- Roles sound like real relationships, not job titles unless required.
- Exactly one "is_primary": true."""


def persona_concept(topic: str, name: str, role: str, is_primary: bool,
                    trait_description_block: str) -> str:
    primary_note = (
        f"{name} is the central person -- the decision affects them most directly."
        if is_primary else f"{name} is involved, but not the central person."
    )
    return f"""You are creating a participant profile for a group chat.

Topic: {topic}
Participant: {name}
Role: {role}
{primary_note}

Big Five traits (treat as tendencies, not stereotypes):
{trait_description_block}

Return JSON only:
{{
  "backstory": "2 short sentences with one specific detail relevant to the topic.",
  "goal": "One sentence in third person -- what {name} wants from the chat."
}}

Rules:
- Specific to the topic.
- Reflect traits subtly; don't name them.
- One concrete reason they might resist at least one option.
- No caricature."""


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


def agent_beliefs(name: str, role: str, goal: str, backstory: str,
                  personality_summary: str, options_text: str) -> str:
    return f"""You are building a private preference model for a chat participant.

Participant: {name}
Role: {role}
Goal: {goal}
Background: {backstory}
Personality: {personality_summary}

Options:
{options_text}

Decide what they would naturally prefer before the chat. Real people are flexible:
they have a top pick but can live with a few others.

Rules:
- preferred: the option that fits goal + concern best.
- acceptable: a wide list. Include preferred PLUS every option that doesn't clearly
  conflict with the concern. Aim for 3 acceptable options out of 4. Only drop one
  if it actively works against them.
- rejected: typically empty. Only fill if something genuinely violates a hard line.
- key_concern: short phrase; must be consistent with preferred.
- concession: concrete condition under which a non-preferred option becomes OK.

Return JSON only:
{{
  "preferred": "A|B|C|D",
  "acceptable": ["A", "B", "C"],
  "rejected": [],
  "key_concern": "short phrase",
  "concession": "concrete condition"
}}"""


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
# Turn prompt -- compact speaker-card
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

    return f"""You are writing the next message in a casual group chat among friends.

Voice rules (apply every turn):
- Friends chatting, not a panel. Match how people actually text.
- MIX LENGTHS naturally. Most turns are short (a few words, a fragment). When you're actually explaining a real point, a 2-3 sentence message is fine. Avoid same-length-every-turn -- that's robotic.
- Punctuation: use it when it helps -- a comma, a period, an exclamation, a question mark. Skipping the final period or capital is fine. Don't avoid punctuation on purpose; just don't force it where it'd feel stiff.
- Casing is your call: lowercase, sentence case, mixed -- whatever fits. Contractions are natural ("don't", "i'm", "won't").
- Filler words and reactions are welcome when natural ("tbh", "ngl", "lol", "ok so", "yeah but", "wait", "hmm").
- If you agree, a short "yeah" / "fair" / "ok" / "same" is enough -- don't justify what nobody disputed.
- If you disagree, say it plainly. One real reason beats three weak ones.
- Don't restate your option-plus-reason after you've said it. Vary your move: react to someone, ask one thing, concede, push back, name a downside of your own pick.
- No corporate-speak. No "great point", "absolutely", "I completely agree". No name prefix. No markdown. No em dashes.
- Only use facts that appear in the option text below. Do not invent specifics.

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
# Moderator prompts
# =============================================================================

def moderator_intervention(
    topic: str,
    participant_names: list[str],
    recent_dialogue: str,
    reason: str,
    target_participant: Optional[str] = None,
    escalation_level: int = 0,
) -> str:
    target_note = (f"\nDraw {target_participant} into the conversation."
                   if target_participant else "")
    escalation = {
        0: f"Point at the unresolved issue, ask {target_participant or 'them'} for one concrete answer.",
        1: f"Ask {target_participant or 'them'} for one concrete condition that would unblock things.",
        2: "Be firmer, still conversational: ask for a decision-relevant answer.",
        3: "Ask for a final position. One line.",
    }.get(escalation_level, "Point at the unresolved issue, ask for one concrete answer.")

    return f"""Neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}
Situation: {reason}{target_note}
Approach: {escalation}

Recent:
{recent_dialogue}

Write ONE short moderator line. Casual, no recap, no favoring. <=20 words.

Do NOT write any participant attribution like "Name: ..." -- you are the moderator, write only your own line.
Return only the line."""


def moderator_stall(
    topic: str,
    participant_names: list[str],
    recent_dialogue: str,
    current_votes: dict,
    escalation_level: int = 1,
) -> str:
    """Group is stuck. Move things forward without forcing yet."""
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


def moderator_emergence(
    topic: str,
    participant_names: list[str],
    options: list[str],
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


def moderator_compromise_test(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
    compromise_option: str,
    holdout_names: list[str],
) -> str:
    holdout_text = ", ".join(holdout_names) if holdout_names else "the group"
    return f"""Neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}
Possible compromise: Option {compromise_option}
People who need to react: {holdout_text}

Recent:
{recent_dialogue}

Test Option {compromise_option} as the compromise. Ask for yes-with-condition or no-with-specific-objection.

Strict rules:
- Reference only Option {compromise_option}. Do NOT combine it with another option.
- Do NOT invent a new option, hybrid, or attribute not in the original options.
- One sentence. <=24 words. Don't declare it final.

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
