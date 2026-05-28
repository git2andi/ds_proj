"""
prompts.py
----------
Single registry of every LLM-facing template. All prose lives here.

Sections:
  - Phase / interaction / position-discipline blocks (turn-level)
  - Setup prompts (options, names+roles, persona concepts, beliefs)
  - The compact speaker-card turn prompt
  - Moderator templates (facilitator, narrowing, force-close)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


# =============================================================================
# Phase instructions (injected into sim_turn_compact)
# =============================================================================

def phase_instruction_text(phase: str, has_voted: bool = False,
                           final_option: Optional[str] = None) -> str:
    """Phase-level guidance.

    Update.md §4.1 / §4.5: stop forcing every turn to acknowledge + name an
    option + give pro/con + restate preference. Negotiation now permits any
    natural move; an option name is required ONLY when adding a real point
    about a specific option, voting, or clarifying a candidate.
    """
    if phase == "narrowing":
        if has_voted:
            return ("You already named your pick. Don't repeat it. Either react to "
                    "someone else's pick, raise one new consideration, or note a condition.")
        return ("Time to land. Say the option you lean toward and one honest reason "
                "(use one of your real reasons, not a generic one). One sentence is plenty.")
    if phase == "closure":
        if final_option:
            return (f"The group landed on Option {final_option}. Sign off in one natural line -- "
                    f"warm if it was your pick, gracious if it wasn't. Don't re-argue anything.")
        return "Wrap up with a short, natural goodbye. One line."
    return {
        "opening": (
            "The discussion is just starting. Say a quick, natural hello and the ONE thing that "
            "matters most to YOU -- your priority, worry, or what you're hoping for. Talk about "
            "that, not the specific options. A sentence or two, in your own voice."
        ),
        "negotiation": (
            "This is the discussion phase. You do NOT have to evaluate the last message. "
            "Reply only if you actually have something to add. Natural moves include:\n"
            "- a short yes or no reaction (\"yeah\", \"not sold\", \"that might work\");\n"
            "- a brief question if you're genuinely unsure about something;\n"
            "- a genuinely new reason (not a rephrase of a point already made);\n"
            "- a compromise framed as one (\"I still prefer X, but I can live with Y\");\n"
            "- moving the decision forward (\"then we're basically between A and D\");\n"
            "- or just push back plainly when you disagree.\n"
            "Do NOT open with \"valid point\", \"good point\", \"fair point\", \"I agree\", "
            "or \"X is right\" -- they make the chat sound robotic. Don't restate something you "
            "(or someone else) already said. Name an option only when you're making a real point "
            "about that specific option, voting, or clarifying a candidate -- not for every short "
            "reaction. You may use your own knowledge and lived experience as reasons; do NOT "
            "invent specific facts about the OPTIONS themselves."
        ),
        "emergence": (
            "The group is closing in on one option. Say whether it works for you, what would make it "
            "work, or the one thing still bothering you. Brief is fine."
        ),
        "confirmation": (
            "Give a clear yes or no in your own words. If yes and it isn't your top pick, you can "
            "say so honestly (\"I'd still prefer X but I can live with this\"). Only say no if you "
            "have a real, specific objection."
        ),
    }.get(phase, "React naturally and briefly.")


# =============================================================================
# Interaction instruction (computed from dialogue state)
# =============================================================================

def interaction_instruction_block(
    last_has_question: bool,
    last_claim_speaker: Optional[str],
    repetition_high: bool,
    open_challenge_from: Optional[str] = None,
) -> str:
    """Computed per-turn interaction guidance.

    Update.md §3.2 / §4.1: drop the unconditional 'engage with the last point'
    instruction. Only force engagement when there is a real obligation -- an
    unanswered question or an open challenge aimed at this speaker. Letting a
    point sit without explicit acknowledgement is natural and necessary.
    """
    del last_claim_speaker  # intentionally unused — no automatic-engagement push
    parts: list[str] = []

    if open_challenge_from:
        parts.append(
            f" {open_challenge_from} pushed back on something you said. Engage with that "
            f"directly -- concede, defend with a specific reason, or name what would change "
            f"your mind. Do not open with \"valid point\" or \"fair point\"."
        )
    elif last_has_question:
        parts.append(" There's an open question -- answer it directly first. If the options "
                     "don't say enough to answer, say that plainly.")

    if repetition_high:
        parts.append(
            " The thread has stalled -- the same point keeps coming back. Either move toward "
            "a pick (\"then we're basically between X and Y\"), say a short yes/no, or raise "
            "a genuinely new angle. Don't restate."
        )

    return "\n" + "".join(parts) if parts else ""


# =============================================================================
# Position discipline (3 branches)
# =============================================================================

def position_discipline_block(
    phase: str,
    anchor: str,
    candidate: Optional[str],
    candidate_in_acceptable: bool,
    candidate_in_rejected: bool,
    candidate_is_anchor: bool,
    reconsider_text: str,
) -> str:
    """Per-turn position guidance keyed by phase + private belief state.

    Update.md §4.7: when the candidate on the table is acceptable but is NOT
    the speaker's preferred, the prompt explicitly asks for a compromise frame
    ("I still prefer X, but I can live with Y"). This makes preference movement
    visible instead of looking like a sudden flip.
    """
    if phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
        return ""

    cond = f" The condition that would move you: {reconsider_text}." if reconsider_text else ""

    if phase in ("emergence", "confirmation", "narrowing") and candidate:
        if candidate_is_anchor:
            return (f"\nYou lean Option {anchor}, and that's what's on the table. Say so briefly "
                    f"and concretely.")
        if candidate_in_acceptable:
            # Compromise framing -- explicit. Update.md §4.7.
            return (f"\nYou still prefer Option {anchor}, but Option {candidate} is something "
                    f"you can live with. Frame it as a compromise -- e.g. \"I'd still prefer "
                    f"{anchor}, but I can live with {candidate}\" -- so it doesn't look like "
                    f"a flip. Brief.{cond}")
        if candidate_in_rejected:
            return (f"\nYou lean Option {anchor}. Option {candidate} is one you can't accept -- "
                    f"say no and give the one specific reason.{cond}")
        return (f"\nYou lean Option {anchor}. Option {candidate} isn't your pick and isn't a clear "
                f"yes for you. Don't just block it: say what one change would make it acceptable, "
                f"or hedge honestly.")

    # Negotiation phase without a fixed candidate.
    return (f"\nYou lean Option {anchor}. You've already made that point -- don't restate it. "
            f"Either add something new, react briefly, or stay quiet on it.")


# =============================================================================
# Surface-move hints (update.md §4.2) -- short, prose-only nudges that bias the
# next turn toward a specific natural shape. The probabilistic sampler in
# prompt_context.pick_surface_move_kind() decides whether to inject one.
#
# Hard rule: prose lives here, not in prompt_context. The sampler only chooses
# which KIND to use; the prose comes from this table.
# =============================================================================

_SURFACE_MOVE_HINTS: dict[str, str] = {
    "ack_only": (
        "A brief one-line yes/agreement is fine if you basically agree -- don't "
        "manufacture a reason."
    ),
    "short_no": (
        "A brief plain \"not sold\" / \"not for me\" / \"still not convinced\" is fine "
        "if you don't agree. Keep it short; don't justify unless asked."
    ),
    "question": (
        "If you're genuinely unsure about something, ask ONE short, useful question."
    ),
    "compromise": (
        "If you're moving toward something you don't actually prefer, mark it as a "
        "compromise: \"I still prefer X, but I can live with Y\"."
    ),
    "decision_move": (
        "Try moving the decision forward -- e.g. \"then we're basically between X and Y\" "
        "or \"can we rule that one out?\"."
    ),
    "new_reason": (
        "Add a genuinely NEW reason -- something nobody has said yet, drawn from your "
        "own knowledge or experience. Don't rephrase what's already on the table."
    ),
}


def surface_move_hint(kind: str) -> str:
    """Return the prose nudge for a surface-move kind, or empty string."""
    return _SURFACE_MOVE_HINTS.get(kind, "")


def narrowing_lines() -> list[str]:
    return [
        "Okay, where is everyone landing -- which option do you each lean toward?",
        "Let's go around: name your pick and why, briefly.",
        "What's everyone leaning toward at this point?",
        "Time to narrow it down -- which option works best for each of you?",
        "Let's hear it: one option each, and a backup if you're unsure.",
    ]


# =============================================================================
# Setup prompts (options, names+roles, persona concepts, beliefs)
# =============================================================================

def option_generation(topic: str) -> str:
    """Two-step generation: classify the decision kind, then produce options
    that ARE the thing being chosen.

    Update.md §4.6: options now carry FIVE bounded qualitative fields instead
    of three. Thin option cards (upside + tradeoff + best_for) gave sims so
    little to chew on that they kept rephrasing the same two attributes. The
    richer card -- adding a practical concern, a fit hint, and an uncertainty
    -- gives real material for disagreement without inviting fabricated
    numbers.
    """
    return f"""You are preparing a small-group chat about a decision.

Topic: {topic}

STEP 1 -- classify the decision:
  - "concrete_pick"  : the group is choosing one of N real-world items
                       (restaurant, hotel, book, product, movie, destination).
  - "abstract_pick"  : the group is choosing a topic, approach, strategy,
                       theme, policy, or angle (a presentation topic, a research
                       direction, a marketing approach, a study method).

STEP 2 -- generate four options that ARE valid answers to the literal decision.

Each option must be RICHER than just upside + tradeoff. Use this single-line
shape with semicolon-separated fields (do not break across lines):

  Option X - [Name or Topic]: [upside]; tradeoff: [qualitative]; concern: [practical worry]; fit: [who/when it suits]; uncertainty: [what's unclear]; best for: [priority]

Hard rules:
- Each option line is the ONLY information the group will have. Self-contained
  and decision-ready.
- For "concrete_pick": each option gets a real-sounding proper name (a venue,
  product, etc.). Example: "The Marriott Downtown".
- For "abstract_pick": do NOT force a proper-noun venue. The option IS the
  topic/approach/theme. Example for "biology presentation topic":
    "CRISPR gene-editing ethics" -- NOT "The Helix Institute".
- Use QUALITATIVE descriptors only. Do NOT invent numbers, prices, dates,
  durations, or fake percentages. Trade-offs and concerns are in plain words
  ("higher cost", "narrower scope", "less flexible if plans change").
- The four options must be MEANINGFULLY DIFFERENT -- each upside should serve a
  distinct priority a reasonable person could weight differently.
- "concern" is a real, practical worry distinct from the tradeoff (the tradeoff
  is the headline cost; concern is the second-order worry: "less flexibility if
  plans change", "depends on one person showing up", "social pressure to join").
- "fit" names the kind of priority or person this suits best -- "low-disruption
  travellers", "people who want depth", "anyone bringing kids".
- "uncertainty" is the honest thing the group can't be sure about from the
  option alone -- "baggage policy unclear", "weather dependent", "vendor
  reliability is variable".
- "best for" is a short priority phrase ("depth-over-breadth"), never a
  person's name.

Opening question:
- One short, natural moderator question inviting priorities (not a vote).

Return JSON only:
{{
  "decision_kind": "concrete_pick" | "abstract_pick",
  "options": [
    "Option A - [Name or Topic]: [upside]; tradeoff: [qualitative]; concern: [practical worry]; fit: [who/when it suits]; uncertainty: [what's unclear]; best for: [priority]",
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
- A 1-sentence backstory with ONE CONCRETE, SPECIFIC EXPERIENCE tied to the topic
  (something they did or saw -- this experience becomes argumentative evidence).
  Avoid generic interests. Keep it tight -- one focused sentence only.
- One sentence describing what they want from the chat (third person).

Rules:
- Specific to the topic; reflect traits subtly; no caricature.
- The concrete experience must be something they could cite in an argument
  (a project, an event, a course, a trip, an incident).
- The experience should hint at one reason they could resist at least one option.

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
    """Belief generation produces the Toulmin argument kit:
    preferred + acceptable + rejected + key_concern + reasons + reservation +
    would_reconsider_if. The deterministic divergence enforcement in persona.py
    then spreads preferred options after generation."""
    return f"""You are building private belief models for chat participants.

Topic: {topic}

Participants:
{personas_text}

Options:
{options_text}

Real people in a chat have a top pick, can live with some others, give CONCRETE
REASONS for what they prefer, raise honest concerns about rivals, and stay open
to updating when the right point is made. Give each participant:

- preferred: best fit for their goal + concern + backstory experience.
- acceptable: a moderately wide list. Include preferred PLUS the other options
  that don't clearly conflict with their concern. Aim for 2-3 options total.
- rejected: typically empty. Only fill when an option genuinely violates a hard
  line for that person.
- key_concern: short phrase summarizing what they're optimizing for.
- reasons: 1-2 CONCRETE reasons for `preferred`, phrased AS the participant's
  knowledge / experience / role. Each reason must be substantive enough to use
  in an argument -- not "I like it" or "it's good". Pull from their backstory.
- reservation: one HONEST concern about an option that is NOT their preferred
  -- a thing they'd want addressed before they could accept it. Frame it as a
  concern, not a refusal ("I'd worry about the depth", not "I refuse Option X").
- would_reconsider_if: the concrete thing that would move them off `preferred`.
  Must be specific and possible to satisfy.

IMPORTANT: It is GOOD for participants to start in different places. Different
preferred options across the group is desired -- the discussion exists to
reconcile real differences. Do NOT make them all prefer the same option.

Return JSON only:
{{
  "beliefs": {{
    "NAME": {{
      "preferred": "A|B|C|D",
      "acceptable": ["A", "B"],
      "rejected": [],
      "key_concern": "short phrase",
      "reasons": ["concrete reason from backstory", "second reason"],
      "reservation": "honest concern about a rival option",
      "would_reconsider_if": "concrete condition that would change my pick"
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
    memory_block: str,
    move_instruction: str,
    output_contract: str,
) -> str:
    """Compact turn prompt. ~400-600 input tokens with memory block.

    Voice rules permit world-knowledge warrants (Toulmin) but forbid invented
    option attributes (Stage 4 grounding). The memory block replaces a raw
    transcript dump with a relevance-filtered view (Park 2023 scaled down).
    """
    group_section = f"\nGROUP STATE\n{group_state}\n" if group_state.strip() else ""
    memory_section = f"\nYOUR MEMORY\n{memory_block}\n" if memory_block.strip() else ""

    return f"""You are writing the next message in a relaxed but real group chat where a few people are deciding something together.

Voice rules (apply every turn):
- Talk like a real person in a group chat -- relaxed, but articulate when needed. Not a formal panel, not exaggerated slang.
- Use normal punctuation and capitalization. Full sentences are fine; so is the occasional fragment. Write the way a thoughtful adult texts.
- Vary length naturally. Many turns are short -- a "yeah", a "not sold", a quick question. Go longer only when you're actually adding a real reason or explaining something. Two or three sentences is the upper end of normal.
- Contractions are natural ("I'd", "don't", "it's"). A light filler now and then is okay, but don't lean on slang.
- Don't open with "valid point", "good point", "fair point", "I agree", "X is right", or "that's a concern" -- they make the chat feel robotic.
- You don't have to evaluate the previous message. If you have nothing genuinely new to add, prefer a short reaction, a question, or just engaging with someone else's earlier point.
- If you agree, say so briefly and move on -- don't manufacture reasons.
- If you disagree, say it plainly with one real reason. One good reason beats three weak ones.
- Don't restate a point you (or someone else) already made. Each turn should add something: an answer, a reaction, a new consideration, a compromise, or a decision move.
- Name an option (its name or letter) only when you're making a real point about it, voting, or clarifying a candidate -- not for short reactions.
- No corporate-speak ("great point", "absolutely", "I completely agree"). No name prefix. No markdown. No em dashes.
- You may use your own knowledge and lived experience to back a point -- that's how real arguments work. Do NOT invent specific facts about the OPTIONS themselves (no fake prices, fake names of services, fake dates tied to an option).

SPEAKER CARD
{speaker_card}

OPTIONS
{relevant_options}
{group_section}{memory_section}
YOUR MOVE
{move_instruction}

RECENT TURNS
{local_context}

OUTPUT
{output_contract}"""


# =============================================================================
# Moderator prompts -- facilitator style (Stage 3)
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
        1: f"{split} Ask the holdout(s) for one concrete condition that would move them.",
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


def moderator_facilitate_disagreement(
    topic: str,
    participant_names: list[str],
    recent_dialogue: str,
    contested_summary: str,
) -> str:
    """Facilitator move: surface the live disagreement, ask sims to engage each
    other rather than restate. Used when two sims have stated conflicting
    priorities/positions without addressing each other."""
    return f"""Neutral facilitator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}

Live disagreement: {contested_summary}

Recent:
{recent_dialogue}

Surface the disagreement by name and ask them to engage each other's point
directly -- not restate their own. Don't pick a side. Don't summarise; nudge.

One sentence. <=24 words. Sound real, not formal.

Do NOT write any participant attribution like "Name: ..." -- you are the moderator, write only your own line.
Return only the line."""


def moderator_what_would_change_mind(
    topic: str,
    participant_names: list[str],
    target_name: str,
    target_position: str,
    recent_dialogue: str,
) -> str:
    """Facilitator move: ask a holdout what would change their mind. Routes
    directly into AgentBeliefs.would_reconsider_if so it's answerable."""
    return f"""Neutral facilitator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}

{target_name} has been holding {target_position}.

Recent:
{recent_dialogue}

Ask {target_name} directly what concrete thing would change their mind. Not "do
you agree" -- "what would move you". One sentence. <=22 words.

Do NOT write any participant attribution like "Name: ..." -- you are the moderator, write only your own line.
Return only the line."""


def moderator_reframe_missing_detail(
    topic: str,
    participant_names: list[str],
    options: list[str],
    recent_dialogue: str,
) -> str:
    """When the group is chasing a missing option attribute the options DON'T
    contain, reframe toward judgment instead of declaring it absent. Stage 3
    replacement for the blunt 'isn't specified, decide on what's listed' line."""
    options_block = "\n".join(options)
    return f"""You are the neutral facilitator of a small group chat. The group is
chasing a specific detail about the options.

Topic: {topic}
Participants: {", ".join(participant_names)}

The options, with everything that is known about them:
{options_block}

Recent:
{recent_dialogue}

If the detail is in the option text above, restate it plainly.
Otherwise, reframe the question into a judgment call grounded in what IS known.
Example reframings (do not copy verbatim):
  - "no exact figures here -- which matters more to you, X or Y?"
  - "the options don't pin that down. given that, what would tip your pick?"
Never invent a number, price, date, or feature that isn't written above.
Never end the line of inquiry with "decide based on what's listed" -- that
shuts the discussion down. Nudge them toward a judgment, not a verdict.

One sentence. <=28 words.

Do NOT write any participant attribution like "Name: ..." -- you are the moderator, write only your own line.
Return only the line."""


def moderator_ask_holdout(
    topic: str,
    participant_names: list[str],
    candidate_option: str,
    holdout_name: str,
    holdout_pick: Optional[str],
    recent_dialogue: str,
) -> str:
    """Update.md §4.8 -- targeted holdout question, NOT a generic group ask.

    Used at confirmation / split-vote moments. The moderator names the holdout
    and asks specifically whether they can live with the candidate, instead of
    "everyone good with Option X?" (which produces weak fake-consensus).
    """
    holdout_clause = (
        f"{holdout_name} picked Option {holdout_pick}"
        if holdout_pick else f"{holdout_name} hasn't landed yet"
    )
    return f"""Neutral moderator in a small group chat.

Topic: {topic}
Participants: {", ".join(participant_names)}

Most of the group is on Option {candidate_option}. {holdout_clause}.

Recent:
{recent_dialogue}

Ask {holdout_name} directly whether they could live with Option {candidate_option},
or whether it's a no for them. Don't ask the whole group. Don't editorialise --
just the question, by name, real and concrete.

One sentence. <=22 words. Sound real, not formal.

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


# =============================================================================
# Repair prompts -- called when verifier flags a generated turn for correction.
# Each prompt includes the specific issue and asks for one clean rewrite.
# =============================================================================

def repair_repetition(original_text: str) -> str:
    """Prompt for self-repetition: ask for a pivot, not a restatement."""
    return f"""A chat message was rejected because it repeats a point already made.

Original (rejected):
{original_text}

Write ONE natural chat message that says something new:
- React to what someone else just said, or
- Name a trade-off you haven't raised yet, or
- Move toward a pick.
Do NOT repeat the same reason or argument.
No name prefix. No markdown. Under 25 words.

Write only the rewritten message."""


def repair_ack_loop(original_text: str) -> str:
    """Update.md §4.3 -- repair an acknowledgement loop.

    Triggered when the turn opens with acknowledgement language AND the recent
    participant turns also do. Do NOT ban acknowledgement globally; only ask
    for a different move when the loop is forming.
    """
    return f"""A chat message was rejected because it only acknowledges the previous point,
and the last few messages already contain acknowledgement language. The chat is
turning into a loop of "fair point / I agree / valid concern".

Original (rejected):
{original_text}

Rewrite this as a DIFFERENT natural move (pick ONE):
- ask one short, useful question;
- say a brief yes or no ("not sold", "yeah, that works", "not for me");
- add a genuinely new reason (something NOT yet said);
- compromise explicitly ("I'd still prefer X, but I can live with Y");
- or push the decision forward ("then we're basically between A and D").

Do NOT start with "valid point", "good point", "fair point", "I agree",
"that's a concern", "X is right", or "makes sense".
No name prefix. No markdown. Under 22 words.

Write only the rewritten message."""


def repair_semantic_repeat(original_text: str, prior_point: str) -> str:
    """Update.md §4.4 -- repair a repeated point (same option + same attribute,
    rephrased). The current message restates a point the same speaker already
    made.
    """
    return f"""A chat message was rejected because the same speaker already made
this point (just rephrased).

Earlier point: "{prior_point}"

Original (rejected):
{original_text}

Rewrite this turn so it does NOT restate the same option-attribute argument.
Pick ONE genuinely new move:
- respond to what someone ELSE said, with a specific reason or pushback;
- name a trade-off you haven't raised;
- ask one short useful question;
- or move toward a pick.

No name prefix. No markdown. Under 22 words.

Write only the rewritten message."""


def repair_invalid_option(original_text: str, options: list[str]) -> str:
    """Prompt when an option reference is invalid or a valid option is denied."""
    options_block = "\n".join(options)
    return f"""A chat message was rejected because it referenced a non-existent option
or incorrectly claimed a listed option is unavailable.

The ONLY valid options are:
{options_block}

Original (rejected):
{original_text}

Rewrite the message so it only refers to the listed options and treats all of
them as genuinely available. No name prefix. Similar length.

Write only the rewritten message."""


def repair_vote(options: list[str]) -> str:
    """Prompt for a missing vote during narrowing."""
    letters = ", ".join(
        m.group(1).upper()
        for opt in options
        if (m := re.match(r"Option\s+([A-D])", opt, re.I))
    )
    options_block = "\n".join(options)
    return f"""A narrowing-phase message was rejected because it doesn't clearly vote
for one of the options.

Available options:
{options_block}

Write one natural chat message that names exactly one option ({letters}) as your pick.
Example: "I'd go with Option B." or "My pick is Option C -- it fits what I need."
No name prefix. Under 22 words.

Write only the message."""


def repair_confirmation(candidate: str) -> str:
    """Prompt for an unclear confirmation (must be yes or no)."""
    return f"""A confirmation message was rejected because it was unclear.

The group is deciding on Option {candidate}.

Write one clear response -- pick one:
  YES: "yeah, works for me" / "that's fine" / "I'm on board" / etc.
  NO: "no, [specific one-line reason]"

Only say no if you have a real objection. No name prefix. Under 18 words.

Write only the message."""


def repair_invented_fact(original_prompt: str) -> str:
    """Append to original prompt to repair an invented option attribute."""
    return (
        original_prompt
        + "\n\nIMPORTANT: Your previous response invented specific facts about an option "
        "(a fake price, number, or named feature not in the option text). "
        "Rewrite the same turn WITHOUT those invented details. "
        "You may use general knowledge and personal experience as reasons, "
        "but do NOT attach specific numbers or fake named features to any option."
    )


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
