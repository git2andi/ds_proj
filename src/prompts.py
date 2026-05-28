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
        return ("Now say which option you actually lean toward, and one honest reason "
                "(use one of your real reasons, not a generic one). Name it (its name "
                "or letter) so it's clear. A sentence is plenty.")
    if phase == "closure":
        if final_option:
            return (f"The group landed on Option {final_option}. Sign off in one natural line -- "
                    f"warm if it was your pick, gracious if it wasn't. Don't re-argue anything.")
        return "Wrap up with a short, natural goodbye. One line."
    return {
        "opening": (
            "The discussion is just starting. Say a quick, natural hello and the ONE thing that "
            "matters most to YOU for this decision -- your priority, worry, or what you're hoping "
            "for. Talk about that, not the specific options yet. A sentence or two, in your own voice."
        ),
        "negotiation": (
            "This is the discussion phase. You should be talking ABOUT THE OPTIONS BY NAME -- "
            "which one fits which priority, which one's trade-off worries you, which one's upside "
            "you'd take, why one beats another for the group. Don't trade abstract priority words "
            "('depth', 'flexibility', 'safety', 'engagement') back and forth -- every point you "
            "make should be tied to a SPECIFIC OPTION (by its name or letter). If someone made a "
            "claim about an option, engage with that claim: build, push back on one specific "
            "point, name a trade-off they glossed over. If you have a position, state it as a "
            "CLAIM, not as a rhetorical 'what about X?' question -- those are dodges. You may "
            "use your own knowledge as a reason; do NOT invent specific details about the options."
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
    open_challenge_from: Optional[str] = None,
) -> str:
    parts: list[str] = []

    if open_challenge_from:
        parts.append(
            f" {open_challenge_from} pushed back on something you said. Engage with that "
            f"directly -- concede the point, defend it with a specific reason, or name what would "
            f"change your mind."
        )
    elif last_has_question:
        parts.append(" There's an open question -- answer it directly first. If the options "
                     "don't say enough to answer, say that plainly.")
    elif last_claim_speaker:
        parts.append(
            f" {last_claim_speaker} just made a point. Engage with it specifically: agree with a "
            f"detail, push back on one claim, or name a trade-off they glossed over. Don't pivot "
            f"back to your own position without addressing theirs."
        )

    if repetition_high:
        parts.append(
            " This point has been made already -- don't repeat it. Either move toward a decision, "
            "concede something, or introduce a genuinely new angle."
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
    if phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
        return ""

    cond = f" The condition that would move you: {reconsider_text}." if reconsider_text else ""

    if phase in ("emergence", "confirmation", "narrowing") and candidate:
        if candidate_is_anchor or candidate_in_acceptable:
            return (f"\nYou lean Option {anchor}, and Option {candidate} works for you. "
                    f"Say so briefly and concretely.{cond}")
        if candidate_in_rejected:
            return (f"\nYou lean Option {anchor}. Option {candidate} is one you can't accept -- "
                    f"say no and give the one specific reason.{cond}")
        return (f"\nYou lean Option {anchor}. Option {candidate} isn't your pick and isn't a clear "
                f"yes for you. Don't just block it: say what one change would make it acceptable.")

    return (f"\nYou lean Option {anchor}. You've made that point -- don't restate it. Engage with "
            f"what others are arguing: challenge a specific claim, concede a real trade-off, or "
            f"acknowledge an honest weakness of Option {anchor} itself.")


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
    that ARE the thing being chosen. Concrete picks get real names; abstract
    picks get descriptive options without forced proper nouns."""
    return f"""You are preparing a small-group chat about a decision.

Topic: {topic}

STEP 1 -- classify the decision:
  - "concrete_pick"  : the group is choosing one of N real-world items
                       (restaurant, hotel, book, product, movie, destination).
  - "abstract_pick"  : the group is choosing a topic, approach, strategy,
                       theme, policy, or angle (a presentation topic, a research
                       direction, a marketing approach, a study method).

STEP 2 -- generate four options that ARE valid answers to the literal decision.

Hard rules:
- Each option line is the ONLY information the group will have. Self-contained
  and decision-ready.
- For "concrete_pick": each option gets a real-sounding proper name (a venue,
  product, etc.). Example: "The Marriott Downtown".
- For "abstract_pick": do NOT force a proper-noun venue. The option IS the
  topic/approach/theme. Example for "biology presentation topic":
    "CRISPR gene-editing ethics" -- NOT "The Helix Institute".
  Example for "marketing strategy":
    "Influencer partnerships on Instagram" -- NOT "The Brandify Agency".
- Each option must carry 2-3 GENUINELY CONTESTABLE dimensions: one real upside,
  one real trade-off, and the kind of priority/audience it suits. Trade-offs in
  qualitative terms ("higher cost", "narrower scope", "harder to research"); do
  NOT invent numbers, prices, or dates.
- The four options must be MEANINGFULLY DIFFERENT -- each upside should serve a
  distinct priority a reasonable person could weight differently.
- "best for" is a priority phrase ("best for: depth-over-breadth"), never an
  invented person's name.

Opening question:
- One short, natural moderator question inviting priorities (not a vote).

Return JSON only:
{{
  "decision_kind": "concrete_pick" | "abstract_pick",
  "options": [
    "Option A - [Name or Topic]: [upside]; trade-off: [qualitative]; best for: [priority]",
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
- A 2-sentence backstory with ONE CONCRETE, SPECIFIC EXPERIENCE tied to the topic
  (something they did or saw -- this experience will later become argumentative
  evidence). Avoid generic interests.
- One sentence describing what they want from the chat (third person).

Rules:
- Specific to the topic; reflect traits subtly; no caricature.
- The concrete experience should be something they could genuinely cite in an
  argument (a project, an event, a course, a trip, an incident).
- One concrete reason they could resist at least one option, drawn from that
  experience.

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
- Vary your length naturally. Most turns are a sentence or two; a quick "yeah, that works for me" is fine when you agree; go to three sentences only when you're actually explaining something.
- Contractions are natural ("I'd", "don't", "it's"). A light filler now and then is okay, but don't lean on slang.
- If you agree, say so briefly and move on -- don't manufacture reasons.
- If you disagree, say it plainly with one real reason. One good reason beats three weak ones.
- Don't restate a point you (or someone else) already made. Each turn should add something: an answer, a reaction to a specific claim, a new consideration, or a decision.
- When you mean a specific option, name it (its name or letter) so others know which one.
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
