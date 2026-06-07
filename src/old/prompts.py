"""
prompts.py
----------
Single registry of LLM-facing prose.
"""

from __future__ import annotations

import re
from typing import Optional

from config_loader import cfg


def _pc_int(name: str, default: int) -> int:
    return int(getattr(cfg.prompt_contracts, name, default))


def phase_instruction_text(
    phase: str,
    has_voted: bool = False,
    final_option: Optional[str] = None,
) -> str:
    if phase == "opening":
        return (
            "The discussion is starting. Say a quick hello and the one thing "
            "that matters most to you. Do not vote yet."
        )
    if phase == "negotiation":
        return (
            "This is the discussion phase. Add only something useful: a short "
            "reaction, a concrete reason, a direct answer, a small pushback, or "
            "a move toward deciding. Do not restate old points. Name an option "
            "only when making a real point about it. Do not ask people to call, "
            "check, look up, or wait for outside facts; use the listed facts."
        )
    if phase == "narrowing":
        if has_voted:
            return (
                "You already named your pick. Do not repeat it. React briefly "
                "or name a backup only if that adds something."
            )
        return "Name your current pick and one honest reason."
    if phase == "confirmation":
        return (
            "Give a clear yes or no. If yes and it is not your top pick, say "
            "briefly that it is a compromise. Only say no for a real blocker."
        )
    if phase == "closure":
        if final_option:
            return (
                f"The group landed on Option {final_option}. Sign off in one "
                "natural line. Do not re-argue."
            )
        return "Wrap up with one short natural line."
    return "React naturally and briefly."


def interaction_instruction_block(
    last_has_question: bool,
    last_claim_speaker: Optional[str],
    repetition_high: bool,
) -> str:
    del last_claim_speaker
    parts: list[str] = []
    if last_has_question:
        parts.append(
            "There is an open question. Answer it directly first, and do not "
            "ask another question."
        )
    if repetition_high:
        parts.append(
            "The thread is getting repetitive. Move toward a pick, give a short "
            "yes/no, or add a genuinely new angle."
        )
    return "\n".join(parts)


def position_discipline_block(
    phase: str,
    anchor: str,
    candidate: Optional[str],
    candidate_in_acceptable: bool,
    candidate_in_rejected: bool,
    candidate_is_anchor: bool,
    reconsider_text: str,
) -> str:
    if phase not in {"negotiation", "narrowing", "confirmation"}:
        return ""
    cond = f" What could move you: {reconsider_text}." if reconsider_text else ""
    if phase in {"narrowing", "confirmation"} and candidate:
        if candidate_is_anchor:
            return f"You prefer Option {anchor}, and it is on the table. Be clear and brief."
        if candidate_in_acceptable:
            return (
                f"You still prefer Option {anchor}, but Option {candidate} is "
                f"acceptable as a compromise. Make that honest.{cond}"
            )
        if candidate_in_rejected:
            return (
                f"You prefer Option {anchor}. Option {candidate} is not acceptable "
                f"to you; give the concrete blocker.{cond}"
            )
        return (
            f"You prefer Option {anchor}. Option {candidate} is not a clear yes "
            f"for you, so hedge honestly or say what would make it acceptable."
        )
    return (
        f"You lean Option {anchor}. Do not simply repeat that; add a new point, "
        "answer someone, or move the decision forward."
    )


def option_generation(topic: str) -> str:
    option_count = int(cfg.option_generation.option_count)
    attr_min = int(cfg.option_generation.attribute_count_min)
    attr_max = int(cfg.option_generation.attribute_count_max)
    score_min = int(cfg.option_generation.score_min)
    score_max = int(cfg.option_generation.score_max)
    return f"""Create a self-contained fictional decision scenario for a small-group chat.

Topic: {topic}

Classify the topic into one decision_kind:
flight_booking, hotel_booking, restaurant_choice, hiking_trip, travel_destination,
presentation_topic, study_or_work_plan, tool_or_product_choice,
game_or_activity_choice, generic_decision.

Generate {option_count} options that are valid answers to the literal decision.
Each option must be one line with this exact shape:

Option X - [Name or Topic]: attrs: key=value, key=value; upside: ...; tradeoff: ...; concern: ...; fit: ...; risk: ...; best for: ...

Rules:
- The option cards are the full fictional decision world.
- Concrete logistics topics need {attr_min}-{attr_max} stable attributes per option.
- Abstract topics need {attr_min}-{attr_max} scored dimensions using {score_min}-{score_max} scales.
- Do not use or imply live lookup facts: availability, waitlists, booking status,
  uncertain schedules, unknown policies, call-ahead checks, live weather, or
  anything the group must check outside the chat.
- The options must differ meaningfully across priorities.
- The concern and risk fields must be stable tradeoffs, not missing information.
- The opening question should invite priorities, not votes.

Return JSON only:
{{
  "decision_kind": "...",
  "options": [
    "Option A - ...",
    "Option B - ...",
    "Option C - ...",
    "Option D - ..."
  ],
  "opening_question": "..."
}}"""


def names_and_roles(topic: str, n: int) -> str:
    role_min = _pc_int("role_words_min", 1)
    role_max = _pc_int("role_words_max", 4)
    return f"""Cast participants for a small-group chat simulation.

Topic: {topic}
Number of participants: {n}

Pick {n} distinct natural first names. Assign each a {role_min}-{role_max} word
relationship/role tied to the decision. Exactly one participant is the primary,
meaning the person most directly affected.

If the topic names a person as the subject of the decision, do not include that
person as a participant.

Return JSON only:
{{
  "participants": [
    {{"name": "Firstname", "role": "relationship", "is_primary": true}},
    {{"name": "Firstname", "role": "relationship", "is_primary": false}}
  ]
}}"""


def persona_group_generation(topic: str, names_roles_traits: list[dict]) -> str:
    experience_sentences = _pc_int("persona_experience_sentences", 1)
    goal_sentences = _pc_int("persona_goal_sentences", 1)
    participants_block = ""
    for entry in names_roles_traits:
        primary_note = "(primary)" if entry["is_primary"] else "(not primary)"
        participants_block += (
            f"\n{entry['name']} -- {entry['role']} {primary_note}\n"
            f"Traits:\n{entry['trait_description_block']}\n"
        )
    names = ", ".join(e["name"] for e in names_roles_traits)
    first_name = names_roles_traits[0]["name"]
    return f"""Create participant profiles for a group chat.

Topic: {topic}
Participants: {names}
{participants_block}
For each participant write:
- A backstory of exactly {experience_sentences} sentence(s) with one concrete
  experience tied to the topic.
- A goal of exactly {goal_sentences} sentence(s), written in third person.

Keep them realistic, ordinary, and distinct. No caricatures.

Return JSON only:
{{
  "personas": {{
    "{first_name}": {{
      "backstory": "...",
      "goal": "..."
    }}
  }}
}}

Include all {len(names_roles_traits)} participants."""


def agent_beliefs_group(topic: str, personas_text: str, options_text: str) -> str:
    reasons_min = int(cfg.argument_kit.reasons_min)
    reasons_max = int(cfg.argument_kit.reasons_max)
    acceptable_min = int(cfg.divergence.target_acceptable_min)
    acceptable_max = int(cfg.divergence.target_acceptable_max)
    return f"""Build private belief models for chat participants.

Topic: {topic}

Participants:
{personas_text}

Options:
{options_text}

For each participant produce:
- preferred: one of A-D.
- acceptable: {acceptable_min}-{acceptable_max} options total, including preferred.
- rejected: usually empty; only use for a real hard line.
- key_concern: short phrase.
- reasons: {reasons_min}-{reasons_max} concrete reasons for preferred, grounded
  in their role/backstory and option facts.
- reservation: one honest concern about a rival option.
- would_reconsider_if: a concrete thing that could move them.

Participants should start with different plausible preferences, but normal
dialogues should still have overlap so compromise is possible.

Return JSON only:
{{
  "beliefs": {{
    "NAME": {{
      "preferred": "A|B|C|D",
      "acceptable": ["A", "B"],
      "rejected": [],
      "key_concern": "...",
      "reasons": ["...", "..."],
      "reservation": "...",
      "would_reconsider_if": "..."
    }}
  }}
}}"""


def sim_turn_compact(
    speaker_card: str,
    relevant_options: str,
    group_state: str,
    local_context: str,
    memory_block: str,
    move_instruction: str,
    output_contract: str,
) -> str:
    group_section = f"\nGROUP STATE\n{group_state}\n" if group_state.strip() else ""
    memory_section = f"\nYOUR MEMORY\n{memory_block}\n" if memory_block.strip() else ""
    return f"""Write the next message in a relaxed but real group chat.

Voice rules:
- Natural adult chat. Casual but not slangy.
- Many turns are short. Go longer only for a real reason.
- Do not open with "valid point", "good point", "fair point", "I agree", or
  "X is right".
- Do not restate a point already made.
- Do not invent option facts. Use only values shown in the option cards.
- No name prefix. No markdown. No em dashes.

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


def structured_vote_turn(
    speaker_card: str,
    relevant_options: str,
    group_state: str,
    local_context: str,
    memory_block: str,
    max_words: int,
) -> str:
    memory_section = f"\nYOUR MEMORY\n{memory_block}\n" if memory_block.strip() else ""
    group_section = f"\nGROUP STATE\n{group_state}\n" if group_state.strip() else ""
    return f"""Write one participant's vote in a relaxed group chat.

Return JSON only. Choose exactly one listed option. The visible message should
sound human, name the option, and give one honest reason. Do not invent facts.

SPEAKER CARD
{speaker_card}

OPTIONS
{relevant_options}
{group_section}{memory_section}
RECENT TURNS
{local_context}

OUTPUT JSON
{{
  "message": "one chat message, no name prefix, <= {max_words} words",
  "action": "vote",
  "option": "A|B|C|D"
}}"""


def structured_confirmation_turn(
    speaker_card: str,
    relevant_options: str,
    group_state: str,
    local_context: str,
    memory_block: str,
    candidate: str,
    preferred: str,
    acceptable: list[str],
    rejected: list[str],
    is_firm_holdout: bool,
    max_words: int,
) -> str:
    acceptable_text = ", ".join(acceptable) if acceptable else "none"
    rejected_text = ", ".join(rejected) if rejected else "none"
    firmness = (
        "This participant is unusually firm; accept only the preferred option."
        if is_firm_holdout else
        "This participant is cooperative, but should still be honest."
    )
    memory_section = f"\nYOUR MEMORY\n{memory_block}\n" if memory_block.strip() else ""
    group_section = f"\nGROUP STATE\n{group_state}\n" if group_state.strip() else ""
    return f"""Write one participant's answer to a direct compromise check.

Return JSON only. The moderator asks whether Option {candidate} works.

PRIVATE STANCE
Preferred: Option {preferred}
Acceptable: {acceptable_text}
Rejected: {rejected_text}
Firmness: {firmness}

SPEAKER CARD
{speaker_card}

OPTIONS
{relevant_options}
{group_section}{memory_section}
RECENT TURNS
{local_context}

OUTPUT JSON
{{
  "message": "one chat message, no name prefix, <= {max_words} words",
  "action": "accept|reject",
  "option": "{candidate}"
}}"""


def repair_repetition(original_text: str) -> str:
    cap = _pc_int("repair_repetition_words", 25)
    return f"""A chat message was rejected because it repeats an earlier point.

Original:
{original_text}

Rewrite as one natural message that adds something new: a reaction, a different
tradeoff, or a move toward a pick. No name prefix. Under {cap} words."""


def repair_ack_loop(original_text: str) -> str:
    cap = _pc_int("repair_ack_loop_words", 22)
    return f"""A chat message was rejected because the discussion is becoming an acknowledgement loop.

Original:
{original_text}

Rewrite as a different natural move: brief yes/no, new reason, direct answer,
compromise, or decision move. Do not start with acknowledgement language.
No name prefix. Under {cap} words."""


def repair_question_chain(original_text: str) -> str:
    cap = _pc_int("repair_question_chain_words", 22)
    return f"""A chat message was rejected because it asks another question while recent questions are unresolved.

Original:
{original_text}

Rewrite as one message with no question: answer, react, compare listed facts, or
move toward a decision. No name prefix. Under {cap} words."""


def repair_semantic_repeat(original_text: str, prior_point: str) -> str:
    cap = _pc_int("repair_semantic_repeat_words", 22)
    return f"""A chat message was rejected because it repeats this earlier point:
{prior_point}

Original:
{original_text}

Rewrite without restating that same option-attribute argument. No name prefix.
Under {cap} words."""


def repair_invalid_option(original_text: str, options: list[str]) -> str:
    return f"""A chat message referenced a non-existent option or denied a listed option.

Valid options:
{chr(10).join(options)}

Original:
{original_text}

Rewrite using only the listed options. No name prefix."""


def repair_vote(options: list[str]) -> str:
    cap = _pc_int("repair_vote_words", 22)
    letters = ", ".join(
        m.group(1).upper()
        for opt in options
        if (m := re.match(r"Option\s+([A-D])", opt, re.I))
    )
    return f"""A narrowing message did not clearly vote.

Available options:
{chr(10).join(options)}

Write one natural message naming exactly one option ({letters}) as the pick.
No name prefix. Under {cap} words."""


def repair_inconsistent_vote(
    original_text: str,
    options: list[str],
    rejected_options: set[str],
) -> str:
    cap = _pc_int("repair_inconsistent_vote_words", 24)
    rejected = ", ".join(f"Option {o}" for o in sorted(rejected_options)) or "an earlier rejected option"
    return f"""A vote message chose {rejected}, which the same speaker ruled out earlier.

Available options:
{chr(10).join(options)}

Original:
{original_text}

Rewrite as one vote for an option not ruled out, or explicitly say they changed
their mind and why. No name prefix. Under {cap} words."""


def repair_repeated_rule_out(original_text: str) -> str:
    cap = _pc_int("repair_repeated_rule_out_words", 22)
    return f"""A message repeated a rule-out that already happened.

Original:
{original_text}

Rewrite as an answer, comparison, compromise, or decision move. No new question.
No name prefix. Under {cap} words."""


def repair_attribute_mismatch(original_text: str, options: list[str]) -> str:
    cap = _pc_int("repair_attribute_mismatch_words", 24)
    return f"""A message changed a listed option fact.

Option facts:
{chr(10).join(options)}

Original:
{original_text}

Rewrite using only listed values. No name prefix. Under {cap} words."""


def repair_confirmation(candidate: str) -> str:
    cap = _pc_int("repair_confirmation_words", 24)
    return f"""A confirmation message was unclear or too thin.

The candidate is Option {candidate}. Write either:
- yes, with one short reason if it is a compromise;
- no, with one concrete blocker.

No name prefix. Under {cap} words."""


def repair_invented_fact(original_prompt: str) -> str:
    return (
        original_prompt
        + "\n\nYour previous response invented option facts. Rewrite the same "
        "turn using only listed option values. General experience is okay; fake "
        "prices, times, policies, services, or numbers are not."
    )


def repair_fact_chasing_question(original_text: str, options: list[str]) -> str:
    cap = _pc_int("repair_fact_chasing_words", 24)
    return f"""A message asked for outside facts the group cannot check.

Option cards:
{chr(10).join(options)}

Original:
{original_text}

Rewrite as one natural message that decides from the listed facts. Do not ask a
new question or suggest calling/checking/looking up. No name prefix. Under
{cap} words."""
