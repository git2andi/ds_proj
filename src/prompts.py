"""All LLM-facing prose and dialogue templates.

Project rule: every prompt sent to an LLM must be produced here.  Other modules
may pass structured data into these functions, but they should not contain
instructional wording for the model.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from config_loader import cfg
from schemas import MoveIntent, OptionCard, Persona, Phase, Scenario


def _json_schema_hint(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def render_option_card(option: OptionCard) -> str:
    return option.source_text()


def render_options(options: Iterable[OptionCard]) -> str:
    return "\n".join(render_option_card(o) for o in options)


def option_generation(topic: str) -> str:
    labels = list(cfg.scenario.option_labels)
    schema = {
        "decision_kind": "restaurant_choice | travel_destination | hotel_booking | flight_booking | study_or_work_plan | presentation_topic | tool_or_product_choice | game_or_activity_choice | generic_decision",
        "opening_question": "one casual moderator question asking for priorities, not votes",
        "options": [
            {
                "id": label,
                "name": "short option name",
                "attrs": {"stable_attribute": "value"},
                "upside": "concrete benefit",
                "tradeoff": "concrete tradeoff",
                "concern": "stable concern people can argue about",
                "fit": "who/what this fits",
                "risk": "stable risk, not missing information",
                "best_for": "priority this option serves",
            }
            for label in labels
        ],
    }
    forbidden = ", ".join(str(x) for x in cfg.scenario.forbidden_live_fact_terms)
    return f"""Create a fictional but grounded decision scenario for a small casual group chat.

Topic: {topic}

Generate exactly {len(labels)} options with ids {labels}. The options are the full source of truth for the later chat.
Each option must be distinct and must give people concrete trade-offs to discuss.
Fill every field (upside, tradeoff, concern, fit, risk, best_for) with a concrete, self-contained clause. Do not leave any field vague, generic, or empty.
Attributes must be specific and comparable across options: use numbers with units where it makes sense (price, time, distance, scores) so people can weigh options against each other.
Use stable facts only. Do not include live-lookup or missing-information facts such as: {forbidden}.
Concrete logistics topics should use {cfg.scenario.attr_min}-{cfg.scenario.attr_max} attributes. Abstract topics should use scored dimensions from {cfg.scenario.score_min} to {cfg.scenario.score_max}.
The opening question should ask what people care about, not which option they vote for.

Return JSON only with this shape:
{_json_schema_hint(schema)}"""


def names_and_roles(topic: str, n: int) -> str:
    schema = {
        "participants": [
            {"id": f"p{i+1}", "name": "short first name", "role": "1-4 words, topic-relevant or friend-group role"}
            for i in range(n)
        ]
    }
    return f"""Create {n} participants for a casual friend-group discussion.

Topic: {topic}

Use plausible short first names. Do not use stereotypes or demographic labels. Roles should give light conversational variety, not professional expertise unless the topic requires it.
Return JSON only:
{_json_schema_hint(schema)}"""


def belief_generation(topic: str, options: list[OptionCard], participant_summaries: list[dict]) -> str:
    schema = {
        "beliefs": [
            {
                "id": "p1",
                "private_goal": "what this participant wants from the decision",
                "preferred_option": "A",
                "acceptable_options": ["A", "B"],
                "soft_rejections": ["D"],
                "hard_rejections": [],
                "reasons": {"A": ["reason grounded in option cards"], "B": ["why it could work"]},
                "reservation": "one addressable concern about a non-preferred option",
                "reconsider_if": "what would make them move from their favorite",
            }
        ]
    }
    summaries = json.dumps(participant_summaries, ensure_ascii=False, indent=2)
    return f"""Assign internally consistent private belief states for the participants.

Topic: {topic}

Option cards:
{render_options(options)}

Participants with traits and behavioral controls:
{summaries}

Rules:
- Everyone should try to reach a compromise if possible.
- Normal participants should have at least two acceptable options, including their preferred option.
- Soft rejections are concerns, not permanent vetoes.
- hard_rejections must usually be empty. Only use a hard rejection when the supplied participant summary says hard_blocker=true.
- Reasons must match the preferred and acceptable options. Do not change a preference without also giving matching reasons.
- Use only facts from the option cards. Do not invent prices, times, availability, policies, weather, ratings, or external facts.

Return JSON only:
{_json_schema_hint(schema)}"""


def speaker_card(persona: Persona) -> str:
    t = persona.traits
    hard = "yes" if persona.is_hard_blocker else "no"
    reasons = json.dumps(persona.reasons, ensure_ascii=False)
    return f"""Name: {persona.name}
Role: {persona.role}
Speech style: {persona.speech_style}
Private goal: {persona.private_goal}
Traits: openness={t.openness}, conscientiousness={t.conscientiousness}, extraversion={t.extraversion}, agreeableness={t.agreeableness}, neuroticism={t.neuroticism}
Behavior controls: compromise_willingness={t.compromise_willingness:.2f}, patience={t.patience:.2f}, initiative={t.initiative:.2f}, conflict_directness={t.conflict_directness:.2f}, detail_level={t.detail_level:.2f}, hard_blocker={hard}
Current private stance: prefers Option {persona.preferred_option}; can probably live with {persona.acceptable_options}; soft concerns about {persona.soft_rejections}; hard rejects {persona.hard_rejections}
Reasons available: {reasons}
Reservation: {persona.reservation}
Would reconsider if: {persona.reconsider_if}"""


def recent_turns_block(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "No previous participant turns yet."


def group_state_summary(state_summary: dict) -> str:
    return json.dumps(state_summary, ensure_ascii=False, indent=2)


def move_instruction(intent: MoveIntent, speaker_name: str, addressee_name: Optional[str]) -> str:
    focus = ", ".join(f"Option {x}" for x in intent.option_focus) if intent.option_focus else "no specific option"
    addressee = addressee_name or "the group"
    return f"""You are {speaker_name}. Generate exactly one chat message.
Local move: {intent.act.value}
Why you are speaking: {intent.reason}
Address: {addressee}
Option focus: {focus}
Length hint: {intent.length_hint}"""


def output_contract(max_words: int, speaker_name: str) -> str:
    banned = ", ".join(str(x) for x in cfg.utterances.banned_register_phrases)
    return f"""Output only the message text, without '{speaker_name}:' and without quotation marks.
Maximum {max_words} words.
Sound like a normal casual friend-group chat: natural, compact, not corporate, not Gen-Z-heavy, not essay-like.
Do not use numbered lists. Do not create multiple turns. Do not invent facts outside the option cards.
Avoid these phrases unless genuinely unavoidable: {banned}."""


def sim_utterance(
    persona: Persona,
    scenario: Scenario,
    recent_lines: list[str],
    state_summary: dict,
    intent: MoveIntent,
    addressee_name: Optional[str],
    max_words: int,
) -> str:
    return f"""You are generating one participant message in a synthetic group decision chat.

Topic:
{scenario.topic}

Option cards, the only source of truth:
{render_options(scenario.options)}

Speaker card:
{speaker_card(persona)}

Recent chat:
{recent_turns_block(recent_lines)}

Structured group state:
{group_state_summary(state_summary)}

Move instruction:
{move_instruction(intent, persona.name, addressee_name)}

Output contract:
{output_contract(max_words, persona.name)}"""


def repair_utterance(
    original_text: str,
    issue_codes: list[str],
    persona: Persona,
    scenario: Scenario,
    recent_lines: list[str],
    state_summary: dict,
    intent: MoveIntent,
    addressee_name: Optional[str],
    max_words: int,
) -> str:
    base = sim_utterance(persona, scenario, recent_lines, state_summary, intent, addressee_name, max_words)
    return f"""{base}

The previous draft was invalid.
Previous draft: {original_text}
Issue codes: {issue_codes}
Rewrite the message once. Keep the same local move, fix the issues, and output only the corrected message text."""


def moderator_opening(scenario: Scenario) -> str:
    option_lines = "\n".join(f"- {o.source_text()}" for o in scenario.options)
    return (
        f"Moderator: Hey everyone — let's get started. We need to decide: {scenario.topic}.\n"
        f"Here are the {len(scenario.options)} options on the table:\n\n"
        f"{option_lines}\n\n"
        f"{scenario.opening_question}"
    )


def moderator_nudge(question: str) -> str:
    return f"Moderator: {question}"


def moderator_close_consensus(option: OptionCard) -> str:
    return f"Moderator: Alright, then we’ll go with Option {option.id}, {option.name}. Sounds like that works as the compromise."


def moderator_close_unresolved() -> str:
    return "Moderator: Looks like we do not have a clean compromise this time, so let’s leave it unresolved for now."


def deterministic_fallback(intent: MoveIntent, persona: Persona, candidate: Optional[str]) -> str:
    preferred = persona.preferred_option
    if intent.act.value == "vote":
        return f"I’d pick Option {preferred} for now."
    if intent.act.value == "accept" and candidate:
        if candidate == preferred:
            return f"Yeah, Option {candidate} works for me."
        return f"I still prefer Option {preferred}, but I can live with Option {candidate}."
    if intent.act.value == "reject" and candidate:
        return f"I’m not quite there on Option {candidate}; my main worry is {persona.reservation}."
    if intent.act.value == "answer":
        return f"Yeah, for me it mostly comes down to Option {preferred} fitting what I care about."
    if intent.act.value == "propose_compromise":
        option = candidate or (persona.acceptable_options[0] if persona.acceptable_options else preferred)
        return f"Maybe Option {option} is the middle ground if everyone can live with it."
    return f"I’m still leaning toward Option {preferred}, but I’m open to a workable compromise."
