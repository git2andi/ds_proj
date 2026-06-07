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
        "opening_question": "one casual moderator question asking for priorities and trade-offs, not votes",
        "options": [
            {
                "id": label,
                "name": "specific short option name",
                "attrs": {
                    "cost_or_effort": "concrete value",
                    "time_or_duration": "concrete value",
                    "main_benefit_metric": "concrete value or 1-5 score",
                    "comfort_or_difficulty": "concrete value or 1-5 score",
                    "flexibility_or_risk_control": "concrete value or 1-5 score",
                    "topic_specific_factor": "concrete value"
                },
                "upside": "specific benefit people can argue for",
                "tradeoff": "specific downside people can weigh against the upside",
                "concern": "stable objection or worry, not missing information",
                "fit": "which kind of group priority this option fits",
                "risk": "stable risk from the fictional scenario",
                "best_for": "one clear decision priority this option serves"
            }
            for label in labels
        ],
    }
    forbidden = ", ".join(str(x) for x in cfg.scenario.forbidden_live_fact_terms)
    return f"""Create a fictional but grounded decision scenario for a small casual group chat.

Topic: {topic}

Generate exactly {len(labels)} options with ids {labels}. The options are the full source of truth for the later chat.
The goal is not a short menu; the goal is four compact option cards with enough concrete material for disagreement, comparison, and compromise.

Required option-card quality:
- Every option needs {cfg.scenario.attr_min}-{cfg.scenario.attr_max} stable attributes in attrs.
- Attribute keys must be topic-specific, not placeholders like stable_attribute_1.
- For logistics topics, include concrete values such as price/cost, duration, distance/time, comfort, flexibility, reliability, risk level, effort, capacity, or similar relevant factors.
- For abstract topics, use concrete 1-{cfg.scenario.score_max} scores or clearly named qualitative levels for dimensions like societal impact, feasibility, cost, risk, accessibility, learning value, novelty, or long-term benefit.
- Every option must expose a real trade-off: a clear upside, a clear downside, a stable concern, a fit, a risk, and a best_for priority.
- Options must differ meaningfully. Do not make four variants of the same priority.
- Do not use missing-info placeholders. The group must be able to decide from the cards alone.
- Use stable fictional facts only. Do not include live-lookup or unknown facts such as: {forbidden}.
- The opening question should ask what people care about and invite trade-offs, not votes.

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
                "backstory": "one short prior experience or memory that explains the priority",
                "main_concern": "the main decision criterion this participant cares about",
                "preferred_option": "A",
                "acceptable_options": ["A", "B"],
                "soft_rejections": ["D"],
                "hard_rejections": [],
                "reasons": {
                    "A": ["2-3 grounded reasons for the preferred option"],
                    "B": ["1-2 grounded reasons why this fallback could work"]
                },
                "reservation": "one addressable concern about a non-preferred or fallback option",
                "reconsider_if": "a realistic condition, based on group priorities/trade-offs, that would make them move from their favorite"
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

Rules for useful persona state:
- Everyone should try to reach a compromise if possible. A rare hard blocker is allowed only when hard_blocker=true.
- Normal participants need at least two acceptable options, including their preferred option.
- The backstory must be short and useful for chat behavior: one concrete prior experience, habit, or memory that explains the participant's priority. Do not add demographic identity labels or irrelevant biography.
- main_concern should be a reusable decision criterion, such as budget, comfort, feasibility, social impact, safety, effort, fairness, speed, or novelty.
- Reasons must be grounded in the option cards and should give the participant argument material, not slogans. Give {cfg.personas.min_reasons_preferred} or more reasons for the preferred option when possible and at least {cfg.personas.min_reasons_acceptable} reason for each acceptable fallback.
- Soft rejections are addressable concerns, not permanent vetoes.
- hard_rejections must usually be empty. Only use them for hard_blocker=true or a genuine dealbreaker grounded in the cards.
- reconsider_if must not change fixed facts from the option cards. Do not write things like "if the price decreased" when the price is fixed. Instead use group-priority conditions, e.g. "if everyone cares more about speed than cost" or "if the group accepts the extra risk for the stronger benefit".
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
Backstory/memory anchor: {persona.backstory}
Main concern: {persona.main_concern}
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
    extra = ""
    if intent.act.value == "accept":
        extra = "\nFor this move, answer for yourself only. Do not ask whether others are okay. Use a clear first-person yes/no."
    elif intent.act.value == "vote":
        extra = "\nFor this move, name your current pick, but keep it provisional unless the group has clearly converged."
    elif intent.act.value == "push_back":
        extra = "\nFor this move, raise one concrete concern while staying open to compromise."
    elif intent.act.value == "compare":
        extra = "\nFor this move, compare real trade-offs between the focused options, not just your favorite."
    elif intent.act.value == "opening":
        extra = "\nFor this move, state a priority in your own words. Do not copy the phrasing of earlier participants."
    return f"""You are {speaker_name}. Generate exactly one chat message.
Local move: {intent.act.value}
Why you are speaking: {intent.reason}
Address: {addressee}
Option focus: {focus}
Length hint: {intent.length_hint}{extra}"""


def output_contract(max_words: int, speaker_name: str) -> str:
    banned = ", ".join(str(x) for x in cfg.utterances.banned_register_phrases)
    discouraged = ", ".join(str(x) for x in cfg.utterances.discouraged_starts)
    return f"""Output only the message text, without '{speaker_name}:' and without quotation marks.
Maximum {max_words} words.
Sound like a normal adult friend-group chat: casual, compact, slightly imperfect, not corporate, not Gen-Z-heavy, not essay-like.
Do not use numbered lists. Do not create multiple turns. Do not invent facts outside the option cards.
Do not repeat the shape of recent turns. Avoid starting with: {discouraged}.
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


def _moderator_option_line(option: OptionCard) -> str:
    attrs = list(option.attrs.items())[: int(cfg.scenario.display_attr_limit)]
    attr_text = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in attrs)
    parts = [
        f"Option {option.id} - {option.name}: {attr_text}" if attr_text else f"Option {option.id} - {option.name}",
        f"upside: {option.upside}" if option.upside else "",
        f"tradeoff: {option.tradeoff}" if option.tradeoff else "",
        f"concern: {option.concern}" if option.concern else "",
        f"fit: {option.fit}" if option.fit else "",
        f"risk: {option.risk}" if option.risk else "",
        f"best for: {option.best_for}" if option.best_for else "",
    ]
    return "; ".join(p for p in parts if p)


def moderator_opening(scenario: Scenario) -> str:
    option_lines = "\n".join(f"- {_moderator_option_line(o)}" for o in scenario.options)
    return f"Moderator: We need to decide: {scenario.topic}. Here are the options I have:\n{option_lines}\n{scenario.opening_question}"


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
        return f"No, Option {candidate} does not work for me yet; my main worry is {persona.reservation}."
    if intent.act.value == "answer":
        return f"For me it mostly comes down to Option {preferred} fitting what I care about, but I’m not trying to block alternatives."
    if intent.act.value == "compare":
        option = candidate or preferred
        return f"Option {option} seems worth comparing properly against the others before we settle too quickly."
    if intent.act.value == "push_back":
        option = candidate or preferred
        return f"My only hesitation with Option {option} is that the downside might matter more once we actually choose."
    if intent.act.value == "react":
        option = candidate or preferred
        return f"I would keep Option {option} in the mix for now; it covers a different priority."
    if intent.act.value == "ask":
        option = candidate or preferred
        return f"Would Option {option} solve enough of what people care about, or is the trade-off too annoying?"
    if intent.act.value == "propose_compromise":
        option = candidate or (persona.acceptable_options[0] if persona.acceptable_options else preferred)
        return f"Maybe we could use Option {option} as the middle ground if everyone can live with it."
    return f"I’m still leaning toward Option {preferred}, but I’m open to a workable compromise."
