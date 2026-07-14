"""Prompt construction for setup and action-to-language realization."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable

from aliases import validated_short_alias
from config_loader import cfg
from models import (
    ActionType,
    DialogueState,
    IssueEffect,
    OpeningMode,
    Persona,
    QuestionMode,
    ResponseMode,
    RunOutcome,
    Scenario,
    StanceUpdateKind,
    UserAction,
)


def _schema(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def setup_scenario(topic: str, n: int) -> str:
    labels = [str(value) for value in cfg.scenario.option_labels]
    example = {
        "scenario": {
            "shared_context": ["one brief public fact"],
            "options": [
                {
                    "id": label,
                    "name": f"full option name {label}",
                    "short_name": f"short name {label}",
                    "attrs": {"topic-specific attribute": "public value"},
                    "upside": "brief public advantage",
                    "concern": "brief public drawback",
                }
                for label in labels
            ],
        }
    }
    return f"""Create a fixed public option board for an option-grounded group decision.

Topic: {topic}
Number of deciding participants: {n}
Required option IDs in this exact order: {labels}

Rules:
- Return JSON only.
- Create exactly {len(labels)} distinct, realistic options.
- Choose attributes that are natural for this topic.
- Every objective fact must appear in shared_context or an option card.
- Each option needs {int(cfg.scenario.public_attr_min)}..{int(cfg.scenario.public_attr_max)} concise factual attributes.
- short_name is required, natural, unique, and at most {int(cfg.scenario.short_alias_max_words)} words.
- Keep shared_context to at most {int(cfg.scenario.shared_context_max_items)} short items.
- Do not include an opening question, decision kind, hidden facts, winner, or recommendation.

JSON shape:
{_schema(example)}
"""


def alias_repair(*, topic: str, option_rows: list[dict], invalid: dict[str, str], duplicates: dict[str, str]) -> str:
    labels = [str(row.get("id", "")) for row in option_rows]
    return f"""Repair only the short_name values of these option cards.
Return JSON only as {{"short_names": {{"A": "..."}}}} for every option ID {labels}.
Do not change full names or facts. Each short name must be unique, natural, and at most
{int(cfg.scenario.short_alias_max_words)} words.

Topic: {topic}
Invalid aliases: {_schema(invalid)}
Duplicate aliases: {_schema(duplicates)}
Options: {_schema(option_rows)}
"""


def setup_personas(
    topic: str,
    n: int,
    trait_rows: list[dict],
    required_preferences: dict[str, str],
    options_json: list[dict],
    shared_context: list[str],
    hard_blocker_id: str | None = None,
) -> str:
    examples = []
    for row in trait_rows:
        pid = row["id"]
        examples.append({
            "id": pid,
            "name": row["name"],
            "background": "one short relevant background sentence",
            "private_goal": "one short private decision priority",
            "age": row.get("age", 35),
            "preferred_options": [required_preferences[pid]],
            "option_stances": {
                option["id"]: {
                    "rank": 5 if option["id"] == required_preferences[pid] else 3,
                    "reason_for": "brief grounded reason or empty",
                    "reason_against": "brief grounded concern or empty",
                }
                for option in options_json
            },
        })
    blocker_rule = (
        f"Participant {hard_blocker_id} is the sole hard blocker: exactly one preferred option, "
        "rank every alternative 1, and give a clear grounded reason_against for every alternative."
        if hard_blocker_id else
        "No participant is a hard blocker. Do not rank every alternative 1 for anyone."
    )
    return f"""Create {n} concise participant persona cards for the fixed option board.
Return JSON only. Behavioral traits are already supplied and must not be rewritten.

Topic: {topic}
Shared public context: {_schema(shared_context)}
Public options: {_schema(options_json)}
Fixed participant identities and direct traits: {_schema(trait_rows)}
Required initial primary preferences: {_schema(required_preferences)}

Rules:
- Preserve every id and name exactly.
- Each background and private_goal must be short, plausible, and relevant.
- Private persona content may motivate preferences but must not invent hidden option facts.
- preferred_options must begin with the required option for that participant.
- Rank every option from 1..5: 5 preferred, 4 acceptable, 3 neutral, 2 disliked, 1 hard rejected.
- Normal participants may have uneven reasons; do not force one pro and con for every option.
- Reasons must be grounded in public option attributes or clearly personal priorities.
- {blocker_rule}
- Age may be supplied when absent; do not create implausible age/backstory combinations.
- Do not output speech_style or behavioral traits.

JSON shape:
{_schema({"participants": examples})}
"""


def word_budget(action: ActionType, verbosity: int) -> tuple[int, int]:
    """Return a soft minimum and configured maximum word count."""
    maximum = int(cfg.level_value(
        "language", "max_words_by_verbosity", verbosity, cast=int
    ))
    if action is ActionType.ACKNOWLEDGE:
        maximum = min(maximum, 12)
    elif action is ActionType.VOTE:
        maximum = min(maximum, 10)
    elif action in {ActionType.ASK, ActionType.ANSWER, ActionType.FINAL_POSITION}:
        maximum = min(maximum, 20)
    minimum = 2 if action in {ActionType.ACKNOWLEDGE, ActionType.VOTE} else 4
    return minimum, maximum


def directness_instruction(level: int) -> str:
    return str(cfg.level_value(
        "language", "directness_instructions", level, cast=str
    ))


def _option_name(state: DialogueState, option_id: str) -> str:
    option = state.scenario.option(option_id)
    safe_short = validated_short_alias(option.name, option.short_name)
    return safe_short or option.name


def _selected_action(state: DialogueState, action: UserAction) -> str:
    names = [_option_name(state, option_id) for option_id in action.option_focus]
    focus = " and ".join(names)
    reason = action.reason.strip()
    decisive = action.decisive_reason.strip() or reason
    condition = action.condition.strip()

    if action.act is ActionType.OPENING:
        if action.opening_mode is OpeningMode.ALIGN:
            return (
                f"Join the opening naturally. Another participant already prefers {focus}; align with that choice "
                f"and give your own brief reason: {reason}. A greeting is optional."
            )
        if action.opening_mode is OpeningMode.CONTRAST:
            return (
                f"Join the opening naturally with a different preference: {focus}. Give your brief reason: {reason}. "
                "A greeting is optional; respond as part of the ongoing chat rather than restarting it."
            )
        return f"Start the discussion naturally, briefly greet the group, state that you prefer {focus}, and give this reason: {reason}."
    if action.act is ActionType.SUPPORT:
        return f"Add one useful supporting point for {focus}: {reason}."
    if action.act is ActionType.CONCERN:
        if action.issue_effect is IssueEffect.MAINTAIN:
            return f"React to the response and make clear that this concern about {focus} still matters: {reason}."
        if state.phase.value == "NARROWING":
            return f"Briefly state that this unresolved concern about {focus} still blocks agreement: {reason}."
        return f"Raise this concrete concern about {focus}: {reason}."
    if action.act is ActionType.ASK:
        target = state.persona(action.addressee_id).name if action.addressee_id else "the group"
        if action.question_mode is QuestionMode.TRADEOFF:
            return (
                f"Ask {target} naturally which factor matters more for {focus}. "
                f"Concern: {reason}. Positive reason: {decisive}. Do not answer for them or prescribe wording."
            )
        if action.question_mode is QuestionMode.CONDITION:
            return (
                f"Ask {target} whether any known condition would make {focus} workable despite this concern: {reason}. "
                "Do not invent or suggest a solution yourself."
            )
        return (
            f"Ask {target} naturally whether this concern changes their choice of {focus}: {reason}. "
            "Do not require any particular phrase."
        )
    if action.act is ActionType.ANSWER:
        target = state.persona(action.addressee_id).name if action.addressee_id else "the previous speaker"
        if action.response_mode is ResponseMode.KNOWN_MITIGATION:
            return f"Answer {target} directly. Known information that addresses the concern: {condition or decisive}."
        if action.response_mode is ResponseMode.ACCEPT_TRADEOFF:
            return (
                f"Answer {target} directly. Position: you recognize the concern and still prefer {focus}. "
                f"Decisive reason: {decisive}. Choose your own natural sentence structure."
            )
        if action.response_mode is ResponseMode.MAINTAIN_CONCERN:
            return f"Answer {target} directly. Position: the concern still affects your choice. Concern: {reason}."
        if action.response_mode is ResponseMode.UNKNOWN:
            return f"Answer {target} directly that the available information is insufficient. Do not invent a fact or solution."
        return f"Answer {target}'s exact question directly. Actual position: {reason}."
    if action.act is ActionType.COMPARE:
        return f"Compare {focus} using this trade-off: {reason}. A useful one-sided point is acceptable if a full comparison sounds forced."
    if action.act in {ActionType.ACKNOWLEDGE, ActionType.COMMENT}:
        target = state.persona(action.addressee_id).name if action.addressee_id else None
        if action.act is ActionType.ACKNOWLEDGE:
            return f"React briefly and naturally to the proposed common ground around {focus}. Do not restate the full reason."
        if action.issue_id:
            if action.response_mode is ResponseMode.ACCEPT_TRADEOFF:
                return (
                    f"Respond to the concern about {focus}. Position: you recognize the concern and still support the option. "
                    f"Decisive reason: {decisive}. Use your own natural structure."
                )
            if action.response_mode is ResponseMode.MAINTAIN_CONCERN:
                return f"Respond that the concern about {focus} still matters to you: {decisive or reason}."
            if action.response_mode is ResponseMode.UNKNOWN:
                return f"Respond that you do not know enough to judge the concern about {focus}. Do not invent a solution."
            return f"Respond to the active issue about {focus}. State this position honestly: {reason}."
        if target:
            return f"React to {target}'s last point about {focus}, then add this distinct point: {reason}."
        return f"Continue the current exchange with this relevant point about {focus}: {reason}."
    if action.act is ActionType.COMPROMISE:
        if action.stance_update and action.stance_update.kind is StanceUpdateKind.SWITCH_PREFERRED:
            detail = f" A brief reason may be used: {reason}." if reason else ""
            return f"Clearly say that you are moving to {focus} after the discussion.{detail}"
        movement_reason = decisive or reason
        detail = f" You may briefly mention: {movement_reason}." if movement_reason else ""
        if action.issue_effect is IssueEffect.RESOLVE:
            return f"Say that the response addressed your concern and make {focus} visibly acceptable.{detail}"
        return f"Make {focus} visibly acceptable as common ground without claiming it was your first choice.{detail}"
    if action.act is ActionType.FINAL_POSITION:
        return f"Briefly state that you are staying with {focus}. Do not repeat earlier reasons and do not use a voting formula."
    if action.act is ActionType.VOTE:
        if action.stance_update:
            return f"State one clear vote for {focus} and briefly indicate that your choice changed. Use natural group-chat wording."
        return f"State only one short, natural choice for {focus}. Three to eight words are enough; do not repeat your reason."
    return reason or "Make one relevant contribution."


def _relevant_facts(state: DialogueState, action: UserAction) -> str:
    if (
        not action.option_focus
        or (action.act is ActionType.FINAL_POSITION and not action.reason)
        or (action.act is ActionType.VOTE and not action.stance_update)
    ):
        return "- No option fact is required for this message."
    lines: list[str] = []
    for option_id in action.option_focus:
        option = state.scenario.option(option_id)
        if action.reason_source and action.reason_source.option_id == option_id:
            source = action.reason_source
            if source.attribute_name in option.attrs:
                detail = f"{source.attribute_name.replace('_', ' ')}: {option.attrs[source.attribute_name]}"
            elif source.attribute_name == "upside":
                detail = f"upside: {option.upside}"
            elif source.attribute_name == "concern":
                detail = f"concern: {option.concern}"
            else:
                detail = source.public_value
            lines.append(f"- {option.id}) {option.name}: {detail}")
        else:
            pieces = [f"- {option.id}) {option.name}"]
            if option.upside:
                pieces.append(f"upside: {option.upside}")
            if option.concern:
                pieces.append(f"concern: {option.concern}")
            lines.append("; ".join(pieces))
    return "\n".join(lines)


def _recent_chat(state: DialogueState) -> str:
    limit = int(cfg.conversation.recent_turns_in_prompt)
    turns = state.turns[-limit:]
    return "\n".join(f"{turn.speaker_name}: {turn.text}" for turn in turns) or "(No previous messages.)"


def _target_question(state: DialogueState, action: UserAction) -> str:
    if action.act is not ActionType.ANSWER or not state.active_issue:
        return ""
    target = state.active_issue.source_text.strip()
    if not target:
        return ""
    recent = "\n".join(turn.text for turn in state.turns[-int(cfg.conversation.recent_turns_in_prompt):])
    return "" if target in recent else f"\nExact question being answered:\n{target}\n"


def _active_issue(state: DialogueState) -> str:
    issue = state.active_issue
    if not issue:
        return ""
    opener = state.persona(issue.opened_by).name
    return (
        f"\nActive issue:\n- {issue.kind.value} raised by {opener}: {issue.summary}\n"
    )


def _recent_own_language(state: DialogueState, speaker_id: str) -> str:
    own = [turn.text for turn in state.turns if not turn.moderator and turn.speaker_id == speaker_id][-2:]
    if not own:
        return ""
    return "\nRecent messages by you (do not repeat their wording):\n" + "\n".join(f"- {text}" for text in own) + "\n"


def realization_prompt(
    state: DialogueState,
    persona: Persona,
    action: UserAction,
) -> str:
    _, maximum = word_budget(action.act, persona.sim_params.verbosity)
    if action.act is ActionType.VOTE:
        maximum = min(maximum, 14 if action.stance_update else 8)
    personal = (
        f"\nRelevant personal context: {action.personal_context.strip()}\n"
        if action.personal_context else ""
    )
    target = _target_question(state, action)
    own = _recent_own_language(state, persona.id)
    return f"""You are {persona.name} in a group decision chat.
Voice: {persona.speech_style}.
Directness: {persona.sim_params.directness}/5 — {directness_instruction(persona.sim_params.directness)}
{personal}
Selected action:
{_selected_action(state, action)}

Relevant public option facts:
{_relevant_facts(state, action)}
{_active_issue(state)}
Recent chat:
{_recent_chat(state)}
{target}{own}
Write exactly one natural group-chat message that continues from the local context.
Express the selected meaning, but do not copy the instruction's sentence structure or rely on one fixed question or contrast formula.
Use only the supplied public option facts. Do not invent numbers, values, features, solutions, or guarantees.
Personal opinions and the supplied personal context are allowed. If a fact or mitigation is unknown, say so.
Do not add a speaker label or summarize the discussion.
Vary the sentence opening naturally; do not mechanically begin with a name, “I”, or the option name.
Short reactions, contractions, and “we”/“us” are valid when natural.
Maximum {maximum} words; finish the thought naturally.
"""


def repair_prompt(
    state: DialogueState,
    persona: Persona,
    action: UserAction,
    original: str,
    errors: list[str],
) -> str:
    _, maximum = word_budget(action.act, persona.sim_params.verbosity)
    if action.act is ActionType.VOTE:
        maximum = min(maximum, 14 if action.stance_update else 8)
    return f"""Rewrite one group-chat message for {persona.name}.

Selected action:
{_selected_action(state, action)}

Relevant public facts:
{_relevant_facts(state, action)}

Original message:
{original}

Problems to fix:
- """ + "\n- ".join(errors) + f"""

Return one corrected message only.
Do not invent facts, values, solutions, or guarantees. Preserve the selected action and any required vote or stance change.
Use natural group-chat wording; the word “formal” is unnecessary.
No speaker label. Maximum {maximum} words; finish the thought naturally.
"""


def moderator_opening(scenario: Scenario) -> str:
    return "Let’s begin with each person’s current preference and main reason."


def moderator_stall_prompt() -> str:
    return "Is there another important reason, concern, or question before we narrow the choices?"


def moderator_compromise_prompt() -> str:
    return "We seem stuck. Is there an option anyone could accept even if it is not their first choice?"


def moderator_coverage_prompt(scenario: Scenario, option_id: str) -> str:
    name = scenario.option(option_id).short_name or scenario.option(option_id).name
    return f"We have not really considered {name}. Is there a reason to keep it or rule it out?"


def moderator_unanimous_narrowing(scenario: Scenario, option_id: str) -> str:
    name = scenario.option(option_id).short_name or scenario.option(option_id).name
    return f"{name} already has everyone’s current support. Let’s vote."


def moderator_narrowing(scenario: Scenario, options: tuple[str, ...]) -> str:
    if not options:
        return "No clear leader has emerged. Anyone may propose an option they could genuinely accept before we vote."
    names = [scenario.option(option_id).short_name or scenario.option(option_id).name for option_id in options]
    if len(names) == 1:
        return f"{names[0]} currently leads. Anyone not there yet can say whether they can accept it or what still prevents that."
    return f"The group is split between {names[0]} and {names[1]}. Is either one workable common ground before we vote?"


def moderator_revote_narrowing(scenario: Scenario, options: tuple[str, ...]) -> str:
    if not options:
        return "The first vote had no majority. Is there any option someone can newly accept as common ground?"
    names = [scenario.option(option_id).short_name or scenario.option(option_id).name for option_id in options]
    return f"The first vote had no majority. Can anyone newly accept {' or '.join(names)} as common ground?"


def moderator_vote_request(*, revote: bool = False, had_revote_discussion: bool = False) -> str:
    if revote:
        return "Now give one final clear choice."
    return "Now give one clear choice."


def moderator_closure(outcome: RunOutcome, scenario: Scenario) -> str:
    if outcome.final_option:
        name = scenario.option(outcome.final_option).short_name or scenario.option(outcome.final_option).name
        return f"The result is {outcome.status}: {name}."
    return "No option reached a majority, so the result is unresolved."
