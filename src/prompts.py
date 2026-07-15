"""Prompt construction for setup and action-to-language realization."""

from __future__ import annotations

import json
import re
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
            "shared_context": (
                "One or two complete sentences describing the shared situation, "
                "constraints, and stakes relevant to every option."
            ),
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
- shared_context must be one paragraph containing 1..{int(cfg.scenario.get("shared_context_max_sentences", 2))} complete sentences and at most {int(cfg.scenario.shared_context_max_words)} words; never output it as a list or bullets.
- Describe the shared situation, relevant constraints, and stakes rather than repeating option-card facts.
- Every context statement must be able to coexist with every option. Do not state an exact cost, duration, arrival time, availability, capacity, or outcome unless it is a genuine shared constraint compatible with all options.
- When the context contains a hard limit, every option must satisfy it. Avoid unsupported deadlines or timing assumptions that make otherwise valid options impossible.
- Do not include an opening question, participant count, decision kind, hidden facts, winner, or recommendation.

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
    shared_context: str,
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
Shared scenario context: {shared_context}
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
    cap_name = {
        ActionType.ACKNOWLEDGE: "acknowledge",
        ActionType.ASK: "ask",
        ActionType.ANSWER: "answer",
        ActionType.FINAL_POSITION: "final_position",
        ActionType.VOTE: "vote",
    }.get(action)
    if cap_name is not None:
        maximum = min(maximum, cfg.action_word_cap(cap_name))
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


def _previous_priority(state: DialogueState, persona: Persona, action: UserAction) -> tuple[str, str]:
    update = action.stance_update
    if update is None or not update.previous_option_id or update.previous_option_id == update.option_id:
        return "", ""
    previous_name = _option_name(state, update.previous_option_id)
    stance = persona.option_stances.get(update.previous_option_id)
    previous_reason = stance.reason_for.strip() if stance is not None else ""
    return previous_name, previous_reason


def _selected_action(state: DialogueState, action: UserAction, persona: Persona | None = None) -> str:
    names = [_option_name(state, option_id) for option_id in action.option_focus]
    focus = " and ".join(names)
    reason = action.reason.strip()
    decisive = action.decisive_reason.strip() or reason

    if action.act is ActionType.OPENING:
        greeting = (
            "For this two-person chat, use a simple ‘Hi’/‘Hey’ or no greeting; do not say ‘everyone’ or ‘all’."
            if len(state.personas) == 2
            else "A short group greeting is optional."
        )
        if action.opening_mode is OpeningMode.ALIGN:
            return (
                f"Join the opening naturally. Another participant already prefers {focus}; align with that choice "
                f"and give your own brief reason: {reason}. {greeting}"
            )
        if action.opening_mode is OpeningMode.CONTRAST:
            return (
                f"Join the opening naturally with a different preference: {focus}. Give your brief reason: {reason}. "
                f"Respond as part of the ongoing chat rather than restarting it. {greeting}"
            )
        return (
            f"Start the discussion naturally, state that you prefer {focus}, and give this reason: {reason}. "
            f"{greeting}"
        )
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
                f"{target} already publicly prefers {focus}. Address {target} clearly by name somewhere natural in the sentence, then ask whether this exact concern weakens or changes that preference. "
                f"Concern: {reason}. Known benefit they previously gave: {decisive}. Name the concern explicitly; do not ask "
                "whether the option is acceptable, split a fact into sub-comparisons, or introduce a new premise."
            )
        if action.question_mode is QuestionMode.CONDITION:
            return (
                f"Address {target} clearly by name somewhere natural in the sentence. Ask whether any known condition is stated in the provided information that would make "
                f"{focus} workable despite this concern: {reason}. Do not invent or suggest a solution yourself."
            )
        return (
            f"{target} already publicly prefers {focus}. Address {target} clearly by name somewhere natural in the sentence, then ask whether this exact concern changes or weakens that preference: {reason}. "
            "Name the concern explicitly instead of saying only ‘this concern’. Do not ask whether the option is acceptable."
        )
    if action.act is ActionType.ANSWER:
        target = state.persona(action.addressee_id).name if action.addressee_id else "the previous speaker"
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
            if target:
                return f"Briefly agree with {target}'s latest point about {focus}. Do not repeat the full reason or claim a stance change."
            return f"React briefly and naturally to the proposed common ground around {focus}. Do not restate the full reason."
        if action.issue_id:
            if action.response_mode is ResponseMode.ACCEPT_TRADEOFF:
                return (
                    f"Respond to the concern about {focus} with this relevant positive point: {decisive}. "
                    "Acknowledge the concern without claiming that this option is your preferred or top choice, "
                    "unless the selected action explicitly changes your stance."
                )
            if action.response_mode is ResponseMode.MAINTAIN_CONCERN:
                if action.issue_effect is IssueEffect.PARTIAL and decisive:
                    return (
                        f"Acknowledge this response about {focus}: {decisive}. "
                        f"Then make clear that your original concern still remains: {reason}."
                    )
                return f"Respond that this concern about {focus} still matters to you: {reason}."
            if action.response_mode is ResponseMode.UNKNOWN:
                return f"Respond that you do not know enough to judge the concern about {focus}. Do not invent a solution."
            return f"Respond to the active issue about {focus}. State this position honestly: {reason}."
        if target:
            return f"React to {target}'s last point about {focus}, then add this distinct point: {reason}."
        return f"Continue the current exchange with this relevant point about {focus}: {reason}."
    if action.act is ActionType.COMPROMISE:
        update = action.stance_update
        movement_reason = (
            update.movement_reason.strip() if update is not None else ""
        ) or decisive or reason
        remaining_concern = update.remaining_concern.strip() if update is not None else ""
        previous_name, previous_reason = (
            _previous_priority(state, persona, action) if persona is not None else ("", "")
        )
        previous_clause = ""
        if previous_name and previous_reason:
            previous_clause = (
                f" You previously preferred {previous_name} because {previous_reason}. "
                "Briefly preserve that priority while explaining why the new option is still acceptable."
            )
        elif previous_name:
            previous_clause = (
                f" You previously preferred {previous_name}. Briefly preserve that preference while explaining "
                "why the new option is still acceptable."
            )
        if update and update.kind is StanceUpdateKind.SWITCH_PREFERRED:
            already_public = (
                " The reason was already stated publicly, so do not invent a replacement reason."
                if update.reason_already_public else ""
            )
            return (
                f"Clearly state that you are moving to {focus}. The concrete reason: {movement_reason}.{already_public} "
                "The sentence structure is your choice: use a direct conclusion, a short acknowledgment, or a trade-off statement that fits this persona. "
                "Do not rely on repetitive 'I'm switching' or 'changing my vote' formulas."
            )
        movement_shape = (
            "The sentence structure is your choice. Choose a natural structure that fits the movement basis: a brief acknowledgment, a practical consequence, "
            "or a clear trade-off. Avoid the fixed ‘I still prefer X, but I can accept Y’ formula."
        )
        if action.issue_effect is IssueEffect.RESOLVE:
            concern = f" You may acknowledge the earlier concern: {remaining_concern}." if remaining_concern else ""
            return (
                f"Say that the response addressed your concern and make {focus} visibly acceptable. "
                f"The concrete reason: {movement_reason}.{concern}{previous_clause} {movement_shape} "
                "Do not give only a generic fairness reason."
            )
        return (
            f"Make {focus} visibly acceptable as common ground. The concrete reason: {movement_reason}. "
            f"{previous_clause} {movement_shape} Do not claim it was always your first choice or give only a vague fairness reason."
        )
    if action.act is ActionType.FINAL_POSITION:
        return f"Briefly state that you are staying with {focus}. Do not repeat earlier reasons and do not use a voting formula."
    if action.act is ActionType.VOTE:
        if action.stance_update:
            update = action.stance_update
            movement_reason = update.movement_reason.strip() or decisive or reason
            if update.reason_already_public:
                return (
                    f"State one short, clear vote for {focus} and indicate that your choice changed. "
                    "The reason was already explained publicly, so do not invent a new reason."
                )
            return (
                f"State one clear vote for {focus} and make the changed choice clear. If you include a reason, use only this exact "
                f"structured reason: {movement_reason}. Do not summarize other options or introduce a fresh argument."
            )
        return f"State only one short, natural choice for {focus}. Three to eight words are enough; do not repeat your reason or add a new factual argument."
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
    openings = [" ".join(re.findall(r"[A-Za-z0-9’'-]+", text)[:5]) for text in own]
    return (
        "\nRecent openings to avoid repeating:\n"
        + "\n".join(f"- {opening}" for opening in openings if opening)
        + "\n"
    )



def _conversation_connection(state: DialogueState, action: UserAction) -> str:
    """Return a compact interpersonal cue for reaction-like actions."""

    if action.act not in {ActionType.ACKNOWLEDGE, ActionType.COMMENT, ActionType.COMPROMISE}:
        return ""

    target_id = action.addressee_id
    source_text = ""
    if action.issue_id and state.active_issue is not None and state.active_issue.id == action.issue_id:
        target_id = target_id or state.active_issue.opened_by
        source_text = state.active_issue.source_text.strip() or state.active_issue.summary.strip()

    if target_id is None:
        for turn in reversed(state.turns):
            if not turn.moderator and turn.speaker_id != action.speaker_id:
                target_id = turn.speaker_id
                source_text = turn.text.strip()
                break
    elif not source_text:
        for turn in reversed(state.turns):
            if not turn.moderator and turn.speaker_id == target_id:
                source_text = turn.text.strip()
                break

    if target_id is None or target_id not in state.runtimes:
        return ""
    name = state.persona(target_id).name
    visible = source_text or "the point just raised"
    own_priority = action.personal_context or action.decisive_reason or action.reason or "your own decision priority"
    return (
        f"\nConversation connection:\n- Previous speaker: {name}\n"
        f"- {name}'s visible point: {visible}\n"
        f"- Your relationship to that point: connect it to {own_priority}. Show whether it supports, conflicts with, or matters differently to your priority.\n"
        "- Respond to their reasoning; do not merely repeat the option-card wording. Mention their name only when natural.\n"
    )

def _style_tendency_block(persona: Persona) -> str:
    if not persona.style_tendencies:
        return ""
    listed = "\n".join(f"- {item}" for item in persona.style_tendencies)
    return (
        f"\nStable style tendencies:\n{listed}\n"
        "Use them as light tendencies, not mandatory phrases in every message.\n"
    )


def _movement_basis_instruction(action: UserAction) -> str:
    update = action.stance_update
    if update is None:
        return ""
    basis = update.movement_basis
    if basis == "concern_resolved" or action.issue_effect is IssueEffect.RESOLVE:
        return (
            "Movement basis: concern resolved. Explain that the response directly settled or reduced the earlier concern; "
            "do not describe an unrelated benefit as solving it."
        )
    if basis in {"common_ground", "common_ground_proposal", "stagnation_compromise"}:
        return (
            "Movement basis: group compromise. The drawback may remain; do not claim it was solved. "
            "Say the option is workable for common ground or that its benefit now outweighs the remaining trade-off."
        )
    if basis == "previous_acceptance":
        return (
            "Movement basis: previously acceptable option. It was already workable; now make the new final preference clear "
            "without inventing a fresh persuasion story."
        )
    return (
        "Movement basis: new acceptability. Explain why the target option now works for you without claiming that an unrelated concern was resolved."
    )


def realization_prompt(
    state: DialogueState,
    persona: Persona,
    action: UserAction,
) -> str:
    _, maximum = word_budget(action.act, persona.sim_params.verbosity)
    if action.act is ActionType.VOTE and action.stance_update is None:
        maximum = min(maximum, cfg.action_word_cap("simple_vote"))
    personal = (
        f"\nRelevant personal context: {action.personal_context.strip()}\n"
        if action.personal_context else ""
    )
    target = _target_question(state, action)
    own = _recent_own_language(state, persona.id)
    connection = _conversation_connection(state, action)
    style_tendencies = _style_tendency_block(persona)
    movement_basis = _movement_basis_instruction(action)
    movement_block = f"\n{movement_basis}\n" if movement_basis else ""
    return f"""You are {persona.name} in a group decision chat.
Voice: {persona.speech_style}. Maintain this voice through word choice and sentence shape; do not change facts or length for style.
{style_tendencies}Directness: {persona.sim_params.directness}/5 — {directness_instruction(persona.sim_params.directness)}
{personal}
Selected action:
{_selected_action(state, action, persona)}
{movement_block}
Relevant public option facts:
{_relevant_facts(state, action)}
{_active_issue(state)}{connection}
Recent chat:
{_recent_chat(state)}
{target}{own}
Write exactly one natural group-chat message that continues from the local context.
Express the selected meaning, but do not copy the instruction's sentence structure or rely on one fixed question or contrast formula.
Use only supplied public option facts; preserve their literal type, scope, and strength.
Use the supplied option name or short name without adding a subtype.
Treat each supplied fact as atomic: do not infer a new cost, schedule, facility, use case, consequence, guarantee, absence, or comparison.
Do not strengthen facts: “lower” must not become “significantly lower”, and “moderate” must not become “reliable”.
Do not infer cheapest, shortest, fastest, best value, balanced, or middle ground unless that exact relation is supplied.
Do not invent numbers, values, option subtypes, facilities, solutions, guarantees, or stronger/weaker versions of a fact.
Personal implications are allowed only as personal judgments such as “for me”. If information is unknown, say so.
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
    if action.act is ActionType.VOTE and action.stance_update is None:
        maximum = min(maximum, cfg.action_word_cap("simple_vote"))
    return f"""Rewrite one group-chat message for {persona.name}.

Selected action:
{_selected_action(state, action, persona)}

Relevant public facts:
{_relevant_facts(state, action)}

Original message:
{original}

Problems to fix:
- """ + "\n- ".join(errors) + f"""

Return one corrected message only.
Preserve each supplied fact's literal type, scope, and strength. Use the supplied option name or short name without adding a subtype.
Do not strengthen a fact or add unsupported relative claims such as cheapest, shortest, fastest, safest, most reliable, best value, balanced, or middle ground.
Do not invent facts, values, subtypes, facilities, use cases, consequences, guarantees, absences, comparisons, or solutions.
Preserve the selected action and any required vote or stance change.
When the action changes stance, use the concrete movement reason from the selected action; do not replace it with vague wording such as “fair enough”.
Use natural group-chat wording and preserve the participant's voice. When reacting to someone, connect their point to this participant's priority instead of merely repeating the option fact. The word “formal” is unnecessary.
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
        return "There is no clear leader yet. Is there another option anyone could accept as common ground?"
    names = [scenario.option(option_id).short_name or scenario.option(option_id).name for option_id in options]
    if len(names) == 1:
        return f"{names[0]} currently leads. Anyone not there yet can say whether they can accept it or what still prevents that."
    return f"{names[0]} and {names[1]} currently have the most support. Could either work for those currently elsewhere?"


def moderator_split_compromise_prompt(
    scenario: Scenario,
    options: tuple[str, ...],
    *,
    revote: bool = False,
) -> str:
    names = [
        scenario.option(option_id).short_name or scenario.option(option_id).name
        for option_id in options
    ]
    if not names:
        return moderator_revote_narrowing(scenario, ()) if revote else moderator_narrowing(scenario, ())
    if len(names) == 1:
        listed = names[0]
    elif len(names) == 2:
        listed = f"{names[0]} and {names[1]}"
    else:
        listed = f"{', '.join(names[:-1])}, and {names[-1]}"
    prefix = "The first vote had no majority, and we are" if revote else "We are"
    return (
        f"{prefix} still split between {listed}. "
        "Is anyone willing to accept another of these as common ground?"
    )



def moderator_no_movement_bridge(*, revote: bool = False) -> str:
    if revote:
        return "No one changed position, so the split remains."
    return "No one? All right, let’s record the final votes."

def moderator_revote_narrowing(scenario: Scenario, options: tuple[str, ...]) -> str:
    if not options:
        return "The first vote had no majority. Is there any option someone can newly accept as common ground?"
    names = [scenario.option(option_id).short_name or scenario.option(option_id).name for option_id in options]
    return f"The first vote had no majority. Can anyone newly accept {' or '.join(names)} as common ground?"


def moderator_vote_request(
    *,
    revote: bool = False,
    scenario: Scenario | None = None,
    unanimous_option: str | None = None,
) -> str:
    if revote:
        return "Let’s take the final vote again. Please name the one option you’re choosing."
    if scenario is not None and unanimous_option:
        name = scenario.option(unanimous_option).short_name or scenario.option(unanimous_option).name
        return f"{name} already has everyone’s support. Let’s confirm it with a final vote."
    return "Let’s take the final vote. Please name the one option you’re choosing."


def moderator_closure(outcome: RunOutcome, scenario: Scenario) -> str:
    if outcome.final_option:
        name = scenario.option(outcome.final_option).short_name or scenario.option(outcome.final_option).name
        return f"The result is {outcome.status}: {name}."
    return "No option reached a majority, so the result is unresolved."
