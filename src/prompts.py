"""Compact setup, realization, repair, and moderator prompts."""

from __future__ import annotations

import json
import re

from aliases import option_aliases
from config_loader import cfg
from models import ActionType, DialogueState, RunOutcome, Scenario, UserAction


def _schema(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def setup_scenario(
    topic: str,
    n: int,
    *,
    validation_feedback: str = "",
) -> str:
    labels = [str(value) for value in cfg.scenario.option_labels]
    feedback = (
        f"\nThe previous board was invalid: {validation_feedback}\nCorrect that error in the new board."
        if validation_feedback
        else ""
    )
    return f"""Create one public option board for a group decision.

Topic: {topic}
Deciding participants: {n}
Required option IDs in order: {labels}
{feedback}
Return JSON only:
{{
  "scenario": {{
    "shared_context": "one or two complete sentences",
    "options": [
      {{
        "id": "A",
        "name": "full unique name",
        "attrs": {{"topic-specific attribute": "public value"}},
        "upside": "brief public advantage",
        "concern": "brief public drawback"
      }}
    ]
  }}
}}

Rules:
- Create exactly {len(labels)} realistic, distinct options.
- Give every option {int(cfg.scenario.public_attr_min)}..{int(cfg.scenario.public_attr_max)} concise factual attributes.
- shared_context must contain 1..{int(cfg.scenario.get('shared_context_max_sentences', 2))} sentences and at most {int(cfg.scenario.shared_context_max_words)} words.
- Put every objective fact in shared_context or an option card.
- Use consistent attribute names for comparable facts.
- Every option must satisfy the hard requirements in the shared context. A drawback may reduce appeal, but must not make the option unusable for the stated group.
- Avoid superlatives and rankings such as cheapest, fastest, best, nearest, highest, or lowest.
- Do not recommend a winner or mention generated participants.
"""


def setup_aliases(options: list[dict[str, str]]) -> str:
    return f"""Create natural short references for four fixed option names.
Return JSON only as {{"aliases": [{{"id": "A", "aliases": ["short reference", "alternative reference"]}}]}}.

Fixed option names:
{_schema(options)}

Rules:
- Return every supplied option ID exactly once.
- Give each option 1..3 natural aliases of at most {int(cfg.scenario.short_alias_max_words)} words.
- Each alias must use words from that option's exact full name, in the same order.
- Prefer distinctive references people would naturally use in chat, such as "Chicago" for "Chicago City Stay".
- Do not use generic category words such as "the hotel", "the restaurant", or "the option" when they could describe another option.
- Aliases must be unique across options after case, punctuation, article, and accent normalization.
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
    blocker_rule = (
        f"{hard_blocker_id} is the only hard blocker: rank its preferred option 5 and every alternative 1 with a grounded reason_against."
        if hard_blocker_id
        else "No participant is a hard blocker; do not rank every alternative 1."
    )
    return f"""Create {n} concise persona cards for this fixed decision.
Return JSON only as {{"participants": [...]}}.

Topic: {topic}
Shared context: {shared_context}
Options: {_schema(options_json)}
Fixed IDs, optional names, and traits: {_schema(trait_rows)}
Required primary preferences: {_schema(required_preferences)}

Each participant object must contain:
- id and one unique short first name;
- one-sentence background;
- one-sentence private_goal;
- age from 18 to 80;
- preferred_options beginning with the required option;
- option_stances for every option with rank 1..5, reason_for, and reason_against.

Rules:
- Preserve supplied names exactly; generate a unique first name when absent.
- Reasons must use public option facts or clearly personal priorities.
- Do not treat an option as a primary preference when it clearly violates a hard shared requirement.
- 5 means preferred, 4 acceptable, 3 neutral, 2 disliked, 1 hard rejected.
- {blocker_rule}
- Do not output traits or speech_style.
"""


def word_budget(action: ActionType, verbosity: int) -> tuple[int, int]:
    maximum = int(
        cfg.level_value("language", "max_words_by_verbosity", verbosity, cast=int)
    )
    cap_name = {
        ActionType.ASK: "ask",
        ActionType.ANSWER: "answer",
        ActionType.VOTE: "vote",
    }.get(action)
    if cap_name:
        maximum = min(maximum, cfg.action_word_cap(cap_name))
    return (2 if action is ActionType.VOTE else 4, maximum)


def directness_instruction(level: int) -> str:
    return str(
        cfg.level_value("language", "directness_instructions", level, cast=str)
    )


def _option_name(state: DialogueState, option_id: str) -> str:
    return state.scenario.option(option_id).short_name


def _recent_chat(state: DialogueState) -> str:
    limit = int(cfg.conversation.recent_turns_in_prompt)
    turns = state.turns[-limit:]
    if not turns:
        return "(no previous turns)"
    return "\n".join(f"{turn.speaker_name}: {turn.text}" for turn in turns)


def _recent_own_openings(state: DialogueState, speaker_id: str) -> str:
    own = [
        turn.text
        for turn in state.participant_turns
        if turn.speaker_id == speaker_id
    ][-2:]
    openings = [
        " ".join(re.findall(r"[A-Za-z0-9’'-]+", text)[:5])
        for text in own
    ]
    openings = [opening for opening in openings if opening]
    if not openings:
        return "(none yet)"
    return "\n".join(f"- {opening}" for opening in openings)


def _focused_facts(state: DialogueState, action: UserAction) -> str:
    ids = action.option_focus or tuple(state.scenario.option_ids[:1])
    return "\n".join(state.scenario.option(option_id).public_line() for option_id in ids)


def _allowed_references(state: DialogueState, action: UserAction) -> str:
    lines: list[str] = []
    for option_id in action.option_focus:
        refs = ", ".join(option_aliases(state.scenario, option_id))
        lines.append(f"- {option_id}: {refs}")
    return "\n".join(lines) or "- no explicit option reference required"


def _last_other_turn(state: DialogueState, speaker_id: str):
    return next(
        (
            turn
            for turn in reversed(state.participant_turns)
            if turn.speaker_id != speaker_id
        ),
        None,
    )


def _connection_block(state: DialogueState, action: UserAction) -> str:
    if action.act not in {
        ActionType.REACT,
        ActionType.SUPPORT,
        ActionType.OBJECT,
        ActionType.COMPARE,
        ActionType.ACCEPT,
    }:
        return ""
    previous = _last_other_turn(state, action.speaker_id)
    if previous is None:
        return ""
    return (
        "\nConversation connection:\n"
        f"- Previous visible point from {previous.speaker_name}: {previous.text}\n"
        "- Continue from that point: acknowledge, qualify, contrast, or add a distinct consequence for your own priority.\n"
        "- Do not merely restate the option-card wording. Mention the speaker's name only when natural.\n"
    )


def _action_instruction(state: DialogueState, action: UserAction) -> str:
    names = [_option_name(state, option_id) for option_id in action.option_focus]
    focus = " and ".join(names) or "the current point"
    reason = action.reason or "the supplied public trade-off"
    if action.act is ActionType.OPENING:
        return f"Join the opening briefly. Your current choice is {focus}; explain it using: {reason}."
    if action.act is ActionType.SUPPORT:
        return f"Continue the discussion with a distinct supporting point about {focus}: {reason}."
    if action.act is ActionType.OBJECT:
        return f"Respond with a concrete concern about {focus}: {reason}. Do not turn it into a question."
    if action.act is ActionType.REACT:
        target = state.persona(action.addressee_id).name if action.addressee_id else "the previous speaker"
        return f"React to {target}'s visible point about {focus}, connecting it to your priority with: {reason}."
    if action.act is ActionType.COMPARE:
        return f"Compare the trade-off between {focus} and explain which consideration matters to you: {reason}."
    if action.act is ActionType.ASK:
        target = state.persona(action.addressee_id).name if action.addressee_id else "the group"
        return f"Ask {target} one natural question about how this concrete point affects {focus}: {reason}."
    if action.act is ActionType.ANSWER:
        question = state.active_thread.source_text if state.active_thread else "the current question"
        return (
            f"Answer this exact question directly in the first sentence: {question}\n"
            f"State how {focus} works or does not work for you, then explain with: {reason}."
        )
    if action.act is ActionType.ACCEPT:
        previous = (
            _option_name(state, action.stance_update.previous_option_id)
            if action.stance_update and action.stance_update.previous_option_id
            else "your earlier choice"
        )
        return (
            f"Make the movement explicit: say that {focus} is now acceptable or preferred over {previous}. "
            f"Explain it using: {reason}."
        )
    if action.act is ActionType.VOTE:
        return f"State one short, unambiguous final vote for {focus}; do not repeat the full argument."
    raise ValueError(f"unsupported action: {action.act}")


def _explicit_reference_required(action: UserAction) -> bool:
    return action.act in {
        ActionType.OPENING,
        ActionType.ASK,
        ActionType.COMPARE,
        ActionType.ACCEPT,
        ActionType.VOTE,
    }


def realization_prompt(state: DialogueState, action: UserAction) -> str:
    persona = state.persona(action.speaker_id)
    minimum, maximum = word_budget(action.act, persona.sim_params.verbosity)
    style = "; ".join(persona.style_tendencies) or persona.speech_style
    addressee_rule = (
        f"Name {state.persona(action.addressee_id).name} when directly addressing them."
        if action.addressee_id
        else "Do not invent a specific addressee."
    )
    thread = (
        f"Active sub-discussion: {state.active_thread.source_text}"
        if state.active_thread
        else "No active sub-discussion."
    )
    reference_rule = (
        "Use one allowed option reference somewhere in the message. It does not need to begin the sentence."
        if _explicit_reference_required(action)
        else "A pronoun or local reference is allowed when the previous message or active sub-discussion clearly identifies the option; otherwise use an allowed reference."
    )
    return f"""Write exactly one natural group-chat message for {persona.name}.
Output only the message.

Persona:
- background: {persona.background}
- private priority: {persona.private_goal}
- voice: {style}
- directness: {directness_instruction(persona.sim_params.directness)}

Selected action:
{_action_instruction(state, action)}
{addressee_rule}

Relevant public option facts:
{_focused_facts(state, action)}
Allowed option references:
{_allowed_references(state, action)}
{thread}{_connection_block(state, action)}
Recent chat:
{_recent_chat(state)}

Recent openings by {persona.name} to avoid repeating:
{_recent_own_openings(state, persona.id)}

Rules:
- Use {minimum}..{maximum} words when natural and finish the thought.
- {reference_rule}
- Continue from the local exchange rather than restating the action instruction.
- Vary the sentence opening; do not mechanically start with “I”, a participant name, or the option name.
- Do not copy the wording or sentence structure of your recent turns.
- Short acknowledgments, contractions, and “we”/“us” are valid when natural.
- Use only supplied public facts; frame implications and judgments as personal.
- Do not invent numbers, dishes, services, facilities, guarantees, missing information, comparisons, or outside facts.
- Do not add a speaker label or summarize the discussion.
"""


def repair_prompt(
    state: DialogueState,
    action: UserAction,
    raw_text: str,
    errors: list[str],
) -> str:
    return f"""Rewrite one group-chat message so it realizes the same selected action.
Output only the corrected message.

Original: {raw_text or '(empty)'}
Problems:
- """ + "\n- ".join(errors) + f"""

{realization_prompt(state, action)}
"""


def _pick_variant(variant: int, choices: tuple[str, ...]) -> str:
    return choices[variant % len(choices)]


def moderator_opening(scenario: Scenario, *, variant: int = 0) -> str:
    return _pick_variant(
        variant,
        (
            f"Today we need to decide: {scenario.topic}.",
            f"Let’s work through the options for: {scenario.topic}.",
            f"Our decision is: {scenario.topic}.",
            f"We’re here to choose between the listed options for: {scenario.topic}.",
        ),
    )


def moderator_stall_prompt(*, variant: int = 0) -> str:
    return _pick_variant(
        variant,
        (
            "What trade-off matters most before we decide?",
            "Does anyone have a different reaction to what has been said?",
            "Let’s hear one more concrete point before we narrow this down.",
            "Is there a remaining concern or useful comparison?",
        ),
    )


def moderator_compromise_prompt(
    scenario: Scenario,
    options: tuple[str, ...],
    *,
    variant: int = 0,
) -> str:
    names = " and ".join(scenario.option(option_id).short_name for option_id in options)
    verb = "offers" if len(options) == 1 else "offer"
    return _pick_variant(
        variant,
        (
            f"The group is still split. Could anyone accept {names} as a workable group choice?",
            f"Before the final vote, is anyone willing to move toward {names}?",
            f"Let’s test whether {names} {verb} a workable compromise.",
            f"Does anyone see enough common ground around {names} to move?",
        ),
    )


def moderator_narrowing(
    scenario: Scenario, options: tuple[str, ...], *, variant: int = 0
) -> str:
    names = " and ".join(scenario.option(option_id).short_name for option_id in options)
    return _pick_variant(
        variant,
        (
            f"The discussion has narrowed to {names}.",
            f"The clearest remaining choice is {names}.",
            f"We now have a clearer direction around {names}.",
            f"The group has moved toward {names}.",
        ),
    )


def moderator_decisive_lead(
    scenario: Scenario,
    option_id: str,
    support: int,
    total: int,
    *,
    variant: int = 0,
) -> str:
    name = scenario.option(option_id).short_name
    return _pick_variant(
        variant,
        (
            f"{name} has a clear {support}–{total - support} lead.",
            f"The group has a clear majority for {name}.",
            f"{support} of {total} participants currently support {name}.",
            f"There is a decisive lead for {name}.",
        ),
    )


def moderator_vote_request(*, scenario: Scenario, variant: int = 0) -> str:
    del scenario
    return _pick_variant(
        variant,
        (
            "Let’s record everyone’s final vote.",
            "Please state your final choice.",
            "We’ll move to the final vote now.",
            "Let’s confirm the final choices.",
        ),
    )


def moderator_closure(
    outcome: RunOutcome, scenario: Scenario, *, variant: int = 0
) -> str:
    if outcome.final_option:
        name = scenario.option(outcome.final_option).short_name
        if outcome.status == "successful":
            return _pick_variant(
                variant,
                (
                    f"The group unanimously chose {name}.",
                    f"Everyone agreed on {name}.",
                    f"The final decision is unanimously {name}.",
                    f"We have full agreement on {name}.",
                ),
            )
        return _pick_variant(
            variant,
            (
                f"The majority chose {name}.",
                f"{name} received the final majority.",
                f"The decision closes with a majority for {name}.",
                f"The final result is a majority for {name}.",
            ),
        )
    return _pick_variant(
        variant,
        (
            "No option reached a majority, so the result remains unresolved.",
            "The group remains split and no final option was selected.",
            "The discussion closes without a majority decision.",
            "The final vote remained split, so the decision is unresolved.",
        ),
    )
