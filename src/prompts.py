"""Compact setup, realization, repair, and moderator prompts."""

from __future__ import annotations

import json
import re

from aliases import option_aliases, resolve_visible_vote
from config_loader import cfg
from models import ActionType, DialogueState, Phase, RunOutcome, Scenario, UserAction


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


def setup_aliases(options: list[dict[str, str]], participant_ids: list[str]) -> str:
    return f"""Create concise setup metadata for fixed option names and participants.
Return JSON only as {{"aliases": [{{"id": "A", "aliases": ["short reference"]}}], "participant_names": [{{"id": "p1", "name": "Maya"}}]}}.

Fixed option names:
{_schema(options)}
Participant IDs:
{_schema(participant_ids)}

Rules:
- Return every supplied option ID and participant ID exactly once.
- Give each option 1 or 2 natural aliases of at most {int(cfg.scenario.short_alias_max_words)} words.
- Each alias must use words from that option's exact full name, in the same order.
- Prefer distinctive references people would naturally use in chat.
- Do not use generic category words when they could describe another option.
- Every alias must contain at least two words and no numbers.
- Do not return incomplete phrases ending in words such as "to", "with", "and", or "the".
- Aliases must be unique across options after normalization.
- Give each participant one unique short first name containing letters only.
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
    if action is ActionType.VOTE:
        return (2, 10)
    maximum = int(
        cfg.level_value("language", "max_words_by_verbosity", verbosity, cast=int)
    )
    cap_name = {
        ActionType.ASK: "ask",
        ActionType.ANSWER: "answer",
    }.get(action)
    if cap_name:
        maximum = min(maximum, cfg.action_word_cap(cap_name))
    return (4, maximum)


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
    lines = [
        f"- {option_id}: {state.scenario.option(option_id).name}"
        for option_id in action.option_focus
    ]
    if action.comparison_sources:
        for source in action.comparison_sources:
            name = state.scenario.option(source.option_id).short_name
            lines.append(
                f"- Fact for {name} only: {source.attribute_name}: {source.public_value}"
            )
    elif action.reason_source:
        lines.append(
            f"- Grounded source: {action.reason_source.attribute_name}: "
            f"{action.reason_source.public_value}"
        )
    if action.reason:
        lines.append(f"- Intended point: {action.reason}")
    return "\n".join(lines) or "- use only the selected action point"


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
    if (
        previous is None
        or previous.action is None
        or not set(previous.action.option_focus) & set(action.option_focus)
    ):
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
        return f"Give a short, natural first contribution. Your choice is {focus}; explain it with: {reason}. A greeting is optional."
    if action.act is ActionType.SUPPORT:
        return (
            f"Continue the exchange with why the focused choice suits you, grounded in: {reason}. "
            "Do not mechanically lead with the option name."
        )
    if action.act is ActionType.OBJECT:
        return (
            f"Continue the exchange with a concern about the focused choice, grounded in: {reason}. "
            "Do not turn it into a question or lead mechanically with the option name."
        )
    if action.act is ActionType.REACT:
        return (
            f"Respond to the previous point using: {reason}. Agree, qualify, disagree, or add a personal consequence "
            "without restating the option card."
        )
    if action.act is ActionType.COMPARE:
        return (
            "Contrast the two focused options using the supplied values for the same public attribute. "
            "State both values accurately and say which fits your priority better. "
            "Keep the wording conversational; do not infer another fact or use one fixed contrast template."
        )
    if action.act is ActionType.ASK:
        target = state.persona(action.addressee_id).name if action.addressee_id else "the group"
        return f"Ask {target} one natural question that connects the grounded point to the current exchange: {reason}."
    if action.act is ActionType.ANSWER:
        if state.phase is Phase.NARROWING:
            return (
                f"Answer the moderator's question about whether {focus} fits your requirements. "
                f"Say no naturally or explain the remaining concern, grounded in: {reason}."
            )
        question = state.active_thread.source_text if state.active_thread else "the current question"
        return (
            f"Reply naturally and directly to this question: {question}\n"
            f"Use this grounded point in your answer: {reason}. A forced yes/no opening is not required."
        )
    if action.act is ActionType.ACCEPT:
        previous = (
            _option_name(state, action.stance_update.previous_option_id)
            if action.stance_update and action.stance_update.previous_option_id
            else "your earlier choice"
        )
        switching = bool(
            action.stance_update
            and action.stance_update.kind.value == "switch_preferred"
        )
        movement = "now prefer" if switching else "could now accept"
        if state.phase is Phase.NARROWING and not switching:
            return (
                f"Answer the moderator directly: say naturally that {focus} could fit your requirements. "
                f"Explain why using the discussion and this grounded point: {reason}. "
                "Do not claim that everyone agrees."
            )
        return (
            f"Show naturally that you {movement} {focus} rather than simply repeating {previous}. "
            f"Ground the reconsideration in the discussion and this point: {reason}. "
            "It may help the group move forward, but it must still sound personally plausible."
        )
    raise ValueError(f"unsupported action: {action.act}")


def _explicit_reference_required(action: UserAction) -> bool:
    # Only turns that establish or change a public preference must repeat an
    # explicit option reference. Ordinary questions and comparisons may rely on
    # the live exchange instead of sounding like option-card narration.
    return action.act in {ActionType.OPENING, ActionType.ACCEPT}



def realization_prompt(state: DialogueState, action: UserAction) -> str:
    persona = state.persona(action.speaker_id)
    minimum, maximum = word_budget(action.act, persona.sim_params.verbosity)
    style = "; ".join(persona.style_tendencies) or persona.speech_style
    if action.act is ActionType.ASK and action.addressee_id:
        addressee_rule = (
            f"Include {state.persona(action.addressee_id).name} naturally somewhere in the question."
        )
    elif action.addressee_id:
        addressee_rule = (
            "Respond to the previous speaker without automatically starting with their name."
        )
    else:
        addressee_rule = "Do not invent a specific addressee."
    thread = (
        f"Active sub-discussion: {state.active_thread.source_text}"
        if state.active_thread
        else "No active sub-discussion."
    )
    reference_rule = (
        "Use one allowed option reference somewhere in the message; it does not need to begin the sentence."
        if _explicit_reference_required(action)
        else "Do not repeat an option name only for formality; use it when clarity needs it, otherwise continue contextually."
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
- Use {minimum}..{maximum} words and finish the thought. {reference_rule}
- Write like this persona continuing a live group chat. When relevant, begin with a short reaction or continuation instead of restarting from the option card.
- Vary sentence structure with the persona style. Do not routinely begin with the option name, another participant's name, or “I”; names and option references can appear later.
- Short acknowledgements, direct additions, “still”, “though”, or a separate sentence are useful. “But” is fine occasionally, not as the default structure.
- Avoid repeatedly writing “Option + fact + helps/limits/makes ...” or “X makes sense, but Y”. State the fact naturally and keep the conclusion personal.
- Use only supplied public facts. Do not invent details, copy recent wording, add a speaker label, or narrate the act of speaking.
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
    leader: str,
    holdout_names: tuple[str, ...],
    *,
    preference_count: int,
    participant_count: int,
    selected_from_tie: bool = False,
    variant: int = 0,
) -> str:
    option_name = scenario.option(leader).short_name
    if len(holdout_names) == 1:
        people = holdout_names[0]
    elif len(holdout_names) == 2:
        people = f"{holdout_names[0]} and {holdout_names[1]}"
    else:
        people = f"{', '.join(holdout_names[:-1])}, and {holdout_names[-1]}"
    if selected_from_tie:
        return _pick_variant(
            variant,
            (
                f"The strongest common-ground options are tied, so I’ll use {option_name} as the compromise target. {people}, would it fit your requirements?",
                f"Public support is tied at the top, so let’s test {option_name} as the option to narrow around. {people}, could you accept it for the group?",
                f"There is no unique leader, so I’m selecting {option_name} from the tied options for this compromise check. {people}, would that choice work for you?",
                f"The leading options remain tied. Let’s use {option_name} as the bounded tie-break target. {people}, could it meet your main requirements?",
            ),
        )
    return _pick_variant(
        variant,
        (
            f"{option_name} currently has {preference_count} of {participant_count} public preferences and the broadest overall support, but not a majority. {people}, would it fit your requirements?",
            f"{option_name} is currently the leading option, though it is still short of a majority. {people}, could you accept it for the group?",
            f"The discussion currently leans most toward {option_name}, without a majority yet. {people}, would that choice work for you?",
            f"{option_name} has the strongest public support so far, but the group is still divided. {people}, could it meet your main requirements?",
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


def deterministic_vote_text(
    scenario: Scenario,
    option_id: str,
    *,
    variant: int = 0,
) -> str:
    name = scenario.option(option_id).short_name
    choices = (
        f"My final vote is {name}.",
        f"I’m voting for {name}.",
        f"{name} is my final choice.",
        f"I choose {name}.",
    )
    text = _pick_variant(variant, choices)
    if resolve_visible_vote(text, scenario) == option_id:
        return text

    # A valid short name can still overlap another option's alias, making the
    # generic visible-vote resolver intentionally ambiguous. The explicit
    # option identifier is guaranteed and keeps deterministic voting safe.
    name = f"Option {option_id}"
    return _pick_variant(
        variant,
        (
            f"My final vote is {name}.",
            f"I’m voting for {name}.",
            f"{name} is my final choice.",
            f"I choose {name}.",
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
