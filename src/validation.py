"""Minimal hard validation for structured actions and their utterances."""

from __future__ import annotations

import re

from aliases import resolve_option_mentions, resolve_visible_vote
from models import (
    ActionType,
    DialogueState,
    Persona,
    StanceUpdateKind,
    UserAction,
)


_NUMBER_RE = re.compile(r"(?<!\w)(?:[$€£]\s*)?\d+(?:[.,]\d+)?(?:\s*%)?")
_PERSONAL_ACCEPTANCE_PATTERNS = (
    "works for me",
    "works better for me",
    "better fit for me",
    "i can go with",
    "i could go with",
    "i'd go with",
    "i’d go with",
    "i would go with",
    "i am willing to",
    "i'm willing to",
    "i’m willing to",
    "i can accept",
    "i could accept",
    "i'm fine with",
    "i’m fine with",
    "i'm okay with",
    "i’m okay with",
    "now prefer",
    "prefer this now",
    "switch to",
    "move to",
    "lean toward",
    "leaning toward",
)


def option_mentioned(text: str, state: DialogueState, option_id: str) -> bool:
    return option_id in resolve_option_mentions(text, state.scenario)


def mentioned_options(text: str, state: DialogueState) -> set[str]:
    return resolve_option_mentions(text, state.scenario)


def validate_action(
    state: DialogueState, persona: Persona, action: UserAction
) -> list[str]:
    errors: list[str] = []
    if action.speaker_id != persona.id:
        errors.append("action speaker does not match persona")
    invalid = [
        option_id
        for option_id in action.option_focus
        if option_id not in state.scenario.option_ids
    ]
    if invalid:
        errors.append(f"action references invalid options: {invalid}")
    if action.addressee_id and action.addressee_id not in state.runtimes:
        errors.append("action addressee does not exist")
    if action.addressee_id == action.speaker_id:
        errors.append("action cannot address its own speaker")
    if action.act is ActionType.VOTE:
        if action.vote_option not in state.scenario.option_ids:
            errors.append("vote requires one valid option")
        if persona.hard_blocker and action.vote_option != persona.preferred_option:
            errors.append("hard blocker cannot vote for another option")
    if action.stance_update is not None:
        target = action.stance_update.option_id
        runtime = state.runtimes[persona.id]
        if target not in state.scenario.option_ids:
            errors.append("stance update targets an invalid option")
        if persona.hard_blocker:
            errors.append("hard blocker cannot change stance")
        if target in runtime.hard_rejected_options:
            errors.append("stance update targets a hard-rejected option")
        if action.stance_update.kind is StanceUpdateKind.SWITCH_PREFERRED and target == runtime.preferred_option:
            errors.append("preference switch must target a different option")
    if action.act in {ActionType.OPENING, ActionType.SUPPORT, ActionType.OBJECT, ActionType.ASK, ActionType.ANSWER, ActionType.ACCEPT, ActionType.VOTE} and not action.option_focus:
        errors.append(f"{action.act.value} requires an option focus")
    if action.act is ActionType.COMPARE:
        if len(action.option_focus) != 2 or len(set(action.option_focus)) != 2:
            errors.append("comparison requires two different option focuses")
        if len(action.comparison_sources) != 2:
            errors.append("comparison requires two grounded sources")
        elif {source.option_id for source in action.comparison_sources} != set(action.option_focus):
            errors.append("comparison sources must match the two focused options")
    elif action.comparison_sources:
        errors.append("comparison sources are only valid for comparison actions")
    return errors


def validate_realization(
    state: DialogueState,
    persona: Persona,
    action: UserAction,
    text: str,
) -> list[str]:
    errors: list[str] = []
    clean = str(text or "").strip()
    if not clean:
        return ["empty output"]
    if len(clean.split()) > 80:
        errors.append("utterance is unusably long")
    if re.match(r"^(moderator|assistant|system|user)\s*:", clean, flags=re.I):
        errors.append("utterance contains an invalid speaker label")

    # Missing an exact alias is a minor realization issue for ordinary discussion
    # turns. Keep explicit references hard only where public state changes or the
    # protocol depends on an unambiguous option.
    if (
        action.option_focus
        and action.act in {ActionType.OPENING, ActionType.ACCEPT, ActionType.VOTE}
        and not _focus_is_visible_or_contextual(state, action, clean)
    ):
        errors.append("focused option is not visible")

    if action.act is ActionType.ASK:
        if "?" not in clean:
            errors.append("question is not visibly phrased as a question")
        if action.addressee_id:
            name = state.persona(action.addressee_id).name
            if not re.search(rf"\b{re.escape(name)}\b", clean, flags=re.I):
                errors.append("direct question does not name the addressee")

    if action.act is ActionType.VOTE:
        visible_vote = resolve_visible_vote(clean, state.scenario)
        if visible_vote != action.vote_option:
            errors.append("visible vote does not match the structured vote")

    if action.stance_update is not None and not _stance_update_visible(state, action, clean):
        errors.append("stance movement is not visible")

    if persona.hard_blocker and _contradicts_hard_blocker(state, persona, clean):
        errors.append("hard blocker visibly accepts a rejected option")

    errors.extend(_numeric_grounding_errors(state, action, clean))
    return errors



def _focus_is_visible_or_contextual(
    state: DialogueState, action: UserAction, text: str
) -> bool:
    required = set(action.option_focus)
    if not required:
        return True
    visible = mentioned_options(text, state)
    if action.act is ActionType.COMPARE:
        return required <= visible
    if required & visible:
        return True
    if action.act not in {
        ActionType.REACT,
        ActionType.SUPPORT,
        ActionType.OBJECT,
        ActionType.ANSWER,
    }:
        return False
    thread = state.active_thread
    if (
        thread is not None
        and len(thread.option_focus) == 1
        and required == set(thread.option_focus)
    ):
        return True
    previous = next(
        (turn for turn in reversed(state.participant_turns) if turn.action is not None),
        None,
    )
    return bool(
        previous
        and len(previous.action.option_focus) == 1
        and required == set(previous.action.option_focus)
    )

def _stance_update_visible(
    state: DialogueState, action: UserAction, text: str
) -> bool:
    update = action.stance_update
    return bool(update and option_mentioned(text, state, update.option_id))


def _contradicts_hard_blocker(
    state: DialogueState, persona: Persona, text: str
) -> bool:
    lowered = text.lower()
    acceptance = any(pattern in lowered for pattern in _PERSONAL_ACCEPTANCE_PATTERNS)
    if not acceptance:
        return False
    mentioned = mentioned_options(text, state)
    return any(option_id != persona.preferred_option for option_id in mentioned)


def _allowed_numeric_text(state: DialogueState, action: UserAction) -> str:
    pieces = [state.scenario.context_text, action.reason]
    for option_id in action.option_focus:
        option = state.scenario.option(option_id)
        pieces.extend(option.public_values())
    if action.reason_source:
        pieces.append(action.reason_source.public_value)
    pieces.extend(source.public_value for source in action.comparison_sources)
    return " ".join(pieces).lower().replace(",", ".")


def _numeric_phrase_is_in_focused_option_reference(
    state: DialogueState, action: UserAction, phrase: str
) -> bool:
    normalized = " ".join(phrase.lower().replace(",", ".").split())
    for option_id in action.option_focus:
        option = state.scenario.option(option_id)
        references = (option.name, option.short_name, *option.aliases)
        for reference in references:
            candidate = " ".join(str(reference).lower().replace(",", ".").split())
            if normalized and normalized in candidate:
                return True
    return False


def _numeric_grounding_errors(
    state: DialogueState, action: UserAction, text: str
) -> list[str]:
    allowed = _allowed_numeric_text(state, action)
    errors: list[str] = []
    for match in _NUMBER_RE.findall(text):
        normalized = " ".join(match.lower().replace(",", ".").split())
        if _numeric_phrase_is_in_focused_option_reference(state, action, normalized):
            continue
        number = re.search(r"\d+(?:\.\d+)?", normalized)
        if number and number.group(0) not in allowed:
            errors.append(f"unsupported numeric claim: {match.strip()}")
    return errors
