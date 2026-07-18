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


_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"(?<!\w)(?:[$€£]\s*)?\d+(?:[.,]\d+)?(?:\s*%|\s*[a-zA-Z]+)?")
_ACCEPTANCE_PATTERNS = (
    "acceptable",
    "works for me",
    "workable",
    "viable",
    "i can go with",
    "i could go with",
    "i am willing to",
    "i'm willing to",
    "i can accept",
    "i could accept",
    "fine with",
    "okay with",
    "now prefer",
    "switch to",
    "move to",
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

    if action.option_focus and not _focus_is_visible_or_contextual(state, action, clean):
        errors.append("focused option is not visible")

    if action.act is ActionType.ASK:
        if "?" not in clean:
            errors.append("question is not visibly phrased as a question")
        if action.addressee_id:
            name = state.persona(action.addressee_id).name
            if not re.search(rf"\b{re.escape(name)}\b", clean, flags=re.I):
                errors.append("direct question does not name the addressee")

    if action.act is ActionType.ANSWER and not _answer_is_relevant(state, action, clean):
        errors.append("answer does not address the active question")

    if action.act is ActionType.VOTE:
        visible_vote = resolve_visible_vote(clean, state.scenario)
        if visible_vote != action.vote_option:
            errors.append("visible vote does not match the structured vote")

    if action.stance_update is not None and not _stance_update_visible(state, action, clean):
        errors.append("stance movement is not visible")

    if persona.hard_blocker and _contradicts_hard_blocker(state, persona, clean):
        errors.append("hard blocker visibly accepts a rejected option")

    errors.extend(_numeric_grounding_errors(state, action, clean))
    errors.extend(_strengthening_errors(action, clean))
    return errors



def _focus_is_visible_or_contextual(
    state: DialogueState, action: UserAction, text: str
) -> bool:
    required = set(action.option_focus)
    if not required:
        return True
    if required & mentioned_options(text, state):
        return True
    if action.act not in {
        ActionType.REACT,
        ActionType.SUPPORT,
        ActionType.OBJECT,
        ActionType.ANSWER,
    }:
        return False
    thread = state.active_thread
    if thread is not None and required <= set(thread.option_focus):
        return True
    previous = next(
        (turn for turn in reversed(state.participant_turns) if turn.action is not None),
        None,
    )
    return bool(previous and required <= set(previous.action.option_focus))

def _terms(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "for", "with", "that",
        "this", "it", "is", "are", "was", "were", "be", "me", "my", "our",
        "option", "works", "work", "choice", "because", "still", "would", "could",
    }
    return {word for word in _WORD_RE.findall(text.lower()) if len(word) >= 3 and word not in stop}


def _answer_is_relevant(
    state: DialogueState, action: UserAction, text: str
) -> bool:
    thread = state.active_thread
    if thread is None or not thread.option_focus:
        return False
    if not _focus_is_visible_or_contextual(state, action, text):
        return False
    answer_terms = _terms(text)
    target_terms = _terms(thread.source_text)
    if action.reason_source:
        target_terms |= _terms(action.reason_source.public_value)
        target_terms |= _terms(action.reason_source.attribute_name)
    direct = any(
        marker in text.lower()
        for marker in (
            "yes", "no", "works", "doesn't", "does not", "acceptable", "not acceptable",
            "fine", "not enough", "enough", "matters", "concern", "prefer",
        )
    )
    return direct and bool(answer_terms & target_terms)


def _stance_update_visible(
    state: DialogueState, action: UserAction, text: str
) -> bool:
    update = action.stance_update
    if update is None or not option_mentioned(text, state, update.option_id):
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in _ACCEPTANCE_PATTERNS)


def _contradicts_hard_blocker(
    state: DialogueState, persona: Persona, text: str
) -> bool:
    lowered = text.lower()
    acceptance = any(pattern in lowered for pattern in _ACCEPTANCE_PATTERNS)
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
    return " ".join(pieces).lower().replace(",", ".")


def _numeric_grounding_errors(
    state: DialogueState, action: UserAction, text: str
) -> list[str]:
    allowed = _allowed_numeric_text(state, action)
    errors: list[str] = []
    for match in _NUMBER_RE.findall(text):
        normalized = " ".join(match.lower().replace(",", ".").split())
        number = re.search(r"\d+(?:\.\d+)?", normalized)
        if number and number.group(0) not in allowed:
            errors.append(f"unsupported numeric claim: {match.strip()}")
    return errors


def _strengthening_errors(action: UserAction, text: str) -> list[str]:
    lowered = text.lower()
    source = " ".join(
        part
        for part in (
            action.reason,
            action.reason_source.public_value if action.reason_source else "",
        )
        if part
    ).lower()
    markers = ("guarantees", "guaranteed", "ensures", "no risk", "everyone will", "always")
    return [
        f"unsupported strengthened claim: {marker}"
        for marker in markers
        if marker in lowered and marker not in source
    ]
