"""Minimal deterministic validation and option grounding.

The structured :class:`UserAction` is authoritative. Validation blocks only
hard correctness failures and required visible state changes; ordinary act
wording is not reconstructed from natural language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from aliases import resolve_option_mentions, resolve_visible_vote
from config_loader import cfg
from models import (
    ActionType,
    DialogueState,
    IssueEffect,
    Persona,
    StanceUpdateKind,
    UserAction,
)


@dataclass(slots=True)
class ValidationResult:
    errors: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def option_mentioned(text: str, state: DialogueState, option_id: str) -> bool:
    return option_id in resolve_option_mentions(text, state.scenario)


def mentioned_options(text: str, state: DialogueState) -> set[str]:
    return resolve_option_mentions(text, state.scenario)


def validate_action(state: DialogueState, persona: Persona, action: UserAction) -> list[str]:
    errors: list[str] = []
    if action.speaker_id != persona.id:
        errors.append("action speaker does not match persona")
    unknown = [option_id for option_id in action.option_focus if option_id not in state.scenario.option_ids]
    if unknown:
        errors.append(f"action references unknown option(s): {unknown}")
    if action.vote_option and action.vote_option not in state.scenario.option_ids:
        errors.append("vote references an unknown option")
    if action.addressee_id and action.addressee_id not in state.runtimes:
        errors.append("action references an unknown addressee")
    if persona.hard_blocker:
        allowed = persona.preferred_option
        if action.vote_option and action.vote_option != allowed:
            errors.append("hard blocker cannot vote for a nonpreferred option")
        if action.stance_update and action.stance_update.option_id != allowed:
            errors.append("hard blocker cannot accept or switch to a nonpreferred option")
    return errors


def validate_realization(
    state: DialogueState,
    persona: Persona,
    action: UserAction,
    text: str,
) -> list[str]:
    errors = validate_action(state, persona, action)
    clean = " ".join(str(text).strip().split())
    if not clean:
        return [*errors, "empty output"]
    if "\n" in str(text).strip() and len([line for line in str(text).splitlines() if line.strip()]) > 1:
        errors.append("output contains multiple lines or speakers")
    if re.match(rf"^(?:{re.escape(persona.name)}|speaker|assistant)\s*:", clean, re.I):
        errors.append("output includes a speaker label")

    unknown_labels = {
        match.upper() for match in re.findall(r"\bOption\s+([A-Z0-9]+)\b", clean)
        if match.upper() not in state.scenario.option_ids
    }
    if unknown_labels:
        errors.append(f"mentions unknown option(s): {sorted(unknown_labels)}")

    errors.extend(_numeric_grounding_errors(state, action, clean))

    if action.act is ActionType.OPENING:
        if not action.option_focus or not option_mentioned(clean, state, action.option_focus[0]):
            errors.append("opening does not visibly state the preferred option")
        if len(clean.split()) < 5:
            errors.append("opening does not visibly include a reason")

    if action.act is ActionType.ANSWER and not _answer_is_relevant(state, action, clean):
        errors.append("direct answer is unrelated to the target question")

    if (
        action.stance_update
        and action.act is not ActionType.VOTE
        and not _stance_update_visible(state, action, clean)
    ):
        errors.append("required stance change is not visible")
    if (
        action.stance_update
        and action.stance_update.movement_reason.strip()
        and not action.stance_update.reason_already_public
        and not _movement_reason_visible(state, action, clean)
    ):
        errors.append("stance change lacks its grounded movement reason")

    if action.issue_effect is IssueEffect.RESOLVE and not re.search(
        r"\b(?:address(?:es|ed)?|resolv(?:e|ed|es)|works?\s+for\s+me|acceptable|can\s+accept|could\s+accept|fine\s+with|okay\s+with|convinced|could\s+live\s+with|can\s+live\s+with|go\s+along\s+with|can\s+support|could\s+support|willing\s+to\s+(?:try|support|accept)|works?\s+as\s+(?:a\s+)?compromise|settle\s+on|on\s+board\s+with)\b",
        clean,
        re.I,
    ):
        errors.append("issue resolution is not visible")
    if action.issue_effect is IssueEffect.MAINTAIN and not re.search(
        r"\b(?:still|remain(?:s|ing)?|not\s+(?:fully\s+)?(?:solved|addressed|convinced)|doesn['’]?t\s+(?:solve|address)|deal[- ]?breaker|continues?\s+to)\b",
        clean,
        re.I,
    ):
        errors.append("continued concern is not visible")

    # Ordinary comparison drift is not a hard error. The structured action may
    # intend a comparison while the endpoint realizes only one useful side of
    # it. Public comparison evidence is added later only when both options are
    # actually visible in the accepted text.

    if action.act is ActionType.VOTE:
        visible = resolve_visible_vote(clean, state.scenario)
        if visible is None:
            # In the formal voting phase the structured action already fixes
            # the intended choice. A short natural utterance is sufficient when
            # it mentions exactly that one option and no competitor.
            visible_mentions = mentioned_options(clean, state)
            if action.vote_option and visible_mentions == {action.vote_option}:
                visible = action.vote_option
        if visible is None:
            errors.append("clear vote is ambiguous or missing")
        elif visible != action.vote_option:
            errors.append("visible vote does not match the structured vote")

    repetition_exempt = (
        action.act in {ActionType.VOTE, ActionType.ANSWER, ActionType.OPENING, ActionType.FINAL_POSITION}
        or (state.phase.value == "NARROWING" and action.act is ActionType.CONCERN)
    )
    if not repetition_exempt and _near_duplicate_of_recent_own_turn(state, action.speaker_id, clean):
        errors.append("near-verbatim repetition of recent own message")
    return errors


def _stance_update_visible(state: DialogueState, action: UserAction, text: str) -> bool:
    update = action.stance_update
    if update is None or not option_mentioned(text, state, update.option_id):
        return False
    if update.kind is StanceUpdateKind.SWITCH_PREFERRED:
        return bool(re.search(
            r"\b(?:now\s+(?:prefer|choose|lean)|changed?\s+my\s+mind|switch(?:ing|ed)?|moving\s+to|go(?:ing)?\s+with|vote\s+for|choose|pick|settle\s+on|support|on\s+board\s+with)\b",
            text,
            re.I,
        ))
    if update.kind is StanceUpdateKind.MAKE_ACCEPTABLE:
        return bool(re.search(
            r"\b(?:acceptable|reasonable|workable|works?\s+for\s+me|can\s+accept|could\s+accept|fine\s+with|okay\s+with|could\s+live\s+with|can\s+live\s+with|go\s+along\s+with|can\s+support|could\s+support|willing\s+to\s+(?:try|support|accept)|works?\s+as\s+(?:a\s+)?compromise|settle\s+on|on\s+board\s+with)\b",
            text,
            re.I,
        ))
    if update.kind is StanceUpdateKind.REJECT:
        return bool(re.search(r"\b(?:reject|rule\s+out|won['’]?t\s+accept|not\s+acceptable)\b", text, re.I))
    return True


def _movement_reason_visible(state: DialogueState, action: UserAction, text: str) -> bool:
    update = action.stance_update
    if update is None or update.reason_already_public:
        return True
    reason = update.movement_reason.strip()
    if not reason:
        return True

    generic = {
        "accept", "acceptable", "choice", "common", "fair", "ground", "option",
        "prefer", "reasonable", "switch", "work", "works", "working",
    }
    option_terms: set[str] = set()
    option = state.scenario.option(update.option_id)
    for value in (option.id, option.name, option.short_name):
        option_terms.update(_semantic_terms(value or ""))
    reason_terms = _semantic_terms(reason) - generic - option_terms
    if not reason_terms:
        return True
    return bool(reason_terms & _semantic_terms(text))


def _semantic_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        if len(token) < 4:
            continue
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s", "ly"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                token = token[: -len(suffix)]
                break
        terms.add(token)
    return terms


def _answer_is_relevant(state: DialogueState, action: UserAction, text: str) -> bool:
    if re.search(r"\b(?:yes|no|not\s+really|probably|maybe|unsure|not\s+sure|don['’]?t\s+know|for\s+me|my)\b", text, re.I):
        return True
    if any(option_mentioned(text, state, option_id) for option_id in action.option_focus):
        return True
    words = set(re.findall(r"[a-z]{4,}", text.casefold()))
    reason_blob = " ".join(filter(None, [action.reason, action.decisive_reason, action.condition]))
    reason_terms = {term for term in re.findall(r"[a-z]{4,}", reason_blob.casefold())}
    if reason_terms & words:
        return True
    issue = state.active_issue
    if issue:
        terms = {term for term in re.findall(r"[a-z]{4,}", issue.summary.casefold())}
        if terms & words:
            return True
    return False


def _numeric_grounding_errors(state: DialogueState, action: UserAction, text: str) -> list[str]:
    """Reject only clear objective numeric inventions.

    First-person schedule/experience numbers are treated as personal statements,
    not option-card facts. This deliberately favors avoiding false positives.
    """
    numbers = re.findall(r"(?<!\w)(?:\d+(?:[.,]\d+)?|\d{1,2}:\d{2})(?!\w)", text)
    if not numbers:
        return []
    public_blob = " ".join(
        [*state.scenario.shared_context]
        + [value for option in state.scenario.options for value in option.public_values()]
        + ([action.reason_source.public_value] if action.reason_source else [])
    ).casefold()
    personal_blob = " ".join(
        filter(None, [action.personal_context, state.persona(action.speaker_id).background, state.persona(action.speaker_id).private_goal])
    ).casefold()
    errors: list[str] = []
    for number in numbers:
        normalized = number.replace(",", ".")
        if number.casefold() in public_blob or normalized in public_blob:
            continue
        if number.casefold() in personal_blob or normalized in personal_blob:
            continue
        sentence = next((part for part in re.split(r"(?<=[.!?])\s+", text) if number in part), text)
        if re.search(r"\b(?:i|me|my|mine|after\s+work|before\s+work|my\s+schedule|for\s+me)\b", sentence, re.I):
            continue
        if re.search(r"\b(?:costs?|price|takes?|duration|distance|closes?|opens?|hours?|minutes?|km|miles?|euros?|dollars?|pages?|year)\b", sentence, re.I):
            errors.append(f"unsupported concrete value: {number}")
    return errors


def _near_duplicate_of_recent_own_turn(state: DialogueState, speaker_id: str, text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.casefold())
    normalized = " ".join(normalized.split())
    if len(normalized.split()) < 4:
        return False
    own = [turn.text for turn in state.turns if not turn.moderator and turn.speaker_id == speaker_id][-3:]
    for previous in own:
        prior = re.sub(r"[^a-z0-9 ]+", " ", previous.casefold())
        prior = " ".join(prior.split())
        threshold = float(cfg.language.near_duplicate_similarity_threshold)
        if normalized == prior or SequenceMatcher(None, normalized, prior).ratio() >= threshold:
            return True
    return False
