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
    errors.extend(_qualitative_grounding_errors(state, action, clean))
    errors.extend(_cross_option_reason_errors(state, action, clean))

    if action.act is ActionType.OPENING:
        if not action.option_focus or not option_mentioned(clean, state, action.option_focus[0]):
            errors.append("opening does not visibly state the preferred option")
        if len(clean.split()) < 5:
            errors.append("opening does not visibly include a reason")

    if action.act is ActionType.ASK and action.issue_effect is IssueEffect.OPEN:
        if not _question_is_visible(clean):
            errors.append("opened question is not visibly phrased as a question")
        if action.addressee_id:
            addressee_name = state.persona(action.addressee_id).name
            other_names = [
                other.name for other in state.personas
                if other.id != action.addressee_id
            ]
            if not _direct_addressee_visible(clean, addressee_name, other_names):
                errors.append("direct question does not clearly address its intended addressee")

    if (
        action.act is ActionType.CONCERN
        and action.issue_effect is IssueEffect.OPEN
        and not _concern_is_visible(clean)
    ):
        errors.append("opened concern is not visibly expressed")

    if action.act is ActionType.ANSWER and not _answer_is_relevant(state, action, clean):
        errors.append("direct answer is unrelated to the target question")

    if (
        action.stance_update is None
        and action.act is not ActionType.VOTE
        and action.option_focus
        and _claims_own_strong_preference(clean)
    ):
        current = state.runtimes[action.speaker_id].public_preference or state.runtimes[action.speaker_id].preferred_option
        if action.option_focus[0] != current:
            errors.append("visible preference claim has no matching structured stance change")

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

    if action.issue_effect is IssueEffect.RESOLVE and not _acceptance_visible(clean):
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


def _direct_addressee_visible(
    text: str,
    addressee_name: str,
    other_names: list[str],
) -> bool:
    """Accept a clear vocative anywhere while rejecting a different addressee.

    Natural forms include ``Mira, ...``, ``... what do you think, Mira?`` and
    ``Could you say, Mira, whether ...``. Merely mentioning the name inside a
    statement is not enough, and a different participant may not occupy a
    vocative position.
    """

    intended_patterns = (
        rf"(?:^|[,.!?;:—-]\s*){re.escape(addressee_name)}\b\s*[,—:;-]",
        rf"[,—:;-]\s*{re.escape(addressee_name)}\b(?=\s*[?.!,;:]|\s*$)",
        rf"\b(?:you|your)\b[^?!.]{{0,40}}[,—:;-]\s*{re.escape(addressee_name)}\b",
    )
    if not any(re.search(pattern, text, re.I) for pattern in intended_patterns):
        return False

    for name in other_names:
        other_patterns = (
            rf"(?:^|[,.!?;:—-]\s*){re.escape(name)}\b\s*[,—:;-]",
            rf"[,—:;-]\s*{re.escape(name)}\b(?=\s*[?.!,;:]|\s*$)",
        )
        if any(re.search(pattern, text, re.I) for pattern in other_patterns):
            return False
    return True

def _claims_own_strong_preference(text: str) -> bool:
    return bool(re.search(
        r"\b(?:my\s+(?:top|first|preferred)\s+choice|remains?\s+my\s+choice|"
        r"i\s+(?:still\s+)?(?:prefer|choose|pick)|i['’]?m\s+(?:still\s+)?(?:going\s+with|leaning\s+toward))\b",
        text,
        re.I,
    ))


def _question_is_visible(text: str) -> bool:
    if "?" in text:
        return True
    if re.search(
        r"^(?:[^,?!]{1,40},\s*)?(?:who|what|when|where|why|how|would|could|can|do|does|did|is|are|will|should|have|has)\b",
        text.strip(),
        re.I,
    ):
        return True
    return bool(re.search(
        r"\b(?:i\s+(?:wonder|want\s+to\s+know)\s+whether|can\s+you|could\s+you|would\s+you|tell\s+me|let\s+me\s+know)\b",
        text,
        re.I,
    ))


def _concern_is_visible(text: str) -> bool:
    return bool(re.search(
        r"\b(?:concern(?:ed)?|worr(?:y|ied|ies)|problem|risk|drawback|downside|issue|"
        r"difficult|impractical|unworkable|unsafe|too\s+\w+|not\s+(?:work|enough|practical|"
        r"suitable|acceptable)|can['’]?t|cannot|won['’]?t|keeps?\s+me\s+from|makes?\s+it\s+hard)\b",
        text,
        re.I,
    ))


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
        return _acceptance_visible(text)
    if update.kind is StanceUpdateKind.REJECT:
        return bool(re.search(r"\b(?:reject|rule\s+out|won['’]?t\s+accept|not\s+acceptable)\b", text, re.I))
    return True



def _acceptance_visible(text: str) -> bool:
    """Recognize personal/group acceptance without treating praise as movement."""

    patterns = (
        r"\b(?:acceptable|reasonable|workable)\s+(?:for\s+(?:me|us)|as\s+(?:a\s+)?(?:compromise|middle\s+ground|common\s+ground))\b",
        r"\b(?:works?|would\s+work)\s+(?:well\s+)?for\s+(?:me|us)\b",
        r"\b(?:works?|would\s+work)\s+(?:well\s+)?as\s+(?:a\s+)?(?:reasonable\s+|solid\s+|better\s+)?(?:compromise|middle\s+ground|common\s+ground)\b",
        r"\b(?:suits?|fits?)\s+(?:my|our)\s+(?:needs?|priorities|group)\b",
        r"\b(?:good|solid|reasonable|better)\s+(?:choice|fit|option|compromise|middle\s+ground|common\s+ground)\s+for\s+(?:me|us|our\s+group)\b",
        r"\b(?:can|could|will|would)\s+(?:accept|support|live\s+with|go\s+along\s+with|get\s+behind|work\s+with|settle\s+on)\b",
        r"\b(?:willing|happy|ready)\s+to\s+(?:accept|support|try|go\s+with|settle\s+on)\b",
        r"\b(?:fine|okay|good)\s+(?:with|by)\s+(?:me|us)\b",
        r"\b(?:i|we)(?:['’]m|\s+am|['’]re|\s+are)?\s+(?:on\s+board\s+with|good\s+with|going\s+with|willing\s+to\s+take)\b",
        r"\b(?:makes?|would\s+make)\s+sense\s+for\s+(?:me|us|our\s+group)\b",
    )
    return any(re.search(pattern, text, re.I) for pattern in patterns)

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
    reason_blob = " ".join(filter(None, [action.reason, action.decisive_reason]))
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



def _cross_option_reason_errors(
    state: DialogueState,
    action: UserAction,
    text: str,
) -> list[str]:
    """Reject a reason clearly copied from a different option.

    This narrow check applies only to formal votes and visible stance movement.
    It compares semantic overlap with individual public option values and fires
    only when another option has a substantially stronger two-term match than
    the intended target.
    """
    if action.act is not ActionType.VOTE and action.stance_update is None:
        return []
    target_id = action.vote_option or (action.stance_update.option_id if action.stance_update else None)
    if not target_id or target_id not in state.scenario.option_ids:
        return []
    text_terms = _semantic_terms(text)
    if not text_terms:
        return []

    def best_overlap(option_id: str) -> int:
        option = state.scenario.option(option_id)
        return max((len(text_terms & _semantic_terms(value)) for value in option.public_values() if value), default=0)

    target_score = best_overlap(target_id)
    mismatches = [
        option_id for option_id in state.scenario.option_ids
        if option_id != target_id
        and best_overlap(option_id) >= 2
        and best_overlap(option_id) > target_score
    ]
    if not mismatches:
        return []
    return [f"reason appears to belong to another option: {', '.join(mismatches)}"]


def _qualitative_grounding_errors(
    state: DialogueState,
    action: UserAction,
    text: str,
) -> list[str]:
    """Reject a small set of clear unsupported factual strengthenings.

    The check is intentionally narrow. Personal judgments remain allowed; only
    high-risk modifiers and relative claims absent from the focused option's
    public data are rejected.
    """

    public_parts = list(state.scenario.shared_context)
    for option_id in action.option_focus:
        if option_id in state.scenario.option_ids:
            public_parts.extend(state.scenario.option(option_id).public_values())
    if action.reason_source is not None:
        public_parts.append(action.reason_source.public_value)
    public_blob = " ".join(public_parts).casefold()

    high_risk = (
        "significantly",
        "dramatically",
        "substantially",
        "cheapest",
        "lowest price",
        "fastest",
        "shortest",
        "safest",
        "most reliable",
        "best value",
    )
    lowered = text.casefold()
    unsupported = [phrase for phrase in high_risk if phrase in lowered and phrase not in public_blob]
    if not unsupported:
        return []
    return [f"unsupported qualitative strengthening: {', '.join(sorted(set(unsupported)))}"]

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
