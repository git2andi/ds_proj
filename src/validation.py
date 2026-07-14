"""Minimal structured-action and hard-failure text validation.

This module intentionally does not infer dialogue acts or reconstruct hidden
state from language.  It only checks whether a rendering is safe to commit for
its already-authoritative :class:`UserAction`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from models import (
    ActionType,
    DialogueState,
    IssueEffect,
    IssueKind,
    IssueResponseKind,
    Persona,
    StanceUpdateKind,
    UserAction,
)
from utils import clean_generated, jaccard_text, tokenize


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    text: str = ""
    errors: list[str] = field(default_factory=list)


_NUMERIC = re.compile(
    r"(?<!\w)(?:[$€£]\s*)?\d+(?:[.,]\d+)?(?:\s*[-–—]\s*|\s*)"
    r"(?:%|euros?|dollars?|pounds?|km|kilometers?|miles?|m|minutes?|mins?|hours?|hrs?|people|persons?|seats?|stops?|gb|tb|mbps|w|kw)?\b",
    re.I,
)
_OPTION_TOKEN = re.compile(r"\boption\s+([A-Z0-9]+)\b", re.I)
_ASSERTS_FEATURE = re.compile(
    r"\b(?:has|have|includes?|offers?|provides?|features?|comes\s+with|contains?)\b[^.!?]{0,55}\b"
    r"(wifi|wi-fi|parking|sauna|pool|gym|breakfast|shuttle|charger|warranty|bedrooms?|bathrooms?|seats?|workstations?|equipment|facility|facilities)\b",
    re.I,
)
_POSITIVE_COMMITMENT = re.compile(
    r"\b(?:prefer|choose|vote(?:\s+for)?|accept|support|go\s+with|settle\s+on|works?\s+for\s+me|fine\s+with)\b",
    re.I,
)
_METADATA = re.compile(r"(?:^|\s)[\[{].*(?:act|stance|option_focus|speaker_id|urgency).*[\]}](?:\s|$)", re.I | re.S)
_FORMAL_VOTE_LANGUAGE = re.compile(r"\b(?:vote(?:d|s|ing)?|my\s+vote|ballot)\b", re.I)
_META_ACT_LANGUAGE = re.compile(
    r"\bi\s+(?:open\s+the\s+discussion(?:\s+by)?|acknowledge\b|compare\b)",
    re.I,
)


def _option_aliases(state: DialogueState, option_id: str) -> tuple[str, ...]:
    option = state.scenario.option(option_id)
    aliases = {f"Option {option_id}", option.name, option.short_name}
    # Bare one-character labels such as A/B/C are too ambiguous in ordinary
    # prose (especially the English article "a"). The renderer is instructed
    # to use "Option A" or a public name, so omit those bare labels here.
    if len(option_id) > 1:
        aliases.add(option_id)
    return tuple(alias for alias in aliases if alias)


def option_mentioned(text: str, state: DialogueState, option_id: str) -> bool:
    if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.I) for alias in _option_aliases(state, option_id)):
        return True
    if len(option_id) != 1:
        return False

    # Accept compact references only in an option-list or preference/switch
    # context. This recognizes "Options A and B" and "preferred A, but now
    # choose B" without treating the English article "a" as Option A.
    for match in re.finditer(r"\boptions?\s+([^.!?;:]{0,70})", text, re.I):
        segment = match.group(1)
        if re.search(rf"(?<!\w){re.escape(option_id)}(?!\w)", segment, re.I):
            return True
    contextual = rf"\b(?:prefer(?:red|s|ring)?|lean(?:ed|ing)?\s+(?:toward|towards)|from|to|choose|choosing|choice\s+was|moving\s+to|switch(?:ed|ing)?\s+to|between|over)\s+(?:option\s+)?{re.escape(option_id)}\b"
    return bool(re.search(contextual, text, re.I))


def mentioned_options(text: str, state: DialogueState) -> set[str]:
    return {option_id for option_id in state.scenario.option_ids if option_mentioned(text, state, option_id)}


def validate_action(state: DialogueState, persona: Persona, action: UserAction) -> list[str]:
    errors: list[str] = []
    participant_ids = {candidate.id for candidate in state.personas}
    option_ids = set(state.scenario.option_ids)
    if action.speaker_id != persona.id or action.speaker_id not in participant_ids:
        errors.append("unknown or mismatched speaker")
    invalid_focus = set(action.option_focus) - option_ids
    if invalid_focus:
        errors.append(f"invalid option focus: {sorted(invalid_focus)}")
    if action.addressee_id is not None:
        if action.addressee_id not in participant_ids:
            errors.append("unknown addressee")
        elif action.addressee_id == action.speaker_id:
            errors.append("participant cannot address itself")
    if action.reason_source:
        source = action.reason_source
        if source.option_id not in option_ids:
            errors.append("reason source uses an unknown option")
        else:
            option = state.scenario.option(source.option_id)
            if source.attribute_name == "upside":
                actual = option.upside
            elif source.attribute_name == "concern":
                actual = option.concern
            else:
                actual = option.attrs.get(source.attribute_name)
            if actual is None or str(actual) != str(source.public_value):
                errors.append("reason source does not match the public option card")
    if action.stimulus_id is not None:
        if not state.group_stimulus or action.stimulus_id != state.group_stimulus.id:
            errors.append("unknown group stimulus reference")
    if action.issue_id:
        known = {issue.id for issue in state.issue_history}
        if state.active_issue:
            known.add(state.active_issue.id)
        if action.issue_id not in known:
            errors.append("unknown issue reference")
    if action.issue_effect == IssueEffect.OPEN:
        if action.act not in {ActionType.ASK, ActionType.CONCERN, ActionType.COMPARE}:
            errors.append("only a question, concern, or comparison may open an issue")
        if action.issue_id is not None:
            errors.append("a new issue must not reference an existing issue id")
    elif action.issue_effect in {
        IssueEffect.CONTINUE,
        IssueEffect.ANSWERED,
        IssueEffect.PARTIAL,
        IssueEffect.RESOLVE,
        IssueEffect.MAINTAIN,
    }:
        if not state.active_issue or action.issue_id != state.active_issue.id:
            errors.append("issue continuation must reference the active issue")
        elif state.active_issue.kind.value == "concern":
            if action.issue_effect in {IssueEffect.PARTIAL, IssueEffect.RESOLVE, IssueEffect.MAINTAIN} and action.speaker_id != state.active_issue.opened_by:
                errors.append("only the concern owner may partially address, resolve, or maintain the concern")
        elif state.active_issue.kind.value == "question" and action.issue_effect == IssueEffect.ANSWERED:
            if action.speaker_id == state.active_issue.opened_by:
                errors.append("question author cannot answer its own group question")
    if action.issue_response_kind is not None:
        issue = state.active_issue
        if not issue or issue.kind is not IssueKind.CONCERN or action.issue_id != issue.id:
            errors.append("concern response must reference the active concern")
        elif action.speaker_id == issue.opened_by:
            errors.append("concern owner cannot be its own responder")
        elif action.issue_response_kind is IssueResponseKind.MITIGATION:
            if action.reason_source is None or issue.reason_source is None:
                errors.append("mitigation requires public issue provenance")
            elif action.reason_source.option_id != issue.reason_source.option_id:
                errors.append("mitigation does not match the active concern option")
            elif action.reason_source.attribute_name != issue.reason_source.attribute_name:
                issue_words = {
                    token for token in re.findall(r"[a-z0-9]+", issue.reason_source.public_value.casefold())
                    if len(token) >= 4
                }
                source_text = (
                    action.reason_source.attribute_name.replace("_", " ")
                    + " " + action.reason_source.public_value
                ).casefold()
                if issue_words and not any(word in source_text for word in issue_words):
                    errors.append("mitigation does not match the active concern provenance")

    if action.act == ActionType.VOTE:
        if action.vote_option not in option_ids:
            errors.append("vote must select one valid option")
        if action.option_focus != (action.vote_option,):
            errors.append("vote focus must contain exactly the vote option")
    elif action.vote_option is not None:
        errors.append("only a vote action may contain vote_option")
    if action.stance_update:
        update = action.stance_update
        if update.option_id not in option_ids:
            errors.append("stance update uses an unknown option")
        runtime = state.runtimes[persona.id]
        if update.option_id in runtime.hard_rejected_options:
            errors.append("stance update targets a hard-rejected option")
        if update.kind == StanceUpdateKind.SWITCH_PREFERRED:
            if update.previous_option_id != runtime.preferred_option:
                errors.append("switch does not identify the current preferred option")
            if update.option_id == runtime.preferred_option:
                errors.append("switch target is already preferred")
    if persona.hard_blocker:
        preferred = state.runtimes[persona.id].preferred_option
        if action.vote_option and action.vote_option != preferred:
            errors.append("hard blocker may only vote for its preferred option")
        if action.stance_update is not None:
            errors.append("hard blocker may not change stance")
        positive_acts = {ActionType.OPENING, ActionType.SUPPORT, ActionType.COMPROMISE, ActionType.VOTE}
        if action.act in positive_acts and any(option_id != preferred for option_id in action.option_focus):
            errors.append("hard blocker action positively targets an alternative")
    if action.act == ActionType.OPENING:
        preferred = state.runtimes[persona.id].preferred_option
        if action.option_focus != (preferred,):
            errors.append("opening must state the simulator's initial preferred option")
    if state.response_obligation:
        if action.speaker_id == state.response_obligation and action.act != ActionType.ANSWER:
            errors.append("response obligation requires an answer action")
        if action.act == ActionType.ANSWER and action.speaker_id != state.response_obligation:
            errors.append("mandatory answer is assigned to another participant")
    return errors


def validate_realization(
    raw_text: str,
    state: DialogueState,
    persona: Persona,
    action: UserAction,
    *,
    target_question: str | None = None,
) -> ValidationResult:
    text = clean_generated(raw_text, persona.name)
    errors: list[str] = []
    if not text:
        errors.append("empty output")
        return ValidationResult(False, text, errors)
    if "\n" in str(raw_text).strip() and sum(
        bool(re.match(r"^\s*[A-Za-z][\w -]{0,35}:\s+", line))
        for line in str(raw_text).splitlines() if line.strip()
    ) > 1:
        errors.append("multiple speaker turns")
    if _METADATA.search(text) or text.startswith("{") or text.startswith("["):
        errors.append("metadata instead of dialogue")
    names = [candidate.name for candidate in state.personas]
    if any(re.search(rf"(?:^|[.!?]\s+){re.escape(name)}\s*:", text, re.I) for name in names):
        errors.append("multiple speaker turns")
    if action.act is not ActionType.VOTE and _FORMAL_VOTE_LANGUAGE.search(text):
        errors.append("formal vote language is not allowed outside voting")
    if _META_ACT_LANGUAGE.search(text):
        errors.append("dialogue-act label is exposed instead of natural wording")

    if action.issue_effect is IssueEffect.MAINTAIN and not re.search(
        r"\b(?:still|remain(?:s|ed|ing)?|unresolved|not (?:addressed|solved)|"
        r"cannot accept|can't accept|rule(?:s|d)? out|non-negotiable)\b",
        text, re.I,
    ):
        errors.append("maintained concern is not visible")
    elif action.issue_effect is IssueEffect.PARTIAL and not (
        re.search(r"\b(?:help(?:s|ed)?|partly|partially|somewhat|to a degree)\b", text, re.I)
        and re.search(r"\b(?:but|still|not fully|not completely|remains?)\b", text, re.I)
    ):
        errors.append("partial concern response is not visible")
    elif action.issue_effect is IssueEffect.RESOLVE and not re.search(
        r"\b(?:address(?:es|ed)?|resolv(?:e|ed|es)|enough|acceptable|workable|"
        r"can accept|could accept|can go with|could go with|no longer a concern)\b",
        text, re.I,
    ):
        errors.append("resolved concern is not visible")

    valid_ids = set(state.scenario.option_ids)
    for match in _OPTION_TOKEN.finditer(text):
        raw_token = match.group(1)
        token = raw_token.upper()
        # Ordinary prose such as "public option details" is not an option ID.
        # Treat only short/symbolic or explicitly all-caps tokens as labels.
        looks_like_id = len(raw_token) == 1 or raw_token.isdigit() or raw_token.isupper()
        if looks_like_id and token not in valid_ids:
            errors.append(f"nonexistent option {token}")

    required_mentions = {
        ActionType.OPENING,
        ActionType.SUPPORT,
        ActionType.CONCERN,
        ActionType.COMPARE,
        ActionType.COMPROMISE,
        ActionType.VOTE,
    }
    if action.act in required_mentions:
        required_focus = list(action.option_focus)
        if action.stance_update and action.stance_update.kind == StanceUpdateKind.SWITCH_PREFERRED:
            # The public state already carries the previous preference. During
            # discussion, a visible switch only needs to unambiguously name the
            # new preference and use change language; repeating the old option
            # is optional. Formal vote switches are checked more strictly below.
            required_focus = [action.stance_update.option_id]
        missing = [option_id for option_id in required_focus if not option_mentioned(text, state, option_id)]
        if missing:
            errors.append(f"missing required option mention: {', '.join(missing)}")

    public_text = " ".join([
        *state.scenario.shared_context,
        *(option.public_line() for option in state.scenario.options),
    ])
    public_numbers = {_numeric_key(value) for value in _NUMERIC.findall(public_text)}
    private_text = " ".join(filter(None, [
        persona.background,
        persona.private_goal,
        action.personal_context,
        action.reason,
    ]))
    private_numbers = {_numeric_key(value) for value in _NUMERIC.findall(private_text)}
    private_numbers.add(str(persona.age))
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence_options = mentioned_options(sentence, state)
        option_specific_text = " ".join(state.scenario.shared_context)
        if len(sentence_options) == 1:
            option_specific_text += " " + state.scenario.option(next(iter(sentence_options))).public_line()
        option_specific_numbers = {
            _numeric_key(value) for value in _NUMERIC.findall(option_specific_text)
        }
        objective_option_claim = bool(re.search(
            r"\b(?:costs?|price|closes?|opens?|until|distance|away|takes?|duration|capacity|seats?|people|km|miles?|minutes?|hours?)\b",
            sentence,
            re.I,
        ))
        for token in _NUMERIC.findall(sentence):
            key = _numeric_key(token)
            if not key:
                continue
            if len(sentence_options) == 1 and objective_option_claim and key not in option_specific_numbers:
                errors.append(
                    f"invented concrete value contradicts the public card for Option {next(iter(sentence_options))}: {token.strip()}"
                )
            elif key not in public_numbers and key not in private_numbers:
                errors.append(f"invented concrete value: {token.strip()}")

        feature_match = _ASSERTS_FEATURE.search(sentence)
        if feature_match:
            feature = feature_match.group(1).casefold().replace("wi-fi", "wifi")
            grounding_text = public_text
            if len(sentence_options) == 1:
                grounding_text = " ".join(state.scenario.shared_context) + " " + state.scenario.option(
                    next(iter(sentence_options))
                ).public_line()
            if feature not in grounding_text.casefold().replace("wi-fi", "wifi"):
                errors.append(f"invented option feature: {feature_match.group(1)}")

        errors.extend(_concrete_comparison_errors(sentence, state, action))

    if action.act == ActionType.ANSWER and not _answer_is_relevant(text, state, action, target_question):
        errors.append("direct answer is unrelated")

    if action.stance_update:
        update = action.stance_update
        if not option_mentioned(text, state, update.option_id):
            errors.append("stance change does not visibly name its option")
        if (
            update.kind == StanceUpdateKind.SWITCH_PREFERRED
            and action.act is not ActionType.VOTE
            and not _switch_is_visible(text, state, update)
        ):
            errors.append("preferred-option switch is not visible")
        elif update.kind == StanceUpdateKind.MAKE_ACCEPTABLE and not _acceptance_is_visible(text):
            errors.append("acceptance change is not visible")
        elif update.kind == StanceUpdateKind.REJECT and not re.search(
            r"\b(?:reject|cannot accept|won't accept|not an option|rule out)\b",
            text,
            re.I,
        ):
            errors.append("rejection change is not visible")

    if action.act == ActionType.VOTE:
        mentioned = mentioned_options(text, state)
        if action.vote_option not in mentioned:
            errors.append("formal vote does not name the structured option")
        targets = _explicit_vote_targets(text, state)
        if targets != {action.vote_option}:
            errors.append("formal vote is ambiguous or contradicts the structured vote")
        runtime = state.runtimes[persona.id]
        if runtime.public_preference and runtime.public_preference != action.vote_option:
            previous = action.stance_update.previous_option_id if action.stance_update else runtime.public_preference
            if not _vote_switch_bridge_visible(text, state, previous, action.vote_option):
                errors.append("vote switch lacks a visible bridge")

    if persona.hard_blocker:
        preferred = state.runtimes[persona.id].preferred_option
        for option_id in mentioned_options(text, state) - {preferred}:
            if _positive_for_option(text, state, option_id):
                errors.append("hard-blocker contradiction")
                break

    repetition_exempt = action.act is ActionType.VOTE or action.issue_effect is IssueEffect.MAINTAIN
    if not repetition_exempt:
        previous = [
            turn.text for turn in state.participant_turns
            if turn.speaker_id == persona.id
        ][-4:]
        for old in previous:
            if jaccard_text(text, old, min_len=2) >= 0.88 or SequenceMatcher(None, text.casefold(), old.casefold()).ratio() >= 0.91:
                errors.append("near-verbatim repetition")
                break
    return ValidationResult(not errors, text, errors)


def _acceptance_is_visible(text: str) -> bool:
    """Return whether a structured acceptance is broadly visible in wording.

    The stance update is authoritative.  This check deliberately accepts
    ordinary willingness language instead of trying to classify the whole
    utterance semantically.
    """
    return bool(re.search(
        r"\b(?:acceptable|reasonable|workable|viable|suitable(?: choice)?|"
        r"could live with|can accept|could accept|willing to accept|fine with|okay with|ok with|"
        r"open to|can go with|could go with|willing to go with|happy to go with|prepared to go with|"
        r"works? for me|would work for me|could work (?:well )?for (?:me|us|the group)(?: too)?|"
        r"I(?:'m| am) good with|I(?:'m| am) comfortable with)\b",
        text,
        re.I,
    ))


def _switch_is_visible(text: str, state: DialogueState, update) -> bool:
    """Check only target naming plus an explicit movement marker.

    The public state already records the prior preference. During discussion,
    the old option therefore need not be repeated. This is intentionally a
    small surface-consistency check, not a semantic stance interpreter.
    """
    if not option_mentioned(text, state, update.option_id):
        return False
    patterns = (
        r"\bchanged my mind\b",
        r"\bchang(?:e|ed|ing) my (?:mind|preference|preferred choice|vote)\b",
        r"\b(?:willing|ready|prepared) to (?:switch|move)(?: my preference)?(?: from [^.!?]{0,50})? to\b",
        r"\bswitch(?:ing|ed)?(?: my preference)?(?: from [^.!?]{0,50})? to\b",
        r"\b(?:move|moving|moved)(?: my preference)?(?: from [^.!?]{0,50})? to\b",
        r"\b(?:now|actually)\s+(?:prefer|favor|choose)\b",
        r"\b(?:i(?:'m| am)\s+)?(?:now|actually)\s+lean(?:ing)?\s+(?:a bit\s+|somewhat\s+|more\s+)?(?:toward|towards)\b",
        r"\b(?:starting|beginning) to (?:prefer|favor|lean)\b",
        r"\blean(?:ing)?(?: a bit| somewhat)? more (?:toward|towards)\b",
        r"\blean(?:ing)? (?:toward|towards) [^.!?]{0,45} instead\b",
        r"\blean(?:ing)? (?:toward|towards) [^.!?]{0,45}\bnow\b",
        r"\bprefer [^.!?]{0,45} instead\b",
        r"\bhas become my preferred (?:choice|option)\b",
        r"\bmy preferred (?:choice|option) is now\b",
        r"\bbut now\b[^.!?]{0,70}\b(?:better|preferred|best fit)\b",
    )
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _vote_switch_bridge_visible(
    text: str,
    state: DialogueState,
    previous_option: str | None,
    vote_option: str | None,
) -> bool:
    """Require an explicit old-to-new bridge only for a formal vote switch."""
    if not vote_option or not option_mentioned(text, state, vote_option):
        return False
    # When the previous preference is already public, ordinary change-of-mind
    # wording is enough. The validator does not require the speaker to restate
    # public history in the same sentence as the vote.
    public_history_patterns = (
        r"\bchang(?:e|ed|ing) my (?:mind|preference|preferred choice|vote)\b",
        r"\bchanged the balance\b",
        r"\breconsider(?:ed|ing)?\b",
    )
    if any(re.search(pattern, text, re.I) for pattern in public_history_patterns):
        return True

    # Other bridges explicitly contrast the former and current choice, so the
    # old option must be visible as well as the vote target.
    if previous_option and not option_mentioned(text, state, previous_option):
        return False
    explicit_old_to_new_patterns = (
        r"\bswitch(?:ing|ed)?(?: my (?:preference|preferred choice|vote))?\b",
        r"\b(?:moved|moving) from\b",
        r"\bpreviously prefer(?:red|ring)\b",
        r"\binitially preferred\b",
        r"\bused to prefer\b",
        r"\bfirst choice\b",
        r"\bpreferred [^.!?]{0,45}\bbut\b",
        r"\bfrom [^.!?]{0,35} to\b",
    )
    return any(re.search(pattern, text, re.I) for pattern in explicit_old_to_new_patterns)


def _numeric_key(value: str) -> str:
    value = value.casefold().replace(",", "").strip()
    currency = ""
    if value.startswith("$"):
        currency, value = "usd", value[1:]
    elif value.startswith("€"):
        currency, value = "eur", value[1:]
    elif value.startswith("£"):
        currency, value = "gbp", value[1:]
    value = re.sub(r"\s*[-–—]\s*", "", value)
    value = re.sub(r"\s+", "", value)
    unit_aliases = (
        (r"hours?$|hrs?$", "h"),
        (r"minutes?$|mins?$", "min"),
        (r"dollars?$", "usd"),
        (r"euros?$", "eur"),
        (r"pounds?$", "gbp"),
        (r"kilometers?$|kms?$", "km"),
        (r"miles?$", "mile"),
        (r"persons?$|people$", "people"),
        (r"seats?$", "seat"),
        (r"stops?$", "stop"),
    )
    for pattern, replacement in unit_aliases:
        value = re.sub(pattern, replacement, value)
    return value + currency if currency else value


def _explicit_vote_targets(text: str, state: DialogueState) -> set[str]:
    """Return explicit final vote targets while excluding a visible old choice.

    In ordinary vote wording, every option governed by a vote/choice phrase is
    a target. In an explicit ``switch my vote from X to Y`` bridge, X is public
    history and only Y is the final target. This keeps ambiguous alternatives
    invalid without misclassifying natural switch language.
    """
    targets: set[str] = set()
    vote_words = r"(?:vote(?:d|s|\s+is|\s+for|\s+goes\s+to)?|voting\s+for|choose|choosing|choice\s+is|select|selecting|ballot|go(?:ing)?\s+with)"
    for option_id in state.scenario.option_ids:
        for alias in _option_aliases(state, option_id):
            escaped = re.escape(alias)
            before = rf"{vote_words}[^.!?]{{0,35}}(?<!\w){escaped}(?!\w)"
            after = rf"(?<!\w){escaped}(?!\w)[^.!?]{{0,25}}(?:gets?\s+my\s+vote|is\s+my\s+vote|has\s+my\s+vote)"
            if re.search(before, text, re.I) or re.search(after, text, re.I):
                targets.add(option_id)

    old_choices: set[str] = set()
    new_choices: set[str] = set()
    movement = r"(?:switch(?:ing|ed)?|chang(?:e|ed|ing)|mov(?:e|ed|ing))"
    object_phrase = r"(?:\s+(?:my\s+)?(?:vote|preference|preferred\s+choice))?"
    for old_id in state.scenario.option_ids:
        for new_id in state.scenario.option_ids:
            if old_id == new_id:
                continue
            for old_alias in _option_aliases(state, old_id):
                for new_alias in _option_aliases(state, new_id):
                    bridge = (
                        rf"{movement}{object_phrase}\s+from\s+[^.!?]{{0,30}}"
                        rf"(?<!\w){re.escape(old_alias)}(?!\w)[^.!?]{{0,180}}"
                        rf"\bto\s+[^.!?]{{0,30}}(?<!\w){re.escape(new_alias)}(?!\w)"
                    )
                    if re.search(bridge, text, re.I):
                        old_choices.add(old_id)
                        new_choices.add(new_id)
    targets.update(new_choices)
    targets.difference_update(old_choices - new_choices)
    return targets


def _concrete_comparison_errors(sentence: str, state: DialogueState, action: UserAction) -> list[str]:
    """Block only clear, measurable comparative contradictions.

    Pairwise claims are evaluated against the explicitly named or structured
    comparison peer. Global superlatives are evaluated across all measurable
    options. Ambiguous qualitative language is allowed rather than treated as
    an unsupported fact: the structured action and reason provenance already
    carry the intended meaning.
    """
    claim_specs = (
        ("cheaper", "cost", -1, r"\b(?:cheaper|lower[- ]priced?|lower price)\b"),
        ("expensive", "cost", 1, r"\b(?:more expensive|higher[- ]priced?|higher price|costlier)\b"),
        ("shorter", "duration", -1, r"\b(?:shorter (?:flight|trip|journey|duration|travel time)|faster)\b"),
        ("longer", "duration", 1, r"\b(?:longer (?:flight|trip|journey|duration|travel time)|slower)\b"),
        ("earlier", "time", -1, r"\bearlier\s+(?:departure|arrival|closing|opening|time)\b"),
        ("later", "time", 1, r"\blater\s+(?:departure|arrival|closing|opening|time)\b"),
    )
    superlative_specs = (
        ("cost", -1, r"\b(?:cheapest|lowest[- ]priced?|lowest price)\b"),
        ("cost", 1, r"\b(?:most expensive|highest[- ]priced?|highest price|costliest)\b"),
        ("duration", -1, r"\b(?:shortest (?:flight|trip|journey|duration|travel time)|fastest)\b"),
        ("duration", 1, r"\b(?:longest (?:flight|trip|journey|duration|travel time)|slowest)\b"),
        ("time", -1, r"\bearliest\s+(?:departure|arrival|closing|opening|time)\b"),
        ("time", 1, r"\blatest\s+(?:departure|arrival|closing|opening|time)\b"),
    )
    errors: list[str] = []
    mentioned = mentioned_options(sentence, state)

    for claim_name, kind, direction, pattern in claim_specs:
        for match in re.finditer(pattern, sentence, re.I):
            subject = _comparison_claim_subject(sentence, state, match.start(), match.end())
            pair = _comparison_subjects(sentence, state, match.start())
            if pair is not None:
                pair_subject, peer = pair
                if _measures_available(state, pair_subject, [peer], kind) and not _direction_is_true(
                    state, pair_subject, [peer], kind, direction
                ):
                    errors.append("concrete comparison contradicts public values")
                continue

            if subject is None and action.reason_source and action.reason_source.option_id in mentioned:
                subject = action.reason_source.option_id
            if subject is None and len(mentioned) == 1:
                subject = next(iter(mentioned))
            if subject is None:
                continue

            peers = [option_id for option_id in mentioned if option_id != subject]
            if not peers and action.act is ActionType.COMPARE and subject in action.option_focus:
                peers = [option_id for option_id in action.option_focus if option_id != subject]
            measurable_peers = [
                option_id for option_id in peers
                if _public_measure(state, option_id, kind) is not None
            ]
            if measurable_peers:
                if _public_measure(state, subject, kind) is not None and not _direction_is_true(
                    state, subject, measurable_peers, kind, direction
                ):
                    errors.append("concrete comparison contradicts public values")
                continue

            # A card descriptor or reason source may intentionally say "lower
            # price" or "earlier closing time" without naming a comparison
            # target. Accept that provenance. Otherwise block only when public
            # measures prove the exact opposite extreme.
            if _option_card_supports_claim(state, subject, claim_name):
                continue
            if (
                action.reason_source
                and action.reason_source.option_id == subject
                and _structured_claim_supports(claim_name, _structured_action_fact_text(action))
            ):
                continue
            values = _all_public_measures(state, kind)
            subject_value = values.get(subject)
            if subject_value is None or len(values) < 2:
                continue
            other_values = [value for option_id, value in values.items() if option_id != subject]
            clearly_opposite = (
                subject_value >= max(other_values) if direction < 0
                else subject_value <= min(other_values)
            )
            if clearly_opposite:
                errors.append("concrete comparison contradicts public values")

    for kind, direction, pattern in superlative_specs:
        for match in re.finditer(pattern, sentence, re.I):
            subject = _comparison_claim_subject(sentence, state, match.start(), match.end())
            if subject is None and action.reason_source and action.reason_source.option_id in mentioned:
                subject = action.reason_source.option_id
            if subject is None and len(mentioned) == 1:
                subject = next(iter(mentioned))
            values = _all_public_measures(state, kind)
            if subject is None or subject not in values or len(values) < 2:
                continue
            expected = min(values.values()) if direction < 0 else max(values.values())
            if values[subject] != expected:
                errors.append("concrete comparison contradicts public values")
    return list(dict.fromkeys(errors))


def _measures_available(
    state: DialogueState, subject: str, peers: list[str], kind: str
) -> bool:
    return _public_measure(state, subject, kind) is not None and all(
        _public_measure(state, peer, kind) is not None for peer in peers
    )


def _option_card_supports_claim(state: DialogueState, option_id: str, claim_name: str) -> bool:
    option = state.scenario.option(option_id)
    card_text = " ".join([
        *(f"{key.replace('_', ' ')} {value}" for key, value in option.attrs.items()),
        option.upside,
        option.concern,
    ]).casefold()
    return _structured_claim_supports(claim_name, card_text)


def _structured_action_fact_text(action: UserAction) -> str:
    return " ".join(filter(None, [
        action.reason,
        action.reason_source.public_value if action.reason_source else None,
    ])).casefold()


def _direction_is_true(
    state: DialogueState,
    subject: str,
    peers: list[str],
    kind: str,
    direction: int,
) -> bool:
    subject_value = _public_measure(state, subject, kind)
    peer_values = [_public_measure(state, peer, kind) for peer in peers]
    peer_values = [value for value in peer_values if value is not None]
    if subject_value is None or not peer_values:
        return False
    return all(subject_value < value for value in peer_values) if direction < 0 else all(
        subject_value > value for value in peer_values
    )


def _all_public_measures(state: DialogueState, kind: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for option_id in state.scenario.option_ids:
        value = _public_measure(state, option_id, kind)
        if value is not None:
            result[option_id] = value
    return result


def _comparison_claim_subject(
    sentence: str,
    state: DialogueState,
    comparative_start: int,
    comparative_end: int,
) -> str | None:
    """Return the option locally described by a comparative phrase."""
    before = sentence[:comparative_start]
    after = sentence[comparative_end:comparative_end + 70]
    before_hits: list[tuple[int, str]] = []
    for option_id in state.scenario.option_ids:
        for alias in _option_aliases(state, option_id):
            for hit in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", before, re.I):
                before_hits.append((hit.start(), option_id))
            if re.search(rf"^\s*(?:of|for)\s+(?:the\s+)?(?<!\w){re.escape(alias)}(?!\w)", after, re.I):
                return option_id
    if not before_hits:
        return None
    position, option_id = max(before_hits, key=lambda item: item[0])
    return option_id if len(before) - position <= 100 else None


def _structured_claim_supports(name: str, structured: str) -> bool:
    equivalents = {
        "cheaper": ("cheaper", "lower price", "lowest price", "low cost", "free"),
        "expensive": ("more expensive", "higher price", "highest price", "costlier"),
        "shorter": ("shorter", "shortest", "faster", "fastest"),
        "longer": ("longer", "longest", "slower", "slowest"),
        "earlier": ("earlier", "earliest"),
        "later": ("later", "latest"),
    }
    return any(term in structured for term in equivalents[name])


def _comparison_subjects(
    sentence: str,
    state: DialogueState,
    comparative_start: int,
) -> tuple[str, str] | None:
    before = sentence[:comparative_start]
    after = sentence[comparative_start:]
    than_part = re.split(r"\bthan\b", after, maxsplit=1, flags=re.I)
    if len(than_part) != 2:
        return None
    before_hits: list[tuple[int, str]] = []
    after_hits: list[tuple[int, str]] = []
    for option_id in state.scenario.option_ids:
        for alias in _option_aliases(state, option_id):
            for hit in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", before, re.I):
                before_hits.append((hit.start(), option_id))
            for hit in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", than_part[1], re.I):
                after_hits.append((hit.start(), option_id))
    if not before_hits or not after_hits:
        return None
    subject = max(before_hits, key=lambda item: item[0])[1]
    peer = min(after_hits, key=lambda item: item[0])[1]
    return None if subject == peer else (subject, peer)


def _public_measure(state: DialogueState, option_id: str, kind: str) -> float | None:
    option = state.scenario.option(option_id)
    for key, raw in option.attrs.items():
        key_text = key.casefold().replace("_", " ")
        value_text = str(raw).casefold().replace("–", "-").replace("—", "-")
        if kind == "cost" and any(token in key_text for token in ("cost", "price", "fare")):
            if "free" in value_text:
                return 0.0
            match = re.search(r"\d+(?:[.,]\d+)?", value_text)
            return float(match.group().replace(",", ".")) if match else None
        if kind == "duration" and any(token in key_text for token in ("duration", "travel", "journey", "flight time")):
            hours = re.search(r"(\d+(?:[.,]\d+)?)\s*-?\s*(?:hours?|hrs?)", value_text)
            minutes = re.search(r"(\d+(?:[.,]\d+)?)\s*-?\s*(?:minutes?|mins?)", value_text)
            if hours or minutes:
                return (float(hours.group(1).replace(",", ".")) * 60 if hours else 0.0) + (float(minutes.group(1).replace(",", ".")) if minutes else 0.0)
        if kind == "time" and any(token in key_text for token in ("departure", "arrival", "closing", "opening", "time")):
            clock = re.search(r"\b(\d{1,2}):(\d{2})\b", value_text)
            if clock:
                return float(int(clock.group(1)) * 60 + int(clock.group(2)))
    return None


def _positive_for_option(text: str, state: DialogueState, option_id: str) -> bool:
    for alias in _option_aliases(state, option_id):
        for match in re.finditer(re.escape(alias), text, re.I):
            start = max(0, match.start() - 55)
            end = min(len(text), match.end() + 55)
            if _POSITIVE_COMMITMENT.search(text[start:end]):
                return True
    return False


def _positive_option_mentions(text: str, state: DialogueState) -> set[str]:
    return {
        option_id for option_id in state.scenario.option_ids
        if option_mentioned(text, state, option_id) and _positive_for_option(text, state, option_id)
    }


def _answer_is_relevant(
    text: str,
    state: DialogueState,
    action: UserAction,
    target_question: str | None,
) -> bool:
    lowered = text.casefold()
    if re.search(r"\b(?:yes|no|not really|it depends|because|i think|for me|my concern|my reason)\b", lowered):
        return True
    if any(option_mentioned(text, state, option_id) for option_id in action.option_focus):
        return True
    reference = " ".join(filter(None, [target_question, state.active_issue.summary if state.active_issue else None, action.reason]))
    reference_tokens = set(tokenize(reference, min_len=4))
    answer_tokens = set(tokenize(text, min_len=4))
    return bool(reference_tokens & answer_tokens)
