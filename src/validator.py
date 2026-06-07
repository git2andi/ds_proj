"""Deterministic validation of generated participant messages."""

from __future__ import annotations

import re
from typing import Optional

from config_loader import cfg
from schemas import ActType, DialogueState, MoveIntent, ValidationIssue, ValidationResult
from state import own_recent_similarity
from utils import OptionResolver, extract_numbers


class MessageValidator:
    def __init__(self, resolver: OptionResolver) -> None:
        self.resolver = resolver

    def validate(self, text: str, state: DialogueState, intent: MoveIntent) -> ValidationResult:
        if not bool(cfg.validation.enabled):
            return ValidationResult(ok=True)
        issues: list[ValidationIssue] = []
        stripped = text.strip()
        if not stripped:
            issues.append(ValidationIssue("EMPTY", "fatal", "Generated message is empty."))
            return ValidationResult(ok=False, issues=issues)
        self._check_structure(stripped, intent, issues)
        self._check_option_refs(stripped, issues)
        self._check_grounding_numbers(stripped, issues)
        self._check_repetition(stripped, state, intent, issues)
        self._check_decision_clarity(stripped, intent, issues)
        self._check_style(stripped, issues)
        ok = not any(issue.severity in {"repair", "fatal"} for issue in issues)
        return ValidationResult(ok=ok, issues=issues)

    def _check_structure(self, text: str, intent: MoveIntent, issues: list[ValidationIssue]) -> None:
        pattern = str(cfg.validation.multi_turn_prefix_pattern)
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > 1 and any(re.search(pattern, line.strip()) for line in lines[1:]):
            issues.append(ValidationIssue("MULTI_TURN_OUTPUT", "fatal", "Message appears to contain multiple chat turns."))
        # Prefixing own name is repairable because it can be stripped, but repeated prefixes indicate prompt non-compliance.
        speaker_name = ""
        try:
            speaker_name = intent.speaker_id
        except Exception:
            pass
        if re.search(r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß .'-]{1,30}:\s*", text):
            issues.append(ValidationIssue("SPEAKER_PREFIX", "repair", "Message starts with a speaker prefix."))

    def _check_option_refs(self, text: str, issues: list[ValidationIssue]) -> None:
        invalid = self.resolver.invalid_option_refs(text)
        if invalid:
            issues.append(ValidationIssue("INVALID_OPTION_REFERENCE", "fatal", f"Invalid option ids referenced: {invalid}"))

    def _check_grounding_numbers(self, text: str, issues: list[ValidationIssue]) -> None:
        numbers = extract_numbers(text)
        if not numbers:
            return
        refs = self.resolver.ids_in_text(text)
        if not refs:
            # A single vague number can occur in casual text, but multiple numbers usually means invented facts.
            if len(numbers) > int(cfg.validation.max_new_numbers_without_option_source):
                issues.append(ValidationIssue("UNGROUNDED_NUMERIC_FACT", "repair", "Numeric facts appear without an option source."))
            return
        source = "\n".join(self.resolver.option_text(ref) for ref in refs).lower()
        invented = [n for n in numbers if _normalise_number(n) not in _normalise_number(source)]
        if invented:
            issues.append(ValidationIssue("INVENTED_OPTION_ATTRIBUTE", "repair", f"Numeric facts not present in option source: {invented}"))

    def _check_repetition(self, text: str, state: DialogueState, intent: MoveIntent, issues: list[ValidationIssue]) -> None:
        score = own_recent_similarity(state, intent.speaker_id, text)
        if score >= float(cfg.validation.own_turn_similarity_threshold):
            issues.append(ValidationIssue("SELF_REPETITION", "repair", f"Too similar to the speaker's previous turn: {score:.2f}"))

    def _check_decision_clarity(self, text: str, intent: MoveIntent, issues: list[ValidationIssue]) -> None:
        refs = self.resolver.ids_in_text(text)
        lower = text.lower()
        if intent.act == ActType.VOTE:
            vote_cue = re.search(r"\b(?:pick|choose|vote|go with|prefer|my choice|my pick|i’d|i would|i'll)\b", lower)
            if not refs or not vote_cue:
                issues.append(ValidationIssue("UNCLEAR_VOTE", "repair", "Vote move must clearly name an option."))
        if intent.act == ActType.ACCEPT:
            yes = re.search(r"\b(?:yes|yeah|works|can live with|accept|fine|okay|ok|good|no objection)\b", lower)
            no = re.search(r"\b(?:no|nope|not|can't|cannot|rather not|doesn't work)\b", lower)
            if not yes and not no:
                issues.append(ValidationIssue("UNCLEAR_CONFIRMATION", "repair", "Confirmation move must give a clear yes or no."))

    def _check_style(self, text: str, issues: list[ValidationIssue]) -> None:
        lower = text.lower()
        for phrase in cfg.utterances.banned_register_phrases:
            if str(phrase).lower() in lower:
                severity = "fatal" if bool(cfg.validation.style_banned_phrase_is_fatal) else "warn"
                issues.append(ValidationIssue("BANNED_REGISTER_PHRASE", severity, f"Avoid phrase: {phrase}"))


def _normalise_number(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)
