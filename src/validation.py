"""Candidate assessment and act-specific fallback — side-effect free.

ValidationMixin owns the single correctness-critical assessment path that decides what
happens to a candidate utterance (ACCEPT / REPAIR / FALLBACK / DROP):
deterministic structural, option-reference, question, commitment, blocker,
switch, hybrid-option, and exact contradiction checks. Ordinary conversational
semantics do not block runtime turns. It also builds narrow safe fallbacks.
It returns structured assessments, never mutates dialogue state, and never
resolves threads; the observer decides what an accepted turn realized.
"""

from __future__ import annotations

import re

from aliases import short_alias_map
from models import (
    ActType,
    AssessmentAction,
    DialogueState,
    MoveIntent,
    Persona,
    TurnAssessment,
    ValidationIssue,
    VisibleEvidence,
    _DECISION_ACTS,
)
from parsing import hybrid_blend_detected, switch_bridge_ok
from utils import usable_reason_fragment


_ACTION_SEVERITY = {
    AssessmentAction.ACCEPT: 0,
    AssessmentAction.ACCEPT_WITH_METRIC: 1,
    AssessmentAction.REPAIR: 2,
    AssessmentAction.FALLBACK: 3,
    AssessmentAction.DROP: 4,
}


def assessment_severity(assessment: TurnAssessment) -> tuple[int, int, int, int]:
    """Orderable severity: action first, then blocking issues, then issue count.

    Candidate selection compares these tuples — never raw issue counts. One
    unsupported factual claim (blocking) outweighs any number of harmless
    metric-only deviations.
    """
    blocking = sum(1 for issue in assessment.issues if issue.blocking)
    return (
        _ACTION_SEVERITY.get(assessment.action, 4),
        1 if blocking else 0,
        blocking,
        len(assessment.issues),
    )


# P7: a discourse-marker head that promises content but delivers none
# ("Just to be clear.", "Actually.", "Oh, and."). Genuine short reactions
# ("Fair point.", "Not for me.") do not match.
_BARE_MARKER_TURN = re.compile(
    r"^\W*(?:just\s+to\s+be\s+clear|to\s+be\s+clear|actually|oh(?:,\s*and)?|"
    r"one\s+more\s+thing|by\s+the\s+way|that\s+said|on\s+top\s+of\s+that)\W*$",
    re.I,
)

# P7: a lone subordinate clause printed as a whole turn ("Since the Museum
# plan is low effort."). As an answer to a question this shape is natural
# ("Because it's cheap."), so the check skips answer acts.
_EXACT_QUANTIFIED_DETAIL = re.compile(
    r"(?:[$€£]\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*[$€£]|"
    r"\d+\s*h(?:\s*\d+\s*m)?|"
    r"\d+(?:[.,]\d+)?\s*[- ]?(?:%|percent|seconds?|secs?|minutes?|mins?|"
    r"hours?|hrs?|days?|weeks?|months?|euros?|dollars?|pounds?|people|persons?|"
    r"attendees?|seats?|rooms?|stops?|kilomet(?:er|re)s?|km|miles?|mb|gb|tb|"
    r"cups?|lit(?:er|re)s?))\b",
    re.I,
)

_TEXTUAL_TIME_DETAIL = re.compile(
    r"\b(?:(?P<relation>under|less\s+than|over|more\s+than)\s+)?"
    r"(?P<amount>half\s+an?|an?|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?P<unit>hours?|hrs?|minutes?|mins?)\b",
    re.I,
)

_WORD_NUMBER = {
    "a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0,
    "four": 4.0, "five": 5.0, "six": 6.0, "seven": 7.0,
    "eight": 8.0, "nine": 9.0, "ten": 10.0, "eleven": 11.0,
    "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "sixteen": 16.0, "seventeen": 17.0,
    "eighteen": 18.0, "nineteen": 19.0, "twenty": 20.0,
    "thirty": 30.0, "forty": 40.0, "fifty": 50.0,
    "sixty": 60.0, "seventy": 70.0, "eighty": 80.0,
    "ninety": 90.0, "hundred": 100.0,
}


def _fact_norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9%€$£]+", " ", text.lower()).split())


def _duration_minutes(text: str) -> float | None:
    """Parse one explicit card duration such as ``1h 40m`` or ``25 minutes``."""
    normalized = text.lower().replace(",", ".")
    combined = re.search(
        r"(?P<h>\d+(?:\.\d+)?)\s*(?:h|hours?|hrs?)\s*"
        r"(?P<m>\d+(?:\.\d+)?)?\s*(?:m|minutes?|mins?)?",
        normalized,
    )
    if combined:
        hours = float(combined.group("h"))
        minutes = float(combined.group("m") or 0.0)
        return hours * 60.0 + minutes
    minutes = re.search(r"(?P<m>\d+(?:\.\d+)?)\s*(?:m|minutes?|mins?)\b", normalized)
    if minutes:
        return float(minutes.group("m"))
    return None


def _textual_time_bound(match: re.Match[str]) -> tuple[str, float]:
    amount_text = _fact_norm(match.group("amount"))
    if amount_text.startswith("half"):
        amount = 0.5
    else:
        amount = _WORD_NUMBER[amount_text]
    unit = match.group("unit").lower()
    minutes = amount * (60.0 if unit.startswith(("h", "hr")) else 1.0)
    relation = _fact_norm(match.group("relation") or "exact")
    return relation, minutes


def _time_claim_conflicts(actual_minutes: float, relation: str, claimed_minutes: float) -> bool:
    if relation in {"under", "less than"}:
        return actual_minutes >= claimed_minutes
    if relation in {"over", "more than"}:
        return actual_minutes <= claimed_minutes
    return abs(actual_minutes - claimed_minutes) > 1.0


def _attribute_key_spans(key: str, text: str) -> list[tuple[int, int]]:
    """Return conservative lexical spans referring to one card attribute."""
    key_norm = _fact_norm(key.replace("_", " "))
    patterns: list[str] = []
    if key_norm:
        patterns.append(rf"\b{re.escape(key_norm)}\b")
    key_tokens = set(key_norm.split())
    if key_tokens & {"cost", "price", "fee", "usd", "eur"}:
        patterns.append(r"\b(?:cost|costs|price|priced|fee|euros?|dollars?|pounds?)\b|[$€£]")
    if "layover" in key_tokens:
        patterns.append(r"\b(?:layovers?|connections?)\b")
    elif "departure" in key_tokens and "time" in key_tokens:
        patterns.append(r"\b(?:departure(?:\s+time)?|departs?)\b")
    elif "travel" in key_tokens:
        patterns.append(r"\b(?:travel(?:\s+time)?|journey|commute)\b")
    elif key_tokens & {"duration", "length"}:
        patterns.append(r"\b(?:duration|length|lasts?|takes?)\b")
    elif "time" in key_tokens:
        patterns.append(r"\btime\b")
    if key_tokens & {"capacity", "seats", "places"}:
        patterns.append(r"\b(?:capacity|seats?|places?)\b")
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        spans.extend((m.start(), m.end()) for m in re.finditer(pattern, text, re.I))
    return sorted(set(spans))


def _attribute_key_visible(key: str, text: str) -> bool:
    return bool(_attribute_key_spans(key, text))


def _value_spans(value: str, text: str) -> list[tuple[int, int]]:
    """Find a listed attribute value in natural equivalent formatting.

    Card values and dialogue commonly differ only by hyphenation/pluralisation:
    ``45`` beside ``travel_time_minutes`` appears as ``45-minute travel time``;
    ``12 euros`` appears as ``12 euro``.  Matching these forms by raw substring
    caused listed facts to be treated as cross-option transfers.
    """
    raw = str(value).strip().lower()
    if not raw:
        return []
    tokens = re.findall(r"[a-z]+|\d+(?:[.,]\d+)?|[%€$£]", raw)
    if not tokens:
        return []
    unit_aliases = {
        "euro": r"euros?", "euros": r"euros?",
        "dollar": r"dollars?", "dollars": r"dollars?",
        "pound": r"pounds?", "pounds": r"pounds?",
        "hour": r"(?:hours?|hrs?|h)", "hours": r"(?:hours?|hrs?|h)",
        "minute": r"(?:minutes?|mins?|m)", "minutes": r"(?:minutes?|mins?|m)",
        "person": r"(?:people|persons?)", "people": r"(?:people|persons?)",
    }
    parts = [unit_aliases.get(token, re.escape(token)) for token in tokens]
    pattern = r"(?<![a-z0-9])" + r"[\s-]*".join(parts) + r"(?![a-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.I)]


def _numbers_in_text(text: str) -> set[str]:
    values = set(re.findall(r"\d+(?:[.,]\d+)?", _fact_norm(text)))
    for word in re.findall(r"[a-z]+", text.lower()):
        number = _WORD_NUMBER.get(word)
        if number is not None:
            values.add(str(int(number)) if number.is_integer() else str(number))
    return values


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0


def _locally_attached(text: str, left: tuple[int, int], right: tuple[int, int], *, max_gap: int = 18) -> bool:
    if _span_distance(left, right) > max_gap:
        return False
    lo = min(left[1], right[1]) if left[0] <= right[0] else min(right[1], left[1])
    hi = max(left[0], right[0])
    between = text[lo:hi]
    return not bool(re.search(r"\b(?:and|but|plus|while|whereas|versus|vs)\b", between, re.I))


_ASSERTED_OPTION_FACT = re.compile(
    r"\b(?:has|have|offers?|includes?|provides?|features?|supports?|sends?|"
    r"allows?|requires?|uses?|fits?\s+(?:on|in|inside|under|the|a|an)|guarantees?|can|could|might|will|comes?\s+with|"
    r"is\s+located(?:\s+in|\s+near|\s+at)?|is\s+downtown|is\s+nearby|"
    r"(?:schedule|availability|capacity|location|security|connection|layover|"
    r"baggage|noise|reliability|delays?)\s+(?:is|looks?|seems?|can|could|might|will))"
    r"\b(?P<claim>[^.!?;,—]*)",
    re.I,
)

_FACT_CLAIM_STOP_HEADS = {
    "a concern", "concerns", "a problem", "problems", "a point", "points",
    "a reason", "reasons", "a preference", "preferences", "a worry", "worries",
}

_FACTUAL_DETAIL_NOUN = re.compile(
    r"\b(?:amenit(?:y|ies)|lounges?|baggage|allowances?|capabilit(?:y|ies)|"
    r"features?|alerts?|updates?|locations?|downtown|nearby|capacity|seats?|seating|rooms?|"
    r"parking|accounts?|apps?|availability|schedules?|security|strikes?|delays?|warrant(?:y|ies)|"
    r"reliability|noise|services?|equipment|stations?|brew(?:ing)?|counter|"
    r"airport|terminal|connection|layover)\b",
    re.I,
)


_DETAIL_SEMANTIC_ALIASES = {
    "room": {"room", "roomy", "space", "capacity", "seat", "seats"},
    "rooms": {"room", "roomy", "space", "capacity", "seat", "seats"},
    "attendee": {"attendee", "attendees", "people", "person", "participants"},
    "attendees": {"attendee", "attendees", "people", "person", "participants"},
    "person": {"attendee", "attendees", "people", "person", "participants"},
    "persons": {"attendee", "attendees", "people", "person", "participants"},
    "alert": {"alert", "alerts", "update", "updates", "notification", "notifications"},
    "alerts": {"alert", "alerts", "update", "updates", "notification", "notifications"},
    "update": {"alert", "alerts", "update", "updates", "notification", "notifications"},
    "updates": {"alert", "alerts", "update", "updates", "notification", "notifications"},
    "connection": {"connection", "connections", "layover", "layovers"},
    "connections": {"connection", "connections", "layover", "layovers"},
    "layover": {"connection", "connections", "layover", "layovers"},
    "layovers": {"connection", "connections", "layover", "layovers"},
    "counter": {"counter", "compact", "footprint", "space-saving"},
    "counters": {"counter", "compact", "footprint", "space-saving"},
}


def _feature_detail_terms(text: str) -> list[str]:
    return [_fact_norm(match.group(0)) for match in _FACTUAL_DETAIL_NOUN.finditer(text)]


def _detail_is_listed(term: str, allowed_norm: str) -> bool:
    candidates = {term}
    if term.endswith("ies"):
        candidates.add(term[:-3] + "y")
    if term.endswith("es"):
        candidates.add(term[:-2])
    if term.endswith("s"):
        candidates.add(term[:-1])
    candidates.update(_DETAIL_SEMANTIC_ALIASES.get(term, set()))
    return any(
        re.search(rf"\b{re.escape(candidate)}\b", allowed_norm)
        for candidate in candidates if candidate
    )

_SUBORDINATE_ONLY_TURN = re.compile(
    r"^(?:since|because|although|even\s+if|even\s+though|whereas|unless)\b[^,.!?]*[.!?]?$",
    re.I,
)


class ValidationMixin:
    # ------------------------------------------------------------------
    # Evidence-based assessment (todo_validation item 9)
    # ------------------------------------------------------------------

    def _assess_candidate(
        self,
        *,
        text: str,
        state: DialogueState,
        persona: Persona,
        intent: MoveIntent,
        evidence: VisibleEvidence | None,
        verification_issues: list[str] | None = None,
        structure_flags: list[str] | None = None,
        operational_failure: bool = False,
    ) -> TurnAssessment:
        """Assess only correctness-critical runtime properties.

        Soft realization quality is telemetry. It must not trigger repairs of
        usable support, concern, answer, comparison, or opinion turns.
        """
        issues: list[ValidationIssue] = []

        def add(code: str, explanation: str = "", *, span: str = "",
                option: str | None = None, blocking: bool = True) -> None:
            if not any(i.code == code and i.option_id == option for i in issues):
                issues.append(ValidationIssue(code, explanation, span, option, blocking))

        stripped = text.strip()
        if not stripped:
            add("EMPTY_UTTERANCE")
        for flag in structure_flags or []:
            if flag in {"MULTI_TURN_OUTPUT", "LEAKED_METADATA", "MALFORMED_ENVELOPE"}:
                add("MALFORMED_UTTERANCE", flag)
        if _BARE_MARKER_TURN.match(stripped) or (
            intent.act != ActType.ANSWER and _SUBORDINATE_ONLY_TURN.match(stripped)
        ):
            add("MALFORMED_UTTERANCE", "lead-in or lone subordinate clause")
        if self._resolver and self._resolver.invalid_option_refs(text):
            add("INVALID_OPTION_REFERENCE")

        if operational_failure or evidence is None:
            add("CRITICAL_INTERPRETATION_UNAVAILABLE")
            return TurnAssessment(action=AssessmentAction.REPAIR, issues=issues)

        mentioned = [m.option_id for m in evidence.mentions]
        commitment = evidence.sole_commitment()
        if evidence.commitments and commitment is None:
            add("CONFLICTING_COMMITMENTS")

        focus = [o for o in intent.option_focus if o in state.scenario.option_ids]
        if intent.route_source == "coverage" and focus and focus[0] not in mentioned:
            add("MISSING_REQUIRED_OPTION_FOCUS", option=focus[0])
        if intent.act is ActType.COMPARE and len(focus) >= 2:
            missing = [oid for oid in focus[:2] if oid not in mentioned]
            if missing:
                add("MISSING_REQUIRED_OPTION_FOCUS", ",".join(missing))
        if intent.act is ActType.ASK and not evidence.questions:
            add("QUESTION_REQUIRED")

        rt = state.runtimes[persona.id]
        resolved = {b.option_id for b in evidence.blockers if b.action == "resolved"}
        for entry in evidence.commitments:
            if entry.option_id in rt.rejected_options() and entry.option_id not in resolved:
                add("BLOCKED_OPTION_ACCEPTED", option=entry.option_id)

        if (intent.act == ActType.COMPROMISE or evidence.proposals) and self._resolver and hybrid_blend_detected(text, self._resolver):
            add("HYBRID_COMPROMISE")

        if intent.act is ActType.VOTE:
            if commitment is None or commitment.kind not in {"vote", "accept"}:
                add("UNCLEAR_VISIBLE_COMMITMENT")
            elif intent.required_vote and commitment.option_id != intent.required_vote:
                add("REQUIRED_VOTE_MISMATCH", option=commitment.option_id)

        if intent.allow_vote_change and commitment and focus:
            allowed = set(focus)
            if intent.required_vote:
                allowed.add(intent.required_vote)
            if rt.explicit_vote:
                allowed.add(rt.explicit_vote)
            if commitment.option_id not in allowed:
                add("OFF_TARGET_SWITCH", option=commitment.option_id)

        previous_public = rt.explicit_vote
        if (
            commitment
            and intent.act is ActType.VOTE
            and previous_public in state.scenario.option_ids
            and commitment.option_id != previous_public
        ):
            bridged = any(
                s.target == commitment.option_id and (s.source or s.reason_span)
                for s in evidence.switches
            ) or bool(self._resolver and switch_bridge_ok(text, previous_public, self._resolver))
            if not bridged:
                add("UNBRIDGED_SWITCH")

        # High-confidence deterministic contradictions only. The resolver owns
        # exact card-value checks; ordinary implications and opinions are never
        # grounding failures here.
        if self._resolver:
            grounding_context = focus[0] if len(focus) == 1 else None
            for code, option_id, explanation in self._deterministic_fact_issues(
                text, state, context_option=grounding_context
            ):
                add(code, explanation, option=option_id)

        realized = self._intended_move_realized(intent, evidence, commitment, focus, add)
        focus_realized = bool(set(focus) & set(mentioned)) if focus else None
        action = AssessmentAction.REPAIR if any(i.blocking for i in issues) else AssessmentAction.ACCEPT
        return TurnAssessment(
            action=action,
            issues=issues,
            intended_act_realized=realized,
            intended_focus_realized=focus_realized,
            notes="critical deterministic validation",
        )

    def _deterministic_fact_issues(
        self,
        text: str,
        state: DialogueState,
        *,
        context_option: str | None = None,
    ) -> list[tuple[str, str | None, str]]:
        """Detect only high-confidence closed-world fact violations.

        A normal sentence is checked when it names one option. A comparison
        sentence naming several options is checked only when it contains a
        clear clause boundary (for example ``while``/``whereas``/``versus``)
        that leaves one explicit option in each local segment. Ambiguous shared
        predicates such as "A and B both ..." are deliberately ignored rather
        than guessed.
        """
        if self._resolver is None:
            return []
        issues: list[tuple[str, str | None, str]] = []
        full_ids = self._resolver.ids_in_text(text)
        clauses = [c.strip() for c in re.split(r"(?<=[.!?;])\s+|\s*,\s*(?=[A-Z])", text) if c.strip()]
        segments: list[tuple[str, str]] = []
        for clause in clauses:
            mentions = self._resolver.mentions(clause)
            ids = list(dict.fromkeys(m.option_id for m in mentions))
            if not ids and len(full_ids) == 1:
                segments.append((clause, full_ids[0]))
                continue
            if not ids and not full_ids and context_option in state.scenario.option_ids:
                # A routed answer/concern may naturally use "it" or only repeat
                # the visible card values.  The public single-option route may
                # scope grounding, but it still creates no support/commitment
                # evidence in the observer.
                segments.append((clause, context_option))
                continue
            if len(ids) == 1:
                segments.append((clause, ids[0]))
                continue
            # Multi-option clauses are safe to attribute only after an explicit
            # contrast/conjunction boundary. Every retained piece must name one
            # option itself; pronouns and shared "both/respectively" predicates
            # remain intentionally unaudited.
            pieces = [
                piece.strip(" ,:-")
                for piece in re.split(
                    r"\b(?:while|whereas|versus|vs\.?|but|compared\s+(?:with|to)|and)\b",
                    clause,
                    flags=re.I,
                )
                if piece.strip(" ,:-")
            ]
            if len(pieces) < 2:
                continue
            for piece in pieces:
                if re.search(r"\b(?:both|each|respectively)\b", piece, re.I):
                    continue
                piece_ids = list(dict.fromkeys(self._resolver.ids_in_text(piece)))
                if len(piece_ids) != 1:
                    continue
                # A bare option-name fragment created by splitting "A and B"
                # carries no local claim and must not borrow the next fragment's
                # predicate.
                has_local_fact = bool(
                    _EXACT_QUANTIFIED_DETAIL.search(piece)
                    or _ASSERTED_OPTION_FACT.search(piece)
                    or any(
                        str(value).strip().lower() in piece.lower()
                        for option in state.scenario.options
                        for value in option.attrs.values()
                        if len(str(value).strip()) >= 2
                    )
                )
                arithmetic_comparison = bool(
                    _EXACT_QUANTIFIED_DETAIL.search(piece)
                    and re.search(
                        r"\b(?:more|less|cheaper|costlier|longer|shorter|higher|lower)\s+than\b",
                        piece,
                        re.I,
                    )
                )
                if has_local_fact and not arithmetic_comparison:
                    segments.append((piece, piece_ids[0]))

        for clause, target_id in segments:
            target = state.scenario.option(target_id)
            low = clause.lower()
            clause_norm_for_values = _fact_norm(clause)
            for source in state.scenario.options:
                if source.id == target_id:
                    continue
                for attr, raw in source.attrs.items():
                    # Cross-option transfer is blocking only when both options
                    # expose the same attribute and the clause explicitly names
                    # that attribute. Generic values such as "low" or "weekly"
                    # are too ambiguous to police by substring.
                    if attr not in target.attrs or not _attribute_key_visible(attr, clause):
                        continue
                    value = str(raw).strip()
                    target_value = str(target.attrs.get(attr, "")).strip()
                    if not value or _fact_norm(target_value) == _fact_norm(value):
                        continue
                    value_spans = _value_spans(value, clause)
                    key_spans = _attribute_key_spans(attr, clause)
                    if value_spans and key_spans and any(
                        _locally_attached(clause, key_span, value_span)
                        for key_span in key_spans
                        for value_span in value_spans
                    ):
                        issues.append((
                            "CROSS_OPTION_VALUE", target_id,
                            f"{value!r} is listed for {source.id}, not {target_id}",
                        ))

            clause_norm = _fact_norm(clause)
            for time_match in _TEXTUAL_TIME_DETAIL.finditer(clause):
                relation, claimed_minutes = _textual_time_bound(time_match)
                all_temporal_attrs: list[tuple[str, str, float]] = []
                temporal_attrs: list[tuple[str, str, float]] = []
                claim_span = (time_match.start(), time_match.end())
                for attr, raw in target.attrs.items():
                    actual_minutes = _duration_minutes(str(raw))
                    if actual_minutes is None:
                        continue
                    entry = (attr, str(raw), actual_minutes)
                    all_temporal_attrs.append(entry)
                    spans = _attribute_key_spans(attr, clause)
                    if any(_locally_attached(clause, span, claim_span) for span in spans):
                        temporal_attrs.append(entry)
                if not temporal_attrs and len(all_temporal_attrs) == 1:
                    temporal_attrs = all_temporal_attrs
                if len(temporal_attrs) == 1:
                    attr, raw, actual_minutes = temporal_attrs[0]
                    if _time_claim_conflicts(actual_minutes, relation, claimed_minutes):
                        issues.append((
                            "ATTRIBUTE_CONTRADICTION", target_id,
                            f"the stated {attr!r} time conflicts with {raw!r} for {target_id}",
                        ))

            allowed_text = " ".join([
                state.scenario.topic,
                target.name, target.short_name,
                *[f"{key} {value}" for key, value in target.attrs.items()],
                target.upside or "", target.concern or "",
                *state.scenario.shared_context,
            ])
            quantified = list(_EXACT_QUANTIFIED_DETAIL.finditer(clause))
            for attr, raw in target.attrs.items():
                expected_numbers = _numbers_in_text(str(raw))
                key_spans = _attribute_key_spans(attr, clause)
                nearby_claims = [
                    match for match in quantified
                    if any(
                        _locally_attached(clause, span, (match.start(), match.end()))
                        for span in key_spans
                    )
                ]
                local_numbers = {
                    number for match in nearby_claims
                    for number in _numbers_in_text(match.group(0))
                }
                if expected_numbers and local_numbers and expected_numbers.isdisjoint(local_numbers):
                    issues.append((
                        "ATTRIBUTE_CONTRADICTION", target_id,
                        f"the stated {attr!r} value conflicts with {str(raw)!r} for {target_id}",
                    ))

            allowed_numbers = _numbers_in_text(allowed_text)
            for match in quantified:
                claimed = match.group(0)
                claimed_numbers = _numbers_in_text(claimed)
                if claimed_numbers and not claimed_numbers.issubset(allowed_numbers):
                    issues.append((
                        "UNLISTED_NUMERIC_DETAIL", target_id,
                        f"{claimed!r} is not listed for {target_id} or in shared context",
                    ))

            # Conservative existence/capability/location claims are checked
            # only when phrased as an assertive option fact. This catches
            # invented amenities, service features, alerts, and locations
            # without treating opinions or ordinary implications as facts.
            universal_context = [
                item for item in state.scenario.shared_context
                if target_id in self._resolver.ids_in_text(item)
                or re.search(
                    r"\b(?:all|every|each)\s+(?:options?|plans?|choices?|venues?|services?|products?)\b",
                    item,
                    re.I,
                )
            ]
            allowed_feature_text = " ".join([
                target.name, target.short_name,
                *[f"{key} {value}" for key, value in target.attrs.items()],
                target.upside or "", target.concern or "",
                *universal_context,
            ])
            allowed_norm = _fact_norm(allowed_feature_text)
            for match in _ASSERTED_OPTION_FACT.finditer(clause):
                prefix = clause[:match.start()].strip().lower()
                if re.search(r"\b(?:i|we|you)\s*$", prefix):
                    continue
                claim = match.group("claim").strip(" ,:-") or match.group(0)
                claim_norm = _fact_norm(claim)
                if (
                    not claim_norm
                    or claim_norm in _FACT_CLAIM_STOP_HEADS
                    or re.search(r"\b(?:been\s+)?(?:discussed|mentioned|supported|backed)\b", claim_norm)
                ):
                    continue
                # A listed phrase or a phrase made entirely from listed factual
                # tokens is safe. Otherwise the asserted feature/location is
                # not available in the closed-world setup.
                location_predicate = bool(re.search(
                    r"\bis\s+(?:located|downtown|nearby)\b", match.group(0), re.I
                ))
                if not location_predicate and not _FACTUAL_DETAIL_NOUN.search(match.group(0)):
                    continue
                detail_terms = _feature_detail_terms(match.group(0))
                missing_details = [
                    term for term in detail_terms
                    if not _detail_is_listed(term, allowed_norm)
                ]
                guarantee_predicate = bool(re.search(r"\bguarantees?\b", match.group(0), re.I))
                claim_content = {
                    token for token in claim_norm.split()
                    if len(token) > 3 and token not in {
                        "that", "this", "with", "from", "will", "every", "everyone",
                        "attendees", "people", "person", "group", "really", "enough",
                    }
                }
                guarantee_supported = bool(claim_content & set(allowed_norm.split()))
                if (
                    location_predicate and claim_norm not in allowed_norm
                    or missing_details
                    or guarantee_predicate and not guarantee_supported
                ):
                    issues.append((
                        "UNLISTED_FEATURE_DETAIL", target_id,
                        f"asserted feature/location {claim!r} is not listed for {target_id}",
                    ))

            # Possessive noun phrases can assert option features without an
            # explicit verb (for example "Senseo's quick brew"). Check only
            # the short phrase immediately following a resolved option mention
            # and only factual detail nouns from the conservative vocabulary.
            for mention in self._resolver.mentions(clause):
                if mention.option_id != target_id:
                    continue
                mention_end = mention.span.start + len(mention.span.text)
                tail = clause[mention_end:]
                possessive = re.match(
                    r"^\s*(?:'s|’s)\s+(?P<claim>[^.!?;,—]{1,60})",
                    tail,
                    re.I,
                )
                if not possessive:
                    continue
                claim = possessive.group("claim").strip(" ,:-")
                preceding = clause[max(0, mention.span.start - 72):mention.span.start].replace("’", "'")
                # Uncertainty suppresses a feature assertion only when it
                # directly governs this possessive phrase. A prior clause such
                # as "timing is unknown, but Senseo's quick brew ..." does not
                # license the new feature after the contrast boundary.
                local_preceding = re.split(
                    r"[.!?;]|,|\b(?:but|however|though|yet)\b",
                    preceding,
                    flags=re.I,
                )[-1]
                if re.search(
                    r"\b(?:do(?:n't| not)\s+know|unknown|unclear|uncertain|not\s+given|not\s+listed|"
                    r"question(?:ing)?\s+whether)\b",
                    local_preceding,
                    re.I,
                ):
                    continue
                # Only the possessive noun phrase itself asserts a feature.
                # Later consequences ("leaves room", "means delay") are
                # opinions/implications and must not be pulled into grounding.
                noun_phrase = re.split(
                    r"\b(?:is|are|was|were|has|have|had|can|could|might|may|will|would|"
                    r"means?|gives?|leaves?|seems?|looks?|worries?|helps?)\b",
                    claim,
                    maxsplit=1,
                    flags=re.I,
                )[0].strip(" ,:-")
                if re.search(
                    r"\b(?:question|concern|worry|unknown|unclear|uncertain|not\s+given|not\s+listed)\b",
                    noun_phrase,
                    re.I,
                ):
                    continue
                detail_terms = _feature_detail_terms(noun_phrase)
                missing_details = [
                    term for term in detail_terms
                    if not _detail_is_listed(term, allowed_norm)
                ]
                if missing_details:
                    issues.append((
                        "UNLISTED_FEATURE_DETAIL", target_id,
                        f"asserted possessive feature {noun_phrase!r} is not listed for {target_id}",
                    ))
        # Stable deduplication: one clause may trigger both a transferred card
        # value and an unlisted-number check for the same span.
        return list(dict.fromkeys(issues))


    @staticmethod
    def _intended_move_realized(
        intent: MoveIntent,
        evidence: VisibleEvidence,
        commitment,
        focus: list[str],
        add,
    ) -> bool | None:
        """Universal realization check: was the requested FUNCTION visibly
        performed (not: does the primary label match)? Applies on every route."""
        act = intent.act
        if act is ActType.OPENING:
            # A protocol opening is complete only when the participant visibly
            # establishes an initial option position. Merely mentioning an
            # option or giving generic process commentary must not satisfy the
            # mandatory opening round.
            hit = (
                commitment is not None
                and (not focus or commitment.option_id in focus)
            ) or any(not focus or s.option_id in focus for s in evidence.supports)
            if hit:
                return True
            if focus and (evidence.supports or commitment is not None):
                visible = (
                    commitment.option_id if commitment is not None
                    else evidence.supports[0].option_id
                )
                add(
                    "WRONG_OPTION_FOCUS",
                    f"opening position names {visible}, intended {focus[0]}",
                    option=focus[0],
                    blocking=True,
                )
            else:
                add("OPENING_POSITION_NOT_REALIZED", blocking=True)
            return False
        if act is ActType.SUPPORT:
            hit = any(not focus or s.option_id in focus for s in evidence.supports) or (
                commitment is not None and (not focus or commitment.option_id in focus)
            )
            if hit:
                return True
            if focus and evidence.supports:
                add("WRONG_OPTION_FOCUS",
                    f"supports {evidence.supports[0].option_id}, intended {focus[0]}",
                    option=focus[0], blocking=True)
            else:
                add("SUPPORT_NOT_REALIZED", blocking=True)
            return False
        if act is ActType.CONCERN:
            hit = any(not focus or c.option_id in focus for c in evidence.concerns)
            if hit:
                return True
            if focus and evidence.concerns:
                add("WRONG_OPTION_FOCUS",
                    f"concern about {evidence.concerns[0].option_id}, intended {focus[0]}",
                    option=focus[0], blocking=True)
            else:
                add("CONCERN_NOT_REALIZED", blocking=True)
            return False
        if act is ActType.COMPARE:
            need = set(focus[:2])
            has_comparison = any(len(set(c.option_ids)) >= 2 for c in evidence.comparisons)
            hit = any(
                len(set(c.option_ids) & need) >= min(2, len(need)) if need else len(c.option_ids) >= 2
                for c in evidence.comparisons
            )
            if not has_comparison:
                add("COMPARE_NOT_REALIZED", blocking=True)
            elif not hit:
                add(
                    "COMPARE_FOCUS_MISMATCH",
                    f"visible comparison does not cover required options {sorted(need)}",
                    blocking=True,
                )
            return hit
        if act is ActType.ASK:
            hit = any(q.scope in ("direct", "group") for q in evidence.questions)
            if not hit:
                add("ASK_NOT_REALIZED", blocking=True)
            return hit
        if act is ActType.ANSWER:
            hit = any(a.addresses_target for a in evidence.answers)
            if not hit:
                add("ANSWER_DOES_NOT_ADDRESS_QUESTION",
                    blocking=(intent.respond_to_turn is not None))
            return hit
        if act is ActType.COMPROMISE:
            hit = bool(evidence.proposals) or (commitment is not None and commitment.kind == "accept")
            if not hit:
                add("COMPROMISE_NOT_REALIZED", blocking=True)
            return hit
        if act is ActType.VOTE:
            if commitment is None:
                add("UNCLEAR_VISIBLE_COMMITMENT", blocking=True)
                return False
            if intent.required_vote and commitment.option_id != intent.required_vote:
                add("REQUIRED_VOTE_MISMATCH",
                    f"committed to {commitment.option_id}, required {intent.required_vote}",
                    option=intent.required_vote, blocking=True)
                return False
            return True
        return None  # opening / comment / process / closing: no realization contract

    def _fallback_candidate(
        self,
        state: DialogueState,
        persona: Persona,
        intent: MoveIntent,
        issue_codes: list[str],
    ) -> tuple[str | None, str]:
        """Narrow, truthful deterministic fallback (todo_validation item 12).

        Returns ``(text, family)`` — text is None when no truthful fallback
        exists for this intent, in which case the turn is dropped rather than
        printing false evidence. Only families whose complete public evidence
        can be constructed from known grounded data remain:

        - explicit vote / vote switch with the approved grounded reason;
        - hard-blocker restatement from the already grounded reason;
        - coverage request (a question introducing the unprocessed option);
        - exact listed answer, or the explicit does-not-say answer.

        Generic support/concern/compromise stand-ins are gone: a fallback
        must never fabricate a stance, proposal, or objection the sim did not
        visibly produce.
        """
        aliases = short_alias_map(state.scenario.options)
        rt = state.runtimes[persona.id]
        if intent.continuation:
            # An optional addendum with no safe content is simply dropped.
            return None, ""
        if intent.act in _DECISION_ACTS:
            return self._decision_fallback_text(state, persona, intent, issue_codes), "vote"
        if (
            "MISSING_REQUIRED_OPTION_FOCUS" in issue_codes
            and intent.route_source == "coverage"
            and intent.option_focus
        ):
            return self._coverage_question_fallback(state, intent, aliases), "coverage_question"
        focus = next((o for o in intent.option_focus if o in state.scenario.option_ids), None)
        if intent.act is ActType.OPENING:
            target = focus or rt.top_option() or persona.preferred_option
            if target not in state.scenario.option_ids:
                target = next(iter(state.scenario.option_ids), None)
            if target is None:
                return None, ""
            reason = usable_reason_fragment(rt.reason_for(target), aliases[target])
            if reason:
                return f"{aliases[target]} is where I'm leaning — {reason}.", "opening_lean"
            return f"{aliases[target]} is where I'm leaning right now.", "opening_lean"
        if intent.act is ActType.CONCERN and focus is not None and focus in rt.rejected_options():
            reason = usable_reason_fragment(rt.reason_against(focus), aliases[focus])
            if reason:
                # Hard-blocker restatement from the already grounded reason.
                return f"{aliases[focus]} still doesn't work for me — {reason}.", "blocker_restate"
            return None, ""
        if intent.act is ActType.ANSWER:
            text, family = self._answer_fallback_text(state, intent, aliases)
            return (text, family) if text else (None, "")
        # No truthful act-specific form exists (support/concern/compromise/
        # comment/process/ask outside coverage): drop instead of printing.
        return None, ""

    def _coverage_question_fallback(self, state: DialogueState, intent: MoveIntent, aliases: dict) -> str:
        gap = intent.option_focus[0]
        other = next((o for o in state.scenario.option_ids if o != gap), None)
        if other:
            return f"One option we haven't really talked about: {aliases[gap]}. How does it stack up against {aliases[other]}?"
        return f"One option we haven't really talked about: {aliases[gap]}. Worth a quick look before we decide."

    def _answer_fallback_text(self, state: DialogueState, intent: MoveIntent, aliases: dict) -> tuple[str | None, str]:
        """Exact listed answer when the asked attribute is on the card; the
        explicit does-not-say answer when it is not."""
        focus = next((o for o in intent.option_focus if o in state.scenario.option_ids), None)
        if focus is None:
            return None, ""
        option = state.scenario.option(focus)
        question = ""
        if intent.respond_to_turn is not None:
            target = next((t for t in state.turns if t.index == intent.respond_to_turn), None)
            question = target.text if target is not None else ""
        if not question.strip():
            # Without the actual question, neither a listed answer nor an
            # honest "the cards don't say" can be constructed truthfully.
            return None, ""
        question_tokens = set(re.findall(r"[a-zäöüß]+", question.lower()))
        for key, value in option.attrs.items():
            key_lower = key.lower().replace("_", " ")
            key_tokens = set(key_lower.split())
            # Generic English question-word bridges (how long -> duration,
            # how much -> cost); no scenario-specific dimension vocabulary.
            bridged = (
                (any(k in key_lower for k in ("duration", "time", "length"))
                 and question_tokens & {"long", "time", "hours", "minutes", "take", "takes", "last", "lasts"})
                or (any(k in key_lower for k in ("cost", "price", "fee"))
                    and question_tokens & {"much", "cost", "costs", "price", "expensive", "cheap", "euros", "dollars"})
            )
            if key_tokens & question_tokens or bridged:
                return f"For {aliases[focus]}, the card lists {key.replace('_', ' ')}: {value}.", "answer_listed"
        return (
            f"The listed facts don't say — {aliases[focus]}'s card doesn't cover that.",
            "answer_unknown",
        )

    def _decision_fallback_text(self, state: DialogueState, persona: Persona, intent: MoveIntent, issue_codes: list[str]) -> str:
        """Minimal, public, truthful decision-turn replacement (item 4).

        The text is one unambiguous commitment to an allowed option and nothing
        more. It never carries controller rationale (``intent.allowed_reason``),
        route/trace wording, or a prior preference taken from private ranks or
        controller state. A switch is mentioned ONLY when this participant has a
        prior PUBLIC commitment recorded in accepted state, and it carries no
        invented reason. Blocker turns never accept a rejected option.
        """
        aliases = short_alias_map(state.scenario.options)
        rt = state.runtimes[persona.id]
        blocked = next(iter(rt.rejected_options()), None)
        if intent.act in _DECISION_ACTS and intent.required_vote in state.scenario.option_ids:
            target = intent.required_vote
        else:
            candidates = [rt.top_option(), persona.preferred_option, *intent.option_focus, *state.scenario.option_ids]
            target = next(
                (o for o in candidates if o in state.scenario.option_ids and o != blocked and o not in rt.rejected_options()),
                next(o for o in state.scenario.option_ids if o != blocked),
            )
        if target == blocked or target in rt.rejected_options():
            target = next(o for o in state.scenario.option_ids if o != blocked and o not in rt.rejected_options())
        target_name = aliases[target]

        # Truthful hard-block restatement: this sim visibly cannot back `blocked`.
        if blocked and "BLOCKED_OPTION_ACCEPTED" in issue_codes:
            return f"I can't get behind {aliases[blocked]}, so I vote for {target_name}."

        # A public switch is stated only from a prior PUBLIC commitment by this
        # same participant (accepted vote in state) — never from private ranks
        # or controller intent — and only on a sanctioned change turn. The named
        # source is the visible bridge; no fabricated reason is added.
        prior_public = rt.explicit_vote
        if (
            intent.allow_vote_change
            and prior_public in state.scenario.option_ids
            and prior_public != target
        ):
            return f"I'm switching from {aliases[prior_public]} to {target_name}."

        # Default: a minimal, unambiguous vote. A small rotated template pool so
        # several fallback voters in one round do not sound identical; labels
        # match parsing._PHRASE_FAMILIES so avoid_phrases rotation works, and
        # every template parses as a direct vote.
        templates = [
            ("gets my vote", "{o} gets my vote."),
            ("I vote for", "I vote for {o}."),
            ("I'm going with", "I'm going with {o}."),
            ("my pick is", "My pick is {o}."),
        ]
        _label, template = next(
            ((l, t) for l, t in templates if l not in intent.avoid_phrases),
            templates[0],
        )
        return template.format(o=target_name)

