"""Deterministic live interpretation for correctness-critical dialogue state.

The interpreter extracts only visible evidence needed by runtime routing,
stance tracking, blockers, and formal voting. It performs no LLM calls and no
open-ended claim grounding. Ordinary conversational quality is handled by the
generation prompt and evaluation, not by a second semantic model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from models import (
    ActType,
    AnswerEvidence,
    BlockerEvidence,
    CommitmentEvidence,
    ComparisonEvidence,
    ConcernEvidence,
    EvidenceSpan,
    ProposalEvidence,
    QuestionEvidence,
    SofteningEvidence,
    SupportEvidence,
    SwitchEvidence,
    VisibleEvidence,
)
from parsing import (
    OptionResolver,
    active_blocker_option,
    blocker_resolution_option,
    commitment_has_reason,
    commitment_post_checks,
    visible_commitment,
    visible_comparison,
    visible_question,
)


@dataclass(slots=True)
class InterpretationResult:
    evidence: VisibleEvidence | None
    operational_failure: bool = False
    verification_issues: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    fast_path: bool = True
    fast_path_reason: str = "critical deterministic runtime interpretation"
    requested_categories: tuple[str, ...] = ()
    operational_retries: int = 0
    retry_failures: tuple[str, ...] = ()


def derive_primary_act(evidence: VisibleEvidence) -> ActType:
    """Stable display label derived from visible evidence priority."""
    if evidence.commitments:
        return ActType.VOTE
    if evidence.blockers or evidence.concerns:
        return ActType.CONCERN
    if evidence.questions:
        return ActType.ASK
    if evidence.comparisons:
        return ActType.COMPARE
    if evidence.answers:
        return ActType.ANSWER
    if evidence.proposals:
        return ActType.COMPROMISE
    if evidence.supports:
        return ActType.SUPPORT
    return ActType.COMMENT


_CONCERN = re.compile(
    r"\b(?:"
    r"(?:main|biggest|strongest|remaining|real)\s+(?:concern|worry|problem|issue)\b|"
    r"(?:concern|worry|problem|issue)\s+(?:with|about|is|would\s+be|for)\b|"
    r"drawback|downside|dealbreaker|hard\s+no|non-?starter|"
    r"(?:can(?:'t|not)|cannot|won't|doesn't|isn't|aren't|may\s+not|might\s+not)\s+"
    r"(?:work|fit|cover|solve|help|support|include|allow|handle|keep|leave|give|provide|offer|meet|reduce|avoid)|"
    r"too\s+(?:expensive|long|far|loud|risky|limited|quiet|slow|small|large|"
    r"tiring|demanding|intense|stressful)|"
    r"(?:risky|(?:high|serious|significant|unacceptable)\s+risk|"
    r"risk\s+(?:of|that|is\s+(?:high|serious|significant|unacceptable))|"
    r"lacks?|missing|unclear|unknown|not\s+given|not\s+listed)|"
    r"reject|against"
    r")\b",
    re.I,
)
_CONCERN_ACKNOWLEDGEMENT = re.compile(
    r"\b(?:i\s+(?:understand|get|see|hear)|that(?:'s|\s+is)\s+(?:a\s+)?(?:fair|valid|reasonable)|"
    r"fair|valid|reasonable)\s+(?:the|that|your|their|his|her)?\s*(?:concern|worry|issue|point)\b",
    re.I,
)
_NO_CONCERN = re.compile(
    r"\b(?:not|no\s+longer|isn't|is\s+not|won't\s+be|wouldn't\s+be)\s+"
    r"(?:a\s+|an\s+|the\s+)?(?:concern|worry|problem|issue|drawback|risk)\b|"
    r"\b(?:concern|worry|problem|issue|risk)\s+(?:is|has\s+been)\s+"
    r"(?:resolved|addressed|covered|gone|manageable|acceptable|minor|small|low)\b|"
    r"\b(?:benefit|upside|advantage|value)\s+(?:outweighs?|beats?)\s+"
    r"(?:the\s+)?(?:concern|worry|problem|issue|drawback|risk)\b",
    re.I,
)
_POSITIVE_RISK = re.compile(
    r"\b(?:low|minimal|reduced|limited|manageable|acceptable|small)\s+risk\b|"
    r"\brisk\s+(?:is|looks?|seems?)\s+(?:low|minimal|reduced|limited|manageable|acceptable|small)\b|"
    r"\b(?:reduces?|avoids?|limits?|lowers?)\s+(?:the\s+)?risk\b",
    re.I,
)
_PERSONAL_CONCERN = re.compile(
    r"\b(?:my\s+(?:concern|worry|problem|issue)|i\s+(?:worry|can't|cannot|won't)|"
    r"what\s+(?:worries|bothers)\s+me|for\s+me)\b",
    re.I,
)
_SUPPORT = re.compile(
    r"\b(?:prefer|like|good|better|best|works?|fit|suits?|help|benefit|"
    r"practical|reasonable|appealing|favour|favor|support|easy|useful)\b",
    re.I,
)
_CONDITIONAL = re.compile(r"\b(?:if|provided|assuming|as long as|could live with)\b", re.I)
_LEAN_OBJECT_AFTER = re.compile(
    r"\b(?:warming\s+(?:up\s+)?to|leaning\s+(?:toward|towards|to)|"
    r"starting\s+to\s+(?:like|prefer|favor|favour)|coming\s+(?:around\s+)?to|"
    r"i(?:'m|\s+am)\s+on\s+board\s+with)\b",
    re.I,
)
_LEAN_SUBJECT_BEFORE = re.compile(
    r"\b(?:looks?\s+better\s+now|(?:is\s+)?making\s+more\s+sense|"
    r"is\s+starting\s+to\s+look\s+better|is\s+growing\s+on\s+me)\b",
    re.I,
)
_LEAN_SETTLED = re.compile(
    r"\b(?:that\s+settles\s+it(?:\s+for\s+me)?|settles\s+it\s+for\s+me|"
    r"you(?:'ve|\s+have)\s+convinced\s+me)\b",
    re.I,
)
_PROPOSAL = re.compile(
    r"\b(?:common ground|compromise|settle on|meet in the middle|"
    r"could we choose|what about|maybe we (?:choose|go with))\b",
    re.I,
)


def _concern_mask(text: str) -> str:
    """Mask resolved/negated concern phrases without changing character spans."""
    masked = _NO_CONCERN.sub(lambda match: " " * (match.end() - match.start()), text)
    return _POSITIVE_RISK.sub(lambda match: " " * (match.end() - match.start()), masked)


_CLAUSE_BOUNDARY = re.compile(
    r"[.;!?—–]|\s+-{1,2}\s|"
    r",\s*(?=(?:but|while|whereas|though|although|however|since|because)\b)|"
    r"\b(?:but|while|whereas|though|although|however|since|because)\b",
    re.I,
)


def _clause_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Return the local clause containing ``start:end``.

    Evidence binding must not jump across a contrast/reason boundary.  This is
    what previously turned "I'm leaning toward Park Picnic, since Nature Hike
    ..." into a lean toward the later option.
    """
    left = 0
    right = len(text)
    for boundary in _CLAUSE_BOUNDARY.finditer(text):
        if boundary.end() <= start:
            left = boundary.end()
            continue
        if boundary.start() >= end:
            right = boundary.start()
            break
    return left, right


def _mentions_in_span(
    text: str,
    resolver: OptionResolver,
    span: tuple[int, int],
) -> list:
    lo, hi = span
    return [
        mention for mention in resolver.mentions(text)
        if lo <= mention.span.start < hi
    ]


def _visible_lean_option(text: str, resolver: OptionResolver) -> str | None:
    """Bind explicit positive stance movement to the option in its clause.

    The routed focus is deliberately not used: hidden intent cannot decide
    which option the speaker visibly warmed toward.  Object-form phrases bind
    to the first following option, subject-form phrases to the nearest prior
    option, and free "that settles it" phrases use the only/nearest option in
    the same clause.
    """
    for pattern, direction in (
        (_LEAN_OBJECT_AFTER, "after"),
        (_LEAN_SUBJECT_BEFORE, "before"),
        (_LEAN_SETTLED, "either"),
    ):
        for match in pattern.finditer(text):
            clause = _clause_span(text, match.start(), match.end())
            mentions = _mentions_in_span(text, resolver, clause)
            if direction == "either" and not mentions:
                # Natural acceptance often uses an em-dash bridge:
                # "That settles it for me—the Museum is workable."  The dash
                # is a clause boundary for concern attribution, but here it is
                # precisely the visible link to the selected option.
                nearby = [
                    mention for mention in resolver.mentions(text)
                    if abs(mention.span.start - match.end()) <= 90
                ]
                mentions = nearby
            if not mentions:
                continue
            before = [m for m in mentions if m.span.start < match.start()]
            after = [m for m in mentions if m.span.start >= match.end()]
            if direction == "after" and after:
                return after[0].option_id
            if direction == "before" and before:
                return before[-1].option_id
            if direction == "either":
                if after:
                    return after[0].option_id
                if before:
                    return before[-1].option_id
            if len(mentions) == 1:
                return mentions[0].option_id
    return None


def _has_visible_concern(text: str, act: ActType | None) -> bool:
    """Conservative concern detection that ignores acknowledgements/resolutions.

    A supporter saying "I get the worry, but X still works" responds to an
    existing concern; it must not open another concern thread. Likewise,
    "storage won't be an issue" is explicitly reassuring. A direct personal
    objection or a concrete negative predicate still counts regardless of act.
    """
    masked = _concern_mask(text)
    if not _CONCERN.search(masked):
        return False
    if (
        act in {ActType.SUPPORT, ActType.ANSWER, ActType.COMMENT}
        and _CONCERN_ACKNOWLEDGEMENT.search(masked)
        and not _PERSONAL_CONCERN.search(masked)
    ):
        return False
    return True


def _visible_concern_options(
    text: str,
    resolver: OptionResolver,
    *,
    act: ActType | None,
    focused: list[str],
) -> list[str]:
    """Resolve each visible negative claim to its nearest mentioned option.

    Intent focus can contain both the chosen option and the rejected alternative
    on vote/compare turns. Applying one detected negative predicate to every
    focused option created false concern threads. Local visible attribution is
    therefore primary; intent focus is used only as a single-option fallback for
    an explicit concern act.
    """
    if not _has_visible_concern(text, act):
        return []
    masked = _concern_mask(text)
    resolved: list[str] = []
    for match in _CONCERN.finditer(masked):
        clause = _clause_span(masked, match.start(), match.end())
        mentions = _mentions_in_span(text, resolver, clause)
        before = [m for m in mentions if m.span.start < match.start()]
        after = [m for m in mentions if m.span.start >= match.end()]
        # "concern with/about X" grammatically points forward.  Negative
        # predicates ("X doesn't help ... like Y") normally belong to the
        # subject before the predicate, not the comparison object after it.
        points_forward = bool(re.search(
            r"(?:concern|worry|problem|issue)\s+(?:with|about|for)\s*$",
            masked[max(clause[0], match.start() - 30):match.end()],
            re.I,
        ))
        target = None
        if points_forward and after:
            target = after[0]
        elif before:
            target = before[-1]
        elif after:
            target = after[0]
        if target is not None and target.option_id not in resolved:
            resolved.append(target.option_id)
            continue
        if act is ActType.CONCERN and len(focused) == 1 and focused[0] not in resolved:
            resolved.append(focused[0])
    if not resolved and act is ActType.CONCERN and len(focused) == 1:
        resolved.append(focused[0])
    return resolved


class TurnInterpreter:
    """Extract visible runtime evidence without contacting an LLM."""

    def __init__(
        self,
        _llm,
        resolver: OptionResolver,
        scenario,
        participant_names: dict[str, str],
        *,
        mode: str = "critical",
    ) -> None:
        if mode != "critical":
            raise ValueError("TurnInterpreter supports only validation mode 'critical'.")
        self._resolver = resolver
        self._scenario = scenario
        self._participant_names = dict(participant_names)

    def interpret(
        self,
        *,
        text: str,
        speaker_id: str,
        intent=None,
        target_turn_text: str | None = None,
        target_turn_speaker: str | None = None,
        thread_summary: str | None = None,
        previous_vote: str | None = None,
        context_candidates: tuple[str, ...] = (),
        rejected_options: tuple[str, ...] = (),
        previous_speaker_id: str | None = None,
    ) -> InterpretationResult:
        del target_turn_speaker, thread_summary, context_candidates
        stripped = text.strip()
        evidence = VisibleEvidence(
            utterance=stripped,
            mentions=self._resolver.mentions(stripped),
        )
        span = EvidenceSpan(text=stripped, start=0)
        mentioned = list(dict.fromkeys(m.option_id for m in evidence.mentions))
        sanctioned = bool(intent and getattr(intent, "allow_vote_change", False))

        commitment = visible_commitment(
            stripped, self._resolver, sanctioned_switch=sanctioned
        )
        if commitment is not None and commitment[0] in {"vote", "accept"}:
            kind, option_id = commitment
            failed_checks = commitment_post_checks(
                stripped,
                option_id,
                self._resolver,
                kind=kind,
                rejected_options=rejected_options,
                sanctioned_switch=sanctioned,
            )
            if not failed_checks:
                evidence.commitments.append(CommitmentEvidence(kind, option_id, span))
                if previous_vote and previous_vote != option_id:
                    source = next((oid for oid in mentioned if oid != option_id), None)
                    if source or sanctioned:
                        evidence.switches.append(
                            SwitchEvidence(
                                target=option_id,
                                source=source or previous_vote,
                                span=span,
                                reason_span=span if commitment_has_reason(stripped) else None,
                            )
                        )

        normalized = stripped.replace("’", "'").replace("‘", "'")
        raised = active_blocker_option(normalized, self._resolver)
        resolved = blocker_resolution_option(normalized, self._resolver)
        if raised:
            evidence.blockers.append(BlockerEvidence(raised, "raised", span))
            evidence.concerns.append(ConcernEvidence(raised, "hard", span))
        if resolved and resolved != raised:
            evidence.blockers.append(BlockerEvidence(resolved, "resolved", span))

        question = visible_question(
            stripped,
            speaker_id=speaker_id,
            participant_names=self._participant_names,
            previous_speaker_id=previous_speaker_id,
        )
        if question is not None:
            scope, addressee = question
            evidence.questions.append(
                QuestionEvidence(
                    scope=scope,
                    span=span,
                    addressee_id=addressee,
                    option_ids=mentioned[:2],
                )
            )

        comparison = visible_comparison(stripped, self._resolver)
        if comparison:
            evidence.comparisons.append(ComparisonEvidence(comparison, span))

        act = getattr(intent, "act", None)
        focused = [
            oid for oid in (getattr(intent, "option_focus", None) or [])
            if oid in mentioned
        ]
        targets = focused or mentioned[:1]
        concern_options = _visible_concern_options(
            normalized, self._resolver, act=act, focused=focused
        )
        has_support = bool(_SUPPORT.search(normalized))
        lean_target = _visible_lean_option(normalized, self._resolver)
        has_proposal = bool(_PROPOSAL.search(normalized))
        conditional = bool(_CONDITIONAL.search(normalized))

        for option_id in concern_options:
            if not any(c.option_id == option_id for c in evidence.concerns):
                evidence.concerns.append(ConcernEvidence(option_id, "ordinary", span))

        for option_id in targets:
            if has_support and act in {
                ActType.OPENING,
                ActType.SUPPORT,
                ActType.COMMENT,
                ActType.ANSWER,
                ActType.COMPROMISE,
            }:
                evidence.supports.append(
                    SupportEvidence(option_id, "conditional" if conditional else "firm", span)
                )
            if has_proposal and act is ActType.COMPROMISE:
                evidence.proposals.append(ProposalEvidence(option_id, span))

        if lean_target:
            evidence.softenings.append(SofteningEvidence(span=span, option_id=lean_target))

        if act is ActType.ANSWER and target_turn_text:
            evidence.answers.append(AnswerEvidence("full", span, addresses_target=True))

        evidence.primary_act = derive_primary_act(evidence)
        return InterpretationResult(evidence=evidence)
