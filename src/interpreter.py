"""Semantic turn interpretation: deterministic critical layer + selective
validator LLM.

Every candidate gets the deterministic critical layer (strict commitments
with post-checks, explicit blockers/resolutions, genuine questions, exact
mention resolution). In selective mode, simple fully-verifiable turns skip
the validator entirely (fast paths, each with an explicit reason); otherwise
ONE validator call requests only the soft categories the intended move can
change, plus grounding claims. Raw validator output is a PROPOSAL:
deterministic verification checks every span against the utterance, every
option/participant id against the world, context-resolved references against
the public-context resolver, and every critical commitment against the
deterministic parser before anything may reach state. Validator failures
never fail open.

The interpreter never generates public dialogue text and never sees private
ranks or hidden persona preferences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import prompts
from models import (
    ANSWER_COMPLETENESS,
    ActType,
    AnswerEvidence,
    BlockerEvidence,
    CLAIM_KINDS,
    COMMITMENT_KINDS,
    CommitmentEvidence,
    ComparisonEvidence,
    CONCERN_SEVERITIES,
    ConcernEvidence,
    EvidenceSpan,
    GroundingClaim,
    OptionMention,
    ProposalEvidence,
    QuestionEvidence,
    SofteningEvidence,
    SUPPORT_STRENGTHS,
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
    has_commitment_phrase,
    has_implicit_reference,
    visible_commitment,
    visible_comparison,
    visible_question,
)


@dataclass(slots=True)
class InterpretationResult:
    """Outcome of interpreting one candidate utterance.

    ``evidence`` is None exactly when interpretation failed closed (operational
    validator failure with no safe deterministic result). Semantic verification
    drops (``verification_issues``) are recorded separately from operational
    failures — a dropped proposal is an utterance problem, a dead endpoint is
    an infrastructure problem. Intended-move realization is NOT judged here:
    validation compares the verified evidence with controller intent (item 7).
    """

    evidence: VisibleEvidence | None
    operational_failure: bool = False
    verification_issues: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    fast_path: bool = False
    fast_path_reason: str = ""      # why the validator LLM was skipped
    requested_categories: tuple[str, ...] = ()  # validator payload actually requested
    # Structured-output retries against the validator endpoint: operational
    # noise, counted separately from semantic repairs of the utterance.
    operational_retries: int = 0


# ---------------------------------------------------------------------------
# Claim-level grounding (todo_validation item 8)
#
# A normalized fact table binds option ID + attribute + value plus the
# shared-context facts. Verification checks the exact relationship: a number
# must belong to the claimed option's own facts (or the shared context), not
# merely occur somewhere in the world text. "B holds 200 people" fails when
# 200 is A's capacity even though every word occurs in the scenario.
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
# Question shapes that assert a premise ("Isn't A closed on Sundays?",
# "Since it books out, shouldn't we...?") — those need the claim audit.
_PREMISE_QUESTION = re.compile(
    r"\b(?:isn'?t|aren'?t|doesn'?t|don'?t|wasn'?t|weren'?t|won'?t|didn'?t|"
    r"since|because|given\s+that|considering)\b",
    re.I,
)

_GENERIC_CLAIM_WORDS = {
    "euro", "euros", "dollar", "dollars", "hour", "hours", "minute", "minutes",
    "cost", "costs", "price", "takes", "about", "around", "roughly", "person",
    "people", "option", "the", "that", "this", "with", "have", "has",
}

# Comparison-structure vocabulary (item 6): comparative adjectives, relational
# and contrast connectives. A digit-free comparison whose only non-card content
# words come from here carries no groundable concrete claim, so it can skip the
# validator. This mirrors the item-3 grammar and adds no endpoint phrases.
_COMPARISON_WORDS = frozenset({
    "more", "less", "than", "versus", "compared", "cheaper", "pricier", "costlier",
    "dearer", "faster", "slower", "bigger", "larger", "smaller", "closer", "nearer",
    "farther", "further", "longer", "shorter", "easier", "harder", "simpler",
    "higher", "lower", "stronger", "weaker", "safer", "riskier", "nicer", "better",
    "worse", "double", "triple", "twice", "half", "while", "whereas", "however",
    "though", "although", "both", "versus", "vs",
})


def _numbers_in(text: str) -> set[float]:
    found = {
        float(match.replace(",", "."))
        for match in re.findall(r"\d+(?:[.,]\d+)?", text)
    }
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text, re.I):
            found.add(float(value))
    return found


def _content_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-zäöüß'-]{4,}", text.lower())
        if word not in _GENERIC_CLAIM_WORDS
    }


class FactTable:
    """Normalized deterministic fact base for one scenario."""

    def __init__(self, scenario) -> None:
        self.scenario = scenario
        self._option_text: dict[str, str] = {}
        self._option_numbers: dict[str, set[float]] = {}
        self._option_attrs: dict[str, set[str]] = {}
        self._option_attr_values: dict[str, dict[str, str]] = {}
        for option in scenario.options:
            text = " ".join(
                [option.name, option.short_name or "", option.upside or "", option.concern or ""]
                + [f"{key} {value}" for key, value in option.attrs.items()]
            ).lower()
            self._option_text[option.id] = text
            self._option_numbers[option.id] = _numbers_in(text)
            self._option_attrs[option.id] = {
                str(key).lower().replace("_", " ").strip() for key in option.attrs
            }
            self._option_attr_values[option.id] = {
                str(key).lower().replace("_", " ").strip(): str(value)
                for key, value in option.attrs.items()
            }
        self.context_items = [str(item) for item in scenario.shared_context]
        self._context_text = " ".join(self.context_items).lower()
        self._context_numbers = _numbers_in(self._context_text)
        self._all_numbers = set().union(self._context_numbers, *self._option_numbers.values()) \
            if self._option_numbers else set(self._context_numbers)

    # -- sources ---------------------------------------------------------

    def source_exists(self, ref: str) -> bool:
        """True when a claimed source binding ("A.cost", "context:1") exists."""
        ref = str(ref).strip()
        if ref.lower().startswith("context:"):
            try:
                return 0 <= int(ref.split(":", 1)[1]) < len(self.context_items)
            except ValueError:
                return False
        if "." not in ref:
            return False
        option_id, attribute = ref.split(".", 1)
        option_id = option_id.strip().upper()
        attribute = attribute.lower().replace("_", " ").strip()
        if option_id not in self._option_attrs:
            return False
        if attribute in {"upside", "concern", "name"}:
            return True
        return any(
            attribute == known or attribute in known or known in attribute
            for known in self._option_attrs[option_id]
        )

    # -- arithmetic ------------------------------------------------------

    def _derivable(self, value: float) -> bool:
        """A claimed number is reproducible when it is listed, or reachable by
        one simple operation over listed numbers (sum, difference, small
        multiple, or per-k split)."""
        if value in self._all_numbers:
            return True
        numbers = sorted(self._all_numbers)
        for i, a in enumerate(numbers):
            for b in numbers[i:]:
                if value in (a + b, abs(a - b)):
                    return True
            for k in range(2, 9):
                if value in (a * k,) or (abs(a / k - value) < 1e-9):
                    return True
        return False

    # -- contradiction ---------------------------------------------------

    def _attr_value_conflict(self, option_id, attribute, value) -> str | None:
        """When a claimed (attribute, value) names a KNOWN card attribute of the
        option but asserts a number absent from that attribute's listed value,
        report the conflict — a direct contradiction of a listed fact rather
        than a merely absent value (item 5). Non-numeric or unknown attributes
        are handled by the absence/ownership checks, not here."""
        if not option_id or not attribute or not value:
            return None
        attr_norm = str(attribute).lower().replace("_", " ").strip()
        card = self._option_attr_values.get(option_id, {})
        card_val = next(
            (v for k, v in card.items() if attr_norm == k or attr_norm in k or k in attr_norm),
            None,
        )
        if card_val is None:
            return None
        claim_nums = _numbers_in(str(value))
        card_nums = _numbers_in(card_val)
        if claim_nums and card_nums and not (claim_nums & card_nums):
            return (
                f"contradicts listed {attr_norm} of option {option_id} "
                f"({card_val.strip()})"
            )
        return None

    def _soft_premise_conflict(self, claim: GroundingClaim) -> str | None:
        """Guards a subjective conclusion (opinion/uncertainty/inference) against
        an embedded CONCRETE premise it must not smuggle past its soft label
        (item 5): an unreproducible number, or a structured attribute/value that
        contradicts the option card. The qualified conclusion itself passes."""
        for value in _numbers_in(claim.span.text):
            if not self._derivable(value):
                return (
                    f"embeds concrete value {value:g} that is not reproducible "
                    "from listed facts"
                )
        return self._attr_value_conflict(claim.option_id, claim.attribute, claim.value)

    # -- claim verification ----------------------------------------------

    def verify(self, claim: GroundingClaim) -> tuple[bool, str]:
        """(supported, reason-if-unsupported) for one atomic claim."""
        kind = claim.kind
        if kind in ("opinion", "uncertainty", "inference"):
            # Subjective judgments, unknowns, and qualified inferences from
            # listed facts pass (item 5): a reasonable conclusion is not rejected
            # merely because its exact words are not in the scenario. The soft
            # label may not hide a concrete premise, though — an unreproducible
            # number or a value that contradicts the card still fails.
            conflict = self._soft_premise_conflict(claim)
            return (False, conflict) if conflict else (True, "")
        if kind == "invented_detail":
            return False, "concrete detail not present in the scenario"
        if kind == "cross_option_transfer":
            return False, "applies another option's value to this option"
        if kind == "ungrounded_inference":
            return False, "conclusion with no traceable listed support"
        if kind == "contradiction":
            return False, "directly conflicts with a listed option fact"
        if kind == "arithmetic":
            span_numbers = _numbers_in(claim.span.text)
            for value in span_numbers:
                if not self._derivable(value):
                    return False, f"value {value:g} is not reproducible from listed numbers"
            return True, ""
        if kind == "listed_fact":
            return self._verify_listed_fact(claim)
        return False, f"unknown claim kind {kind!r}"

    def _verify_listed_fact(self, claim: GroundingClaim) -> tuple[bool, str]:
        span_numbers = _numbers_in(claim.span.text)
        # A structured (attribute, value) that names a known card attribute but a
        # different number is a direct contradiction, reported ahead of the
        # generic absence/ownership checks (item 5).
        conflict = self._attr_value_conflict(claim.option_id, claim.attribute, claim.value)
        if conflict:
            return False, conflict
        if claim.option_id is None:
            # Shared-context fact: values must come from the shared context.
            for value in span_numbers:
                if value not in self._context_numbers:
                    return False, f"value {value:g} is not in the shared context"
            if not span_numbers:
                missing = _content_words(claim.span.text) - _content_words(self._context_text)
                if missing == _content_words(claim.span.text) and missing:
                    return False, "statement is not traceable to the shared context"
            return True, ""
        option_numbers = self._option_numbers.get(claim.option_id, set())
        for value in span_numbers:
            if value in option_numbers or value in self._context_numbers:
                continue
            owners = [oid for oid, nums in self._option_numbers.items() if value in nums]
            if owners:
                return False, (
                    f"value {value:g} belongs to option {owners[0]}, not {claim.option_id}"
                )
            return False, f"value {value:g} is not listed for option {claim.option_id}"
        if not span_numbers:
            words = _content_words(claim.span.text)
            own = _content_words(self._option_text.get(claim.option_id, "")) | _content_words(self._context_text)
            if words and not (words & own):
                return False, f"no stated fact of option {claim.option_id} matches this claim"
        return True, ""


def _loose_bool(value) -> bool | None:
    """Booleans from structured LLM output: accept true/false and their
    string forms; anything else (null, prose) is None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


# Evidence-type precedence used only to derive the compatibility primary act.
def derive_primary_act(evidence: VisibleEvidence) -> ActType:
    if evidence.commitments:
        return ActType.VOTE
    if any(b.action == "raised" for b in evidence.blockers) or any(
        c.severity == "hard" for c in evidence.concerns
    ):
        return ActType.CONCERN
    if evidence.concerns:
        return ActType.CONCERN
    if any(q.scope in ("direct", "group") for q in evidence.questions):
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


# Semantic categories requested from the validator per intended act (item 7):
# a compact intent-specific payload instead of the full universal schema.
# Commitments, blockers, and questions are caught deterministically on every
# turn (critical parser); the validator adds soft recall where the intended
# state-changing move needs it. Vote turns additionally ask the validator for
# menu-less commitment wording and switch structure.
_ACT_CATEGORIES: dict[ActType, tuple[str, ...]] = {
    ActType.OPENING: ("supports", "concerns"),
    ActType.SUPPORT: ("supports", "concerns"),
    ActType.CONCERN: ("concerns", "supports"),
    ActType.COMMENT: ("supports", "concerns", "softenings"),
    ActType.COMPARE: ("comparisons",),
    ActType.ASK: (),
    ActType.ANSWER: ("answers", "concerns"),
    ActType.COMPROMISE: ("proposals", "concerns", "softenings"),
    ActType.PROCESS: (),
    ActType.VOTE: ("commitments", "switches", "concerns"),
    ActType.CLOSING: (),
}
# Without controller intent (fixture/diagnostic interpretation), request the
# full soft-category set.
_ALL_CATEGORIES: tuple[str, ...] = (
    "supports", "concerns", "comparisons", "answers",
    "softenings", "proposals", "commitments", "switches",
)


class TurnInterpreter:
    """Owns the validator call and the deterministic verification of its output.

    ``mode`` is the validation mode (item 8): "selective" (default) applies
    the deterministic fast paths below and calls the validator LLM only when
    soft natural-language meaning can change state; "full" interprets every
    candidate through the LLM (diagnostics/evaluation).
    """

    def __init__(
        self,
        validator_llm,
        resolver: OptionResolver,
        scenario,
        participant_names: dict[str, str],
        *,
        mode: str = "selective",
    ) -> None:
        self._llm = validator_llm
        self._resolver = resolver
        self._scenario = scenario
        self._participant_names = dict(participant_names)
        self._facts = FactTable(scenario)
        self._mode = str(mode)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

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
        sanctioned = bool(intent and getattr(intent, "allow_vote_change", False))
        if self._mode != "full":
            fast = self._deterministic_fast_path(
                text, intent,
                sanctioned=sanctioned,
                rejected_options=rejected_options,
                speaker_id=speaker_id,
                previous_speaker_id=previous_speaker_id,
            )
            if fast is not None:
                evidence, reason = fast
                return InterpretationResult(
                    evidence=evidence, fast_path=True, fast_path_reason=reason,
                )

        categories = (
            _ACT_CATEGORIES.get(intent.act, _ALL_CATEGORIES) if intent is not None
            else _ALL_CATEGORIES
        )
        # Prompt context reduction (item 9): only the options this candidate
        # can be about — explicitly mentioned, in the routed focus, or public
        # context candidates. Deterministic grounding still checks claimed
        # values against the FULL fact table, so omitting unrelated cards
        # cannot weaken enforcement. With no relevant subset (implicit-only
        # references), the full board is sent.
        mentions = self._resolver.mentions(text)
        relevant_ids = (
            {m.option_id for m in mentions}
            | set(getattr(intent, "option_focus", None) or [])
            | set(context_candidates)
        )
        relevant_options = [o for o in self._scenario.options if o.id in relevant_ids]
        prompt = prompts.validator_interpret(
            utterance=text,
            speaker_name=self._participant_names.get(speaker_id, speaker_id),
            options=relevant_options or list(self._scenario.options),
            shared_context=list(self._scenario.shared_context),
            resolved_mentions=[
                f"{m.option_id} (\"{m.alias_form}\")" for m in mentions
            ],
            categories=categories,
            target_turn_text=target_turn_text,
            target_turn_speaker=target_turn_speaker,
            thread_summary=thread_summary,
            previous_vote=previous_vote if "commitments" in categories else None,
        )
        tokens_in = tokens_out = 0
        data = None
        retries = 0
        for attempt in range(2):  # at most one structured-output retry
            try:
                data = self._llm.generate_json(prompt, profile="validator")
            except Exception:
                data = None
            tokens_in += int(getattr(self._llm, "last_tokens_in", 0))
            tokens_out += int(getattr(self._llm, "last_tokens_out", 0))
            if isinstance(data, dict):
                break
            retries = attempt + 1
        if not isinstance(data, dict):
            # Fail closed: no unvalidated LLM output may shape soft state.
            return InterpretationResult(
                evidence=None, operational_failure=True,
                tokens_in=tokens_in, tokens_out=tokens_out,
                operational_retries=retries, requested_categories=categories,
            )
        evidence, issues = self._verify(
            data, text,
            categories=categories,
            context_candidates=context_candidates,
            rejected_options=rejected_options,
            sanctioned=sanctioned,
        )
        self._merge_deterministic_evidence(
            evidence, text,
            speaker_id=speaker_id,
            sanctioned=sanctioned,
            rejected_options=rejected_options,
            previous_speaker_id=previous_speaker_id,
        )
        self._strip_blocker_conflicts(evidence, issues)
        self._ground_claims(evidence)
        evidence.primary_act = derive_primary_act(evidence)
        return InterpretationResult(
            evidence=evidence,
            verification_issues=issues,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            operational_retries=retries,
            requested_categories=categories,
        )

    # ------------------------------------------------------------------
    # Deterministic critical layer (always on)
    # ------------------------------------------------------------------

    def _merge_deterministic_evidence(
        self,
        evidence: VisibleEvidence,
        text: str,
        *,
        speaker_id: str,
        sanctioned: bool,
        rejected_options: tuple[str, ...],
        previous_speaker_id: str | None,
    ) -> None:
        """Critical visible facts the deterministic layer always contributes:
        strict commitments (post-checked), explicit blockers/resolutions, and
        genuine questions. Validator entries for the same fact are deduped."""
        span = EvidenceSpan(text=text.strip(), start=0)
        commit = visible_commitment(text, self._resolver, sanctioned_switch=sanctioned)
        if commit is not None and commit[0] in ("vote", "accept"):
            kind, option_id = commit
            already = any(c.option_id == option_id for c in evidence.commitments)
            if not already and not commitment_post_checks(
                text, option_id, self._resolver,
                kind=kind, rejected_options=rejected_options, sanctioned_switch=sanctioned,
            ):
                evidence.commitments.append(CommitmentEvidence(kind, option_id, span))

        check = text.replace("’", "'").replace("‘", "'")
        raised = active_blocker_option(check, self._resolver)
        resolved = blocker_resolution_option(check, self._resolver)
        # A same-option raise+resolve conflict is stripped (with an issue)
        # by _strip_blocker_conflicts after the merge.
        if raised and not any(
            b.option_id == raised and b.action == "raised" for b in evidence.blockers
        ):
            evidence.blockers.append(BlockerEvidence(raised, "raised", span))
        if resolved and not any(
            b.option_id == resolved and b.action == "resolved" for b in evidence.blockers
        ):
            evidence.blockers.append(BlockerEvidence(resolved, "resolved", span))

        question = visible_question(
            text, speaker_id=speaker_id, participant_names=self._participant_names,
            previous_speaker_id=previous_speaker_id,
        )
        if question is not None and not evidence.questions:
            scope, addressee = question
            evidence.questions.append(QuestionEvidence(
                scope=scope, span=span, addressee_id=addressee,
                option_ids=[m.option_id for m in self._resolver.mentions(text)][:2],
            ))

        # Basic two-option comparison (item 3): only when the validator did not
        # already return one — the validator still owns subtle direction and
        # dimension. Deterministic recognition keeps COMPARISON_MISSES_OPTIONS
        # reserved for lines that are not visibly comparisons.
        if not evidence.comparisons:
            pair = visible_comparison(text, self._resolver)
            if pair:
                evidence.comparisons.append(ComparisonEvidence(option_ids=pair, span=span))

    @staticmethod
    def _strip_blocker_conflicts(evidence: VisibleEvidence, issues: list[str]) -> None:
        raised = {b.option_id for b in evidence.blockers if b.action == "raised"}
        resolved = {b.option_id for b in evidence.blockers if b.action == "resolved"}
        for option_id in raised & resolved:
            issues.append(f"BLOCKER_RAISED_AND_RESOLVED:{option_id}")
            evidence.blockers = [b for b in evidence.blockers if b.option_id != option_id]

    def _ground_claims(self, evidence: VisibleEvidence) -> None:
        # Claim-level grounding: every classified claim is verified against
        # the normalized fact table inside the same interpretation contract —
        # a validator failure can never produce a false "grounded".
        for claim in evidence.claims:
            supported, reason = self._facts.verify(claim)
            claim.supported = supported
            claim.reason = "" if supported else reason

    # ------------------------------------------------------------------
    # Deterministic fast paths (item 8, selective mode only)
    # ------------------------------------------------------------------

    def _deterministic_fast_path(
        self,
        text: str,
        intent,
        *,
        sanctioned: bool,
        rejected_options: tuple[str, ...],
        speaker_id: str,
        previous_speaker_id: str | None,
    ) -> tuple[VisibleEvidence, str] | None:
        """Skip the validator LLM only when deterministic code can fully
        establish the turn's public semantics and no soft natural-language
        state update is possible. A fast path never fabricates evidence from
        hidden intent: everything it accepts is deterministically visible.

        Lines containing digits always go to the validator — numbers need
        claim-level grounding judgment.
        """
        stripped = text.strip()
        if not stripped or re.search(r"\d", stripped):
            return None
        check = stripped.replace("’", "'").replace("‘", "'")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]
        mentions = self._resolver.mentions(stripped)
        span = EvidenceSpan(text=stripped, start=0)
        intent_act = getattr(intent, "act", None)
        commit = visible_commitment(stripped, self._resolver, sanctioned_switch=sanctioned)
        raised = active_blocker_option(check, self._resolver)

        # 1) Direct unambiguous commitment; on sanctioned switch turns the
        #    old pick may be named as the visible bridge (two mentions).
        if commit is not None and commit[0] in ("vote", "accept"):
            kind, option_id = commit
            simple = (
                "?" not in stripped
                and len(sentences) <= 2
                and len(stripped.split()) <= (26 if sanctioned else 24)
                and len({m.option_id for m in mentions}) <= (2 if sanctioned else 1)
                and not commitment_post_checks(
                    stripped, option_id, self._resolver,
                    kind=kind, rejected_options=rejected_options,
                    sanctioned_switch=sanctioned,
                )
            )
            if not simple:
                return None
            evidence = VisibleEvidence(
                utterance=stripped,
                mentions=mentions,
                commitments=[CommitmentEvidence(kind=kind, option_id=option_id, span=span)],
                primary_act=ActType.VOTE,
            )
            if sanctioned:
                source = next(
                    (m.option_id for m in mentions if m.option_id != option_id), None
                ) or getattr(intent, "old_preference", None)
                if source and source != option_id:
                    evidence.switches.append(SwitchEvidence(
                        target=option_id, span=span, source=source,
                        reason_span=span if commitment_has_reason(check) else None,
                    ))
                return evidence, "direct unambiguous sanctioned switch, verified deterministically"
            return evidence, "direct unambiguous fact-free commitment, verified deterministically"

        # 2) Explicit hard-blocker restatement about exactly one option
        #    (reject wording is part of the blocker vocabulary, so this runs
        #    before the general rejection bail-out).
        if (
            raised is not None
            and intent_act is ActType.CONCERN
            and "?" not in stripped
            and len({m.option_id for m in mentions}) == 1
            and blocker_resolution_option(check, self._resolver) is None
        ):
            evidence = VisibleEvidence(
                utterance=stripped,
                mentions=mentions,
                concerns=[ConcernEvidence(raised, "hard", span)],
                blockers=[BlockerEvidence(raised, "raised", span)],
                primary_act=ActType.CONCERN,
            )
            return evidence, "explicit hard-blocker statement, verified deterministically"
        if commit is not None or raised is not None:
            return None  # rejection/blocker semantics beyond the simple shape -> validator

        # 3) Process/closing text cannot change option state.
        if intent_act in (ActType.PROCESS, ActType.CLOSING):
            evidence = VisibleEvidence(utterance=stripped, mentions=mentions)
            self._add_deterministic_question(evidence, stripped, speaker_id, previous_speaker_id, span)
            evidence.primary_act = intent_act
            return evidence, "process/closing text cannot change option state"

        # 4) Plain single question on an ask turn: question detection is
        #    deterministic, and with no digits and no premise-asserting shape
        #    (negative-polarity or subordinate lead-in) the line carries no
        #    checkable claim to audit.
        if (
            intent_act is ActType.ASK
            and len(sentences) == 1
            and stripped.endswith("?")
            and not _PREMISE_QUESTION.search(check)
        ):
            evidence = VisibleEvidence(utterance=stripped, mentions=mentions)
            self._add_deterministic_question(evidence, stripped, speaker_id, previous_speaker_id, span)
            if evidence.questions:
                evidence.primary_act = ActType.ASK
                return evidence, "plain premise-free question, classified deterministically"
            return None

        # 5) Mention-free light comment: with no option reference (explicit or
        #    implicit) and no commitment wording, no option-bound semantics or
        #    concrete option claims are possible.
        if (
            intent_act is ActType.COMMENT
            and not mentions
            and len(stripped.split()) <= 18
            and not has_commitment_phrase(check)
            and not has_implicit_reference(check)
        ):
            evidence = VisibleEvidence(utterance=stripped)
            self._add_deterministic_question(evidence, stripped, speaker_id, previous_speaker_id, span)
            evidence.primary_act = derive_primary_act(evidence)
            return evidence, "mention-free light comment cannot change option state"

        # 6) Clean two-option comparison (item 6): the required COMPARE evidence
        #    is deterministic (item 3) and — with no digits and no residual
        #    content word beyond the comparison vocabulary and the two options'
        #    own card/context terms — the line carries no groundable concrete
        #    claim, so the validator has nothing left to add.
        if intent_act is ActType.COMPARE and not has_commitment_phrase(check):
            pair = visible_comparison(stripped, self._resolver)
            if pair and self._comparison_carries_no_groundable_claim(stripped, pair):
                evidence = VisibleEvidence(
                    utterance=stripped, mentions=mentions,
                    comparisons=[ComparisonEvidence(option_ids=pair, span=span)],
                )
                self._add_deterministic_question(evidence, stripped, speaker_id, previous_speaker_id, span)
                evidence.primary_act = ActType.COMPARE
                return evidence, "deterministic two-option comparison over listed facts"

        return None

    def _comparison_carries_no_groundable_claim(self, text: str, pair: list[str]) -> bool:
        """True when a digit-free comparison's content words are all comparison
        vocabulary or terms already on the two compared options' cards / shared
        context — i.e. nothing concrete remains for the grounding judge. Any
        residual word (a possible invented capability) returns False so the turn
        still goes to the validator (safe direction)."""
        words = _content_words(text)
        if not words:
            return True
        allowed = set(_COMPARISON_WORDS) | _content_words(self._facts._context_text)
        for oid in pair:
            allowed |= _content_words(self._facts._option_text.get(oid, ""))
        return words <= allowed

    def _add_deterministic_question(
        self, evidence: VisibleEvidence, text: str, speaker_id: str,
        previous_speaker_id: str | None, span: EvidenceSpan,
    ) -> None:
        question = visible_question(
            text, speaker_id=speaker_id, participant_names=self._participant_names,
            previous_speaker_id=previous_speaker_id,
        )
        if question is not None:
            scope, addressee = question
            evidence.questions.append(QuestionEvidence(
                scope=scope, span=span, addressee_id=addressee,
                option_ids=[m.option_id for m in evidence.mentions][:2],
            ))

    # ------------------------------------------------------------------
    # Deterministic verification of validator output
    # ------------------------------------------------------------------

    def _locate(self, span_text, utterance: str) -> EvidenceSpan | None:
        """Exact span in the utterance, tolerant only to whitespace collapse
        and apostrophe variants. None when the validator invented text."""
        wanted = " ".join(str(span_text or "").split())
        if not wanted:
            return None
        idx = utterance.find(wanted)
        if idx < 0:
            normalised = utterance.replace("’", "'").replace("‘", "'")
            wanted_norm = wanted.replace("’", "'").replace("‘", "'")
            idx = normalised.find(wanted_norm)
            if idx < 0:
                return None
            return EvidenceSpan(text=utterance[idx: idx + len(wanted_norm)], start=idx)
        return EvidenceSpan(text=wanted, start=idx)

    def _bind_option(
        self,
        option_id,
        explicit_ids: list[str],
        text: str,
        context_candidates: tuple[str, ...],
        issues: list[str],
        what: str,
    ) -> tuple[str | None, str]:
        """Verify one option binding. Returns (option_id or None, resolution)."""
        if option_id is None:
            return None, "explicit"
        option_id = str(option_id).strip().upper()
        if option_id not in self._resolver.by_id:
            issues.append(f"INVALID_OPTION:{what}:{option_id}")
            return None, "explicit"
        if option_id in explicit_ids:
            return option_id, "explicit"
        resolved, _ambiguous = self._resolver.resolve_reference(
            text, context_candidates=context_candidates
        )
        if resolved == option_id:
            return option_id, "context"
        issues.append(f"UNVERIFIED_CONTEXT_RESOLUTION:{what}:{option_id}")
        return None, "explicit"

    def _verify(
        self,
        data: dict,
        text: str,
        *,
        categories: tuple[str, ...],
        context_candidates: tuple[str, ...],
        rejected_options: tuple[str, ...],
        sanctioned: bool,
    ) -> tuple[VisibleEvidence, list[str]]:
        issues: list[str] = []
        explicit_mentions = self._resolver.mentions(text)
        explicit_ids = [m.option_id for m in explicit_mentions]
        evidence = VisibleEvidence(utterance=text, mentions=list(explicit_mentions))
        context_bound: set[str] = set()
        # Grounding claims are part of the common portion on every call.
        requested = set(categories) | {"claims"}

        def entries(key: str) -> list[dict]:
            # Only requested categories are consumed: unrequested output is
            # never allowed to smuggle semantics past the contract (item 7).
            if key not in requested:
                return []
            raw = data.get(key)
            return [e for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []

        def span_of(entry: dict, what: str) -> EvidenceSpan | None:
            span = self._locate(entry.get("span"), text)
            if span is None:
                issues.append(f"SPAN_NOT_IN_UTTERANCE:{what}")
            return span

        def bound(entry: dict, what: str, key: str = "option") -> tuple[str | None, str]:
            option_id, resolution = self._bind_option(
                entry.get(key), explicit_ids, text, context_candidates, issues, what
            )
            if option_id and resolution == "context":
                context_bound.add(option_id)
            return option_id, resolution

        for entry in entries("supports"):
            span = span_of(entry, "support")
            option_id, _res = bound(entry, "support")
            strength = str(entry.get("strength") or "").lower()
            if span is None or option_id is None:
                continue
            if strength not in SUPPORT_STRENGTHS:
                issues.append(f"INVALID_VOCABULARY:support:{strength}")
                continue
            evidence.supports.append(SupportEvidence(option_id, strength, span))

        for entry in entries("concerns"):
            span = span_of(entry, "concern")
            option_id, _res = bound(entry, "concern")
            severity = str(entry.get("severity") or "").lower()
            if span is None or option_id is None:
                continue
            if severity not in CONCERN_SEVERITIES:
                issues.append(f"INVALID_VOCABULARY:concern:{severity}")
                continue
            evidence.concerns.append(ConcernEvidence(option_id, severity, span))

        for entry in entries("comparisons"):
            span = span_of(entry, "comparison")
            raw_options = entry.get("options") if isinstance(entry.get("options"), list) else []
            ids: list[str] = []
            for raw_id in raw_options:
                option_id, _res = self._bind_option(
                    raw_id, explicit_ids, text, context_candidates, issues, "comparison"
                )
                if option_id:
                    ids.append(option_id)
            favored, _res = bound(entry, "comparison_favored", key="favored") if entry.get("favored") else (None, "explicit")
            if span is None or len(set(ids)) < 2:
                if span is not None:
                    issues.append("COMPARISON_NEEDS_TWO_OPTIONS")
                continue
            dimension = str(entry.get("dimension")).strip() if entry.get("dimension") else None
            evidence.comparisons.append(ComparisonEvidence(
                option_ids=list(dict.fromkeys(ids)), span=span,
                favored=favored if favored in ids else None, dimension=dimension,
            ))

        for entry in entries("answers"):
            span = span_of(entry, "answer")
            completeness = str(entry.get("completeness") or "").lower()
            if span is None:
                continue
            if completeness not in ANSWER_COMPLETENESS:
                issues.append(f"INVALID_VOCABULARY:answer:{completeness}")
                continue
            evidence.answers.append(AnswerEvidence(
                completeness=completeness, span=span,
                addresses_target=bool(entry.get("addresses_target", completeness in ("full", "partial"))),
            ))

        for entry in entries("softenings"):
            span = span_of(entry, "softening")
            option_id, _res = bound(entry, "softening")
            if span is None:
                continue
            evidence.softenings.append(SofteningEvidence(
                span=span, option_id=option_id, concession=bool(entry.get("concession", False)),
            ))

        for entry in entries("proposals"):
            span = span_of(entry, "proposal")
            option_id, _res = bound(entry, "proposal")
            if span is None:
                continue
            evidence.proposals.append(ProposalEvidence(option_id=option_id, span=span))

        # Blockers/resolutions are deterministic-only (merged after _verify);
        # the same-line resolution exception for a rejected-option acceptance
        # uses the deterministic detection directly.
        check = text.replace("’", "'").replace("‘", "'")
        deterministic_resolution = blocker_resolution_option(check, self._resolver)
        resolved_blockers = {deterministic_resolution} if deterministic_resolution else set()
        for entry in entries("commitments"):
            span = span_of(entry, "commitment")
            option_id, resolution = bound(entry, "commitment")
            kind = str(entry.get("kind") or "").lower()
            if span is None or option_id is None:
                continue
            if kind not in COMMITMENT_KINDS:
                issues.append(f"INVALID_VOCABULARY:commitment:{kind}")
                continue
            # Critical conclusions must satisfy the deterministic parser: a
            # validator-proposed vote is still bound by naming, prerequisite,
            # question, conflict, and rejected-option protection (item 6).
            check_kind = "accept" if (kind == "accept" and resolution == "context") else kind
            critical = commitment_post_checks(
                text, option_id, self._resolver,
                kind=check_kind,
                rejected_options=rejected_options,
                resolves_blocker=option_id if option_id in resolved_blockers else None,
                sanctioned_switch=sanctioned,
            )
            if critical:
                issues.extend(f"COMMITMENT_REJECTED:{code}" for code in critical)
                continue
            evidence.commitments.append(CommitmentEvidence(kind, option_id, span))

        committed_ids = {c.option_id for c in evidence.commitments}
        for entry in entries("switches"):
            span = span_of(entry, "switch")
            target, _res = bound(entry, "switch_target", key="target")
            source, _src_res = (
                bound(entry, "switch_source", key="source") if entry.get("source") else (None, "explicit")
            )
            if span is None or target is None:
                continue
            if target not in committed_ids:
                issues.append(f"SWITCH_WITHOUT_COMMITMENT:{target}")
                continue
            reason_span = self._locate(entry.get("reason_span"), text) if entry.get("reason_span") else None
            if entry.get("reason_span") and reason_span is None:
                issues.append("SPAN_NOT_IN_UTTERANCE:switch_reason")
            evidence.switches.append(SwitchEvidence(
                target=target, span=span, source=source, reason_span=reason_span,
            ))

        for entry in entries("claims"):
            span = span_of(entry, "claim")
            kind = str(entry.get("kind") or "").lower()
            if span is None:
                continue
            if kind not in CLAIM_KINDS:
                issues.append(f"INVALID_VOCABULARY:claim:{kind}")
                continue
            # Claims bind through the same rule as every other evidence type:
            # an option neither named nor an unambiguous public context
            # referent stays unbound rather than guessed (the live validator
            # otherwise mis-binds pronoun claims to arbitrary options).
            option_id, _res = bound(entry, "claim")
            sources = [str(s) for s in entry.get("sources") or [] if str(s).strip()]
            evidence.claims.append(GroundingClaim(
                span=span, kind=kind, option_id=option_id,
                attribute=str(entry.get("attribute")).strip() if entry.get("attribute") else None,
                value=str(entry.get("value")).strip() if entry.get("value") else None,
                source_facts=sources,
            ))

        for raw_span in data.get("ambiguous_references") or []:
            span = self._locate(raw_span, text)
            if span is not None:
                evidence.ambiguous_references.append(span)

        # Context-resolved bindings become explicit-marked mentions of their own.
        for option_id in sorted(context_bound):
            evidence.mentions.append(OptionMention(
                option_id=option_id,
                span=EvidenceSpan(text="", start=-1),
                order=len(evidence.mentions),
                alias_form="",
                resolution="context",
            ))

        evidence.thread_relevant = _loose_bool(data.get("thread_relevant"))
        # primary_act is never taken from the validator: it is derived from
        # the verified evidence after the deterministic merge (item 7).
        return evidence, issues
