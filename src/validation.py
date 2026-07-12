"""Candidate assessment and act-specific fallback — side-effect free.

ValidationMixin owns the ONE evidence-based assessment path that decides what
happens to a candidate utterance (ACCEPT / ACCEPT_WITH_METRIC / REPAIR /
FALLBACK / DROP): universal intended-move realization, deterministic
structural/commitment/blocker/switch safety, and claim-level grounding
outcomes. It also builds the narrow act-specific deterministic fallback.
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
from parsing import hybrid_blend_detected, switch_bridge_ok, visible_commitment
from utils import jaccard_text, usable_reason_fragment


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
        """One explicit action for a candidate utterance, decided from the
        validated visible evidence — never from raw issue counts.

        Every intended semantic move gets the same realization check on every
        route. Blocking issues must not print; non-blocking issues earn one
        repair and may print if repair cannot improve them.
        """
        issues: list[ValidationIssue] = []

        def add(code: str, explanation: str = "", *, span: str = "",
                option: str | None = None, blocking: bool = False) -> None:
            if not any(i.code == code and i.option_id == option for i in issues):
                issues.append(ValidationIssue(code, explanation, span, option, blocking))

        if operational_failure or evidence is None:
            # Fail closed: the utterance may be fine, but nothing validated it.
            add("VALIDATOR_UNAVAILABLE",
                "structured interpretation unavailable; failing closed", blocking=True)
            return TurnAssessment(
                action=AssessmentAction.FALLBACK, issues=issues,
                notes="operational validator failure",
            )

        stripped = text.strip()
        if not stripped:
            add("EMPTY", blocking=True)
        for flag in structure_flags or []:
            if flag in {"MULTI_TURN_OUTPUT", "LEAKED_METADATA", "MALFORMED_ENVELOPE"}:
                add(flag)
        if _BARE_MARKER_TURN.match(stripped) or (
            intent.act != ActType.ANSWER and _SUBORDINATE_ONLY_TURN.match(stripped)
        ):
            add("MALFORMED_UTTERANCE", "lead-in or lone subordinate clause", blocking=True)
        if self._resolver and self._resolver.invalid_option_refs(text):
            add("INVALID_OPTION_REFERENCE", blocking=True)

        mentioned = [m.option_id for m in evidence.mentions]
        commitment = evidence.sole_commitment()
        if evidence.commitments and commitment is None:
            add("CONFLICTING_COMMITMENT", "several different commitment targets", blocking=True)

        if (
            intent.option_focus
            and intent.route_source == "coverage"
            and intent.option_focus[0] not in mentioned
        ):
            add("MISSING_REQUIRED_OPTION_FOCUS", option=intent.option_focus[0], blocking=True)

        # Claim-level grounding outcomes: every unsupported claim retains its
        # exact span, subject option, and reason — never reduced to one code.
        for claim in evidence.claims:
            if claim.supported is False:
                add(
                    f"UNSUPPORTED_CLAIM:{claim.kind}", claim.reason,
                    span=claim.span.text, option=claim.option_id, blocking=True,
                )

        focus = [o for o in intent.option_focus if o in state.scenario.option_ids]
        realized = self._intended_move_realized(intent, evidence, commitment, focus, add)

        rt = state.runtimes[persona.id]
        resolved_blockers = {b.option_id for b in evidence.blockers if b.action == "resolved"}
        for entry in evidence.commitments:
            if entry.option_id in rt.rejected_options() and entry.option_id not in resolved_blockers:
                add("BLOCKED_OPTION_ACCEPTED", option=entry.option_id, blocking=True)

        if (intent.act == ActType.COMPROMISE or evidence.proposals) and self._resolver and hybrid_blend_detected(text, self._resolver):
            add("HYBRID_COMPROMISE", "coordinates two options into one plan", blocking=True)

        if intent.allow_vote_change and commitment and intent.option_focus:
            allowed = set(intent.option_focus) | {rt.top_option(), persona.preferred_option}
            if intent.required_vote:
                allowed.add(intent.required_vote)
            if commitment.option_id not in allowed:
                add("OFF_TARGET_SWITCH", option=commitment.option_id, blocking=True)

        bridge_from = intent.old_preference or rt.top_option() or persona.preferred_option
        if (
            commitment
            and commitment.kind == "vote"
            and bridge_from in state.scenario.option_ids
            and commitment.option_id != bridge_from
        ):
            bridged = any(
                s.target == commitment.option_id and (s.reason_span or s.source)
                for s in evidence.switches
            ) or (self._resolver and switch_bridge_ok(text, bridge_from, self._resolver))
            if not bridged:
                add("UNBRIDGED_SWITCH", f"switch away from {bridge_from} without a visible bridge",
                    blocking=True)

        if intent.continuation:
            previous = rt.already_said
            prev_text = previous[-1] if previous else ""
            if prev_text and jaccard_text(text, prev_text) >= 0.5:
                add("CONTINUATION_REPEATS", blocking=True)
            last_turns = [t for t in state.turns if t.speaker_id == persona.id]
            asked = next((q.addressee_id for q in evidence.questions if q.addressee_id), None)
            if last_turns and asked and asked == last_turns[-1].question_target():
                add("CONTINUATION_REPEATS", blocking=True)
            if intent.option_focus and mentioned and not set(mentioned) & set(intent.option_focus):
                add("CONTINUATION_TOPIC_JUMP", blocking=True)

        if (
            intent.route_source in ("thread_hot", "thread_cooling")
            and intent.act in {ActType.SUPPORT, ActType.ANSWER, ActType.CONCERN, ActType.COMMENT}
            and intent.option_focus
            and mentioned
            and not set(intent.option_focus) & set(mentioned)
        ):
            add("THREAD_RESPONSE_MISSES_OPTION", option=intent.option_focus[0])
        if evidence.thread_relevant is False:
            add("THREAD_RESPONSE_OFF_TOPIC")

        focus_realized = bool(set(focus) & set(mentioned)) if focus else None

        # Repair only genuine blocking failures (item 11): the candidate
        # cannot safely realize the required move. Non-blocking findings —
        # an unrealized soft function, a thread-relevance miss, a harmless
        # extra function, a label difference — are telemetry, never repairs.
        if any(issue.blocking for issue in issues):
            action = AssessmentAction.REPAIR
        elif issues or realized is False or (
            evidence.primary_act is not None and evidence.primary_act != intent.act
        ):
            action = AssessmentAction.ACCEPT_WITH_METRIC
        else:
            action = AssessmentAction.ACCEPT
        notes = ""
        if action is AssessmentAction.ACCEPT_WITH_METRIC:
            if evidence.primary_act is not None and evidence.primary_act != intent.act:
                notes = f"primary act {evidence.primary_act.value} differs from intended {intent.act.value}"
            elif realized is False:
                notes = f"intended {intent.act.value} not visibly realized; safe to print"
        if verification_issues:
            notes = (notes + "; " if notes else "") + \
                "validator proposals dropped: " + ", ".join(verification_issues[:4])
        return TurnAssessment(
            action=action,
            issues=issues,
            intended_act_realized=realized,
            intended_focus_realized=focus_realized,
            notes=notes,
        )

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
                add("SUPPORT_NOT_REALIZED")
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
                add("CONCERN_NOT_REALIZED")
            return False
        if act is ActType.COMPARE:
            need = set(focus[:2])
            hit = any(
                len(set(c.option_ids) & need) >= min(2, len(need)) if need else len(c.option_ids) >= 2
                for c in evidence.comparisons
            )
            if not hit:
                add("COMPARISON_MISSES_OPTIONS")
            return hit
        if act is ActType.ASK:
            hit = any(q.scope in ("direct", "group") for q in evidence.questions)
            if not hit:
                add("ASK_NOT_REALIZED")
            return hit
        if act is ActType.ANSWER:
            hit = any(a.addresses_target for a in evidence.answers)
            if not hit:
                add("ANSWER_DOES_NOT_ADDRESS_QUESTION",
                    blocking=(intent.route_source == "answer_required"))
            return hit
        if act is ActType.COMPROMISE:
            hit = bool(evidence.proposals) or (commitment is not None and commitment.kind == "accept")
            if not hit:
                add("COMPROMISE_NOT_REALIZED")
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
        - exact factual comparison from listed attributes;
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
        if "MISSING_REQUIRED_OPTION_FOCUS" in issue_codes and intent.option_focus:
            return self._coverage_question_fallback(state, intent, aliases), "coverage_question"
        focus = next((o for o in intent.option_focus if o in state.scenario.option_ids), None)
        if intent.act is ActType.CONCERN and focus is not None and focus in rt.rejected_options():
            reason = usable_reason_fragment(rt.reason_against(focus), aliases[focus])
            if reason:
                # Hard-blocker restatement from the already grounded reason.
                return f"{aliases[focus]} still doesn't work for me — {reason}.", "blocker_restate"
            return None, ""
        if intent.act is ActType.COMPARE:
            text = self._comparison_fallback_text(state, intent, aliases)
            return (text, "comparison") if text else (None, "")
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

    @staticmethod
    def _comparison_fallback_text(state: DialogueState, intent: MoveIntent, aliases: dict) -> str | None:
        """Factual comparison from exact listed attributes, or None."""
        ids = [o for o in intent.option_focus if o in state.scenario.option_ids][:2]
        if len(ids) < 2:
            return None
        first, second = state.scenario.option(ids[0]), state.scenario.option(ids[1])
        shared = [
            key for key in first.attrs
            if key in second.attrs and str(first.attrs[key]).strip() and str(second.attrs[key]).strip()
        ]
        if not shared:
            return None
        key = shared[0]
        label = key.replace("_", " ")
        return (
            f"On {label}, {aliases[ids[0]]} is listed at {first.attrs[key]} "
            f"versus {second.attrs[key]} for {aliases[ids[1]]}."
        )

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

