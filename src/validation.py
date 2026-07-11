"""Turn-text validation, grounding, and fallback text — side-effect free.

ValidationMixin owns every check that decides whether a generated line may reach
the transcript — structural/commitment/blocker/switch validation, the limited
thread-aware realization checks, the grounding tripwire and LLM fact-judge —
plus the deterministic restate-first fallback used when a blocking issue
survives repair. It returns structured issues, never mutates dialogue state,
and never resolves threads; the observer decides what a turn realized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import prompts
from aliases import short_alias_map
from config_loader import cfg
from models import (
    ActType,
    DialogueAct,
    DialogueState,
    MoveIntent,
    Persona,
    _DECISION_ACTS,
)
from parsing import hybrid_blend_detected, switch_bridge_ok, visible_commitment
from utils import jaccard_text


@dataclass(slots=True)
class ValidationReport:
    issues: list[str]
    block_state_mutation: bool = False


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
    _world_text: str | None = None
    _world_state_id: int | None = None
    _option_tokens: dict[str, set[str]] = {}

    def _validate_turn_text(self, text: str, state: DialogueState, persona: Persona, intent: MoveIntent, act: DialogueAct) -> ValidationReport:
        issues: list[str] = []
        block = False
        if not text.strip():
            issues.append("EMPTY")
            block = True
        if "\n" in text.strip():
            issues.append("MULTI_TURN_OUTPUT")
        if re.search(r"\[\s*(?:act|opt|stance)\s*=", text, re.I):
            issues.append("LEAKED_METADATA")
        # Malformed compressed turn (P7): a bare marker head, or a lone
        # subordinate clause outside answer acts. One repair, then fallback.
        stripped = text.strip()
        if _BARE_MARKER_TURN.match(stripped) or (
            intent.act != ActType.ANSWER and _SUBORDINATE_ONLY_TURN.match(stripped)
        ):
            issues.append("MALFORMED_UTTERANCE")
            block = True
        if self._resolver and self._resolver.invalid_option_refs(text):
            issues.append("INVALID_OPTION_REFERENCE")
            block = True
        # Coverage turns exist to bring one ignored option into the room; the
        # route itself (not a magic reason string, item 10) requires the line
        # to actually name that option.
        if (
            intent.option_focus
            and intent.route_source == "coverage"
            and intent.option_focus[0] not in act.option_refs
        ):
            issues.append("MISSING_REQUIRED_OPTION_FOCUS")
            block = True
        if intent.act in _DECISION_ACTS and not (act.explicit_vote or act.accepts):
            issues.append("UNCLEAR_VISIBLE_COMMITMENT")
            block = True
        if intent.required_vote and intent.act in _DECISION_ACTS:
            committed_to_required = (
                act.explicit_vote == intent.required_vote
                or intent.required_vote in act.accepts
            )
            if not committed_to_required:
                issues.append("REQUIRED_VOTE_MISMATCH")
                block = True
        rejected = state.runtimes[persona.id].rejected_options()
        if any(oid in rejected for oid in ([act.explicit_vote] if act.explicit_vote else []) + list(act.accepts)):
            issues.append("HARD_BLOCKER_ACCEPTED_REJECTED_OPTION")
            block = True
        # A compromise must pin ONE existing option (P6): coordinating two
        # options as a single plan creates an implicit new hybrid option.
        if (
            self._resolver
            and (intent.act == ActType.COMPROMISE or act.offers_compromise)
            and hybrid_blend_detected(text, self._resolver)
        ):
            issues.append("HYBRID_COMPROMISE")
            block = True
        # A visible, unresolved active blocker (I3) binds like a setup rejection:
        # committing to that option needs a resolution in the same line.
        rt = state.runtimes[persona.id]
        committed = set(act.accepts) | ({act.explicit_vote} if act.explicit_vote else set())
        for option_id in committed:
            if option_id in rt.rejected_options() and act.resolves_blocker != option_id:
                issues.append("BLOCKED_OPTION_ACCEPTED")
                block = True
        # A continuation must genuinely add something (issue 6): a near-repeat of
        # the sim's own previous line, or re-asking the same person a question,
        # is exactly the accidental-duplicate failure this feature must prevent.
        if intent.continuation:
            previous = state.runtimes[persona.id].already_said
            prev_text = previous[-1] if previous else ""
            if prev_text and jaccard_text(text, prev_text) >= 0.5:
                issues.append("CONTINUATION_REPEATS")
                block = True
            last_turns = [t for t in state.turns if t.speaker_id == persona.id]
            if (
                last_turns
                and act.question_target_id
                and act.question_target_id == last_turns[-1].act.question_target_id
            ):
                issues.append("CONTINUATION_REPEATS")
                block = True
            # A continuation must stay on its own previous focus (P3): naming
            # only options disjoint from the just-made point is a topic jump,
            # not an addendum.
            if (
                intent.option_focus
                and act.option_refs
                and not set(act.option_refs) & set(intent.option_focus)
            ):
                issues.append("CONTINUATION_TOPIC_JUMP")
                block = True
        # Limited thread-aware checks (Section 15): validation stays side-effect
        # free and never touches threads — it only reports whether the routed
        # thread move was visibly realized, so repair gets one chance before
        # the observer refuses to count the turn.
        if intent.route_source == "answer_required" and text.strip():
            focus_missing = bool(
                intent.option_focus
                and not set(intent.option_focus) & set(act.option_refs)
            )
            only_counter_question = bool(
                act.question_scope and text.strip().endswith("?") and not act.option_refs
            )
            if only_counter_question or (focus_missing and act.question_scope):
                # Blocking: an evasive line must not reach the transcript as a
                # routed answer, or it would falsely resolve the question thread.
                issues.append("ANSWER_DOES_NOT_ADDRESS_QUESTION")
                block = True
        # Local act-realization alignment (todo_prompt item 6): on thread and
        # narrowing routes the routed direction matters for thread progress, so
        # a line whose parsed realization is a generic comment (for SUPPORT) or
        # shows no objection at all (for CONCERN) gets one focused repair.
        # Non-blocking: telemetry + one repair chance, never a fallback.
        if intent.route_source in ("thread_hot", "thread_cooling", "participant_narrowing") and text.strip():
            if intent.act == ActType.SUPPORT and act.act_type == ActType.COMMENT:
                issues.append("SUPPORT_NOT_REALIZED")
            if intent.act == ActType.CONCERN and act.act_type in {ActType.COMMENT, ActType.SUPPORT}:
                issues.append("CONCERN_NOT_REALIZED")
        # An implicit reply inside a thread is fine — the exchange makes the
        # target unambiguous (item 7/10). Only a response that names OTHER
        # options while skipping the thread's focus is talking past the issue.
        if (
            intent.route_source in ("thread_hot", "thread_cooling")
            and intent.act in {ActType.SUPPORT, ActType.ANSWER, ActType.CONCERN, ActType.COMMENT}
            and intent.option_focus
            and act.option_refs
            and not set(intent.option_focus) & set(act.option_refs)
        ):
            issues.append("THREAD_RESPONSE_MISSES_OPTION")
        if (
            intent.act == ActType.COMPARE
            and intent.route_source in ("thread_hot", "thread_cooling")
            and len(intent.option_focus) >= 2
            and sum(1 for oid in intent.option_focus[:2] if oid in act.option_refs) < 2
        ):
            issues.append("COMPARISON_MISSES_OPTIONS")
        # A sanctioned switch may only land on the offered option or the sim's
        # own current/initial preference (restate); never a third option.
        if intent.allow_vote_change and act.explicit_vote and intent.option_focus:
            allowed = set(intent.option_focus) | {rt.top_option(), persona.preferred_option}
            if intent.required_vote:
                allowed.add(intent.required_vote)
            if act.explicit_vote not in allowed:
                issues.append("OFF_TARGET_SWITCH")
                block = True
        # A visible commitment that lands on an option other than the sim's
        # current internal lean is a preference switch; it must bridge the old
        # stance to the new pick with a stated reason (issue 5), or the
        # transcript shows a socially unexplained flip. Blocking: if the LLM
        # cannot produce the bridge, the deterministic fallback restates the
        # current lean rather than fabricating an unexplained switch.
        # Sanctioned vote/switch turns carry the controller's visible previous
        # stance in intent.old_preference. Use that as the bridge source instead
        # of a mutable runtime preference field. Otherwise a valid sentence
        # like "I preferred Ninja, but I vote for Moccamaster because ..." can be
        # falsely checked against a newer latent preference and rejected as
        # UNBRIDGED_SWITCH. This was the root cause of many repair/fallback loops.
        bridge_from = intent.old_preference or rt.top_option() or persona.preferred_option
        if (
            act.explicit_vote
            # A soft acceptance ("X works for me too") is not a switch: it makes
            # the option acceptable without moving the lean, and people accept
            # without bridging (todo_prompt item 5). Only a direct vote away
            # from the current lean needs the visible bridge.
            and act.explicit_vote not in act.accepts
            and bridge_from in state.scenario.option_ids
            and act.explicit_vote != bridge_from
            and not switch_bridge_ok(text, bridge_from, self._resolver)
        ):
            issues.append("UNBRIDGED_SWITCH")
            block = True
        return ValidationReport(list(dict.fromkeys(issues)), block)

    def _collect_report(
        self,
        text: str,
        state: DialogueState,
        persona: Persona,
        intent: MoveIntent,
        act: DialogueAct,
        focus_options: list,
    ) -> tuple[ValidationReport, int, int]:
        """Regex validation plus an optional LLM grounding check; returns extra tokens."""
        report = self._validate_turn_text(text, state, persona, intent, act)
        deterministic_issue = self._deterministic_grounding_issue(text, state)
        if deterministic_issue and deterministic_issue not in report.issues:
            report.issues.append(deterministic_issue)
            # Asserted logistical workarounds are not just telemetry: if repair
            # cannot remove them, the fallback must replace the line before it
            # reaches the transcript. Explicitly uncertain mitigations are
            # allowed by _deterministic_grounding_issue and never get here.
            report.block_state_mutation = True
            return report, 0, 0
        # Do not spend an LLM grounding call on text that already failed a
        # deterministic structural/commitment check. In the previous version, a
        # malformed vote could trigger: utterance -> grounding judge -> repair ->
        # grounding judge -> fallback. That raised token cost without improving
        # the turn. First make the turn parse as the intended act; only then ask
        # the grounding judge about factual support.
        if report.block_state_mutation:
            return report, 0, 0
        issue, gti, gto = self._grounding_issue(text, state, intent, act, focus_options)
        if issue and issue not in report.issues:
            report.issues.append(issue)
        if issue == "UNSUPPORTED_FACT":
            # Grounding is a source-of-truth constraint, not just telemetry. The
            # repair path still gets one chance, but unsupported facts that survive
            # repair are replaced by fallback before reaching the transcript.
            report.block_state_mutation = True
        return report, gti, gto

    def _grounding_issue(
        self,
        text: str,
        state: DialogueState,
        intent: MoveIntent,
        act: DialogueAct,
        focus_options: list,
    ) -> tuple[str | None, int, int]:
        if not bool(cfg.validation.get("enabled", True)) or not bool(cfg.validation.get("grounding_check", False)):
            return None, 0, 0
        allowed = set(cfg.validation.get("grounding_acts", []))
        if allowed and intent.act.value not in allowed:
            return None, 0, 0
        if not text.strip():
            return None, 0, 0
        # Tripwire mode (default): only pay for the LLM judge when the line
        # contains a suspicious concrete claim — a number or a policy/medical/
        # weather-style term that does not occur in the option cards or shared
        # context (issue I11).
        if str(cfg.validation.get("grounding_mode", "tripwire")) == "tripwire" and not self._grounding_tripwire(text, state):
            return None, 0, 0
        # Candidate-specific turns get a smaller fact base, but it must include
        # the options actually mentioned in the generated text. Otherwise a
        # legitimate comparison can be judged against only the routed candidate
        # and create false unsupported-fact repairs.
        option_by_id = {option.id: option for option in state.scenario.options}
        judge_ids: list[str] = []
        for option in list(focus_options):
            option_id = getattr(option, "id", None)
            if option_id in option_by_id and option_id not in judge_ids:
                judge_ids.append(option_id)
        for option_id in act.option_refs:
            if option_id in option_by_id and option_id not in judge_ids:
                judge_ids.append(option_id)
        judge_options = [option_by_id[oid] for oid in judge_ids] if 0 < len(judge_ids) <= 3 else list(state.scenario.options)
        prompt = prompts.grounding_check(utterance=text, state=state, focus_options=judge_options)
        try:
            data = self._llm.generate_json(prompt, profile="repair")
        except Exception:
            # A flaky judge must never block generation; treat as grounded.
            return None, self._llm.last_tokens_in, self._llm.last_tokens_out
        unsupported = bool(data.get("unsupported")) if isinstance(data, dict) else False
        return ("UNSUPPORTED_FACT" if unsupported else None), self._llm.last_tokens_in, self._llm.last_tokens_out

    def _safe_fallback_text(self, state: DialogueState, persona: Persona, intent: MoveIntent, report: ValidationReport) -> str:
        """Deterministic replacement for LLM text that kept blocking issues after repair.

        The wording is chosen so the conservative parser reads it exactly as
        intended: decision turns yield one unambiguous commitment to an allowed
        option, blocker turns never accept the rejected option, and discussion
        turns stay commitment-free. Phrasings avoid every hedge/conditional/
        rejection pattern in parsing.py.
        """
        aliases = short_alias_map(state.scenario.options)
        rt = state.runtimes[persona.id]
        if intent.continuation:
            # A failed continuation add-on gets a neutral closer: no option
            # reference, no commitment vocabulary, nothing the parser reads.
            return "Anyway, that's my two cents for now."
        blocked = next(iter(rt.rejected_options()), None)
        # Required decision targets are controller-selected and validation-safe;
        # prefer them on decision turns. Otherwise restate the sim's own current
        # stance. Never fabricate acceptance of a hard-blocked option.
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
        if intent.act in _DECISION_ACTS:
            # Deterministic last-resort lines only (todo_prompt item 7): a small
            # pool, rotated so several fallback voters in one round do not sound
            # identical. Labels match parsing._PHRASE_FAMILIES so avoid_phrases
            # rotation works; every template parses as a direct vote.
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
            target_name = aliases[target]
            line = template.format(o=target_name)
            if blocked and "HARD_BLOCKER_ACCEPTED_REJECTED_OPTION" in report.issues:
                return f"I can't get behind {aliases[blocked]}, so I vote for {target_name}."
            current = intent.old_preference or rt.top_option() or persona.preferred_option
            if current in state.scenario.option_ids and current != target:
                old_name = aliases[current]
                reason = intent.allowed_reason or f"{target_name} has the clearest visible support now"
                switch_templates = [
                    "I vote for {target}; I was on {old}, but {reason}.",
                    "{target} gets my vote now; I preferred {old}, but {reason}.",
                    "I'm going with {target}; {old} was my earlier pick, but {reason}.",
                    "I can live with {target}; {old} was my first pick, but {reason}.",
                ]
                idx = (state.turn_index + len(persona.id) + len(report.issues)) % len(switch_templates)
                line = switch_templates[idx].format(old=old_name, target=target_name, reason=reason.rstrip('.'))
            # Self-check (todo_prompt item 10): a fallback exists so the decision
            # turn ALWAYS parses. A stored reason can still smuggle in wording
            # that voids the commitment (a second commit phrase, a question, a
            # third option); when the composed line does not parse to the
            # target, emit the minimal guaranteed-parseable form instead.
            if self._resolver is not None:
                commit = visible_commitment(
                    line, self._resolver, sanctioned_switch=bool(intent.allow_vote_change)
                )
                if commit is None or commit[1] != target:
                    if current in state.scenario.option_ids and current != target:
                        line = f"I vote for {target_name}; I was on {aliases[current]}, but I can go with the group here."
                    else:
                        line = f"I vote for {target_name}."
            return line
        if "MISSING_REQUIRED_OPTION_FOCUS" in report.issues and intent.option_focus:
            gap = intent.option_focus[0]
            other = target if target != gap else next((o for o in state.scenario.option_ids if o != gap), None)
            if other:
                return f"One option we haven't really talked about: {aliases[gap]}. How does it stack up against {aliases[other]}?"
            return f"One option we haven't really talked about: {aliases[gap]}. Worth a quick look before we decide."
        return f"I'm sticking with {aliases[target]} on this one."

    _SUSPECT_CLAIM = re.compile(
        r"\b(?:polic(?:y|ies)|includ(?:es|ed|ing)|refund\w*|warrant(?:y|ies)|discount\w*|"
        r"free\s+(?:of|shipping|entry|parking|wifi|drinks?)|allerg\w*|toxic\w*|poison\w*|"
        r"forecast\w*|guarantee[ds]?|certified|award[- ]?winn\w*|complimentary|licens\w*|"
        # Experiential/operational domains that invented facts favor (issue 7):
        # claims about parking, connectivity, weather, crowding, traffic, or
        # staffing that no card states get judged.
        r"parking|wi-?fi|weather|rain\w*|snow\w*|crowd\w*|queue\w*|traffic|"
        r"staff\w*|waiter\w*|servic\w*|jet\s*lag|peak\s+(?:hours?|times?)|rush\s+hour|"
        r"shelter\w*|shade|corner|quiet(?:er)?\s+(?:spot|corner|table|area)|"
        r"route|indoor|outdoor|seating|host)\b",
        re.I,
    )

    _UNCERTAINTY = re.compile(
        r"\b(?:not\s+clear|isn'?t\s+clear|unclear|unknown|not\s+listed|not\s+stated|"
        r"not\s+on\s+the\s+board|the\s+board\s+(?:doesn'?t|does\s+not)\s+say|"
        r"check|ask|see\s+if|not\s+sure|don't\s+know|do\s+not\s+know|"
        r"no\s+guarantee|can't\s+assume|cannot\s+assume|we\s+don't\s+know|we\s+do\s+not\s+know)\b",
        re.I,
    )

    _ASSERTED_WORKAROUND = re.compile(
        r"\b(?:we|you|they|i)\s+(?:can|will|should|just|could|might)\s+(?:just\s+|simply\s+|always\s+)?"
        r"(?:pick|get|find|book|reserve|hold|request|ask\s+for|choose|use|take|plan|add|combine)\b[^.!?]{0,80}"
        r"\b(?:quiet(?:er)?|corner|spot|table|shelter\w*|shade|parking|route|booking|reservation|"
        r"indoor|outdoor|seating|discount|weather|forecast|host|hike|kayak|activity|trail)\b",
        re.I,
    )

    # P8: a quantity whose unit class exists nowhere in the world facts is an
    # invented measurement ("height range is 25-51 inches" on a board that
    # lists no lengths at all). Common units (money, minutes, hours) are left
    # to the LLM judge because sims legitimately do arithmetic with them.
    _UNIT_NUMBER = re.compile(
        r"\b\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?[\s-]*"
        r"(inches|inch|cm|centimeters?|meters?|metres?|feet|foot|km|kilometers?|miles?|"
        r"kg|kilograms?|grams?|lbs?|pounds?|liters?|litres?|ml|gb|tb|mb|mah|ghz|mhz|"
        r"khz|hz|kbps|mbps|bits?|mph|kmh|mpg|db|decibels?|acres?|watts?|volts?|floors?|stories|degrees?|seats?|rooms?|"
        r"sessions?|workshops?|stations?|participants?|attendees?)\b",
        re.I,
    )

    def _deterministic_grounding_issue(self, text: str, state: DialogueState) -> str | None:
        """Cheaply catch unsupported logistical fixes before paying a judge call.

        The pattern only catches asserted workarounds ("we can pick a quieter
        corner"). Uncertain formulations ("maybe", "check if", "we don't know")
        remain allowed and may still be judged by the LLM tripwire if concrete.
        """
        world = getattr(self, "_world_text", None)
        if world is None or self._world_state_id != id(state):
            world = " ".join(
                [option.prompt_card() for option in state.scenario.options] + list(state.scenario.shared_context)
            ).lower()
            self._world_text = world
            self._world_state_id = id(state)
            self._option_tokens = self._distinctive_option_tokens(state)
        # Invented measurement class (P8): number + unit where neither the
        # number nor the unit word occurs in any card or shared-context fact.
        # Not excused by uncertainty wording elsewhere in the message — the
        # asserted quantity itself is still invented.
        for m in self._UNIT_NUMBER.finditer(text):
            unit = m.group(1).lower().rstrip("s")
            number = re.match(r"\d+(?:[.,]\d+)?", m.group(0)).group(0)
            if (
                unit not in world
                and unit + "s" not in world
                and not re.search(rf"\b{re.escape(number)}\b", world)
            ):
                return "UNSUPPORTED_FACT"
        match = self._ASSERTED_WORKAROUND.search(text)
        if not match:
            return None
        phrase = match.group(0).lower()
        # A workaround is allowed only when the text explicitly marks the
        # mitigation as unknown/check-needed. A bare "we could just..." is still
        # an unsupported asserted fix in this simulator.
        if phrase in world or self._UNCERTAINTY.search(text):
            return None
        return "UNSUPPORTED_FACT"

    def _grounding_tripwire(self, text: str, state: DialogueState) -> bool:
        """True when the line makes a concrete claim not present in the world facts,
        or reuses one option's distinctive card facts while talking about another
        option (cross-option fact transfer, I16)."""
        world = getattr(self, "_world_text", None)
        if world is None or self._world_state_id != id(state):
            world = " ".join(
                [option.prompt_card() for option in state.scenario.options] + list(state.scenario.shared_context)
            ).lower()
            self._world_text = world
            self._world_state_id = id(state)
            self._option_tokens = self._distinctive_option_tokens(state)
        for number in re.findall(r"\d+(?:[.,:]\d+)?", text):
            if number not in world:
                return True
        uncertain = bool(self._UNCERTAINTY.search(text))
        asserted_workaround = bool(self._ASSERTED_WORKAROUND.search(text))
        question_like = "?" in text and not asserted_workaround
        for match in self._SUSPECT_CLAIM.finditer(text):
            if match.group(0).lower() not in world:
                # "Parking details are unclear" / genuine questions about
                # unknown logistics are valid uncertainty statements, not
                # unsupported assertions. Asserted fixes remain caught by the
                # deterministic pass before the LLM judge is called.
                if (uncertain or question_like) and not asserted_workaround:
                    continue
                return True
        # Unsupported comparative/detail risks such as "clearly labels", "faster",
        # or "less travel time" should be judged even when generic words like
        # "travel" appear somewhere in the scenario. The judge can allow direct
        # implications from listed distance/time/cost facts.
        if self._COMPARATIVE.search(text) and re.search(
            r"\b(?:label\w*|fast\w*|quick\w*|speed|travel|distance|closer|farther|nearer|commute|cheaper|pricier|expensive)\b",
            text,
            re.I,
        ):
            return True
        if re.search(r"\bclearly\s+label\w*\b", text, re.I):
            return True
        # Cross-option transfer: tokens unique to one card showing up in a line
        # that names a different option (or that compares several cards' facts)
        # are judged — the claim exists in the world but may sit on the wrong
        # option or compare unlike quantities.
        text_tokens = set(re.findall(r"[a-z0-9]{4,}", text.lower()))
        hits = {oid for oid, tokens in self._option_tokens.items() if tokens & text_tokens}
        if len(hits) >= 2 and self._COMPARATIVE.search(text):
            return True
        resolver = getattr(self, "_resolver", None)
        mentioned = set(resolver.ids_in_text(text)) if resolver else set()
        return bool(hits and mentioned and hits - mentioned)

    _COMPARATIVE = re.compile(
        r"\b(?:than|versus|vs\.?|compared?|beats?|bigger|smaller|cheaper|pricier|faster|"
        r"slower|closer|farther|higher|lower|longer|shorter|more|less|fewer)\b",
        re.I,
    )

    @staticmethod
    def _distinctive_option_tokens(state: DialogueState) -> dict[str, set[str]]:
        """Per option: content tokens that appear on no other card and not in
        shared context. Aliases/names are excluded — naming an option is a
        mention, not a fact claim."""
        raw = {
            option.id: set(re.findall(r"[a-z0-9]{4,}", option.prompt_card().lower()))
            for option in state.scenario.options
        }
        shared = set(re.findall(r"[a-z0-9]{4,}", " ".join(state.scenario.shared_context).lower()))
        generic_fact_tokens = {
            "cost", "price", "minutes", "minute", "hours", "hour", "people", "person",
            "group", "option", "upside", "concern", "best", "short",
            "works", "work", "better", "flexible", "flexibility", "quality", "simple",
            "standard", "original", "select", "classic", "premium", "basic", "plus",
            "route", "parking", "booking", "reservation", "weather", "indoor", "outdoor",
        }
        name_tokens = {
            token
            for option in state.scenario.options
            for token in re.findall(r"[a-z0-9]{4,}", f"{option.name} {option.short_name}".lower())
        }
        distinctive: dict[str, set[str]] = {}
        for oid, tokens in raw.items():
            others = set().union(*(raw[o] for o in raw if o != oid)) if len(raw) > 1 else set()
            distinctive[oid] = tokens - others - shared - name_tokens - generic_fact_tokens
        return distinctive
