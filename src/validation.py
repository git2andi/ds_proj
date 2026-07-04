"""Turn-text validation, grounding, and fallback text (issue 8 extraction).

ValidationMixin owns every check that decides whether a generated line may reach
the transcript — structural/commitment/blocker/switch validation, the grounding
tripwire and LLM fact-judge — plus the deterministic restate-first fallback used
when a blocking issue survives repair. Mixed into DialogueRunner.
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
from parsing import switch_bridge_ok
from utils import jaccard_text


@dataclass(slots=True)
class ValidationReport:
    issues: list[str]
    block_state_mutation: bool = False


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
        if self._resolver and self._resolver.invalid_option_refs(text):
            issues.append("INVALID_OPTION_REFERENCE")
            block = True
        if (
            intent.option_focus
            and "not yet been socially processed" in intent.reason
            and intent.option_focus[0] not in act.option_refs
        ):
            issues.append("MISSING_REQUIRED_OPTION_FOCUS")
            block = True
        if intent.act in {ActType.VOTE, ActType.ACCEPT} and not (act.explicit_vote or act.accepts):
            issues.append("UNCLEAR_VISIBLE_COMMITMENT")
            block = True
        if persona.rejection and (act.explicit_vote == persona.rejection or persona.rejection in act.accepts):
            issues.append("HARD_BLOCKER_ACCEPTED_REJECTED_OPTION")
            block = True
        # A visible, unresolved active blocker (I3) binds like a setup rejection:
        # committing to that option needs a resolution in the same line.
        rt = state.runtimes[persona.id]
        committed = set(act.accepts) | ({act.explicit_vote} if act.explicit_vote else set())
        for option_id in committed:
            if option_id in rt.hard_rejections and act.resolves_blocker != option_id:
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
        # A sanctioned switch may only land on the offered option or the sim's
        # own current/initial preference (restate); never a third option.
        if intent.allow_vote_change and act.explicit_vote and intent.option_focus:
            allowed = set(intent.option_focus) | {rt.current_preference, persona.preferred_option}
            if act.explicit_vote not in allowed:
                issues.append("OFF_TARGET_SWITCH")
                block = True
        # A visible commitment that lands on an option other than the sim's
        # current internal lean is a preference switch; it must bridge the old
        # stance to the new pick with a stated reason (issue 5), or the
        # transcript shows a socially unexplained flip. Blocking: if the LLM
        # cannot produce the bridge, the deterministic fallback restates the
        # current lean rather than fabricating an unexplained switch.
        current = rt.current_preference or persona.preferred_option
        if (
            act.explicit_vote
            and current in state.scenario.option_ids
            and act.explicit_vote != current
            and not switch_bridge_ok(text, current, self._resolver)
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
        issue, gti, gto = self._grounding_issue(text, state, intent)
        if issue and issue not in report.issues:
            report.issues.append(issue)  # non-blocking; drives one repair toward grounded text
        return report, gti, gto

    def _grounding_issue(
        self,
        text: str,
        state: DialogueState,
        intent: MoveIntent,
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
        # Always judge against the full option board: comparisons legitimately
        # restate other options' card facts, so a focus-scoped fact base
        # produces false UNSUPPORTED_FACT flags (issue #18).
        prompt = prompts.grounding_check(utterance=text, state=state, focus_options=list(state.scenario.options))
        try:
            data = self._llm.generate_json(prompt, profile="repair")
        except Exception:
            # A flaky judge must never block generation; treat as grounded.
            return None, self._llm.last_tokens_in, self._llm.last_tokens_out
        unsupported = bool(data.get("unsupported")) if isinstance(data, dict) else False
        return ("UNSUPPORTED_FACT" if unsupported else None), self._llm.last_tokens_in, self._llm.last_tokens_out

    @staticmethod
    def _safe_fallback_text(state: DialogueState, persona: Persona, intent: MoveIntent, report: ValidationReport) -> str:
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
        blocked = persona.rejection
        # Restate-first: never fabricate consent. The sim's own current/initial
        # preference wins over the intent's offered options; runtime blockers
        # (parsed dealbreakers) disqualify a target just like the setup rejection.
        candidates = [rt.current_preference, persona.preferred_option, *intent.option_focus, *state.scenario.option_ids]
        target = next(
            (o for o in candidates if o in state.scenario.option_ids and o != blocked and o not in rt.hard_rejections),
            next(o for o in state.scenario.option_ids if o != blocked),
        )
        if intent.act in _DECISION_ACTS:
            # Labels match parsing._PHRASE_FAMILIES so avoid_phrases rotation
            # works; every template parses as a direct vote (I19: a wide pool
            # keeps seven fallback voters in one round from sounding identical).
            templates = [
                ("gets my vote", "{o} gets my vote."),
                ("I'd go with", "I'd go with {o}."),
                ("my pick is", "My pick is {o}."),
                ("I vote for", "I vote for {o}."),
                ("my vote is", "My vote goes to {o}."),
                ("I'm going with", "I'm going with {o}."),
                ("I'm sold on", "I'm sold on {o}."),
                ("count me in for", "Count me in for {o}."),
            ]
            label, template = next(
                ((l, t) for l, t in templates if l not in intent.avoid_phrases),
                templates[0],
            )
            line = template.format(o=aliases[target])
            if blocked and "HARD_BLOCKER_ACCEPTED_REJECTED_OPTION" in report.issues:
                tail = line if line.startswith("I") else line[0].lower() + line[1:]
                return f"I can't get behind {aliases[blocked]}, so {tail}"
            current = rt.current_preference or persona.preferred_option
            if current in state.scenario.option_ids and current != target:
                # The restate target was disqualified (or the intent demands a
                # switch), so the deterministic line is itself a switch — it must
                # carry the bridge or it would fail the same UNBRIDGED_SWITCH
                # check that sent us here.
                body = line.rstrip(".")
                if not body.startswith("I"):
                    body = body[0].lower() + body[1:]
                return f"I still like {aliases[current]}, but {body} — it works better for the group."
            return line
        if "MISSING_REQUIRED_OPTION_FOCUS" in report.issues and intent.option_focus:
            gap = intent.option_focus[0]
            other = target if target != gap else next((o for o in state.scenario.option_ids if o != gap), None)
            if other:
                return f"One option we haven't really talked about: {aliases[gap]}. How does it stack up against {aliases[other]}?"
            return f"One option we haven't really talked about: {aliases[gap]}. Worth a quick look before we decide."
        return f"I'm sticking with {aliases[target]} on this one."

    _SUSPECT_CLAIM = re.compile(
        r"\b(?:polic(?:y|ies)|includ(?:es|ed|ing)|refund\w*|warrant(?:y|ies)|reservation|discount\w*|"
        r"free\s+(?:of|shipping|entry|parking|wifi|drinks?)|allerg\w*|toxic\w*|poison\w*|"
        r"forecast\w*|guarantee[ds]?|certified|award[- ]?winn\w*|complimentary|licens\w*|"
        # Experiential/operational domains that invented facts favor (issue 7):
        # claims about parking, connectivity, weather, crowding, traffic, or
        # staffing that no card states get judged.
        r"parking|wi-?fi|weather|rain\w*|snow\w*|crowd\w*|queue\w*|traffic|"
        r"staff\w*|waiter\w*|servic\w*|jet\s*lag|peak\s+(?:hours?|times?)|rush\s+hour)\b",
        re.I,
    )

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
        for match in self._SUSPECT_CLAIM.finditer(text):
            if match.group(0).lower() not in world:
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
        name_tokens = {
            token
            for option in state.scenario.options
            for token in re.findall(r"[a-z0-9]{4,}", f"{option.name} {option.short_name}".lower())
        }
        distinctive: dict[str, set[str]] = {}
        for oid, tokens in raw.items():
            others = set().union(*(raw[o] for o in raw if o != oid)) if len(raw) > 1 else set()
            distinctive[oid] = tokens - others - shared - name_tokens
        return distinctive

    @staticmethod
    def _semantic_block(persona: Persona, intent: MoveIntent, act: DialogueAct) -> bool:
        if persona.rejection and (act.explicit_vote == persona.rejection or persona.rejection in act.accepts):
            return True
        if intent.act in {ActType.VOTE, ActType.ACCEPT} and not (act.explicit_vote or act.accepts):
            return True
        return False
