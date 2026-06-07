"""
simulator.py
------------
Simulator -- wraps a Persona and generates one dialogue turn via the LLM,
using the compact speaker-card prompt.

Generation flow (per turn):
  1. _generate_raw()    : build prompt + call LLM -> (raw_text, prompt)
  2. strip name prefix  : remove "Name: " if model echoed it
  3. _verify_and_repair : verify; repair once if needed; store result in
                          self._last_verification for the logger to read
  4. _enforce_word_budget : hard cap on word count
"""

from __future__ import annotations

import re
from typing import Any, Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from persona import Persona
from prompt_context import (
    build_group_state,
    build_local_context,
    build_memory_block,
    build_move_instruction,
    build_output_contract,
    build_relevant_options,
    build_speaker_card,
)
from utils import OptionResolver
from verifier import VerificationIssue, VerificationResult, verify_participant_turn

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from state import DialogueMemory


class Simulator:

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options
        self._resolver = OptionResolver(options)
        self._llm = get_llm_client()
        self._last_verification: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------

    def generate_turn(
        self,
        history: list[str],
        state: "DialogueState",
        all_names: Optional[list[str]] = None,
        structured: Optional["DialogueMemory"] = None,
    ) -> tuple[str, int, int]:
        del all_names  # reserved for future use
        raw, gen_prompt = self._generate_raw(history, state, structured)
        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out

        if not raw:
            self._last_verification = None
            return "[SILENCE]", tok_in, tok_out

        # Strip "Name: " prefix if the model echoed it
        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        # Verify and repair (stores result in self._last_verification)
        raw = self._verify_and_repair(raw, gen_prompt, history, state, structured)

        raw = self._enforce_word_budget(raw, state.phase)
        return raw or "[SILENCE]", tok_in, tok_out

    def generate_control_turn(
        self,
        history: list[str],
        state: "DialogueState",
        structured: Optional["DialogueMemory"],
        kind: str,
    ) -> tuple[str, str, Optional[str], int, int, Optional[dict[str, Any]]]:
        """Generate a phase-critical vote/confirmation with structured control.

        Normal discussion remains free-form. For decisions, the LLM returns a
        JSON action so the live outcome is not inferred from broad stance text.
        """
        prompt = self._build_control_prompt(history, state, structured, kind)
        data: dict[str, Any] = {}
        parse_error = ""
        try:
            data = self._llm.generate_json(prompt)
        except Exception as exc:
            parse_error = str(exc)

        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out
        fallback_reason = ""
        action, option, message = self._normalise_control_output(data, state, structured, kind)
        if not message or not action or not option:
            fallback_reason = "invalid_or_empty_control_json"
        else:
            fallback_reason = self._control_message_issue(message, action, option, kind)
        if fallback_reason:
            action, option, message = self._control_fallback(state, structured, kind)

        if message.lower().startswith(f"{self.name.lower()}:"):
            message = message.split(":", 1)[1].strip()
        message = self._enforce_word_budget(message, state.phase)

        verification = self._verify_control_message(message, state, structured)
        if verification and verification.get("needs_fallback"):
            fallback_reason = fallback_reason or "visible_message_failed_verification"
            action, option, message = self._control_fallback(state, structured, kind)
            message = self._enforce_word_budget(message, state.phase)
            verification.pop("needs_fallback", None)

        meta = verification or {"ok": True, "issues": [], "repair_attempted": False, "repair_succeeded": False}
        if parse_error:
            meta["repair_attempted"] = True
            meta["repair_succeeded"] = True
        meta["structured_control"] = {
            "kind": kind,
            "action": action,
            "option": option,
            "parse_error": parse_error,
        }
        if fallback_reason:
            meta["structured_control"]["fallback_reason"] = fallback_reason
        self._last_verification = meta
        return message or "[SILENCE]", action, option, tok_in, tok_out, meta

    # ------------------------------------------------------------------

    def _has_voted(self, history: list[str]) -> bool:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() != self.name:
                continue
            if self._resolver.vote_in(msg):
                return True
        return False

    def _generate_raw(
        self,
        history: list[str],
        state: "DialogueState",
        structured: Optional["DialogueMemory"],
    ) -> tuple[str, str]:
        """Build the full prompt, call the LLM, and return (raw_text, prompt).
        No verification or grounding check here -- that happens in generate_turn.
        """
        is_closure = state.phase == "closure"
        phase_instr = prompts.phase_instruction_text(
            phase=state.phase, has_voted=self._has_voted(history),
            final_option=getattr(state, "preferred_option", None) if is_closure else None,
        )

        n_recent = cfg.prompt_budget.recent_turns_short

        speaker_card = build_speaker_card(self.persona)
        candidate = state.candidate_option or state.current_leading_option
        relevant_opts = build_relevant_options(self.options, self.persona, candidate)
        group_state = "" if is_closure else build_group_state(state)
        local_ctx = build_local_context(history, n_recent=n_recent)
        memory_block = "" if is_closure else build_memory_block(self.name, structured)

        interaction_instr = "" if is_closure else self._interaction_instruction(history, state)
        position_disc = "" if is_closure else self._position_discipline(state)

        move_instr = build_move_instruction(
            phase_instruction=phase_instr,
            interaction_instruction=interaction_instr,
            position_discipline=position_disc,
        )
        output_contract = build_output_contract(
            max_words=self.persona.max_words(state.phase), name=self.name,
        )

        prompt = prompts.sim_turn_compact(
            speaker_card=speaker_card,
            relevant_options=relevant_opts,
            group_state=group_state,
            local_context=local_ctx,
            memory_block=memory_block,
            move_instruction=move_instr,
            output_contract=output_contract,
        )

        result = self._llm.generate(prompt).strip()
        return result, prompt

    def _build_control_prompt(
        self,
        history: list[str],
        state: "DialogueState",
        structured: Optional["DialogueMemory"],
        kind: str,
    ) -> str:
        n_recent = int(getattr(cfg.prompt_budget, "recent_turns_control", cfg.prompt_budget.recent_turns_short))
        speaker_card = build_speaker_card(self.persona)
        candidate = state.candidate_option or state.current_leading_option
        relevant_opts = build_relevant_options(self.options, self.persona, candidate)
        group_state = build_group_state(state)
        local_ctx = build_local_context(history, n_recent=n_recent)
        memory_block = build_memory_block(self.name, structured)
        max_words = self.persona.max_words(state.phase)

        if kind == "vote":
            return prompts.structured_vote_turn(
                speaker_card=speaker_card,
                relevant_options=relevant_opts,
                group_state=group_state,
                local_context=local_ctx,
                memory_block=memory_block,
                max_words=max_words,
            )

        beliefs = self.persona.beliefs
        preferred = beliefs.preferred if beliefs else (self._resolver.letters[0] if self._resolver.letters else "A")
        acceptable = list(beliefs.acceptable or []) if beliefs else []
        rejected = list(beliefs.rejected or []) if beliefs else []
        ps = structured.participants.get(self.name) if structured else None
        return prompts.structured_confirmation_turn(
            speaker_card=speaker_card,
            relevant_options=relevant_opts,
            group_state=group_state,
            local_context=local_ctx,
            memory_block=memory_block,
            candidate=candidate or preferred,
            preferred=preferred,
            acceptable=acceptable,
            rejected=rejected,
            is_firm_holdout=bool(getattr(ps, "is_true_hard_blocker", False)),
            max_words=max_words,
        )

    def _normalise_control_output(
        self,
        data: dict[str, Any],
        state: "DialogueState",
        structured: Optional["DialogueMemory"],
        kind: str,
    ) -> tuple[str, Optional[str], str]:
        sc = cfg.structured_control
        action = str(data.get(sc.action_key, "")).strip().lower()
        option_raw = data.get(sc.option_key)
        option = str(option_raw).strip().upper() if option_raw is not None else ""
        message = str(data.get(sc.message_key, "")).strip()
        valid_options = {l.upper() for l in self._resolver.letters}

        if option not in valid_options:
            return "", None, ""
        if kind == "vote" and action != sc.vote_action:
            return "", None, ""
        if kind == "confirmation":
            candidate = state.candidate_option or state.current_leading_option
            if candidate and option != candidate:
                return "", None, ""
            if action not in {sc.accept_action, sc.reject_action}:
                return "", None, ""
            if action == sc.accept_action and self._must_reject_candidate(option, structured):
                return "", None, ""
            if action == sc.reject_action and not self._must_reject_candidate(option, structured):
                return "", None, ""
        return action, option, message

    def _control_message_issue(
        self,
        message: str,
        action: str,
        option: str,
        kind: str,
    ) -> str:
        """Return a fallback reason if visible text disagrees with control JSON."""
        sc = cfg.structured_control
        if kind == "vote":
            visible_vote = self._vote_ref(message)
            if visible_vote is None:
                return "visible_vote_missing"
            if visible_vote != option:
                return "visible_vote_conflicts_with_json_option"
            return ""

        if kind != "confirmation":
            return ""

        signal = self._confirmation_signal(message)
        if signal == "mixed":
            return "visible_confirmation_mixed_yes_no"
        if action == sc.accept_action and signal == "reject":
            return "visible_confirmation_rejects_json_accept"
        if action == sc.reject_action and signal == "accept":
            return "visible_confirmation_accepts_json_reject"

        mentions = self._resolver.options_in(message)
        if mentions and option not in mentions:
            return "visible_confirmation_mentions_other_option_only"
        return ""

    def _confirmation_signal(self, message: str) -> str:
        """Small local yes/no classifier for structured confirmation text."""
        text = message.strip().lower()
        accept = bool(re.search(
            r"\b(?:yes|yeah|yep|sure|ok|okay|fine|agreed|works?\s+for\s+me|"
            r"works?\s+(?:well\s+enough|fine|okay|ok)|can\s+live\s+with|"
            r"could\s+live\s+with|acceptable|good\s+with|fine\s+with|"
            r"that\s+works|sounds?\s+good)\b",
            text,
            re.I,
        ))
        reject = bool(re.search(
            r"^\s*(?:no|nope|nah|not\s+really|still\s+not|"
            r"i\s+can'?t|i\s+won'?t|can'?t\s+live\s+with|not\s+sold|"
            r"doesn'?t\s+work|won'?t\s+work)\b",
            text,
            re.I,
        ))
        if accept and re.search(
            r"\b(?:not|doesn'?t|does\s+not|won'?t|cannot|can'?t)\s+"
            r"(?:work|works|acceptable|fine|okay|ok|good)\b",
            text,
            re.I,
        ):
            accept = False
        if accept and reject:
            return "mixed"
        if accept:
            return "accept"
        if reject:
            return "reject"
        return ""

    def _must_reject_candidate(
        self,
        candidate: str,
        structured: Optional["DialogueMemory"],
    ) -> bool:
        beliefs = self.persona.beliefs
        if not beliefs:
            return False
        ps = structured.participants.get(self.name) if structured else None
        firm = bool(getattr(ps, "is_true_hard_blocker", False))
        if candidate in (beliefs.rejected or []):
            return True
        if firm and candidate != beliefs.preferred:
            return True
        return candidate not in (beliefs.acceptable or [])

    def _control_fallback(
        self,
        state: "DialogueState",
        structured: Optional["DialogueMemory"],
        kind: str,
    ) -> tuple[str, str, str]:
        sc = cfg.structured_control
        beliefs = self.persona.beliefs
        valid_options = list(self._resolver.letters or ["A", "B", "C", "D"])

        if kind == "vote":
            rejected = self._rejected_options_for_self(state)
            preferred_order: list[str] = []
            if beliefs:
                preferred_order.extend([beliefs.preferred] + list(beliefs.acceptable or []))
            preferred_order.extend(valid_options)
            for option in preferred_order:
                if option and option in valid_options and option not in rejected:
                    return sc.vote_action, option, sc.fallback_vote_template.format(option=option)
            option = valid_options[0]
            return sc.vote_action, option, sc.fallback_vote_template.format(option=option)

        candidate = state.candidate_option or state.current_leading_option or valid_options[0]
        preferred = beliefs.preferred if beliefs else candidate
        if self._must_reject_candidate(candidate, structured):
            reason = (beliefs.reservation if beliefs and beliefs.reservation else sc.fallback_reject_generic_reason)
            return (
                sc.reject_action,
                candidate,
                sc.fallback_reject_template.format(candidate=candidate, reason=reason),
            )
        if candidate == preferred:
            msg = sc.fallback_accept_top_template.format(candidate=candidate)
        else:
            msg = sc.fallback_accept_template.format(preferred=preferred, candidate=candidate)
        return sc.accept_action, candidate, msg

    def _verify_control_message(
        self,
        text: str,
        state: "DialogueState",
        structured: Optional["DialogueMemory"],
    ) -> Optional[dict[str, Any]]:
        if not cfg.verification.enabled:
            return None
        ps = structured.participants.get(self.name) if structured else None
        candidate = state.candidate_option or state.current_leading_option
        result = verify_participant_turn(
            text=text,
            speaker_name=self.name,
            phase=state.phase,
            options=self.options,
            history=[],
            persona_state=ps,
            resolver=self._resolver,
            candidate=candidate,
        )
        extra_issue = self._local_consistency_issue(text, state)
        if extra_issue:
            result.issues.append(extra_issue)
            result.ok = False
            result.needs_repair = True
        if not result.needs_repair:
            return result.as_dict() if result.issues else None
        meta = result.as_dict()
        meta["repair_attempted"] = True
        meta["repair_succeeded"] = True
        meta["needs_fallback"] = True
        return meta

    # ------------------------------------------------------------------
    # Verification + repair
    # ------------------------------------------------------------------

    def _verify_and_repair(
        self,
        text: str,
        gen_prompt: str,
        history: list[str],
        state: "DialogueState",
        structured: Optional["DialogueMemory"],
    ) -> str:
        """Verify the generated text; repair once if needed.

        The logger should reflect the text that is actually emitted. Earlier
        versions logged the *original* failed verification even when a repair or
        deterministic fallback produced a valid final line. That made obvious
        fallback votes such as "I'd go with Option A" appear as
        MISSING_EXPLICIT_VOTE in .eval.json. This method now re-verifies the
        emitted text and stores final-result metadata, with original issue codes
        preserved separately for diagnostics.
        """
        if not cfg.verification.enabled:
            self._last_verification = None
            return text

        ps = structured.participants.get(self.name) if structured else None
        candidate = state.candidate_option or state.current_leading_option

        def _verify(candidate_text: str) -> VerificationResult:
            return verify_participant_turn(
                text=candidate_text,
                speaker_name=self.name,
                phase=state.phase,
                options=self.options,
                history=history,
                persona_state=ps,
                resolver=self._resolver,
                candidate=candidate,
            )

        result = _verify(text)
        extra_issue = self._local_consistency_issue(text, state)
        if extra_issue:
            result.issues.append(extra_issue)
            result.ok = False
            result.needs_repair = True
        if not result.needs_repair:
            self._last_verification = result.as_dict() if result.issues else None
            return text

        original_codes = [i.code for i in result.issues if i.severity == "repair"]

        # --- Attempt LLM repair -----------------------------------------
        repair_prompt = self._build_repair_prompt(text, gen_prompt, result, state)
        repaired = self._llm.generate(repair_prompt).strip()
        if repaired.lower().startswith(f"{self.name.lower()}:"):
            repaired = repaired.split(":", 1)[1].strip()

        if repaired:
            result2 = _verify(repaired)
            extra2 = self._local_consistency_issue(repaired, state)
            if extra2:
                result2.issues.append(extra2)
                result2.ok = False
                result2.needs_repair = True
            if not result2.needs_repair:
                final_meta = result2.as_dict()
                final_meta["repair_attempted"] = True
                final_meta["repair_succeeded"] = True
                final_meta["original_issues"] = original_codes
                self._last_verification = final_meta
                return repaired

        # --- Deterministic fallback for phase-critical failures ----------
        fallback = self._deterministic_fallback(state, result)
        if fallback:
            fallback_result = _verify(fallback)
            extra_fb = self._local_consistency_issue(fallback, state)
            if extra_fb:
                fallback_result.issues.append(extra_fb)
                fallback_result.ok = False
                fallback_result.needs_repair = True
            final_meta = fallback_result.as_dict()
            final_meta["repair_attempted"] = True
            final_meta["repair_succeeded"] = not fallback_result.needs_repair
            final_meta["original_issues"] = original_codes
            self._last_verification = final_meta
            return fallback

        # Accept original if no fallback is applicable; log the failed result.
        result.repair_attempted = True
        result.repair_succeeded = False
        self._last_verification = result.as_dict()
        return text

    def _build_repair_prompt(
        self,
        original_text: str,
        gen_prompt: str,
        result: "VerificationResult",
        state: "DialogueState",
    ) -> str:
        """Choose the right repair prompt based on the primary issue code.

        Priority order (high-impact / phase-critical first):
          phase obligations (vote, confirmation) -> validity (option refs/facts)
          -> naturalness (ack-loop, semantic repeat, self-repetition).
        """
        repair_codes = {i.code for i in result.issues if i.severity == "repair"}

        if "INCONSISTENT_VOTE_WITH_PRIOR_REJECTION" in repair_codes or "INCONSISTENT_VOTE_WITH_PERSONA" in repair_codes:
            rejected = self._rejected_options_for_self(state)
            return prompts.repair_inconsistent_vote(original_text, self.options, rejected)

        if "REPEATED_RULE_OUT" in repair_codes:
            return prompts.repair_repeated_rule_out(original_text)

        if "OPTION_ATTRIBUTE_MISMATCH" in repair_codes:
            return prompts.repair_attribute_mismatch(original_text, self.options)

        if "MISSING_EXPLICIT_VOTE" in repair_codes:
            return prompts.repair_vote(self.options)

        if "UNCLEAR_CONFIRMATION" in repair_codes or "WEAK_COMPROMISE_CONFIRMATION" in repair_codes:
            candidate = state.candidate_option or state.current_leading_option or "?"
            return prompts.repair_confirmation(candidate)

        if "VALID_OPTION_DENIED" in repair_codes or "INVALID_OPTION_REFERENCE" in repair_codes:
            return prompts.repair_invalid_option(original_text, self.options)

        if "INVENTED_OPTION_FACT" in repair_codes:
            return prompts.repair_invented_fact(gen_prompt)

        if "FACT_CHASING_QUESTION" in repair_codes:
            return prompts.repair_fact_chasing_question(original_text, self.options)

        if "QUESTION_CHAIN" in repair_codes:
            return prompts.repair_question_chain(original_text)

        # Group-level acknowledgement loop.
        if "ACK_LOOP" in repair_codes:
            return prompts.repair_ack_loop(original_text)

        # Semantic repeat (use the matched prior point in the
        # repair so the model knows which point to avoid).
        if "SEMANTIC_POINT_REPEAT" in repair_codes:
            prior = next(
                (i.message.split("'")[1] for i in result.issues
                 if i.code == "SEMANTIC_POINT_REPEAT" and "'" in i.message),
                "",
            )
            return prompts.repair_semantic_repeat(original_text, prior)

        if "SELF_REPETITION" in repair_codes:
            return prompts.repair_repetition(original_text)

        # Default: treat like repetition
        return prompts.repair_repetition(original_text)

    def _vote_ref(self, text: str) -> Optional[str]:
        vote = self._resolver.vote_in(text)
        if vote:
            return vote
        m = re.search(
            r"\b(?:i\s*(?:'d|would)?\s*(?:go\s+with|pick|choose|prefer|vote\s+for)|"
            r"i\s*(?:'m|am)\s*(?:leaning\s+(?:toward|towards)|going\s+with)|"
            r"my\s+(?:pick|choice|vote)\s+(?:is|would\s+be))\s+(?:Option\s+)?([A-D])\b",
            text,
            re.I,
        )
        return m.group(1).upper() if m else None

    def _rejected_options_for_self(self, state: "DialogueState") -> set[str]:
        rejects = getattr(state, "explicit_rejects", {}).get(self.name, {}) or {}
        return set(rejects.keys())

    def _changed_mind_marker(self, text: str) -> bool:
        return bool(re.search(
            r"\b(?:changed my mind|change my mind|reconsidered|after thinking|on second thought|actually|despite what I said|I know I ruled it out)\b",
            text,
            re.I,
        ))

    def _local_consistency_issue(self, text: str, state: "DialogueState") -> Optional[VerificationIssue]:
        """Checks that need live dialogue state rather than only the verifier inputs."""
        if state.phase == "narrowing":
            vote = self._vote_ref(text)
            beliefs = self.persona.beliefs
            if vote and vote in self._rejected_options_for_self(state) and not self._changed_mind_marker(text):
                return VerificationIssue(
                    code="INCONSISTENT_VOTE_WITH_PRIOR_REJECTION",
                    severity="repair",
                    message=f"Speaker previously ruled out Option {vote} but now votes for it without saying they changed their mind.",
                )
            if vote and beliefs:
                allowed = {beliefs.preferred, *(beliefs.acceptable or [])}
                if vote not in allowed and not self._changed_mind_marker(text):
                    return VerificationIssue(
                        code="INCONSISTENT_VOTE_WITH_PERSONA",
                        severity="repair",
                        message=f"Speaker voted for Option {vote}, which is neither preferred nor acceptable in their private belief state.",
                    )
        # Do not keep asking to rule out the same option once the discussion has
        # already locally rejected it. This was producing pruning-tree chats.
        m = re.search(r"\b(?:can we |let(?:'s| us) |we should |i(?:'d| would) )?rule out\s+(?:Option\s+)?([A-D])\b", text, re.I)
        if m and state.phase == "negotiation":
            opt = m.group(1).upper()
            # Look at raw dialogue history via state is unavailable here; use
            # explicit rejections already parsed by orchestrator.
            if any(opt in rejects for rejects in getattr(state, "explicit_rejects", {}).values()):
                return VerificationIssue(
                    code="REPEATED_RULE_OUT",
                    severity="repair",
                    message=f"Option {opt} was already ruled out or rejected recently.",
                )
        return None

    def _deterministic_fallback(
        self,
        state: "DialogueState",
        result: "VerificationResult",
    ) -> Optional[str]:
        """Deterministic safe line for phase-critical failures that survived repair."""
        codes = {i.code for i in result.issues}
        beliefs = self.persona.beliefs

        if ("MISSING_EXPLICIT_VOTE" in codes or "INCONSISTENT_VOTE_WITH_PRIOR_REJECTION" in codes or "INCONSISTENT_VOTE_WITH_PERSONA" in codes) and state.phase == "narrowing":
            rejected = self._rejected_options_for_self(state)
            options = []
            if beliefs:
                options.extend([beliefs.preferred] + list(beliefs.acceptable or []))
            options.extend(list(self._resolver.letters or []))
            for opt in options:
                if opt and opt not in rejected:
                    return f"I'd go with Option {opt}."
            preferred = beliefs.preferred if beliefs else (self._resolver.letters[0] if self._resolver.letters else "A")
            return f"I'd go with Option {preferred}."

        if ("UNCLEAR_CONFIRMATION" in codes or "WEAK_COMPROMISE_CONFIRMATION" in codes) and state.phase == "confirmation":
            candidate = state.candidate_option or state.current_leading_option
            if candidate and beliefs and candidate in (beliefs.rejected or []):
                return "No, I'm still not convinced."
            if candidate and beliefs and candidate != beliefs.preferred:
                return f"I still prefer Option {beliefs.preferred}, but Option {candidate} works well enough."
            return "Yeah, that works for me."

        return None

    # ------------------------------------------------------------------

    def _interaction_instruction(
        self,
        history: list[str],
        state: "DialogueState",
    ) -> str:
        if state.phase in {"opening", "closure", "confirmation"}:
            return ""

        last = self._last_participant_line(history)
        last_has_question = bool(
            last and "?" in last
            and last.split(":", 1)[0].strip() != self.name
            and state.pending_question_target in (None, self.name)
        )

        last_claim_speaker: Optional[str] = None
        if not last_has_question and last and ":" in last:
            spk = last.split(":", 1)[0].strip()
            if spk != self.name and spk not in cfg.EXCLUDED_SPEAKERS:
                last_claim_speaker = spk

        repetition_high = state.repetition_pressure >= cfg.repetition.stall_increment_threshold

        return prompts.interaction_instruction_block(
            last_has_question=last_has_question,
            last_claim_speaker=last_claim_speaker,
            repetition_high=repetition_high,
        )

    def _position_discipline(self, state: "DialogueState") -> str:
        if state.phase not in ("negotiation", "narrowing", "confirmation"):
            return ""
        beliefs = self.persona.beliefs
        if not beliefs:
            return ""

        anchor = beliefs.preferred
        candidate = state.candidate_option or state.current_leading_option

        return prompts.position_discipline_block(
            phase=state.phase,
            anchor=anchor,
            candidate=candidate,
            candidate_in_acceptable=bool(
                candidate and candidate in beliefs.acceptable and candidate != anchor
            ),
            candidate_in_rejected=bool(candidate and candidate in (beliefs.rejected or [])),
            candidate_is_anchor=bool(candidate and candidate == anchor),
            reconsider_text=beliefs.would_reconsider_if or "",
        )

    # ------------------------------------------------------------------

    def _last_participant_line(self, history: list[str]) -> str:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() not in cfg.EXCLUDED_SPEAKERS and speaker.strip() != self.name:
                return f"{speaker.strip()}: {msg.strip()}"
        return ""

    def _enforce_word_budget(self, text: str, phase: str) -> str:
        max_words = self.persona.max_words(phase)
        clean = re.sub(r"\s+", " ", text).strip()
        words = clean.split()
        if len(words) <= max_words:
            return clean

        sentences = re.split(r"(?<=[.!?])\s+", clean)
        kept: list[str] = []
        count = 0
        for sentence in sentences:
            s_words = sentence.split()
            if not s_words:
                continue
            if count + len(s_words) <= max_words:
                kept.append(sentence.strip())
                count += len(s_words)
            elif kept:
                break
            else:
                kept = s_words[:max_words]
                break

        trimmed = " ".join(kept).strip(" ,;:")
        if trimmed and trimmed[-1] not in (".", "!", "?"):
            trimmed += "."
        return trimmed
