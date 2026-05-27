"""
simulator.py
------------
Simulator -- wraps a Persona and generates one dialogue turn via the LLM,
using the compact speaker-card prompt only (legacy monolithic path removed).
"""

from __future__ import annotations

import random
import re
from typing import Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from persona import Persona
from prompt_context import (
    build_group_state,
    build_local_context,
    build_move_instruction,
    build_output_contract,
    build_relevant_options,
    build_speaker_card,
)
from reasoning import fact_check, repair_directive

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from policy import TurnPlan


class Simulator:

    _GOODBYES = ["Later!", "See ya!", "Bye!", "Bye everyone!", "See you!", "Cheers!", "Take care!"]

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options
        self._llm = get_llm_client()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_turn(
        self, history: list[str], state: "DialogueState",
        all_names: list[str] | None = None,
        forced_adaptation: bool = False,
        turn_plan: Optional["TurnPlan"] = None,
    ) -> tuple[str, int, int]:
        if state.phase == "closure":
            return self._closure_line(state), 0, 0

        raw = self._generate(history, state, forced_adaptation, turn_plan)
        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out

        if not raw:
            return "[SILENCE]", tok_in, tok_out

        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        raw = self._enforce_word_budget(raw, state.phase)
        return raw or "[SILENCE]", tok_in, tok_out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _has_voted(self, history: list[str]) -> bool:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() != self.name:
                continue
            if re.search(r"\boption\s+([a-d])(?![\w'\-])", msg.lower()):
                return True
        return False

    def _generate(self, history: list[str], state: "DialogueState",
                  forced_adaptation: bool,
                  turn_plan: Optional["TurnPlan"]) -> str:
        add_brevity = (
            (state.repetition_pressure >= cfg.repetition.add_brevity_threshold
             or state.post_narrowing_rounds >= 2)
            and state.phase not in {"greeting", "opening", "closure", "confirmation"}
        )
        phase_instr = prompts.phase_instruction_text(
            phase=state.phase, add_brevity=add_brevity, has_voted=self._has_voted(history),
        )

        is_closure = state.phase == "closure"
        n_recent = cfg.prompt_budget.recent_turns_short

        speaker_card = build_speaker_card(self.persona)
        candidate = state.candidate_option or state.current_leading_option
        relevant_opts = build_relevant_options(self.options, self.persona, candidate)
        group_state = "" if is_closure else build_group_state(state)
        local_ctx = build_local_context(history, n_recent=n_recent)

        interaction_instr = "" if is_closure else self._interaction_instruction(history, state)
        position_disc = "" if is_closure else self._position_discipline(state)
        forbidden_openers = self._recent_openers(history)

        move_instr = build_move_instruction(
            phase_instruction=phase_instr,
            interaction_instruction=interaction_instr,
            position_discipline=position_disc,
            forced_adaptation=forced_adaptation,
            forbidden_openers=forbidden_openers,
            turn_plan=turn_plan,
        )
        output_contract = build_output_contract(
            max_words=self.persona.max_words(state.phase), name=self.name,
        )

        prompt = prompts.sim_turn_compact(
            speaker_card=speaker_card, relevant_options=relevant_opts,
            group_state=group_state, local_context=local_ctx,
            move_instruction=move_instr, output_contract=output_contract,
        )

        try:
            result = self._llm.generate(prompt).strip()
            return self._ground_check(result, prompt)
        except Exception as exc:
            print(f"!! Turn generation error for {self.name}: {exc}")
            return "[SILENCE]"

    # ------------------------------------------------------------------
    # Grounding -- only on longer turns; short reactions can't invent facts
    # ------------------------------------------------------------------

    def _ground_check(self, turn_text: str, original_prompt: str) -> str:
        if not cfg.grounding.enable_fact_check:
            return turn_text
        if len(turn_text.split()) < cfg.grounding.min_words_to_check:
            return turn_text

        flags = fact_check(turn_text, self.options, self.topic)
        if not flags:
            return turn_text

        if cfg.grounding.repair_attempts < 1:
            return turn_text

        try:
            repaired = self._llm.generate(original_prompt + "\n\n" + repair_directive()).strip()
            return repaired
        except Exception:
            return turn_text

    # ------------------------------------------------------------------
    # Interaction instruction (delegates prose to prompts.py)
    # ------------------------------------------------------------------

    def _interaction_instruction(self, history: list[str], state: "DialogueState") -> str:
        if state.phase in {"greeting", "closure", "confirmation"}:
            return ""

        last = self._last_participant_line(history)
        recent = self._recent_participant_messages(history, limit=5)
        question_count = sum(1 for msg in recent if "?" in msg)
        last_has_question = bool(last and "?" in last and last.split(":", 1)[0].strip() != self.name)

        # When the last turn was a claim (not a question), surface the speaker
        # so the prompt can oblige a direct reaction rather than a restatement.
        last_claim_speaker: Optional[str] = None
        if not last_has_question and last and ":" in last:
            spk = last.split(":", 1)[0].strip()
            if spk != self.name and spk not in cfg.EXCLUDED_SPEAKERS:
                last_claim_speaker = spk

        compromise = getattr(state, "compromise_option", None) if state.phase == "emergence" else None
        b = self.persona.beliefs
        compromise_in_acceptable = bool(compromise and b and compromise in b.acceptable)

        rejected = getattr(state, "last_rejected_option", None)
        rejecting_speaker = getattr(state, "last_rejecting_speaker", None)
        rejecting_self = bool(rejected and rejecting_speaker == self.name
                              and state.phase in {"narrowing", "emergence"})
        turns_since_rejection = len(self._own_recent_turns(history, limit=10)) if rejecting_self else 0

        own_last_was_question = bool(self._last_own_turn(history) and "?" in self._last_own_turn(history))
        own_turns = self._own_recent_turns(history, limit=4)
        speculative_count = sum(1 for t in own_turns if re.search(r"\bwhat if\b", t, re.IGNORECASE))
        repeated_kws = self._repeated_concern_keywords(history)
        self_repeated = self._own_recent_repetition(history) if len(repeated_kws) < 2 else False

        return prompts.interaction_instruction_block(
            last_has_question=last_has_question,
            question_count=question_count,
            compromise_option=compromise,
            compromise_in_acceptable=compromise_in_acceptable,
            rejected_option=rejected if rejecting_self else None,
            rejecting_self=rejecting_self,
            turns_since_rejection=turns_since_rejection,
            escalation_threshold=cfg.repetition.turns_since_rejection_escalation,
            own_last_was_question=own_last_was_question,
            speculative_count=speculative_count,
            repeated_kws=repeated_kws,
            self_repeated=self_repeated,
            last_claim_speaker=last_claim_speaker,
        )

    def _position_discipline(self, state: "DialogueState") -> str:
        if state.phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
            return ""
        beliefs = self.persona.beliefs
        if not beliefs:
            return ""

        preferred = beliefs.preferred
        flips = state.vote_changes.get(self.name, 0)
        current_vote = state.last_known_vote.get(self.name)

        if current_vote and current_vote != preferred and flips >= 1:
            anchor = current_vote
            prefix = f"You switched to Option {anchor}."
        else:
            anchor = preferred
            prefix = f"You lean Option {anchor} ({beliefs.key_concern})."

        # Compute candidate for ALL decision phases so position_discipline can
        # honor acceptable-list flexibility during narrowing and confirmation too.
        # Negotiation stays anchor-driven.
        candidate = None
        if state.phase in ("narrowing", "emergence", "confirmation"):
            candidate = state.candidate_option or state.current_leading_option

        rejected_list = beliefs.rejected or []
        can_soften = (
            (self.persona.agreeableness >= 3 or self.persona.conscientiousness <= 2)
            and state.emergence_rounds >= 3
        )

        return prompts.position_discipline_block(
            phase=state.phase, prefix=prefix, anchor=anchor, flips=flips,
            candidate=candidate,
            candidate_in_acceptable=bool(candidate and candidate in beliefs.acceptable and candidate != anchor),
            candidate_in_rejected=bool(candidate and candidate in rejected_list),
            candidate_is_anchor=bool(candidate and candidate == anchor),
            can_soften=can_soften,
            concession_text=beliefs.concession or "",
            high_agreeableness=self.persona.agreeableness >= 4,
            low_agreeableness_or_high_neuroticism=(
                self.persona.agreeableness <= 2 or self.persona.neuroticism >= 4
            ),
        )

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _own_recent_repetition(self, history: list[str]) -> bool:
        turns = self._own_recent_turns(history, limit=2)
        if len(turns) < 2:
            return False
        a = set(re.sub(r"[^\w\s]", "", turns[0].lower()).split())
        b = set(re.sub(r"[^\w\s]", "", turns[1].lower()).split())
        if not a or not b:
            return False
        return len(a & b) / max(1, min(len(a), len(b))) >= cfg.repetition.jaccard_threshold_self

    def _repeated_concern_keywords(self, history: list[str]) -> list[str]:
        turns = self._own_recent_turns(history, limit=5)
        if len(turns) < 2:
            return []
        stopwords = {
            "that", "this", "with", "from", "have", "would", "could", "should",
            "there", "their", "about", "option", "think", "just", "like", "work",
            "good", "need", "want", "make", "sure", "know", "okay", "also", "even",
            "very", "more", "some", "than", "when", "what", "which", "been", "will",
            "still", "into", "back", "each", "much", "both", "your", "they", "them",
            "then", "well", "here", "feel", "does", "might", "only", "most", "over",
            "really", "right", "actually", "though", "going", "mean",
        }
        from collections import Counter as _Counter
        word_counts: _Counter = _Counter()
        for turn in turns:
            words = set(re.sub(r"[^\w\s]", "", turn.lower()).split())
            meaningful = {w for w in words if len(w) >= 4 and w not in stopwords}
            word_counts.update(meaningful)
        return [w for w, c in word_counts.items() if c >= 2]

    def _own_recent_turns(self, history: list[str], limit: int) -> list[str]:
        turns: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == self.name:
                turns.append(msg.strip())
                if len(turns) >= limit:
                    break
        return turns

    def _recent_participant_messages(self, history: list[str], limit: int) -> list[str]:
        messages: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            messages.append(msg.strip())
            if len(messages) >= limit:
                break
        return messages

    def _last_own_turn(self, history: list[str]) -> str:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == self.name:
                return msg.strip()
        return ""

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

    def _recent_openers(self, history: list[str], n: int = 6) -> str:
        openers: list[str] = []
        seen: set[str] = set()
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            words = msg.strip().split()
            if not words:
                continue
            w1 = words[0].rstrip(",.!?").lower()
            w2 = (words[1].rstrip(",.!?").lower() if len(words) > 1 else "")
            two_word = f"{w1} {w2}".strip() if w2 else w1
            if two_word and two_word not in seen:
                openers.append(two_word)
                seen.add(two_word)
                seen.add(w1)
            if len(openers) >= n:
                break
        return ", ".join(f'"{o}"' for o in openers) if openers else ""

    def _closure_line(self, state: "DialogueState") -> str:
        option = getattr(state, "preferred_option", None)
        if option and random.random() < cfg.moderation.closure.template_probability:
            return random.choice(prompts.closure_templates(option))
        return random.choice(self._GOODBYES)
