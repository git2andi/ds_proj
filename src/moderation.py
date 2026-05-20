"""
moderation.py
-------------
ModerationEngine — all moderator decision logic for a single dialogue.

  Orchestrator owns: phase transitions, vote tracking, consensus, closure.
  ModerationEngine owns: when/how to intervene, escalation, speculative-loop
                         and outlier detection, LLM-driven moderator lines.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from utils import current_votes, last_n_turns_for, extract_option_letters

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator


# (text, tokens_in, tokens_out) — passed to _store_moderator in orchestrator
StoreFn = Callable[[str, int, int], None]


class ModerationEngine:

    def __init__(
        self,
        topic: str,
        options: list[str],
        moderator_style: str,
        sims: list["Simulator"],
    ) -> None:
        self.topic = topic
        self.options = list(options)
        self.moderator_style = moderator_style
        self.sims = sims
        self._llm = get_llm_client()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def escalation_level(self, state: "DialogueState") -> int:
        """0=normal nudges, 1=direct compromise ask, 2=firm demand, 3=force-close."""
        r = state.post_narrowing_rounds
        if r < cfg.turns.escalation_level_1:
            return 0
        if r < cfg.turns.escalation_level_2:
            return 1
        if r < cfg.turns.escalation_level_3:
            return 2
        return 3

    def should_narrow(self, state: "DialogueState", participant_turn_count: int) -> bool:
        if state.has_asked_narrowing or self.moderator_style == "passive":
            return False
        n = len(self.sims)
        if participant_turn_count < max(n * 2, cfg.turns.min_before_narrowing):
            return False
        stalling = state.repetition_pressure >= 0.75 and state.stall_rounds >= 1
        talked_plenty = participant_turn_count >= n * 5
        if self.moderator_style == "minimal":
            return stalling and talked_plenty
        return stalling or talked_plenty

    def should_intervene(
        self,
        state: "DialogueState",
        history: list[str],
        any_sim_stuck: bool,
        participant_turn_count: int,
    ) -> Optional[str]:
        """
        Return an intervention tag or None.
          'clarify:{keyword}' — speculative loop about something outside the options
          'outlier:{name}'    — participant repeating the same position verbatim
          'stall'             — generic high-repetition stall
        """
        if self.moderator_style == "passive":
            return None
        if participant_turn_count < len(self.sims):
            return None

        loop_topic = self._detect_speculative_loop(state, history)
        if loop_topic:
            return f"clarify:{loop_topic}"

        outlier = self._detect_outlier(state, history)
        if outlier:
            return f"outlier:{outlier}"

        if state.has_asked_narrowing and any_sim_stuck:
            return "stall"

        if state.repetition_pressure >= 0.80 and state.stall_rounds >= 2:
            return "stall"

        return None

    def run_intervention(
        self,
        reason: str,
        state: "DialogueState",
        history: list[str],
        store_fn: StoreFn,
    ) -> None:
        """Generate a moderator line via LLM and hand it to store_fn(text, tokens_in, tokens_out)."""
        names = [s.name for s in self.sims]
        recent = "\n".join(history[-10:])
        level = self.escalation_level(state)

        try:
            if reason.startswith("clarify:"):
                keyword = reason.split(":", 1)[1]
                state.clarification_topics_used.add(keyword)
                line = self._llm.generate(
                    prompts.moderator_clarification(
                        topic=self.topic,
                        participant_names=names,
                        options=self.options,
                        recent_dialogue=recent,
                        looping_topic=keyword,
                    )
                ).strip()

            elif reason.startswith("outlier:"):
                outlier_name = reason.split(":", 1)[1]
                state.nudged_participants.add(outlier_name)
                votes = current_votes(history, self.sims)
                context_note = ""
                if votes:
                    n = len(self.sims)
                    counts = Counter(votes.values())
                    top_opt, top_count = counts.most_common(1)[0]
                    has_majority = top_count > n / 2
                    if has_majority:
                        supporters = [nm for nm, o in votes.items() if o == top_opt and nm != outlier_name]
                        if supporters:
                            verb = "both prefer" if len(supporters) > 1 else "prefers"
                            context_note = (
                                f" {' and '.join(supporters)} {verb} Option {top_opt} — "
                                f"ask {outlier_name} what specific trade-off makes their choice worth holding."
                            )
                    else:
                        # Multi-way split — no majority exists
                        context_note = (
                            f" The group is split with no clear majority — "
                            f"ask {outlier_name} what one specific thing matters most to them."
                        )
                outlier_reason = (
                    f"{outlier_name} has been repeating the same position without new reasoning."
                    + context_note
                )
                line = self._llm.generate(
                    prompts.moderator_intervention(
                        topic=self.topic,
                        participant_names=names,
                        recent_dialogue=recent,
                        reason=outlier_reason,
                        target_participant=outlier_name,
                        escalation_level=level,
                    )
                ).strip()

            elif reason == "compromise":
                candidate = getattr(state, "compromise_option", None) or getattr(state, "candidate_option", None)
                line = self._llm.generate(
                    prompts.moderator_compromise_test(
                        topic=self.topic,
                        participant_names=names,
                        options=self.options,
                        recent_dialogue=recent,
                        compromise_option=candidate or "A",
                        holdout_names=[
                            s.name for s in self.sims
                            if current_votes(history, self.sims).get(s.name) != candidate
                        ],
                    )
                ).strip()

            else:  # stall
                votes = current_votes(history, self.sims)
                candidate = getattr(state, "candidate_option", None) or state.current_leading_option
                if state.phase == "emergence" and level < 2 and candidate:
                    # Fisher Phase 3: facilitate softening, don't harden positions
                    line = self._llm.generate(
                        prompts.moderator_emergence(
                            topic=self.topic,
                            participant_names=names,
                            options=self.options,
                            recent_dialogue=recent,
                            candidate_option=candidate,
                        )
                    ).strip()
                else:
                    line = self._llm.generate(
                        prompts.moderator_deadlock(
                            topic=self.topic,
                            participant_names=names,
                            options=self.options,
                            recent_dialogue=recent,
                            current_votes=votes,
                            escalation_level=level,
                        )
                    ).strip()

            if line:
                store_fn(line, self._llm.last_tokens_in, self._llm.last_tokens_out)

        except Exception as exc:
            print(f"!! Moderator intervention error ({reason}): {exc}")

    # ------------------------------------------------------------------
    # Private detection helpers
    # ------------------------------------------------------------------

    def _detect_speculative_loop(
        self, state: "DialogueState", history: list[str]
    ) -> Optional[str]:
        """
        Return a keyword if participants keep speculating about something not in any
        option description for 4+ consecutive turns.
        """
        if not self.options:
            return None

        hedge_words = {"maybe", "could", "might", "possibly", "perhaps", "wonder"}
        stopwords = {
            "the", "and", "for", "that", "this", "with", "have", "they",
            "are", "was", "but", "not", "all", "can", "its", "our", "you",
            "we", "it", "in", "of", "to", "a", "is", "be", "at", "on",
            "do", "if", "or", "so", "as", "by", "option", "think", "also",
            "good", "great", "like", "just", "more", "about", "some",
            "would", "should", "there", "something", "anything", "might",
            "steve", "party", "group", "point", "think", "feel", "make",
        } | hedge_words | {
            "still", "going", "other", "first", "their", "after", "right",
            "every", "since", "which", "while", "where", "again", "those",
            "these", "never", "start", "being", "often", "under", "given",
            "doing", "least", "means", "seems", "keeps", "risks", "really",
        }

        excluded_words: set[str] = set()
        for opt in self.options:
            for w in re.sub(r"[^\w\s]", " ", opt.lower()).split():
                if len(w) >= 5:
                    excluded_words.add(w)
        for w in re.sub(r"[^\w\s]", " ", self.topic.lower()).split():
            if len(w) >= 5:
                excluded_words.add(w)

        threshold = 4
        min_word_len = 5

        recent: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            recent.append(msg.strip().lower())
            if len(recent) >= threshold * 4:
                break

        if len(recent) < threshold:
            return None

        def is_speculative(msg: str) -> bool:
            return "?" in msg or any(w in hedge_words for w in msg.split())

        run: list[str] = []
        for msg in recent:
            if is_speculative(msg):
                run.append(msg)
            else:
                break

        if len(run) < threshold:
            return None

        all_words: list[str] = []
        for msg in run:
            for w in re.sub(r"[^\w]", " ", msg).split():
                if (len(w) >= min_word_len
                        and w not in stopwords
                        and w not in excluded_words
                        and w not in state.clarification_topics_used):
                    all_words.append(w)

        if not all_words:
            return None

        from collections import Counter as _Counter
        top_word, _ = _Counter(all_words).most_common(1)[0]
        return top_word

    def _detect_outlier(
        self, state: "DialogueState", history: list[str]
    ) -> Optional[str]:
        """Name of a participant whose last 2 turns are >55% identical in wording."""
        if not state.has_asked_narrowing:
            return None
        for sim in self.sims:
            turns = last_n_turns_for(sim.name, history, n=2)
            if len(turns) < 2:
                continue
            words0 = set(turns[0].split())
            ratio = len(words0 & set(turns[1].split())) / max(1, len(words0))
            if ratio >= 0.55:
                return sim.name
        return None
