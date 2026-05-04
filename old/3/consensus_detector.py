"""
modules/consensus_detector.py
------------------------------
ConsensusDetector — all detection logic for the dialogue system.

Extracted from orchestrator.py to keep each module focused.
The orchestrator calls into this class; no dialogue state is mutated here.

Covers:
  - Soft consensus (natural agreement language, no option letters required)
  - Regex consensus (explicit option letter mentions, weighted by primary)
  - LLM consensus (fallback model call)
  - Speculative loop detection (group stuck speculating about something)
  - Persistent outlier detection (one participant repeating verbatim)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional, TYPE_CHECKING

import prompts as prompts
from config_loader import cfg
from modules.llm_client import get_llm_client

if TYPE_CHECKING:
    from modules.orchestrator import DialogueState

# How many consecutive speculative turns trigger a clarification.
_SPECULATIVE_LOOP_THRESHOLD = 3


# Words that describe the discussion process itself — can never indicate a
# factual speculative loop regardless of topic.
_META_STOPWORDS: frozenset[str] = frozenset({
    "presentation", "present", "research", "discuss", "discussion", "create",
    "choose", "choosing", "decide", "decision", "option", "topic", "theory",
    "group", "team", "audience", "course", "class", "school", "project",
    "show", "work", "find", "finding", "pick", "selecting", "think",
})


def _words_from_topic(topic: str) -> frozenset[str]:
    """
    Extract significant lowercase words (4+ chars) from the topic string.
    These are permanently excluded from speculative loop detection because
    they appear in every turn by virtue of being the subject matter.
    """
    base_stopwords = {
        "that", "this", "with", "have", "they", "from", "will", "each",
        "their", "here", "needs", "need", "about", "what", "which", "where",
        "when", "would", "could", "should",
    }
    words: set[str] = set()
    for word in re.sub(r"[^\w\s]", " ", topic.lower()).split():
        cleaned = word.strip()
        if len(cleaned) >= 4 and cleaned not in base_stopwords:
            words.add(cleaned)
    return frozenset(words)


class ConsensusDetector:

    def __init__(
        self,
        sims: list[Any],
        options: list[str],
        moderator_style: str,
        topic: str = "",
    ) -> None:
        self.sims = sims
        self.options = options
        self.moderator_style = moderator_style
        self._llm = get_llm_client()
        # Words from the topic string + meta process words — never valid loop signals.
        self._topic_stopwords: frozenset[str] = _words_from_topic(topic) | _META_STOPWORDS

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        history: list[str],
        state: "DialogueState",
    ) -> Optional[tuple[str, Optional[str]]]:
        """
        Run all consensus checks in priority order.
        Returns (preferred_option, backup_option) or None.

        Guards:
        - Everyone must have spoken at least twice (prevents single-round exits).
        - Only runs from negotiation phase onward.
        """
        min_turns = len(self.sims) * 2
        if self._participant_turn_count(history) < min_turns:
            return None

        result = self._soft(history, state)
        if result:
            return result

        result = self._regex(history)
        if result:
            return result

        if state.phase in {"negotiation", "narrowing", "confirmation"}:
            state.llm_check_countdown -= 1
            if state.llm_check_countdown <= 0:
                state.llm_check_countdown = cfg.consensus.llm_check_every_n_turns
                return self._llm_check(history)

        return None

    # ------------------------------------------------------------------
    # Speculative loop detection
    # ------------------------------------------------------------------

    def speculative_loop(
        self, history: list[str], used_topics: set[str]
    ) -> Optional[str]:
        """
        Return a content keyword if participants have been speculating about
        the same topic for _SPECULATIVE_LOOP_THRESHOLD consecutive turns.
        Returns None if no loop or all candidates are already clarified.
        """
        hedge_words = {"maybe", "could", "might", "possibly", "perhaps", "wonder", "potential"}
        stopwords = {
            "the", "and", "for", "that", "this", "with", "have", "they",
            "are", "was", "but", "not", "all", "can", "its", "our", "you",
            "we", "it", "in", "of", "to", "a", "is", "be", "at", "on",
            "do", "if", "or", "so", "as", "by", "option", "think", "also",
            "good", "great", "like", "just", "more", "about", "some",
            "maybe", "could", "might", "possibly", "perhaps", "wonder",
            "potential", "would", "should", "offer", "there", "arrange",
            "something", "anything", "everything",
        } | self._topic_stopwords  # topic vocabulary + meta process words never valid

        recent = self._recent_participant_lines(history, limit=_SPECULATIVE_LOOP_THRESHOLD * 4)
        if len(recent) < _SPECULATIVE_LOOP_THRESHOLD:
            return None

        def is_speculative(msg: str) -> bool:
            return "?" in msg or any(w in hedge_words for w in msg.lower().split())

        def content_words(msg: str) -> list[str]:
            return [
                re.sub(r"[^\w]", "", w)
                for w in msg.lower().split()
                if len(re.sub(r"[^\w]", "", w)) >= 4
                and w.rstrip("?.,!") not in stopwords
            ]

        # `recent` is most-recent-first (from _recent_participant_lines).
        # Walk it forward to find the consecutive speculative run at the tail.
        run: list[str] = []
        for line in recent:
            msg = line.split(":", 1)[1]
            if is_speculative(msg):
                run.append(msg)
            else:
                break

        if len(run) < _SPECULATIVE_LOOP_THRESHOLD:
            return None

        all_words: list[str] = [w for msg in run for w in content_words(msg)]
        for word, _ in Counter(all_words).most_common():
            if word and word not in used_topics:
                return word

        return None

    # ------------------------------------------------------------------
    # Persistent outlier detection
    # ------------------------------------------------------------------

    def persistent_outlier(self, history: list[str], has_asked_narrowing: bool) -> Optional[str]:
        """
        Return the name of a participant who has stated the same option
        2+ consecutive turns almost verbatim (no new reasoning).
        Only fires after narrowing has been asked.
        """
        if not has_asked_narrowing:
            return None

        for sim in self.sims:
            last_two = self._last_n_turns_for(sim.name, history, n=2)
            if len(last_two) < 2:
                continue

            options_per_turn = [self._extract_options(t) for t in last_two]
            if not all(opts for opts in options_per_turn):
                continue
            if not all(opts[0] == options_per_turn[0][0] for opts in options_per_turn):
                continue

            # Turns are nearly identical — no new reasoning added.
            words0 = set(last_two[0].split())
            overlap_ratios = [
                len(words0 & set(t.split())) / max(1, len(words0))
                for t in last_two[1:]
            ]
            if all(r >= 0.55 for r in overlap_ratios):
                return sim.name

        return None

    # ------------------------------------------------------------------
    # Consensus methods
    # ------------------------------------------------------------------

    def _soft(
        self, history: list[str], state: "DialogueState"
    ) -> Optional[tuple[str, Optional[str]]]:
        """
        Heuristic: detect natural agreement language without requiring
        explicit option-letter mentions. Requires a known leading option.
        """
        if state.phase not in {"negotiation", "narrowing", "confirmation"}:
            return None

        leading = state.current_leading_option
        if not leading:
            return None

        speakers_seen = {
            line.split(":", 1)[0].strip()
            for line in history
            if ":" in line and line.split(":", 1)[0].strip() not in cfg.EXCLUDED_SPEAKERS
        }
        if not all(s.name in speakers_seen for s in self.sims):
            return None

        agreement_signals = [
            "sounds perfect", "sounds great", "sounds good", "that works for",
            "i'm in", "i'm good with", "i agree", "let's go", "let's do",
            "perfect choice", "great choice", "good choice", "works for me",
            "that's the one", "i confirm", "yes, i confirm", "i'm happy with",
            "absolutely", "definitely", "for sure", "i support", "i love that",
            "that's perfect", "sounds like a plan", "i'm on board",
            "i am on board", "on board", "great, let's", "perfect for",
        ]
        dissent_signals = [
            "not sure", "don't agree", "do not agree", "i disagree",
            "still think", "have we considered", "what about option",
            "rather go with", "instead of", "not convinced",
        ]

        latest: dict[str, str] = self._latest_turn_per_speaker(history)
        if len(latest) < len(self.sims):
            return None

        agree_count = sum(
            1 for msg in latest.values()
            if any(s in msg for s in agreement_signals)
            and not any(s in msg for s in dissent_signals)
        )

        required = (
            len(self.sims) if self.moderator_style != "active"
            else max(2, len(self.sims) - 1)
        )
        if agree_count < required:
            return None

        # Primary must not be the one dissenting.
        primary = self._primary_sim()
        if primary and primary.name in latest:
            msg = latest[primary.name]
            if any(s in msg for s in dissent_signals) and not any(s in msg for s in agreement_signals):
                return None

        return leading, None

    def _regex(self, history: list[str]) -> Optional[tuple[str, Optional[str]]]:
        """
        Fast path: all participants have explicitly voted for the same option.
        Primary's vote counts double. Window scales with group size.
        """
        window = max(cfg.consensus.regex_window, len(self.sims) * 3)
        recent = self._recent_participant_lines(history, limit=window)

        if len(recent) < len(self.sims):
            return None
        if not self._primary_has_agreed(recent):
            return None

        latest_vote: dict[str, str] = {}
        for line in reversed(recent):
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            mentions = self._extract_options(msg)
            if mentions and speaker not in latest_vote:
                latest_vote[speaker] = mentions[0]

        if len(latest_vote) < len(self.sims):
            return None

        # Primary double-vote.
        primary = self._primary_sim()
        weighted = list(latest_vote.values())
        if primary and primary.name in latest_vote:
            weighted.append(latest_vote[primary.name])

        counts = Counter(weighted)
        top_option, top_count = counts.most_common(1)[0]

        n = len(self.sims)
        max_dissenters = (
            getattr(cfg.consensus, "max_dissenters_active", min(2, n - 2))
            if self.moderator_style == "active"
            else getattr(cfg.consensus, "max_dissenters_other", 0)
        )
        if top_count < max(2, n - max_dissenters):
            return None

        return top_option, None

    def _llm_check(self, history: list[str]) -> Optional[tuple[str, Optional[str]]]:
        """LLM fallback: ask the model whether consensus has been reached."""
        n_needed = max(2, len(self.sims) - 1)
        recent_dialogue = "\n".join(history[-20:])
        names = [s.name for s in self.sims]

        try:
            data = self._llm.generate_json(
                prompts.consensus_check(names, self.options, recent_dialogue, n_needed, len(self.sims))
            )
            if not data.get("consensus_reached"):
                return None
            opt = str(data.get("preferred_option") or "").strip().upper()
            bak_raw = str(data.get("backup_option") or "").strip().upper()
            bak = bak_raw if bak_raw in {"A", "B", "C", "D"} and bak_raw != opt else None
            if opt in {"A", "B", "C", "D"}:
                return opt, bak
        except Exception as exc:
            print(f"!! LLM consensus check error: {exc}")

        return None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _extract_options(self, text: str) -> list[str]:
        return [m.upper() for m in re.findall(r"\boption\s+([a-d])\b", text.lower())]

    def _primary_sim(self) -> Optional[Any]:
        return next((s for s in self.sims if s.persona.get("is_primary", False)), None)

    def _primary_has_agreed(self, recent: list[str]) -> bool:
        primary = self._primary_sim()
        if primary is None:
            return True
        for line in reversed(recent):
            speaker, msg = line.split(":", 1)
            if speaker.strip() == primary.name and self._extract_options(msg):
                return True
        return False

    def _participant_turn_count(self, history: list[str]) -> int:
        return sum(
            1 for line in history
            if ":" in line and line.split(":", 1)[0].strip() not in cfg.EXCLUDED_SPEAKERS
        )

    def _recent_participant_lines(self, history: list[str], limit: int) -> list[str]:
        lines: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            if line.split(":", 1)[0].strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    def _latest_turn_per_speaker(self, history: list[str]) -> dict[str, str]:
        latest: dict[str, str] = {}
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in cfg.EXCLUDED_SPEAKERS:
                continue
            if speaker not in latest:
                latest[speaker] = msg.lower()
            if len(latest) == len(self.sims):
                break
        return latest

    def _last_n_turns_for(self, name: str, history: list[str], n: int) -> list[str]:
        turns: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == name:
                turns.append(msg.strip().lower())
                if len(turns) >= n:
                    break
        return turns