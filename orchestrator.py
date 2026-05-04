"""
orchestrator.py
---------------
Orchestrator — coordinates a single dialogue run.

Responsibilities:
  1. Setup       — generate options + opening history (LLM)
  2. State        — phase, leading option, turn counts
  3. Main loop   — drive rounds, detect consensus, fire moderator lines
  4. Moderator   — narrowing, escalating interventions, confirmation, closure

Detection  → consensus.py
Turn logic → turn_manager.py
Logging    → logger.py
"""

from __future__ import annotations

import datetime
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import prompts
from config_loader import cfg
from consensus import ConsensusDetector
from logger import DialogueLogger
from llm_client import get_llm_client
from turn_manager import TurnManager


# ---------------------------------------------------------------------------
# Dialogue state
# ---------------------------------------------------------------------------

@dataclass
class DialogueState:
    phase: str = "opening"
    mode: str = "decision"              # "decision" | "open"
    turn_index: int = 0

    has_asked_narrowing: bool = False
    agreement_reached: bool = False

    preferred_option: Optional[str] = None
    backup_option: Optional[str] = None
    current_leading_option: Optional[str] = None

    last_addressed: Optional[str] = None
    pending_question_target: Optional[str] = None

    repetition_pressure: float = 0.0

    stall_rounds: int = 0                # rounds of high repetition after narrowing
    post_narrowing_rounds: int = 0       # total rounds after narrowing (for escalation)
    llm_check_countdown: int = 0

    # Prevent re-firing the same moderator clarification topic
    clarification_topics_used: set[str] = field(default_factory=set)

    # Sims scheduled for a forced-adaptation instruction on their next turn
    nudged_participants: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:

    def __init__(self, topic: str, moderator_style: str = "active", mode: str = "decision") -> None:
        self.topic = topic
        self.moderator_style = moderator_style.lower()
        self.mode = mode
        self.sims: list[Any] = []

        self._llm = get_llm_client()
        self._turn_mgr = TurnManager()
        self.state = DialogueState()
        self.state.mode = mode
        self.state.llm_check_countdown = cfg.consensus.llm_check_every_n_turns

        self.dialogue_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if mode == "open":
            self.options = []
            self.opening_question = f"What's your honest take on: {topic}?"
        else:
            self.options, self.opening_question = self._generate_options()

        self.history: list[str] = self._build_opening_history()

        self._logger = DialogueLogger(self.dialogue_id, topic, moderator_style)
        self._detector: Optional[ConsensusDetector] = None

    def add_sim(self, sim: Any) -> None:
        self.sims.append(sim)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _generate_options(self) -> tuple[list[str], str]:
        fallback_options = [
            "Option A - Budget: lowest cost, basic features.",
            "Option B - Convenience: faster or easier, moderately priced.",
            "Option C - Quality: best outcome, higher cost.",
            "Option D - Flexible: adaptable trade-offs.",
        ]
        fallback_q = "What matters most to you personally among these options?"

        try:
            data = self._llm.generate_json(prompts.option_generation(self.topic))
            options_raw = data.get("options", [])
            question = str(data.get("opening_question", "")).strip() or fallback_q

            if not isinstance(options_raw, list) or len(options_raw) != 4:
                return fallback_options, fallback_q

            cleaned: list[str] = []
            for i, raw in enumerate(options_raw):
                if not isinstance(raw, str) or not raw.strip():
                    return fallback_options, fallback_q
                label = chr(ord("A") + i)
                text = raw.strip()
                if not text.lower().startswith(f"option {label.lower()}"):
                    text = f"Option {label} - {text}"
                cleaned.append(text)

            return cleaned, question

        except Exception as exc:
            print(f"!! Option generation error: {exc}")
            return fallback_options, fallback_q

    def _build_opening_history(self) -> list[str]:
        if self.mode == "open":
            lines = [
                "Moderator: Hey everyone, glad you could join.",
                f"Moderator: Today we are talking about: {self.topic}",
                "Moderator: No agenda, no right answers — just share whatever is on your mind.",
                f"Moderator: {self.opening_question}",
            ]
        else:
            lines = [
                "Moderator: Hey everyone, let's get started.",
                f"Moderator: Today we are deciding: {self.topic}",
                "Moderator: Here are the options on the table:",
            ]
            lines.extend(f"Moderator: {opt}" for opt in self.options)
            lines.append(f"Moderator: {self.opening_question}")
        return lines

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _store_line(self, line: str, selected_reason: str = "") -> None:
        self.history.append(line)
        print(f"-> {line}")
        self._logger.append_line(line)
        self._logger.buffer(line, selected_reason, self.state, self.sims)

    def _store_moderator(self, text: str) -> None:
        self._store_line(f"Moderator: {text}", selected_reason="moderator")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _participant_turn_count(self) -> int:
        return sum(
            1 for line in self.history
            if ":" in line
            and line.split(":", 1)[0].strip() not in cfg.EXCLUDED_SPEAKERS
        )

    def _extract_option_letters(self, text: str) -> list[str]:
        return [m.upper() for m in re.findall(r"\boption\s+([a-d])\b", text.lower())]

    def _primary_sim(self) -> Optional[Any]:
        return next((s for s in self.sims if s.persona.is_primary), None)

    def _update_leading_option(self) -> None:
        mentions: list[str] = []
        limit = max(5, len(self.sims) * 2)
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            mentions.extend(self._extract_option_letters(msg))
            if len(mentions) >= limit:
                break
        if mentions:
            self.state.current_leading_option = Counter(mentions).most_common(1)[0][0]

    def _update_phase(self) -> None:
        if self.state.agreement_reached:
            self.state.phase = "closure"
            return

        turns = self._participant_turn_count()
        n = len(self.sims)

        if self.state.mode == "open":
            if turns == 0:
                self.state.phase = "opening"
            elif turns < n * 2:
                self.state.phase = "discussion"
            elif turns < n * 4:
                self.state.phase = "deepening"
            else:
                self.state.phase = "closing"
            return

        # Decision mode
        if turns == 0:
            self.state.phase = "opening"
        elif turns < n:
            self.state.phase = "preference_expression"
        elif self.state.has_asked_narrowing:
            self.state.phase = "narrowing"
        else:
            self.state.phase = "negotiation"

    def _update_discourse(self) -> None:
        sim_names = {s.name for s in self.sims}
        result = self._turn_mgr.extract_discourse(self.history, sim_names)
        self.state.last_addressed = result["last_addressed"]
        self.state.pending_question_target = result["pending_question_target"]

    def _update_repetition(self) -> None:
        self.state.repetition_pressure = self._turn_mgr.repetition_pressure(self.history)

    # ------------------------------------------------------------------
    # Stall detection helpers
    # ------------------------------------------------------------------

    def _current_votes(self) -> dict[str, str]:
        """
        Each sim's most recent explicitly stated option (letter A-D).
        Only looks at participant lines; newest-first walk, one vote per name.
        """
        votes: dict[str, str] = {}
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in cfg.EXCLUDED_SPEAKERS or speaker in votes:
                continue
            letters = self._extract_option_letters(msg)
            if letters:
                votes[speaker] = letters[0]
            if len(votes) == len(self.sims):
                break
        return votes

    def _is_split_deadlock(self) -> bool:
        """
        True when every sim has voted and no single option has enough supporters
        to satisfy the consensus threshold — i.e. genuine N-way split.
        Only meaningful after narrowing has been asked.
        """
        if not self.state.has_asked_narrowing:
            return False
        votes = self._current_votes()
        if len(votes) < len(self.sims):
            return False            # not everyone has voted yet

        n = len(self.sims)
        max_dissenters = (
            cfg.consensus.max_dissenters_active
            if self.moderator_style == "active"
            else cfg.consensus.max_dissenters_other
        )
        required = n - max_dissenters
        counts = Counter(votes.values())
        return counts.most_common(1)[0][1] < required

    def _sim_vote_is_stuck(self, name: str, window: int = 4) -> bool:
        """
        True if this sim has stated the same option in all of their last
        `window` turns — semantic stall regardless of word overlap.
        """
        turns = self._last_n_turns_for(name, n=window)
        if len(turns) < window:
            return False
        options_per_turn = [self._extract_option_letters(t) for t in turns]
        if not all(opts for opts in options_per_turn):
            return False
        first = options_per_turn[0][0]
        return all(opts[0] == first for opts in options_per_turn)

    def _any_sim_stuck(self) -> bool:
        return any(self._sim_vote_is_stuck(s.name) for s in self.sims)

    # ------------------------------------------------------------------
    # Moderator logic
    # ------------------------------------------------------------------

    def _should_narrow(self) -> bool:
        if self.state.has_asked_narrowing or self.moderator_style == "passive":
            return False
        turns = self._participant_turn_count()
        n = len(self.sims)
        if turns < max(n * 2, cfg.turns.min_before_narrowing):
            return False
        stalling = self.state.repetition_pressure >= 0.75 and self.state.stall_rounds >= 1
        talked_plenty = turns >= n * 5
        if self.moderator_style == "minimal":
            return stalling and talked_plenty
        return stalling or talked_plenty

    def _narrowing_prompt(self) -> None:
        self.state.has_asked_narrowing = True
        self.state.phase = "narrowing"
        self._store_moderator(
            "Let's narrow this down — which option does each of you prefer? "
            "A backup is fine if you're genuinely unsure."
        )

    def _escalation_level(self) -> int:
        """
        Returns 0, 1, 2, or 3 based on how many post-narrowing rounds have
        passed without resolution. Controls how aggressively the moderator acts.
          0 — normal (Socratic questions, nudges)
          1 — direct (ask for compromise explicitly)
          2 — firm (name the split, demand movement)
          3 — force (moderator picks and closes)
        """
        r = self.state.post_narrowing_rounds
        if r < cfg.turns.escalation_level_1:
            return 0
        if r < cfg.turns.escalation_level_2:
            return 1
        if r < cfg.turns.escalation_level_3:
            return 2
        return 3

    def _should_intervene(self) -> Optional[str]:
        """
        Return an intervention reason string or None. Possible values:
          "clarify:{keyword}"   — speculative loop about something not in the options
          "outlier:{name}"      — one sim repeating the same position verbatim
          "stall"               — generic high-repetition stall
        Escalation level is checked by the caller and passed to the prompt.
        """
        if self.moderator_style == "passive":
            return None
        if self._participant_turn_count() < len(self.sims):
            return None

        # Speculative loop
        loop_topic = self._detect_speculative_loop()
        if loop_topic:
            return f"clarify:{loop_topic}"

        # Outlier: verbatim repetition
        outlier = self._detect_outlier()
        if outlier:
            return f"outlier:{outlier}"

        # Semantic stall: same vote restated without new reasoning
        if self.state.has_asked_narrowing and self._any_sim_stuck():
            return "stall"

        # Generic word-overlap stall
        if self.state.repetition_pressure >= 0.80 and self.state.stall_rounds >= 2:
            return "stall"

        return None

    def _detect_speculative_loop(self) -> Optional[str]:
        """
        Return a content keyword if participants keep speculating about something
        not present in any option description for 3+ consecutive turns.
        """
        hedge_words = {"maybe", "could", "might", "possibly", "perhaps", "wonder"}
        stopwords = {
            "the", "and", "for", "that", "this", "with", "have", "they",
            "are", "was", "but", "not", "all", "can", "its", "our", "you",
            "we", "it", "in", "of", "to", "a", "is", "be", "at", "on",
            "do", "if", "or", "so", "as", "by", "option", "think", "also",
            "good", "great", "like", "just", "more", "about", "some",
            "would", "should", "there", "something", "anything",
        }

        option_words: set[str] = set()
        for opt in self.options:
            for w in re.sub(r"[^\w\s]", " ", opt.lower()).split():
                if len(w) >= 4:
                    option_words.add(w)

        threshold = 3
        recent: list[str] = []
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            recent.append(msg.strip().lower())
            if len(recent) >= threshold * 3:
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
                if (len(w) >= 4
                        and w not in stopwords
                        and w not in option_words
                        and w not in self.state.clarification_topics_used):
                    all_words.append(w)

        if not all_words:
            return None

        top_word, _ = Counter(all_words).most_common(1)[0]
        return top_word

    def _detect_outlier(self) -> Optional[str]:
        """Name of a participant whose last 2 turns are >55% identical in wording."""
        if not self.state.has_asked_narrowing:
            return None
        for sim in self.sims:
            turns = self._last_n_turns_for(sim.name, n=2)
            if len(turns) < 2:
                continue
            words0 = set(turns[0].split())
            ratio = len(words0 & set(turns[1].split())) / max(1, len(words0))
            if ratio >= 0.55:
                return sim.name
        return None

    def _last_n_turns_for(self, name: str, n: int) -> list[str]:
        turns: list[str] = []
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == name:
                turns.append(msg.strip().lower())
                if len(turns) >= n:
                    break
        return turns

    def _run_moderator_intervention(self, reason: str) -> None:
        names = [s.name for s in self.sims]
        recent = "\n".join(self.history[-10:])
        level = self._escalation_level()

        try:
            if reason.startswith("clarify:"):
                keyword = reason.split(":", 1)[1]
                self.state.clarification_topics_used.add(keyword)
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
                self.state.nudged_participants.add(outlier_name)
                line = self._llm.generate(
                    prompts.moderator_intervention(
                        topic=self.topic,
                        participant_names=names,
                        recent_dialogue=recent,
                        reason=f"{outlier_name} has been repeating the same position without new reasoning",
                        target_participant=outlier_name,
                        escalation_level=level,
                    )
                ).strip()

            else:  # stall — use escalation level
                votes = self._current_votes()
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
                self._store_moderator(line)

        except Exception as exc:
            print(f"!! Moderator intervention error ({reason}): {exc}")

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _max_speakers(self) -> int:
        phase = self.state.phase
        n = len(self.sims)
        if phase in ("opening", "closure"):
            return 1
        if self.state.repetition_pressure >= 0.65:
            return 1
        if phase == "confirmation":
            return min(2, n)
        weights = [0.30, 0.50, 0.20] if n >= 3 else [0.45, 0.55]
        choices = list(range(1, min(4, n + 1)))
        return random.choices(choices, weights=weights[: len(choices)])[0]

    def _run_participant_round(self) -> bool:
        """Run one round of participant turns. Returns True if any sim spoke."""
        self._update_discourse()
        self._update_repetition()
        self._update_phase()
        self._update_leading_option()

        selected = self._turn_mgr.select_speakers(
            self.sims, self.history, self.state, max_speakers=self._max_speakers()
        )

        all_names = [s.name for s in self.sims]
        active = False

        for sim in selected:
            forced_adapt = sim.name in self.state.nudged_participants
            text = sim.generate_turn(
                self.history,
                self.state,
                all_names=all_names,
                forced_adaptation=forced_adapt,
            )
            if text and "[SILENCE]" not in text.upper():
                reason = "forced" if sim.name == self.state.last_addressed else "weighted"
                self._store_line(f"{sim.name}: {text}", selected_reason=reason)
                active = True
                self.state.nudged_participants.discard(sim.name)

        self._update_discourse()
        self._update_repetition()
        return active

    # ------------------------------------------------------------------
    # Conclusion helpers
    # ------------------------------------------------------------------

    def _run_open_closure(self) -> None:
        self.state.phase = "closing"
        all_names = [s.name for s in self.sims]
        primary = self._primary_sim()
        others = [s for s in self.sims if s is not primary]
        ordered = ([primary] if primary else []) + others

        for sim in ordered:
            text = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="closure")

        if self.moderator_style != "passive":
            self._store_moderator("Thanks for sharing — good conversation.")

    def _run_confirmation(self) -> None:
        self.state.phase = "confirmation"
        preferred = self.state.preferred_option
        if preferred is None:
            return

        if self.moderator_style == "active":
            backup = self.state.backup_option
            backup_note = f", with Option {backup} as backup" if backup else ""
            self._store_moderator(
                f"It sounds like Option {preferred} is the preferred choice{backup_note}. "
                "Can everyone confirm briefly?"
            )

        selected = self._turn_mgr.select_speakers(
            self.sims, self.history, self.state, max_speakers=min(2, len(self.sims))
        )
        all_names = [s.name for s in self.sims]
        for sim in selected:
            text = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="confirmation")

        rejection_signals = [
            "no,", "no.", "not quite", "not yet", "still weighing",
            "not sure", "don't agree", "disagree", "not ready",
        ]
        checked = 0
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            if any(sig in msg.lower() for sig in rejection_signals):
                self.state.agreement_reached = False
                self.state.preferred_option = None
                self.state.stall_rounds = 0
                return
            checked += 1
            if checked >= len(self.sims) * 2:
                break

    def _run_closure(self) -> None:
        self.state.phase = "closure"
        primary = self._primary_sim()
        others = [s for s in self.sims if s is not primary]
        candidates = ([primary] if primary else []) + ([random.choice(others)] if others else [])
        all_names = [s.name for s in self.sims]
        for sim in candidates[:2]:
            text = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="closure")

    def _conclude(self, option: str, backup: Optional[str] = None) -> None:
        self.state.preferred_option = option
        self.state.backup_option = backup
        self.state.agreement_reached = True
        self._run_confirmation()
        if self.state.agreement_reached:
            self._run_closure()
            if self.moderator_style != "passive":
                backup_note = f", with Option {backup} as backup" if backup else ""
                self._store_moderator(
                    f"Agreed — Option {option} is the final choice{backup_note}. Discussion concluded."
                )

    def _force_conclusion(self) -> None:
        """
        Force-close when stall escalation reaches level 3 and LLM finds no consensus.
        Priority: primary's preference → leading option → alphabetically first voted option.
        """
        primary = self._primary_sim()
        primary_pref = None
        if primary:
            turns = self._last_n_turns_for(primary.name, n=3)
            for t in turns:
                letters = self._extract_option_letters(t)
                if letters:
                    primary_pref = letters[0]
                    break

        votes = self._current_votes()
        final = (
            primary_pref
            or self.state.current_leading_option
            or (sorted(votes.values())[0] if votes else None)
        )

        if final and self.moderator_style != "passive":
            self.state.preferred_option = final
            self._store_moderator(
                f"We've spent a lot of time on this without reaching agreement. "
                f"I'm going to call it — Option {final} is our final choice. Discussion concluded."
            )
        elif self.moderator_style != "passive":
            self._store_moderator("No clear agreement reached. Discussion concluded.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(self) -> None:
        self._detector = ConsensusDetector(self.sims, self.options, self.moderator_style)

        self._logger.write_header(
            participant_names=[s.name for s in self.sims],
            opening_lines=self.history,
        )
        for line in self.history:
            self._logger.buffer(line, "moderator", self.state, self.sims)

        print(f"\n--- Dialogue started (mode: {self.mode}, moderator: {self.moderator_style}) ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        try:
            for _ in range(cfg.turns.hard_ceiling):
                self.state.turn_index += 1

                active = self._run_participant_round()
                if not active:
                    if self.moderator_style != "passive":
                        self._store_moderator("No further responses. Discussion concluded.")
                    break

                # Open mode: wind down naturally, no consensus logic
                if self.state.mode == "open":
                    if self.state.phase == "closing":
                        self._run_open_closure()
                        break
                    intervention = self._should_intervene()
                    if intervention:
                        self._run_moderator_intervention(intervention)
                    continue

                # ── Decision mode ──────────────────────────────────────────

                # Track rounds after narrowing for escalation
                if self.state.has_asked_narrowing:
                    self.state.post_narrowing_rounds += 1

                # 1. Check for natural consensus
                consensus = self._detector.detect(self.history, self.state)
                if consensus:
                    self._conclude(*consensus)
                    if self.state.agreement_reached:
                        break
                    continue

                # 2. Prompt for narrowing if not yet done
                if self._should_narrow():
                    self._narrowing_prompt()
                    continue

                # 3. Post-narrowing stall and deadlock handling
                if self.state.has_asked_narrowing:

                    # Stall counter (word-overlap based)
                    if self.state.repetition_pressure >= 0.75:
                        self.state.stall_rounds += 1
                    else:
                        self.state.stall_rounds = 0

                    level = self._escalation_level()

                    # Level 3: force-close immediately
                    if level >= 3:
                        forced = self._detector.llm_check(self.history)
                        if forced:
                            self._conclude(*forced)
                        else:
                            self._force_conclusion()
                        break

                    # Split deadlock detected + semantic stall: escalate faster
                    if self._is_split_deadlock() and self._any_sim_stuck():
                        stall_limit = max(1, cfg.consensus.stall_rounds_to_force.get(
                            self.moderator_style, 2
                        ) - 1)
                    else:
                        stall_limit = cfg.consensus.stall_rounds_to_force.get(
                            self.moderator_style, 2
                        )

                    if self.state.stall_rounds >= stall_limit:
                        forced = self._detector.llm_check(self.history)
                        if forced:
                            self._conclude(*forced)
                            if self.state.agreement_reached:
                                break
                            continue
                        # LLM found nothing: fire a deadlock intervention instead of
                        # immediately force-closing (reserve that for level 3)
                        self._run_moderator_intervention("stall")
                        self.state.stall_rounds = 0
                        continue

                # 4. Regular interventions
                intervention = self._should_intervene()
                if intervention:
                    self._run_moderator_intervention(intervention)

            else:
                # Hard ceiling hit
                if self.moderator_style != "passive":
                    forced = self._detector.llm_check(self.history)
                    if forced:
                        self._conclude(*forced)
                    else:
                        self._force_conclusion()

        finally:
            self._logger.flush()
            txt, csv_path = self._logger.paths
            print(f"\n[Saved: {txt} | {csv_path}]")