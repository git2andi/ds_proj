"""
modules/orchestrator.py
-----------------------
Orchestrator — coordinates a single dialogue run.

Responsibilities:
  1. Setup       — generate options and opening history via LLM
  2. State        — track phase, leading option, primary participant helpers
  3. Moderator   — narrowing, targeted interventions, confirmation, closure
  4. Main loop   — drive turns, delegate to ConsensusDetector and DialogueLogger

Detection logic  →  modules/consensus_detector.py
Logging          →  modules/dialogue_logger.py
"""

from __future__ import annotations

import datetime
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import prompts as prompts
from config_loader import cfg
from modules.consensus_detector import ConsensusDetector
from modules.dialogue_logger import DialogueLogger
from modules.llm_client import get_llm_client
from modules.turn_manager import TurnManager


# ---------------------------------------------------------------------------
# Dialogue state
# ---------------------------------------------------------------------------

@dataclass
class DialogueState:
    phase: str = "opening"
    turn_index: int = 0

    has_asked_narrowing: bool = False
    agreement_reached: bool = False

    preferred_option: Optional[str] = None
    backup_option: Optional[str] = None
    current_leading_option: Optional[str] = None

    last_addressed: Optional[str] = None
    pending_question_target: Optional[str] = None
    pending_reply_target: Optional[str] = None

    repetition_pressure: float = 0.0
    important_events: list[str] = field(default_factory=list)

    stall_rounds: int = 0
    turns_in_phase: int = 0
    llm_check_countdown: int = 0

    # Topics the moderator has already clarified — prevents re-firing.
    clarification_topics_used: set[str] = field(default_factory=set)

    # Participants who have already been nudged — their next turn gets a
    # hard adaptation instruction injected into the sim prompt.
    nudged_participants: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:

    def __init__(self, topic: str, moderator_style: str = "active") -> None:
        self.topic = topic
        self.moderator_style = moderator_style.lower()
        self.sims: list[Any] = []

        self._llm = get_llm_client()
        self._turn_manager = TurnManager()
        self.state = DialogueState()

        self.dialogue_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.options, self.opening_question = self._generate_options()
        self.history: list[str] = self._build_opening_history()

        self._logger = DialogueLogger(self.dialogue_id, topic, moderator_style)
        self._detector: Optional[ConsensusDetector] = None  # built after sims are added

    def add_sim(self, sim: Any) -> None:
        self.sims.append(sim)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _generate_options(self) -> tuple[list[str], str]:
        fallback_options = [
            "Option A - Budget choice: lowest cost, basic features, some trade-offs.",
            "Option B - Convenience choice: faster or easier, moderately priced.",
            "Option C - Quality choice: best comfort or outcome, higher cost.",
            "Option D - Flexible choice: mixed trade-offs, adaptable to needs.",
        ]
        fallback_question = "What matters most to you, and which option looks best right now?"

        try:
            data = self._llm.generate_json(prompts.option_generation(self.topic))
            options_raw = data.get("options", [])
            question = str(data.get("opening_question", "")).strip() or fallback_question

            if not isinstance(options_raw, list) or len(options_raw) != 4:
                return fallback_options, fallback_question

            cleaned: list[str] = []
            for i, raw in enumerate(options_raw):
                if not isinstance(raw, str) or not raw.strip():
                    return fallback_options, fallback_question
                label = chr(ord("A") + i)
                text = raw.strip()
                if not text.lower().startswith(f"option {label.lower()}"):
                    text = f"Option {label} - {text}"
                cleaned.append(text)

            return cleaned, question

        except Exception as exc:
            print(f"!! Option generation error: {exc}")
            return fallback_options, fallback_question

    def _build_opening_history(self) -> list[str]:
        lines = [f"Moderator: Let's discuss: {self.topic}"]
        lines.append("Moderator: Here are the options on the table:")
        lines.extend(f"Moderator: {opt}" for opt in self.options)
        lines.append(f"Moderator: {self.opening_question}")
        return lines

    # ------------------------------------------------------------------
    # Logging helpers (thin wrappers around DialogueLogger)
    # ------------------------------------------------------------------

    def _store_line(self, line: str, selected_reason: str = "") -> None:
        self.history.append(line)
        print(f"-> {line}\n")
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
            if ":" in line and line.split(":", 1)[0].strip() not in cfg.EXCLUDED_SPEAKERS
        )

    def _extract_option_letters(self, text: str) -> list[str]:
        return [m.upper() for m in re.findall(r"\boption\s+([a-d])\b", text.lower())]

    def _primary_sim(self) -> Optional[Any]:
        return next((s for s in self.sims if s.persona.get("is_primary", False)), None)

    def _primary_has_spoken_recently(self, window: int = 4) -> bool:
        primary = self._primary_sim()
        if primary is None:
            return True
        for line in reversed(self.history[-(window * 2):]):
            if ":" in line and line.split(":", 1)[0].strip() == primary.name:
                return True
        return False

    def _primary_stated_preference(self) -> Optional[str]:
        """Most recently stated option letter by the primary participant."""
        primary = self._primary_sim()
        if primary is None:
            return None
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == primary.name:
                mentions = self._extract_option_letters(msg)
                if mentions:
                    return mentions[0]
        return None

    def _update_leading_option(self) -> None:
        mentions: list[str] = []
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            mentions.extend(self._extract_option_letters(msg))
            if len(mentions) >= max(5, len(self.sims) * 2):
                break
        if mentions:
            self.state.current_leading_option = Counter(mentions).most_common(1)[0][0]

    def _unanimous_first_round(self) -> Optional[str]:
        """Return the option if every participant named the same one first."""
        if self._participant_turn_count() < len(self.sims):
            return None
        first_mentions: dict[str, str] = {}
        for line in self.history:
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in cfg.EXCLUDED_SPEAKERS or speaker in first_mentions:
                continue
            mentions = self._extract_option_letters(msg)
            if mentions:
                first_mentions[speaker] = mentions[0]
        if len(first_mentions) < len(self.sims):
            return None
        options_named = set(first_mentions.values())
        return options_named.pop() if len(options_named) == 1 else None

    def _update_phase(self) -> None:
        if self.state.agreement_reached:
            self.state.phase = "closure"
            return
        turns = self._participant_turn_count()
        n = len(self.sims)
        if turns == 0:
            self.state.phase = "opening"
        elif turns < n:
            self.state.phase = "preference_expression"
        elif self.state.has_asked_narrowing:
            self.state.phase = "narrowing"
        else:
            if not self.state.has_asked_narrowing:
                unanimous = self._unanimous_first_round()
                if unanimous:
                    self.state.has_asked_narrowing = True
                    self.state.phase = "narrowing"
                    self.state.current_leading_option = unanimous
                    return
            self.state.phase = "negotiation"

    # ------------------------------------------------------------------
    # Moderator — narrowing and targeted interventions
    # ------------------------------------------------------------------

    def _should_narrow(self) -> bool:
        if self.state.has_asked_narrowing or self.moderator_style == "passive":
            return False
        turns = self._participant_turn_count()
        n = len(self.sims)
        if turns < max(n * 2, cfg.turns.min_turns_before_narrowing):
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

    def _should_intervene(self) -> Optional[str]:
        """
        Return an intervention reason or None. Possible reasons:
          "clarify:{keyword}"     — speculative loop detected
          "primary_silent:{name}" — primary hasn't spoken recently
          "outlier:{name}"        — participant repeating verbatim
          "stall"                 — generic repetition stall
        """
        if self.moderator_style == "passive":
            return None
        if self._participant_turn_count() < len(self.sims):
            return None

        loop_topic = self._detector.speculative_loop(
            self.history, self.state.clarification_topics_used
        )
        if loop_topic:
            return f"clarify:{loop_topic}"

        if not self._primary_has_spoken_recently(window=4):
            primary = self._primary_sim()
            if primary:
                return f"primary_silent:{primary.name}"

        outlier = self._detector.persistent_outlier(
            self.history, self.state.has_asked_narrowing
        )
        if outlier:
            return f"outlier:{outlier}"

        if self.state.repetition_pressure >= 0.80 and self.state.stall_rounds >= 2:
            return "stall"

        return None

    def _run_moderator_intervention(self, reason: str) -> None:
        names = [s.name for s in self.sims]
        recent = "\n".join(self.history[-10:])

        try:
            if reason.startswith("clarify:"):
                keyword = reason.split(":", 1)[1]
                self.state.clarification_topics_used.add(keyword)
                line = self._llm.generate(
                    prompts.moderator_clarification(
                        topic=self.topic, participant_names=names,
                        options=self.options, recent_dialogue=recent,
                        looping_topic=keyword,
                    )
                ).strip()

            elif reason.startswith("primary_silent:"):
                name = reason.split(":", 1)[1]
                line = self._llm.generate(
                    prompts.moderator_intervention(
                        topic=self.topic, participant_names=names,
                        recent_dialogue=recent,
                        intervention_reason=f"{name} hasn't contributed recently",
                        quiet_participant=name,
                    )
                ).strip()

            elif reason.startswith("outlier:"):
                outlier_name = reason.split(":", 1)[1]
                # Mark so their next sim turn gets a hard adaptation instruction.
                self.state.nudged_participants.add(outlier_name)
                primary = self._primary_sim()
                primary_pref = self._primary_stated_preference()
                context = (
                    f"{primary.name} (the primary person this decision affects) "
                    f"has clearly stated a preference for Option {primary_pref}."
                    if primary and primary_pref else ""
                )
                line = self._llm.generate(
                    prompts.moderator_outlier_nudge(
                        topic=self.topic, participant_names=names,
                        recent_dialogue=recent, outlier_name=outlier_name,
                        primary_context=context,
                    )
                ).strip()

            else:  # stall
                line = self._llm.generate(
                    prompts.moderator_intervention(
                        topic=self.topic, participant_names=names,
                        recent_dialogue=recent,
                        intervention_reason="the discussion is repeating itself without progress",
                    )
                ).strip()

            if line:
                self._store_moderator(line)

        except Exception as exc:
            print(f"!! Moderator intervention error ({reason}): {exc}")

    # ------------------------------------------------------------------
    # Round execution and conclusion helpers
    # ------------------------------------------------------------------

    def _dynamic_max_speakers(self) -> int:
        phase = self.state.phase
        n = len(self.sims)
        if phase in ("opening", "closure"):
            return 1
        if self.state.repetition_pressure >= 0.65:
            return 1
        if phase == "confirmation":
            return min(2, n)
        weights = [0.25, 0.50, 0.25] if n >= 3 else [0.40, 0.60]
        choices = list(range(1, min(4, n + 1)))
        return random.choices(choices, weights=weights[:len(choices)])[0]

    def _run_participant_round(self) -> bool:
        self._turn_manager.extract_events(self.history, self.state, self.sims)
        self._update_leading_option()
        self._update_phase()

        max_speakers = self._dynamic_max_speakers()
        forced_names = self._turn_manager.forced_names(self.state)
        selected = self._turn_manager.select_speakers(
            self.sims, self.history, self.state, max_speakers=max_speakers
        )

        all_names = [s.name for s in self.sims]
        active = False
        for sim in selected:
            forced_adapt = sim.name in self.state.nudged_participants
            text = sim.generate_turn(
                self.history, self.state,
                all_names=all_names,
                forced_adaptation=forced_adapt,
            )
            if text and "[SILENCE]" not in text.upper():
                reason = "forced" if sim.name in forced_names else "weighted"
                self._store_line(f"{sim.name}: {text}", selected_reason=reason)
                active = True
                # Clear nudge after one use — the hard instruction has fired.
                self.state.nudged_participants.discard(sim.name)

        self._turn_manager.extract_events(self.history, self.state, self.sims)
        self._update_leading_option()
        return active

    def _confirmation_rejected(self) -> bool:
        rejection_signals = [
            "no,", "no.", "not quite", "not yet", "still weighing",
            "haven't decided", "not sure", "don't agree", "disagree",
            "not ready", "we're still", "we are still",
        ]
        checked = 0
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            if any(sig in msg.lower() for sig in rejection_signals):
                return True
            checked += 1
            if checked >= len(self.sims) * 2:
                break
        return False

    def _run_confirmation(self) -> None:
        self.state.phase = "confirmation"
        preferred = self.state.preferred_option
        if preferred is None:
            return

        if self.moderator_style == "active":
            backup = self.state.backup_option
            backup_note = f", with Option {backup} as a backup" if backup else ""
            self._store_moderator(
                f"It sounds like Option {preferred} is the preferred choice{backup_note}. "
                "Can everyone confirm briefly?"
            )

        self._turn_manager.extract_events(self.history, self.state, self.sims)
        selected = self._turn_manager.select_speakers(
            self.sims, self.history, self.state, max_speakers=min(2, len(self.sims))
        )
        all_names = [s.name for s in self.sims]
        forced_names = self._turn_manager.forced_names(self.state)
        for sim in selected:
            text = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                reason = "forced" if sim.name in forced_names else "weighted"
                self._store_line(f"{sim.name}: {text}", selected_reason=reason)

        if self._confirmation_rejected():
            self.state.agreement_reached = False
            self.state.preferred_option = None

    def _run_closure(self) -> None:
        """At most 2 sign-offs — primary first, then one other at random."""
        self.state.phase = "closure"
        self._turn_manager.extract_events(self.history, self.state, self.sims)

        primary = self._primary_sim()
        others = [s for s in self.sims if s is not primary]
        candidates = ([primary] if primary else []) + ([random.choice(others)] if others else [])

        all_names = [s.name for s in self.sims]
        for sim in candidates[:2]:
            text = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="closure")

    def _run_close_moderator(self) -> None:
        if self.moderator_style == "passive":
            return
        preferred = self.state.preferred_option
        backup = self.state.backup_option
        if preferred and backup:
            msg = f"Agreed — Option {preferred} is the final choice, with Option {backup} as backup. Discussion concluded."
        elif preferred:
            msg = f"Agreed — Option {preferred} is the final choice. Discussion concluded."
        else:
            msg = "Discussion concluded."
        self._store_moderator(msg)

    def _conclude(self, option: str, backup: Optional[str] = None) -> None:
        """Set agreement and run the closing sequence."""
        self.state.preferred_option = option
        self.state.backup_option = backup
        self.state.agreement_reached = True
        self._run_confirmation()
        if self.state.agreement_reached:
            self._run_closure()
            self._run_close_moderator()

    def _force_conclusion(self) -> None:
        """Called when stall limit is hit and LLM finds no consensus."""
        final = self._primary_stated_preference() or self.state.current_leading_option
        if final and self.moderator_style != "passive":
            self.state.preferred_option = final
            self._store_moderator(
                f"We've been going in circles. Given the group's priorities, "
                f"Option {final} will be our final choice. Discussion concluded."
            )
        elif self.moderator_style != "passive":
            self._store_moderator("No clear agreement reached. Discussion concluded.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(self) -> None:
        self._detector = ConsensusDetector(self.sims, self.options, self.moderator_style, topic=self.topic)

        self._logger.write_header(
            participant_names=[s.name for s in self.sims],
            opening_lines=self.history,
        )
        for line in self.history:
            self._logger.buffer(line, "moderator", self.state, self.sims)

        print(f"\n--- Simulation Started (moderator: {self.moderator_style}) ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        self.state.llm_check_countdown = cfg.consensus.llm_check_every_n_turns

        try:
            for _ in range(cfg.turns.hard_ceiling):
                self.state.turn_index += 1

                active = self._run_participant_round()
                if not active:
                    if self.moderator_style != "passive":
                        self._store_moderator("No further progress. Discussion concluded.")
                    return

                consensus = self._detector.detect(self.history, self.state)
                if consensus:
                    self._conclude(*consensus)
                    if self.state.agreement_reached:
                        return
                    continue  # Consensus was rejected — keep going.

                if self._should_narrow():
                    self._narrowing_prompt()
                    continue

                if self.state.has_asked_narrowing:
                    self.state.stall_rounds = (
                        self.state.stall_rounds + 1
                        if self.state.repetition_pressure >= 0.75 else 0
                    )
                    stall_limit = cfg.consensus.stall_rounds_to_force.get(self.moderator_style, 2)
                    if self.state.stall_rounds >= stall_limit:
                        forced = self._detector._llm_check(self.history)
                        if forced:
                            self._conclude(*forced)
                        else:
                            self._force_conclusion()
                        return

                intervention = self._should_intervene()
                if intervention:
                    self._run_moderator_intervention(intervention)

            if self.moderator_style != "passive":
                self._store_moderator("Maximum discussion length reached. Discussion concluded.")

        finally:
            self._logger.flush()
            print(f"\n[Logs saved: {self._logger.log_file} | {self._logger.csv_file}]")