"""
orchestrator.py
---------------
Orchestrator — coordinates a single dialogue run.

Responsibilities:
  1. Setup       — generate options + opening history (LLM)
  2. State       — phase, leading option, turn counts
  3. Main loop   — drive rounds, detect consensus, fire moderator lines
  4. Conclusion  — narrowing, confirmation, closure

Detection  → consensus.py
Moderation → moderation.py   (escalation, interventions, speculative-loop/outlier detection)
Turn logic → turn_manager.py
Logging    → logger.py
"""

from __future__ import annotations

import datetime
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import prompts
from config_loader import cfg
from consensus import ConsensusDetector
from logger import DialogueLogger
from llm_client import get_llm_client
from moderation import ModerationEngine
from turn_manager import TurnManager
from utils import (
    extract_preference_vote,
    extract_option_letters,
    participant_turn_count,
    last_n_turns_for,
    current_votes,
)


# ---------------------------------------------------------------------------
# Dialogue state
# ---------------------------------------------------------------------------

@dataclass
class DialogueState:
    phase: str = "greeting"
    turn_index: int = 0

    has_asked_narrowing: bool = False
    agreement_reached: bool = False

    preferred_option: Optional[str] = None
    backup_option: Optional[str] = None
    current_leading_option: Optional[str] = None

    last_addressed: Optional[str] = None
    pending_question_target: Optional[str] = None

    repetition_pressure: float = 0.0

    stall_rounds: int = 0
    post_narrowing_rounds: int = 0
    llm_check_countdown: int = 0

    last_rejected_option: Optional[str] = None
    last_rejecting_speaker: Optional[str] = None
    rejected_options_by_speaker: dict = field(default_factory=dict)
    consensus_cooldown: int = 0

    clarification_topics_used: set[str] = field(default_factory=set)
    nudged_participants: set[str] = field(default_factory=set)

    priority_next_speaker: Optional[str] = None

    vote_changes: dict = field(default_factory=dict)
    last_known_vote: dict = field(default_factory=dict)

    has_entered_emergence: bool = False
    emergence_rounds: int = 0
    candidate_option: Optional[str] = None

    compromise_option: Optional[str] = None
    compromise_tested: bool = False
    compromise_confirmation_tried: bool = False
    compromise_rounds: int = 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:

    def __init__(self, topic: str, moderator_style: str = "active") -> None:
        self.topic = topic
        self.moderator_style = moderator_style.lower()
        self.sims: list[Any] = []

        self._llm = get_llm_client()
        self._turn_mgr = TurnManager()
        self.state = DialogueState()
        self.state.llm_check_countdown = cfg.consensus.llm_check_every_n_turns

        self.dialogue_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.outcome: str = "pending"

        self.options, self.opening_question = self._generate_options()
        self.history: list[str] = self._build_opening_history()

        self._logger = DialogueLogger(self.dialogue_id, topic, moderator_style)
        self._detector: ConsensusDetector = None  # type: ignore[assignment]
        self._mod: ModerationEngine = None  # type: ignore[assignment]

    def add_sim(self, sim: Any) -> None:
        self.sims.append(sim)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _generate_options(self) -> tuple[list[str], str]:
        fallback_options = [
            "Option A - Budget: lowest cost and simplest path; trade-off: fewer extras; best for: minimizing risk.",
            "Option B - Convenience: easier or faster to execute; trade-off: moderate cost; best for: reducing friction.",
            "Option C - Quality: strongest expected outcome; trade-off: higher effort or cost; best for: long-term value.",
            "Option D - Flexible: adaptable middle ground; trade-off: less specialized; best for: keeping options open.",
        ]
        fallback_q = "Before anyone picks, what matters most to you here?"

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

    def _speaker_votes(self, speaker: str) -> tuple[Optional[str], Optional[str]]:
        """Return (primary_vote, backup_vote) for a speaker by scanning history."""
        primary: Optional[str] = None
        backup: Optional[str] = None
        for line in reversed(self.history):
            if ":" not in line:
                continue
            spk, msg = line.split(":", 1)
            if spk.strip() != speaker:
                continue
            vote = extract_preference_vote(msg)
            if vote:
                if primary is None:
                    primary = vote
                elif vote != primary and backup is None:
                    backup = vote
            if primary and backup:
                break
        return primary, backup

    def _annotate_with_vote(self, line: str) -> str:
        """Append the speaker's current vote(s) in parentheses for display."""
        if ":" not in line:
            return line
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        if speaker in cfg.EXCLUDED_SPEAKERS:
            return line
        primary, backup = self._speaker_votes(speaker)
        if primary and backup:
            tag = f"({primary}, {backup})"
        elif primary:
            tag = f"({primary})"
        else:
            return line
        return f"{speaker} {tag}:{text}"

    def _store_line(
        self,
        line: str,
        selected_reason: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        self.history.append(line)
        # Annotated version for console only; .txt receives the clean transcript
        display = self._annotate_with_vote(line)
        print(f"-> {display}")
        self._logger.append_line(line)
        self._logger.buffer(line, selected_reason, self.state, self.sims,
                            tokens_in=tokens_in, tokens_out=tokens_out)

    def _store_moderator(
        self, text: str, tokens_in: int = 0, tokens_out: int = 0
    ) -> None:
        self._store_line(f"Moderator: {text}", selected_reason="moderator",
                         tokens_in=tokens_in, tokens_out=tokens_out)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

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
            mentions.extend(extract_option_letters(msg))
            if len(mentions) >= limit:
                break
        if mentions:
            self.state.current_leading_option = Counter(mentions).most_common(1)[0][0]

        # Track candidate_option from actual votes — more reliable than mention frequency
        votes = current_votes(self.history, self.sims)
        if votes:
            vote_counts = Counter(votes.values())
            self.state.candidate_option = vote_counts.most_common(1)[0][0]

    def _update_phase(self) -> None:
        if self.state.agreement_reached:
            self.state.phase = "closure"
            return

        turns = participant_turn_count(self.history)
        n = len(self.sims)

        # greeting: first full round (all sims say hello once)
        # opening:  second round (first instincts on the topic)
        # negotiation → narrowing → emergence (all voted, no consensus) → confirmation
        if turns < n:
            self.state.phase = "greeting"
        elif turns < n * 2:
            self.state.phase = "opening"
        elif self.state.has_entered_emergence:
            self.state.phase = "emergence"
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
    # Stall / deadlock helpers
    # ------------------------------------------------------------------

    def _is_split_deadlock(self) -> bool:
        if not self.state.has_asked_narrowing:
            return False
        votes = current_votes(self.history, self.sims)
        if len(votes) < len(self.sims):
            return False
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
        turns = last_n_turns_for(name, self.history, n=window)
        if len(turns) < window:
            return False
        options_per_turn = [extract_option_letters(t) for t in turns]
        if not all(opts for opts in options_per_turn):
            return False
        first = options_per_turn[0][0]
        return all(opts[0] == first for opts in options_per_turn)

    def _any_sim_stuck(self) -> bool:
        return any(self._sim_vote_is_stuck(s.name) for s in self.sims)

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _max_speakers(self) -> int:
        phase = self.state.phase
        n = len(self.sims)
        if phase in ("greeting", "opening", "closure"):
            return 1
        if self.state.repetition_pressure >= 0.65:
            return 1
        if phase in ("confirmation", "emergence"):
            return min(2, n)
        weights = [0.30, 0.50, 0.20] if n >= 3 else [0.45, 0.55]
        choices = list(range(1, min(4, n + 1)))
        return random.choices(choices, weights=weights[: len(choices)])[0]

    def _run_participant_round(self) -> bool:
        """Run one round of participant turns. Returns True if any sim spoke."""
        priority_name = self.state.priority_next_speaker
        self.state.priority_next_speaker = None

        self._update_discourse()
        self._update_repetition()
        self._update_phase()
        self._update_leading_option()

        selected = self._turn_mgr.select_speakers(
            self.sims, self.history, self.state, max_speakers=self._max_speakers()
        )

        if priority_name:
            priority_sim = next((s for s in self.sims if s.name == priority_name), None)
            if priority_sim:
                others = [s for s in selected if s.name != priority_name]
                selected = [priority_sim] + others

        all_names = [s.name for s in self.sims]
        active = False

        for sim in selected:
            forced_adapt = sim.name in self.state.nudged_participants
            text, tok_in, tok_out = sim.generate_turn(
                self.history,
                self.state,
                all_names=all_names,
                forced_adaptation=forced_adapt,
            )
            if text and "[SILENCE]" not in text.upper():
                reason = "forced" if sim.name == self.state.last_addressed else "weighted"
                self._store_line(f"{sim.name}: {text}", selected_reason=reason,
                                 tokens_in=tok_in, tokens_out=tok_out)
                active = True
                self.state.nudged_participants.discard(sim.name)

            if self.state.phase == "narrowing":
                votes = current_votes(self.history, self.sims)
                if len(votes) >= len(self.sims):
                    self.state.has_entered_emergence = True
                    break

        self._update_discourse()
        self._update_repetition()
        self._track_vote_flips()

        return active

    def _track_vote_flips(self) -> None:
        for sim in self.sims:
            turns = last_n_turns_for(sim.name, self.history, n=1)
            if not turns:
                continue
            current_vote = extract_preference_vote(turns[0])
            if not current_vote:
                continue

            # Use belief preferred as the baseline — first text vote that matches belief
            # is not a flip; first text vote that differs from belief is flip #1.
            belief_preferred = sim.persona.beliefs.preferred if sim.persona.beliefs else None
            previous_vote = self.state.last_known_vote.get(sim.name, belief_preferred)

            if previous_vote is None:
                self.state.last_known_vote[sim.name] = current_vote
            elif current_vote != previous_vote:
                flips = self.state.vote_changes.get(sim.name, 0) + 1
                self.state.vote_changes[sim.name] = flips
                self.state.last_known_vote[sim.name] = current_vote
                print(f"[vote-flip] {sim.name}: {previous_vote} → {current_vote} (flip #{flips})")
            else:
                self.state.last_known_vote[sim.name] = current_vote

            if self.state.rejected_options_by_speaker.get(sim.name) == current_vote:
                self.state.rejected_options_by_speaker.pop(sim.name, None)

    # ------------------------------------------------------------------
    # Conclusion helpers
    # ------------------------------------------------------------------

    def _narrowing_prompt(self) -> None:
        self.state.has_asked_narrowing = True
        self.state.phase = "narrowing"
        self._store_moderator(
            "Let's narrow this down — which option does each of you prefer? "
            "A backup is fine if you're genuinely unsure."
        )

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
        confirmation_speakers: set[str] = set()

        for sim in selected:
            text, tok_in, tok_out = sim.generate_turn(
                self.history, self.state, all_names=all_names
            )
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="confirmation",
                                 tokens_in=tok_in, tokens_out=tok_out)
                confirmation_speakers.add(sim.name)

        if self.moderator_style == "active":
            for sim in self.sims:
                if sim.name not in confirmation_speakers:
                    self._store_moderator(f"{sim.name}, does Option {self.state.preferred_option} work for you?")
                    text, tok_in, tok_out = sim.generate_turn(
                        self.history, self.state, all_names=all_names
                    )
                    if text and "[SILENCE]" not in text.upper():
                        self._store_line(f"{sim.name}: {text}", selected_reason="confirmation",
                                         tokens_in=tok_in, tokens_out=tok_out)

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
            msg_lower = msg.strip().lower().rstrip("!?.")
            bare_no = msg_lower in {"no", "nope", "nah"}
            if bare_no or any(sig in msg.lower() for sig in rejection_signals):
                self.state.last_rejected_option = preferred
                self.state.last_rejecting_speaker = speaker.strip()
                self.state.rejected_options_by_speaker[speaker.strip()] = preferred
                self.state.priority_next_speaker = speaker.strip()
                self.state.consensus_cooldown = 4
                self.state.agreement_reached = False
                self.state.preferred_option = None
                self.state.stall_rounds = 0
                if self.moderator_style != "passive":
                    self._store_moderator(
                        f"Okay, Option {preferred} doesn't have full buy-in yet. "
                        f"{speaker.strip()}, what is the main blocker for you?"
                    )
                return
            checked += 1
            if checked >= len(self.sims) * 2:
                break

    def _best_compromise_option(self) -> Optional[str]:
        """
        Pick the option most worth testing aloud before force-close.
        Uses private acceptability only to choose what to test; the dialogue
        still has to make that compromise visible.
        """
        scores: dict[str, float] = {opt: 0.0 for opt in ["A", "B", "C", "D"]}
        votes = current_votes(self.history, self.sims)

        for sim in self.sims:
            b = sim.persona.beliefs
            if not b:
                continue
            scores[b.preferred] += 2.0
            for opt in b.acceptable:
                scores[opt] += 1.0
            for opt in b.rejected:
                scores[opt] -= 2.0

        for opt in votes.values():
            scores[opt] += 0.5

        for opt in self.state.rejected_options_by_speaker.values():
            scores[opt] -= 4.0

        primary = self._primary_sim()
        if primary:
            primary_rejected = self.state.rejected_options_by_speaker.get(primary.name)
            if primary_rejected:
                scores[primary_rejected] -= 3.0

        best_score = max(scores.values())
        rejected_in_dialogue = set(self.state.rejected_options_by_speaker.values())
        candidates = [
            opt for opt, score in scores.items()
            if score == best_score and opt not in rejected_in_dialogue
        ]

        # Prefer options that received actual votes over those that only scored via private beliefs
        voted_options = set(votes.values())
        if voted_options:
            voted_candidates = [c for c in candidates if c in voted_options]
            if voted_candidates:
                candidates = voted_candidates

        if self.state.candidate_option in candidates:
            return self.state.candidate_option
        if self.state.current_leading_option in candidates:
            return self.state.current_leading_option
        return candidates[0] if candidates else None

    def _test_compromise(self, option: str) -> None:
        """Ask the group to visibly test a compromise before any forced decision."""
        self.state.has_entered_emergence = True
        self.state.phase = "emergence"
        self.state.candidate_option = option
        self.state.compromise_option = option
        self.state.compromise_tested = True
        self.state.compromise_rounds = 0

        votes = current_votes(self.history, self.sims)
        holdouts = [s.name for s in self.sims if votes.get(s.name) != option]

        try:
            line = self._llm.generate(
                prompts.moderator_compromise_test(
                    topic=self.topic,
                    participant_names=[s.name for s in self.sims],
                    options=self.options,
                    recent_dialogue="\n".join(self.history[-10:]),
                    compromise_option=option,
                    holdout_names=holdouts,
                )
            ).strip()
            self._store_moderator(
                line or f"Before I make any call, could Option {option} work as a compromise, and what would need to be true?",
                tokens_in=self._llm.last_tokens_in,
                tokens_out=self._llm.last_tokens_out,
            )
        except Exception:
            self._store_moderator(
                f"Before I make any call, could Option {option} work as a compromise, and what would need to be true?"
            )

        if holdouts:
            self.state.priority_next_speaker = holdouts[0]

    def _run_closure(self) -> None:
        self.state.phase = "closure"
        primary = self._primary_sim()
        others = [s for s in self.sims if s is not primary]
        ordered = ([primary] if primary else []) + others
        all_names = [s.name for s in self.sims]
        for sim in ordered:
            text, tok_in, tok_out = sim.generate_turn(
                self.history, self.state, all_names=all_names
            )
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="closure",
                                 tokens_in=tok_in, tokens_out=tok_out)

    def _conclude(self, option: str, backup: Optional[str] = None) -> None:
        self.state.preferred_option = option
        self.state.backup_option = backup
        self.state.agreement_reached = True
        self._run_confirmation()
        if self.state.agreement_reached:
            self.outcome = "success"
            if self.moderator_style != "passive":
                backup_note = f", with Option {backup} as backup" if backup else ""
                self._store_moderator(
                    f"Agreed — Option {option} is the final choice{backup_note}. Discussion concluded."
                )
            self._run_closure()

    def _force_conclusion(self) -> None:
        """
        Select the best option using Fisher's method of residues:
        the option with the lowest net resistance (acceptance score minus objection score).
        Primary participant's preference breaks ties among equally-scored options only.
        """
        acceptance: dict[str, float] = {opt: 0.0 for opt in ["A", "B", "C", "D"]}
        objections: dict[str, float] = {opt: 0.0 for opt in ["A", "B", "C", "D"]}

        for sim in self.sims:
            b = sim.persona.beliefs
            if not b:
                continue
            acceptance[b.preferred] += 2.0
            for opt in b.acceptable:
                if opt != b.preferred:
                    acceptance[opt] += 1.0
            for opt in b.rejected:
                objections[opt] += 2.0

        votes = current_votes(self.history, self.sims)
        vote_counts = Counter(votes.values())

        # Actual participant votes dominate the scoring — each vote is worth 5 points.
        # This ensures vote plurality (e.g. 2 votes for A) beats private-belief scoring
        # even when one participant dialogue-rejected A during confirmation.
        for opt, count in vote_counts.items():
            acceptance[opt] += count * 5.0

        for opt in self.state.rejected_options_by_speaker.values():
            objections[opt] += 3.0

        net: dict[str, float] = {opt: acceptance[opt] - objections[opt] for opt in ["A", "B", "C", "D"]}

        primary = self._primary_sim()
        primary_beliefs_pref = primary.persona.beliefs.preferred if (primary and primary.persona.beliefs) else None
        primary_vote = votes.get(primary.name) if primary else None

        # Hard-restrict to options that received actual votes; fall back to
        # participant-mentioned options; last resort is all options.
        voted_options = set(votes.values())
        if voted_options:
            available_net = {opt: sc for opt, sc in net.items() if opt in voted_options}
        else:
            mentioned: set[str] = set()
            for line in self.history:
                if ":" not in line:
                    continue
                spk, msg = line.split(":", 1)
                if spk.strip() not in cfg.EXCLUDED_SPEAKERS:
                    mentioned.update(extract_option_letters(msg))
            available_net = {opt: sc for opt, sc in net.items() if opt in mentioned} if mentioned else net

        best_score = max(available_net.values())
        top_opts = [opt for opt, sc in available_net.items() if sc >= best_score - 0.5]

        # Primary preference is tiebreaker among equally-scored options, not against better ones
        if primary_beliefs_pref and primary_beliefs_pref in top_opts:
            final = primary_beliefs_pref
        elif primary_vote and primary_vote in top_opts:
            final = primary_vote
        elif self.state.candidate_option in top_opts:
            final = self.state.candidate_option
        elif self.state.current_leading_option in top_opts:
            final = self.state.current_leading_option
        else:
            final = top_opts[0]

        self.outcome = "force_close"
        if self.moderator_style != "passive":
            self.state.preferred_option = final

            if final == primary_beliefs_pref or final == primary_vote:
                reason = (f"the primary participant's ({primary.name}'s) preference"
                          if primary else "the primary participant's preference")
            else:
                reason = "the option with the broadest group acceptance and fewest active objections"

            recent = "\n".join(self.history[-6:])
            try:
                line = self._llm.generate(
                    prompts.moderator_force_close(
                        topic=self.topic,
                        participant_names=[s.name for s in self.sims],
                        final_option=final,
                        reason=reason,
                        recent_dialogue=recent,
                    )
                ).strip()
                self._store_moderator(
                    line or f"No consensus reached — I'll call Option {final} as the moderator's choice. Let's wrap up.",
                    tokens_in=self._llm.last_tokens_in,
                    tokens_out=self._llm.last_tokens_out,
                )
            except Exception:
                self._store_moderator(
                    f"We didn't reach agreement, so I'm making the call: Option {final}. Discussion concluded."
                )

        self._run_closure()

    def _fresh_unanswered_question(self) -> bool:
        """True when the latest participant line asks something that deserves air."""
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in cfg.EXCLUDED_SPEAKERS:
                continue
            if "?" not in msg:
                return False
            # If the same speaker already had another turn after this, it is no
            # longer the fresh thing on the table. Since we scan latest-first,
            # getting here means the latest participant line is the question.
            others = [s.name for s in self.sims if s.name != speaker]
            if others:
                self.state.priority_next_speaker = random.choice(others)
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(
        self, setup_tokens_in: int = 0, setup_tokens_out: int = 0
    ) -> None:
        self._detector = ConsensusDetector(self.sims, self.options, self.moderator_style)
        self._mod = ModerationEngine(
            self.topic, self.options, self.moderator_style, self.sims
        )

        self._logger.write_header(
            participant_names=[s.name for s in self.sims],
            opening_lines=self.history,
        )
        for line in self.history:
            self._logger.buffer(line, "moderator", self.state, self.sims)

        print(f"\n--- Dialogue started (moderator: {self.moderator_style}) ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        def _intervene(reason: str) -> None:
            self._mod.run_intervention(
                reason, self.state, self.history,
                lambda t, ti, to: self._store_moderator(t, ti, to),
            )

        try:
            for _ in range(cfg.turns.hard_ceiling):
                self.state.turn_index += 1

                active = self._run_participant_round()
                if not active:
                    self.outcome = "failed"
                    if self.moderator_style != "passive":
                        self._store_moderator("No further responses. Discussion concluded.")
                    break

                # ── Decision mode ──────────────────────────────────────────

                if self.state.has_asked_narrowing:
                    self.state.post_narrowing_rounds += 1

                # Enter emergence when all participants have voted but no consensus yet
                if self.state.has_asked_narrowing and not self.state.has_entered_emergence:
                    _votes = current_votes(self.history, self.sims)
                    if len(_votes) >= len(self.sims):
                        self.state.has_entered_emergence = True

                if self.state.has_entered_emergence:
                    self.state.emergence_rounds += 1
                if self.state.compromise_option:
                    self.state.compromise_rounds += 1

                # 1. Check for natural consensus
                if self.state.consensus_cooldown > 0:
                    self.state.consensus_cooldown -= 1
                consensus = self._detector.detect(self.history, self.state)
                if (consensus
                        and self.state.consensus_cooldown > 0
                        and consensus[0] == self.state.last_rejected_option):
                    consensus = None
                if consensus and consensus[0] in set(self.state.rejected_options_by_speaker.values()):
                    consensus = None
                if consensus:
                    self._conclude(*consensus)
                    if self.state.agreement_reached:
                        break
                    continue

                # 2. Prompt for narrowing if not yet done
                ptc = participant_turn_count(self.history)
                if self._mod.should_narrow(self.state, ptc):
                    if self._fresh_unanswered_question():
                        continue
                    self._narrowing_prompt()
                    continue

                # 3. Post-narrowing stall and deadlock handling
                if self.state.has_asked_narrowing:

                    if self.state.repetition_pressure >= 0.75:
                        self.state.stall_rounds += 1
                    else:
                        self.state.stall_rounds = 0

                    level = self._mod.escalation_level(self.state)

                    if level >= 3:
                        if not self.state.compromise_tested:
                            compromise = self._best_compromise_option()
                            if compromise:
                                self._test_compromise(compromise)
                                continue
                        if self.state.compromise_tested and self.state.compromise_rounds < max(2, len(self.sims) - 1):
                            _intervene("compromise")
                            continue
                        if (self.state.compromise_option
                                and not self.state.compromise_confirmation_tried):
                            self.state.compromise_confirmation_tried = True
                            self._conclude(self.state.compromise_option)
                            if self.state.agreement_reached:
                                break
                            continue
                        forced = self._detector.llm_check(self.history)
                        if forced:
                            self._conclude(*forced)
                        else:
                            self._force_conclusion()
                        break

                    if self._is_split_deadlock() and self._any_sim_stuck():
                        stall_limit = max(1, cfg.consensus.stall_rounds_to_force.get(
                            self.moderator_style, 2
                        ) - 1)
                    else:
                        stall_limit = cfg.consensus.stall_rounds_to_force.get(
                            self.moderator_style, 2
                        )

                    if self.state.stall_rounds >= stall_limit:
                        if not self.state.compromise_tested:
                            compromise = self._best_compromise_option()
                            if compromise:
                                self._test_compromise(compromise)
                                self.state.stall_rounds = 0
                                continue
                        if (self.state.compromise_option
                                and self.state.compromise_rounds >= max(2, len(self.sims) - 1)
                                and not self.state.compromise_confirmation_tried):
                            self.state.compromise_confirmation_tried = True
                            self._conclude(self.state.compromise_option)
                            if self.state.agreement_reached:
                                break
                            continue
                        forced = self._detector.llm_check(self.history)
                        if forced:
                            self._conclude(*forced)
                            if self.state.agreement_reached:
                                break
                            continue
                        _intervene("stall")
                        self.state.stall_rounds = 0
                        # Direct the minority voter to respond to the mod's question first
                        stall_votes = current_votes(self.history, self.sims)
                        if stall_votes:
                            top_opt = Counter(stall_votes.values()).most_common(1)[0][0]
                            for _sim in self.sims:
                                if stall_votes.get(_sim.name) != top_opt:
                                    self.state.priority_next_speaker = _sim.name
                                    break
                        continue

                # 4. Regular interventions
                ptc = participant_turn_count(self.history)
                intervention = self._mod.should_intervene(
                    self.state, self.history, self._any_sim_stuck(), ptc
                )
                if intervention:
                    if intervention == "stall" and self._fresh_unanswered_question():
                        continue
                    _intervene(intervention)
                    if intervention.startswith("outlier:"):
                        self.state.priority_next_speaker = intervention.split(":", 1)[1]

            else:
                # Hard ceiling hit
                if self.moderator_style != "passive":
                    if not self.state.compromise_tested:
                        compromise = self._best_compromise_option()
                        if compromise:
                            self._test_compromise(compromise)
                    forced = self._detector.llm_check(self.history)
                    if forced:
                        self._conclude(*forced)
                    else:
                        self._force_conclusion()

        finally:
            dialogue_tokens_in = self._llm.session_tokens_in
            dialogue_tokens_out = self._llm.session_tokens_out
            self._logger.flush(
                setup_tokens_in, setup_tokens_out,
                dialogue_tokens_in, dialogue_tokens_out,
                outcome=self.outcome,
            )
            txt, csv_path = self._logger.paths
            total_in = setup_tokens_in + dialogue_tokens_in
            total_out = setup_tokens_out + dialogue_tokens_out
            print(f"\n[Tokens]  setup={setup_tokens_in}/{setup_tokens_out}"
                  f"  dialogue={dialogue_tokens_in}/{dialogue_tokens_out}"
                  f"  total={total_in}/{total_out}  (in/out)")
            print(f"[Outcome] {self.outcome}")
            print(f"[Saved]   {txt} | {csv_path}")
