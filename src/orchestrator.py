"""
orchestrator.py
---------------
Orchestrator -- coordinates a single dialogue run.

Responsibilities:
  1. Setup       generate options + opening (one LLM call)
  2. State       phase, leading option, turn counts
  3. Main loop   drive rounds, detect consensus, fire moderator lines
  4. Conclusion  narrowing, confirmation, closure

Single decision path (no A/B flags):
  - Speaker selection : policy.select_next_speakers (SSJ cascade)
  - Act planning      : policy.plan_turn
  - Consensus         : reasoning.ConsensusEngine (public stances only)
  - Phase signal      : reasoning.PhaseDetector  (Fisher ratios; informational)
  - Turn prompt       : prompts.sim_turn_compact via simulator
"""

from __future__ import annotations

import datetime
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from logger import DialogueLogger
from moderation import ModerationEngine, _clean_moderator_line
from policy import (
    plan_turn,
    repetition_pressure as compute_repetition_pressure,
    sample_hard_blockers,
    select_next_speakers,
)
from reasoning import ConsensusEngine, PhaseDetector
from state import StateTracker
from utils import (
    current_votes,
    extract_option_letters,
    extract_preference_vote,
    last_n_turns_for,
    participant_turn_count,
    recent_participant_lines,
)


# ---------------------------------------------------------------------------
# DialogueState
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

    last_rejected_option: Optional[str] = None
    last_rejecting_speaker: Optional[str] = None
    rejected_options_by_speaker: dict = field(default_factory=dict)
    consensus_cooldown: int = 0

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

    confirmation_rejection_count: int = 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:

    def __init__(self, topic: str, moderator_style: str = "active") -> None:
        self.topic = topic
        self.moderator_style = moderator_style.lower()
        self.sims: list[Any] = []

        self._llm = get_llm_client()
        self.state = DialogueState()

        self.dialogue_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.outcome: str = "pending"

        self.options, self.opening_question = self._generate_options()
        self.history: list[str] = self._build_opening_history()

        self._logger = DialogueLogger(self.dialogue_id, topic, moderator_style)
        self._mod: ModerationEngine = None  # type: ignore[assignment]
        self._tracker: Optional[StateTracker] = None
        self._phase_detector: Optional[PhaseDetector] = None
        self._consensus_engine: Optional[ConsensusEngine] = None

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
        fallback_q = "What matters most to you here?"

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
    # Logging
    # ------------------------------------------------------------------

    def _store_line(self, line: str, selected_reason: str = "",
                    tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.history.append(line)
        print(f"-> {line}")
        self._logger.append_line(line)
        self._logger.buffer(line, selected_reason, self.state, self.sims,
                            tokens_in=tokens_in, tokens_out=tokens_out)

        if self._tracker is not None:
            self._tracker.update(line, self.state.phase,
                                 selected_reason=selected_reason,
                                 tokens_in=tokens_in, tokens_out=tokens_out)
            if self._phase_detector is not None:
                ptc = sum(ps.turn_count for ps in self._tracker.state.participants.values())
                self._tracker.state.phase_evidence = self._phase_detector.update(
                    self._tracker.state, ptc
                )
            if self._consensus_engine is not None:
                names = [s.name for s in self.sims]
                self._tracker.state.consensus_state = self._consensus_engine.compute_state(
                    self._tracker.state, names
                )
                if not self._tracker.state.candidate_option:
                    self._tracker.state.candidate_option = self._consensus_engine.leading_candidate(
                        self._tracker.state, names
                    )

    def _store_moderator(self, text: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
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

        votes = current_votes(self.history, self.sims)
        if votes:
            self.state.candidate_option = Counter(votes.values()).most_common(1)[0][0]

    def _update_phase(self) -> None:
        if self.state.agreement_reached:
            self.state.phase = "closure"
            return
        turns = participant_turn_count(self.history)
        n = len(self.sims)
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
        from policy import extract_discourse
        sim_names = {s.name for s in self.sims}
        result = extract_discourse(self.history, sim_names)
        self.state.last_addressed = result["last_addressed"]
        self.state.pending_question_target = result["pending_question_target"]

    def _update_repetition(self) -> None:
        self.state.repetition_pressure = compute_repetition_pressure(self.history)

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
        max_dissenters = (cfg.consensus.max_dissenters_active
                          if self.moderator_style == "active"
                          else cfg.consensus.max_dissenters_other)
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
    # Round
    # ------------------------------------------------------------------

    def _max_speakers(self) -> int:
        phase = self.state.phase
        n = len(self.sims)
        if phase in ("greeting", "opening", "closure"):
            return 1
        if self.state.repetition_pressure >= cfg.turns.high_repetition_max_speakers_threshold:
            return 1
        if phase in ("confirmation", "emergence"):
            return min(2, n)
        weights = cfg.turns.max_speakers_weights if n >= 3 else cfg.turns.max_speakers_weights_2
        choices = list(range(1, min(4, n + 1)))
        return random.choices(choices, weights=weights[: len(choices)])[0]

    def _run_participant_round(self) -> bool:
        priority_name = self.state.priority_next_speaker
        self.state.priority_next_speaker = None

        self._update_discourse()
        self._update_repetition()
        self._update_phase()
        self._update_leading_option()

        discourse = self._tracker.state.discourse if self._tracker else None
        n_max = self._max_speakers()

        # Narrowing: every sim must state their vote before the discussion moves on.
        # Sort unvoted sims by "most stale first" (the one who hasn't spoken longest
        # gets priority) so the primary sim doesn't monopolise just by being first in
        # self.sims insertion order.
        if self.state.phase == "narrowing":
            votes = current_votes(self.history, self.sims)
            unvoted = [s for s in self.sims if s.name not in votes]
            if unvoted:
                def _last_spoken_recency(sim: Any) -> int:
                    for i, line in enumerate(reversed(self.history)):
                        if line.startswith(f"{sim.name}:"):
                            return i  # smaller = more recent
                    return len(self.history)  # never spoken = most stale
                unvoted.sort(key=_last_spoken_recency, reverse=True)  # stale first
                selected = unvoted[:n_max]
            else:
                selected = select_next_speakers(self.sims, self.history, self.state, discourse, max_speakers=n_max)
        else:
            selected = select_next_speakers(self.sims, self.history, self.state, discourse, max_speakers=n_max)

        if priority_name:
            priority_sim = next((s for s in self.sims if s.name == priority_name), None)
            if priority_sim:
                others = [s for s in selected if s.name != priority_name]
                selected = [priority_sim] + others

        all_names = [s.name for s in self.sims]
        active = False

        for sim in selected:
            forced_adapt = sim.name in self.state.nudged_participants

            turn_plan = None
            if self._tracker is not None:
                turn_plan = plan_turn(
                    speaker_name=sim.name,
                    persona=sim.persona,
                    structured=self._tracker.state,
                    legacy_state=self.state,
                    max_words=sim.persona.max_words(self.state.phase),
                )

            text, tok_in, tok_out = sim.generate_turn(
                self.history, self.state,
                all_names=all_names, forced_adaptation=forced_adapt,
                turn_plan=turn_plan,
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
        """
        Vote flips count only PUBLIC vote changes, not private belief vs first
        public commitment. The first detected public vote seeds last_known_vote
        without incrementing the flip counter.
        """
        for sim in self.sims:
            turns = last_n_turns_for(sim.name, self.history, n=1)
            if not turns:
                continue
            current_vote = extract_preference_vote(turns[0])
            if not current_vote:
                continue
            previous = self.state.last_known_vote.get(sim.name)
            if previous is None:
                # First public commitment — seed only, no flip
                self.state.last_known_vote[sim.name] = current_vote
            elif current_vote != previous:
                flips = self.state.vote_changes.get(sim.name, 0) + 1
                self.state.vote_changes[sim.name] = flips
                self.state.last_known_vote[sim.name] = current_vote
            # If unchanged, leave as-is.
            if self.state.rejected_options_by_speaker.get(sim.name) == current_vote:
                self.state.rejected_options_by_speaker.pop(sim.name, None)

    # ------------------------------------------------------------------
    # Consensus check (structured, no LLM)
    # ------------------------------------------------------------------

    def _detect_consensus(self) -> Optional[tuple[str, Optional[str]]]:
        """Returns (preferred, backup) or None. Uses public stances + votes."""
        if participant_turn_count(self.history) < len(self.sims) * 2:
            return None
        if self._tracker is None or self._consensus_engine is None:
            return None

        names = [s.name for s in self.sims]
        cs = self._tracker.state.consensus_state
        if cs in ("full_consensus", "conditional_consensus"):
            best = self._consensus_engine.leading_candidate(self._tracker.state, names)
            if best:
                return best, None

        # Fallback to explicit vote check
        window = max(cfg.consensus.regex_window, len(self.sims) * 4)
        recent = recent_participant_lines(self.history, limit=window)
        if len(recent) < len(self.sims):
            return None

        latest_vote: dict[str, str] = {}
        for line in recent:
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in latest_vote:
                continue
            vote = extract_preference_vote(msg)
            if vote:
                latest_vote[speaker] = vote
        if len(latest_vote) < len(self.sims):
            return None

        primary = self._primary_sim()
        vote_counts: Counter = Counter()
        for spk, opt in latest_vote.items():
            weight = 2 if (primary and spk == primary.name) else 1
            vote_counts[opt] += weight
        top_option = vote_counts.most_common(1)[0][0]
        unique = sum(1 for opt in latest_vote.values() if opt == top_option)
        n = len(self.sims)
        max_dissenters = (cfg.consensus.max_dissenters_active
                          if self.moderator_style == "active"
                          else cfg.consensus.max_dissenters_other)
        if unique < n - max_dissenters:
            return None
        return top_option, None

    # ------------------------------------------------------------------
    # Conclusion helpers
    # ------------------------------------------------------------------

    def _narrowing_prompt(self) -> None:
        self.state.has_asked_narrowing = True
        self.state.phase = "narrowing"
        self._store_moderator(random.choice(prompts.narrowing_lines()))

    def _run_confirmation(self) -> None:
        self.state.phase = "confirmation"
        preferred = self.state.preferred_option
        if preferred is None:
            return

        if self.moderator_style == "active":
            backup = self.state.backup_option
            backup_note = f" (Option {backup} as backup)" if backup else ""
            self._store_moderator(random.choice([
                f"sounds like Option {preferred}{backup_note}. everyone good with that?",
                f"ok so Option {preferred}{backup_note} -- yes or no from each?",
                f"looks like Option {preferred}{backup_note}. quick yes/no?",
                f"settling on Option {preferred}{backup_note}? show of hands",
            ]))

        selected = select_next_speakers(
            self.sims, self.history, self.state,
            self._tracker.state.discourse if self._tracker else None,
            max_speakers=min(2, len(self.sims)),
        )
        all_names = [s.name for s in self.sims]
        confirmation_speakers: set[str] = set()
        for sim in selected:
            text, tok_in, tok_out = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="confirmation",
                                 tokens_in=tok_in, tokens_out=tok_out)
                confirmation_speakers.add(sim.name)

        if self.moderator_style == "active":
            for sim in self.sims:
                if sim.name not in confirmation_speakers:
                    self._store_moderator(f"{sim.name}, you good with Option {self.state.preferred_option}?")
                    text, tok_in, tok_out = sim.generate_turn(self.history, self.state, all_names=all_names)
                    if text and "[SILENCE]" not in text.upper():
                        self._store_line(f"{sim.name}: {text}", selected_reason="confirmation",
                                         tokens_in=tok_in, tokens_out=tok_out)

        # Detect rejection in last N participant lines
        rejection_signals = (
            "no,", "no.", "not quite", "not yet", "still weighing",
            "not sure", "don't agree", "disagree", "not ready",
        )
        checked = 0
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            lower = msg.strip().lower().rstrip("!?.")
            words = lower.split()
            first_word = words[0] if words else ""
            # "no", "nope", "nah" as the entire turn OR as the opening word
            # catches "no too pricey", "nope can't do it", etc.
            bare_no = lower in {"no", "nope", "nah"} or first_word in {"no", "nope", "nah"}
            if bare_no or any(s in msg.lower() for s in rejection_signals):
                self.state.confirmation_rejection_count += 1
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
                        f"OK, Option {preferred} doesn't have buy-in. "
                        f"{speaker.strip()}, what's the blocker?"
                    )
                return
            checked += 1
            if checked >= len(self.sims) * 2:
                break

    def _best_compromise_option(self) -> Optional[str]:
        sc = cfg.scoring.compromise
        scores: dict[str, float] = {opt: 0.0 for opt in ["A", "B", "C", "D"]}
        votes = current_votes(self.history, self.sims)

        for sim in self.sims:
            b = sim.persona.beliefs
            if not b:
                continue
            scores[b.preferred] += sc.private_preferred_weight
            for opt in b.acceptable:
                scores[opt] += sc.private_acceptable_weight
            for opt in b.rejected:
                scores[opt] -= sc.private_rejected_penalty

        for opt in votes.values():
            scores[opt] += sc.vote_weight
        for opt in self.state.rejected_options_by_speaker.values():
            scores[opt] -= sc.dialogue_rejection_penalty

        primary = self._primary_sim()
        if primary:
            primary_rejected = self.state.rejected_options_by_speaker.get(primary.name)
            if primary_rejected:
                scores[primary_rejected] -= sc.primary_rejection_extra

        best_score = max(scores.values())
        rejected_set = set(self.state.rejected_options_by_speaker.values())
        candidates = [opt for opt, sv in scores.items() if sv == best_score and opt not in rejected_set]

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
                _clean_moderator_line(line, [s.name for s in self.sims]) or f"Could Option {option} work? What would need to be true?",
                tokens_in=self._llm.last_tokens_in,
                tokens_out=self._llm.last_tokens_out,
            )
        except Exception:
            self._store_moderator(f"Could Option {option} work? What would need to be true?")

        if holdouts:
            self.state.priority_next_speaker = holdouts[0]

    def _run_closure(self) -> None:
        self.state.phase = "closure"
        primary = self._primary_sim()
        others = [s for s in self.sims if s is not primary]
        ordered = ([primary] if primary else []) + others
        all_names = [s.name for s in self.sims]
        for sim in ordered:
            text, tok_in, tok_out = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="closure",
                                 tokens_in=tok_in, tokens_out=tok_out)

    def _conclude(self, option: str, backup: Optional[str] = None) -> None:
        self.state.preferred_option = option
        self.state.backup_option = backup
        # Align candidate_option with what's actually being confirmed so that
        # position_discipline always names the correct option letter in its prompt.
        self.state.candidate_option = option
        self.state.agreement_reached = True
        self._run_confirmation()
        if self.state.agreement_reached:
            self.outcome = "success"
            if self.moderator_style != "passive":
                backup_note = f", with Option {backup} as backup" if backup else ""
                self._store_moderator(f"Agreed -- Option {option}{backup_note}. Done.")
            self._run_closure()

    def _force_conclusion(self) -> None:
        """Public-stance-only force-close via the consensus engine."""
        names = [s.name for s in self.sims]
        final: Optional[str] = None
        if self._tracker is not None and self._consensus_engine is not None:
            final = self._consensus_engine.best_available_decision(self._tracker.state, names)

        # Vote-plurality fallback if engine returned nothing
        if final is None:
            votes = current_votes(self.history, self.sims)
            if votes:
                final = Counter(votes.values()).most_common(1)[0][0]
            else:
                final = self.state.candidate_option or self.state.current_leading_option or "A"

        self.outcome = "force_close"
        if self.moderator_style != "passive":
            self.state.preferred_option = final
            try:
                line = self._llm.generate(
                    prompts.moderator_force_close(
                        topic=self.topic,
                        participant_names=names,
                        final_option=final,
                        recent_dialogue="\n".join(self.history[-6:]),
                    )
                ).strip()
                self._store_moderator(
                    _clean_moderator_line(line, [s.name for s in self.sims]) or f"No full agreement -- calling Option {final}. Done.",
                    tokens_in=self._llm.last_tokens_in,
                    tokens_out=self._llm.last_tokens_out,
                )
            except Exception:
                self._store_moderator(f"No full agreement -- calling Option {final}. Done.")
        self._run_closure()

    def _fresh_unanswered_question(self) -> bool:
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            if speaker in cfg.EXCLUDED_SPEAKERS:
                continue
            if "?" not in msg:
                return False
            others = [s.name for s in self.sims if s.name != speaker]
            if others:
                self.state.priority_next_speaker = random.choice(others)
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(self, setup_tokens_in: int = 0, setup_tokens_out: int = 0) -> None:
        self._mod = ModerationEngine(self.topic, self.options, self.moderator_style, self.sims)
        self._tracker = StateTracker(
            participant_names=[s.name for s in self.sims], options=self.options,
        )
        self._tracker.attach_personas(self.sims)
        for line in self.history:
            self._tracker.update(line, "greeting", selected_reason="moderator")

        sample_hard_blockers(list(self._tracker.state.participants.values()))

        self._phase_detector = PhaseDetector()
        self._consensus_engine = ConsensusEngine()

        self._logger.write_header(
            participant_names=[s.name for s in self.sims], opening_lines=self.history,
        )
        for line in self.history:
            self._logger.buffer(line, "moderator", self.state, self.sims)

        print(f"\n--- Dialogue started (moderator: {self.moderator_style}) ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        def _intervene(reason: str) -> None:
            self._mod.run_intervention(reason, self.state, self.history,
                                       lambda t, ti, to: self._store_moderator(t, ti, to))

        try:
            for _ in range(cfg.turns.hard_ceiling):
                self.state.turn_index += 1
                active = self._run_participant_round()
                if not active:
                    self.outcome = "failed"
                    if self.moderator_style != "passive":
                        self._store_moderator("No responses. Discussion concluded.")
                    break

                if self.state.has_asked_narrowing:
                    self.state.post_narrowing_rounds += 1
                if self.state.has_asked_narrowing and not self.state.has_entered_emergence:
                    if len(current_votes(self.history, self.sims)) >= len(self.sims):
                        self.state.has_entered_emergence = True
                if self.state.has_entered_emergence:
                    self.state.emergence_rounds += 1
                if self.state.compromise_option:
                    self.state.compromise_rounds += 1

                # 1. Natural consensus
                if self.state.consensus_cooldown > 0:
                    self.state.consensus_cooldown -= 1
                consensus = self._detect_consensus()
                if consensus and self.state.consensus_cooldown > 0 and consensus[0] == self.state.last_rejected_option:
                    consensus = None
                if consensus and consensus[0] in set(self.state.rejected_options_by_speaker.values()):
                    consensus = None
                if consensus:
                    self._conclude(*consensus)
                    if self.state.agreement_reached:
                        break
                    continue

                # 2. Narrowing prompt
                ptc = participant_turn_count(self.history)
                if self._mod.should_narrow(self.state, ptc):
                    if self._fresh_unanswered_question():
                        continue
                    self._narrowing_prompt()
                    continue

                # 3. Post-narrowing stall + deadlock
                if self.state.has_asked_narrowing:
                    if self.state.repetition_pressure >= cfg.repetition.stall_increment_threshold:
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
                        if self.state.compromise_option and not self.state.compromise_confirmation_tried:
                            self.state.compromise_confirmation_tried = True
                            self._conclude(self.state.compromise_option)
                            if self.state.agreement_reached:
                                break
                            continue
                        self._force_conclusion()
                        break

                    if self._is_split_deadlock() and self._any_sim_stuck():
                        stall_limit = max(1, cfg.consensus.stall_rounds_to_force.get(
                            self.moderator_style, 2) - 1)
                    else:
                        stall_limit = cfg.consensus.stall_rounds_to_force.get(self.moderator_style, 2)

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
                        _intervene("stall")
                        self.state.stall_rounds = 0
                        sv = current_votes(self.history, self.sims)
                        if sv:
                            top_opt = Counter(sv.values()).most_common(1)[0][0]
                            for _sim in self.sims:
                                if sv.get(_sim.name) != top_opt:
                                    self.state.priority_next_speaker = _sim.name
                                    break
                        continue

                # 4. Regular interventions
                ptc = participant_turn_count(self.history)
                intervention = self._mod.should_intervene(self.state, self.history,
                                                          self._any_sim_stuck(), ptc)
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
                    self._force_conclusion()

        finally:
            dialogue_in = self._llm.session_tokens_in
            dialogue_out = self._llm.session_tokens_out
            self._logger.flush(
                outcome=self.outcome,
                sims=self.sims, state=self.state,
                structured=(self._tracker.state if self._tracker else None),
                setup_tokens_in=setup_tokens_in,
                setup_tokens_out=setup_tokens_out,
                dialogue_tokens_in=dialogue_in,
                dialogue_tokens_out=dialogue_out,
            )
            txt, csv_path, meta_path = self._logger.paths
            total_in = setup_tokens_in + dialogue_in
            total_out = setup_tokens_out + dialogue_out
            print(f"\n[Tokens] setup={setup_tokens_in}/{setup_tokens_out} "
                  f"dialogue={dialogue_in}/{dialogue_out} total={total_in}/{total_out}")
            print(f"[Outcome] {self.outcome}")
            print(f"[Saved]   {txt} | {csv_path} | {meta_path}")
