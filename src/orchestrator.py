"""
orchestrator.py
---------------
Orchestrator -- coordinates a single dialogue run.

Responsibilities:
  1. Setup       generate options + opening (one LLM call)
  2. State       phase, leading option, turn counts
  3. Main loop   drive rounds, detect consensus, fire moderator lines
  4. Conclusion  narrowing, confirmation, closure, force-close

Single decision path:
  - Speaker selection : policy.select_next_speakers (SSJ cascade, one speaker/round)
  - Consensus         : reasoning.ConsensusEngine (public stances only)
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
from moderation import ModerationEngine, clean_moderator_line
from policy import (
    repetition_pressure as compute_repetition_pressure,
    sample_hard_blocker,
    select_next_speakers,
)
from reasoning import ConsensusEngine
from state import StateTracker
from utils import (
    OptionResolver,
    current_votes,
    last_n_turns_for,
    participant_turn_count,
)


# ---------------------------------------------------------------------------
# DialogueState
# ---------------------------------------------------------------------------

@dataclass
class DialogueState:
    phase: str = "opening"
    turn_index: int = 0

    has_asked_narrowing: bool = False
    agreement_reached: bool = False

    preferred_option: Optional[str] = None
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
    candidate_option: Optional[str] = None

    info_gap_cooldown: int = 0
    facilitate_cooldown: int = 0
    confirmation_rejection_count: int = 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.sims: list[Any] = []

        self._llm = get_llm_client()
        self.state = DialogueState()

        self.dialogue_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.outcome: str = "pending"

        self.options, self.opening_question = self._generate_options()
        self.resolver = OptionResolver(self.options)
        self.history: list[str] = self._build_opening_history()

        self._logger = DialogueLogger(self.dialogue_id, topic)
        self._mod: ModerationEngine = None  # type: ignore[assignment]
        self._tracker: Optional[StateTracker] = None
        self._consensus_engine: Optional[ConsensusEngine] = None

    def add_sim(self, sim: Any) -> None:
        self.sims.append(sim)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _generate_options(self) -> tuple[list[str], str]:
        data = self._llm.generate_json(prompts.option_generation(self.topic))
        options_raw = data.get("options", [])
        question = str(data.get("opening_question", "")).strip()
        if not question:
            raise ValueError("Option generation returned no opening_question.")
        if not isinstance(options_raw, list) or len(options_raw) != 4:
            raise ValueError(f"Option generation expected 4 options, got: {options_raw!r}")

        cleaned: list[str] = []
        for i, raw in enumerate(options_raw):
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Option generation returned an empty option at index {i}.")
            label = chr(ord("A") + i)
            text = raw.strip()
            if not text.lower().startswith(f"option {label.lower()}"):
                text = f"Option {label} - {text}"
            cleaned.append(text)

        decision_kind = str(data.get("decision_kind", "")).strip().lower()
        if decision_kind:
            print(f"  [options] decision_kind={decision_kind}")
        return cleaned, question

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
                    tokens_in: int = 0, tokens_out: int = 0,
                    verification_result: Optional[dict] = None) -> None:
        self.history.append(line)
        print(f"-> {line}")
        self._logger.append_chat_line(line)
        self._logger.buffer(line, selected_reason, self.state,
                            tokens_in=tokens_in, tokens_out=tokens_out,
                            verification_result=verification_result)

        if self._tracker is not None:
            self._tracker.update(line, self.state.phase,
                                 selected_reason=selected_reason,
                                 tokens_in=tokens_in, tokens_out=tokens_out)
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
            mentions.extend(self.resolver.options_in(msg))
            if len(mentions) >= limit:
                break
        if mentions:
            self.state.current_leading_option = Counter(mentions).most_common(1)[0][0]

        votes = current_votes(self.history, self.resolver)
        if votes:
            self.state.candidate_option = Counter(votes.values()).most_common(1)[0][0]

    def _update_phase(self) -> None:
        if self.state.agreement_reached:
            self.state.phase = "closure"
            return
        turns = participant_turn_count(self.history)
        n = len(self.sims)
        if turns < n:
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

    def _decrement_cooldowns(self) -> None:
        self.state.facilitate_cooldown = max(0, self.state.facilitate_cooldown - 1)
        self.state.info_gap_cooldown = max(0, self.state.info_gap_cooldown - 1)
        self.state.consensus_cooldown = max(0, self.state.consensus_cooldown - 1)

    def _set_priority_to_holdout(self) -> None:
        """Route priority_next_speaker to the dissenter (vote != current majority).
        Used by stall + outlier interventions so the moderator nudge actually
        lands on the person whose movement would unblock things -- not on the
        person who just spoke."""
        sv = current_votes(self.history, self.resolver)
        if not sv:
            return
        top_opt = Counter(sv.values()).most_common(1)[0][0]
        for sim in self.sims:
            if sv.get(sim.name) != top_opt:
                self.state.priority_next_speaker = sim.name
                return

    def _max_dissenters(self) -> int:
        """Dissenters tolerated before a vote split counts as deadlock.
        Scales with group size: 0 for a pair, 1 for 3-4, up to (n-1)//2 capped
        by the config ceiling."""
        n = len(self.sims)
        return min(cfg.consensus.max_dissenters, (n - 1) // 2)

    def _is_split_deadlock(self) -> bool:
        if not self.state.has_asked_narrowing:
            return False
        votes = current_votes(self.history, self.resolver)
        if len(votes) < len(self.sims):
            return False
        n = len(self.sims)
        required = n - self._max_dissenters()
        counts = Counter(votes.values())
        return counts.most_common(1)[0][1] < required

    def _sim_vote_is_stuck(self, name: str, window: int = 4) -> bool:
        turns = last_n_turns_for(name, self.history, n=window)
        if len(turns) < window:
            return False
        options_per_turn = [self.resolver.options_in(t) for t in turns]
        if not all(opts for opts in options_per_turn):
            return False
        first = options_per_turn[0][0]
        return all(opts[0] == first for opts in options_per_turn)

    def _any_sim_stuck(self) -> bool:
        return any(self._sim_vote_is_stuck(s.name) for s in self.sims)

    # ------------------------------------------------------------------
    # Round
    # ------------------------------------------------------------------

    def _run_participant_round(self) -> bool:
        priority_name = self.state.priority_next_speaker
        self.state.priority_next_speaker = None

        self._update_discourse()
        self._update_repetition()
        self._update_phase()
        self._update_leading_option()

        discourse = self._tracker.state.discourse if self._tracker else None
        structured = self._tracker.state if self._tracker else None

        # During narrowing, prefer unvoted sims -- "stalest first".
        if self.state.phase == "narrowing":
            votes = current_votes(self.history, self.resolver)
            unvoted = [s for s in self.sims if s.name not in votes]
            if unvoted:
                def _stale(sim: Any) -> int:
                    for i, line in enumerate(reversed(self.history)):
                        if line.startswith(f"{sim.name}:"):
                            return i
                    return len(self.history)
                unvoted.sort(key=_stale, reverse=True)
                selected = [unvoted[0]]
            else:
                selected = select_next_speakers(self.sims, self.history, self.state, discourse)
        else:
            selected = select_next_speakers(self.sims, self.history, self.state, discourse)

        if priority_name:
            priority_sim = next((s for s in self.sims if s.name == priority_name), None)
            if priority_sim:
                selected = [priority_sim]

        all_names = [s.name for s in self.sims]
        active = False

        for sim in selected:
            text, tok_in, tok_out = sim.generate_turn(
                self.history, self.state,
                all_names=all_names,
                structured=structured,
            )
            if text and "[SILENCE]" not in text.upper():
                reason = "forced" if sim.name == self.state.last_addressed else "weighted"
                v_result = getattr(sim, "_last_verification", None)
                self._store_line(f"{sim.name}: {text}", selected_reason=reason,
                                 tokens_in=tok_in, tokens_out=tok_out,
                                 verification_result=v_result)
                active = True
                self.state.nudged_participants.discard(sim.name)

            if self.state.phase == "narrowing":
                votes = current_votes(self.history, self.resolver)
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
            current_vote = self.resolver.vote_in(turns[0])
            if not current_vote:
                continue
            previous = self.state.last_known_vote.get(sim.name)
            if previous is None:
                self.state.last_known_vote[sim.name] = current_vote
            elif current_vote != previous:
                flips = self.state.vote_changes.get(sim.name, 0) + 1
                self.state.vote_changes[sim.name] = flips
                self.state.last_known_vote[sim.name] = current_vote
            if self.state.rejected_options_by_speaker.get(sim.name) == current_vote:
                self.state.rejected_options_by_speaker.pop(sim.name, None)

    # ------------------------------------------------------------------
    # Consensus check -- explicit votes first; private acceptability for compromise
    # ------------------------------------------------------------------

    def _private_acceptable_for(self, option: str, dissenters: list) -> bool:
        """True if every dissenter has `option` in their private acceptable list."""
        if not dissenters:
            return True
        for sim in dissenters:
            if not sim.persona.beliefs:
                return False
            if option not in sim.persona.beliefs.acceptable:
                return False
        return True

    def _detect_consensus(self) -> Optional[str]:
        if participant_turn_count(self.history) < len(self.sims) * 2:
            return None

        names = [s.name for s in self.sims]
        votes = current_votes(self.history, self.resolver)
        n = len(self.sims)

        # 1. All explicit votes agree → clean consensus (works before and after narrowing).
        if len(votes) == n and len(set(votes.values())) == 1:
            return next(iter(votes.values()))

        # 2. Vote majority + all dissenters privately accept the leading option → compromise.
        if len(votes) >= n:
            counts = Counter(votes.values())
            top_opt, top_n = counts.most_common(1)[0]
            if top_n > n // 2:   # strict majority (>50%)
                dissenters = [s for s in self.sims if votes.get(s.name) != top_opt]
                if self._private_acceptable_for(top_opt, dissenters):
                    return top_opt

        if self._tracker is None or self._consensus_engine is None:
            return None

        cs = self._tracker.state.consensus_state

        # 3. Full stance consensus (everyone explicitly supports the same option).
        #    Note: conditional_consensus is NOT used here — vague conditionals are not
        #    treated as real support; only unconditional "support" stance counts.
        if cs == "full_consensus":
            return self._consensus_engine.leading_candidate(self._tracker.state, names)

        # 4. Vote plurality after narrowing with tolerated dissent count.
        if self.state.has_asked_narrowing and cs == "majority_candidate":
            if len(votes) >= n:
                counts = Counter(votes.values())
                top_opt, top_n = counts.most_common(1)[0]
                if n - top_n <= self._max_dissenters():
                    return top_opt

        return None

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

        self._store_moderator(random.choice([
            f"sounds like we're landing on Option {preferred} -- everyone good with that?",
            f"ok, so Option {preferred}? anyone not on board, now's the time.",
            f"looks like Option {preferred} is where we're at. that work for everyone?",
            f"feels like Option {preferred} -- any last objections before we lock it in?",
        ]))

        structured = self._tracker.state if self._tracker else None
        all_names = [s.name for s in self.sims]
        primary = self._primary_sim()
        ordered = ([primary] if primary else []) + [s for s in self.sims if s is not primary]
        for sim in ordered:
            text, tok_in, tok_out = sim.generate_turn(
                self.history, self.state, all_names=all_names, structured=structured,
            )
            if text and "[SILENCE]" not in text.upper():
                v_result = getattr(sim, "_last_verification", None)
                self._store_line(f"{sim.name}: {text}", selected_reason="confirmation",
                                 tokens_in=tok_in, tokens_out=tok_out,
                                 verification_result=v_result)

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
                self._store_moderator(
                    f"OK, Option {preferred} doesn't have buy-in. "
                    f"{speaker.strip()}, what's the blocker?"
                )
                return
            checked += 1
            if checked >= len(self.sims) * 2:
                break

    def _run_closure(self) -> None:
        self.state.phase = "closure"
        primary = self._primary_sim()
        others = [s for s in self.sims if s is not primary]
        ordered = ([primary] if primary else []) + others
        all_names = [s.name for s in self.sims]
        structured = self._tracker.state if self._tracker else None
        for sim in ordered:
            text, tok_in, tok_out = sim.generate_turn(
                self.history, self.state, all_names=all_names, structured=structured,
            )
            if text and "[SILENCE]" not in text.upper():
                v_result = getattr(sim, "_last_verification", None)
                self._store_line(f"{sim.name}: {text}", selected_reason="closure",
                                 tokens_in=tok_in, tokens_out=tok_out,
                                 verification_result=v_result)

    def _conclude(self, option: str) -> None:
        self.state.preferred_option = option
        self.state.candidate_option = option
        self.state.agreement_reached = True
        self._run_confirmation()
        if self.state.agreement_reached:
            self.outcome = "success"
            self._store_moderator(f"Agreed -- Option {option}. Done.")
            self._run_closure()

    def _force_conclusion(self) -> None:
        names = [s.name for s in self.sims]
        votes = current_votes(self.history, self.resolver)
        final: Optional[str] = None
        if self._tracker is not None and self._consensus_engine is not None:
            final = self._consensus_engine.best_available_decision(
                self._tracker.state, names, explicit_votes=votes
            )

        if final is None:
            if votes:
                final = Counter(votes.values()).most_common(1)[0][0]
            else:
                final = self.state.candidate_option or self.state.current_leading_option or "A"

        self.outcome = "force_close"
        self.state.preferred_option = final
        line = self._llm.generate(
            prompts.moderator_force_close(
                topic=self.topic,
                participant_names=names,
                final_option=final,
                recent_dialogue="\n".join(self.history[-cfg.moderation.recent_window_force_close:]),
            )
        ).strip()
        self._store_moderator(
            clean_moderator_line(line, names) or f"No full agreement -- calling Option {final}. Done.",
            tokens_in=self._llm.last_tokens_in,
            tokens_out=self._llm.last_tokens_out,
        )
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
        n = len(self.sims)
        hard_ceiling = max(cfg.turns.min_ceiling, n * cfg.turns.ceiling_per_participant)

        self._mod = ModerationEngine(self.topic, self.options, self.sims, self.resolver)
        self._tracker = StateTracker(
            participant_names=[s.name for s in self.sims], options=self.options,
            resolver=self.resolver,
        )
        self._tracker.attach_personas(self.sims)
        for line in self.history:
            self._tracker.update(line, "opening", selected_reason="moderator")

        sample_hard_blocker(list(self._tracker.state.participants.values()))

        self._consensus_engine = ConsensusEngine()

        self._logger.write_header(
            participant_names=[s.name for s in self.sims],
            personas=[s.persona for s in self.sims],
            opening_lines=self.history,
        )
        for line in self.history:
            self._logger.buffer(line, "moderator", self.state)

        print(f"\n--- Dialogue started ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        def _intervene(reason: str) -> None:
            self._mod.run_intervention(
                reason, self.state, self.history,
                lambda t, ti, to: self._store_moderator(t, ti, to),
                structured=(self._tracker.state if self._tracker else None),
            )

        try:
            for _ in range(hard_ceiling):
                self.state.turn_index += 1
                self._decrement_cooldowns()

                active = self._run_participant_round()
                if not active:
                    self.outcome = "failed"
                    self._store_moderator("No responses. Discussion concluded.")
                    break

                if self.state.has_asked_narrowing:
                    self.state.post_narrowing_rounds += 1
                if self.state.has_asked_narrowing and not self.state.has_entered_emergence:
                    if len(current_votes(self.history, self.resolver)) >= len(self.sims):
                        self.state.has_entered_emergence = True

                # 1. Natural consensus
                final = self._detect_consensus()
                if final and self.state.consensus_cooldown > 0 \
                        and final == self.state.last_rejected_option:
                    final = None
                if final and final in set(self.state.rejected_options_by_speaker.values()):
                    final = None
                if final:
                    self._conclude(final)
                    if self.state.agreement_reached:
                        break
                    continue

                # 1b. Info-chase (sims fishing for a specific missing attribute).
                # Stage 3: the moderator REFRAMES rather than shuts down.
                if self.state.info_gap_cooldown == 0 \
                        and self.state.phase in ("negotiation", "narrowing", "emergence") \
                        and self._mod.detect_info_chase(self.history):
                    _intervene("clarify")
                    self.state.info_gap_cooldown = len(self.sims) + 1
                    continue

                # 2. Narrowing -- deliberation-gated.
                ptc = participant_turn_count(self.history)
                structured = self._tracker.state if self._tracker else None
                if self._mod.should_narrow(self.state, ptc, structured=structured):
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
                        self._force_conclusion()
                        break

                    stall_limit = cfg.consensus.stall_rounds_to_force
                    if self._is_split_deadlock() and self._any_sim_stuck():
                        stall_limit = max(1, stall_limit - 1)

                    if self.state.stall_rounds >= stall_limit:
                        _intervene("stall")
                        self.state.stall_rounds = 0
                        self._set_priority_to_holdout()
                        continue

                # 4. Regular interventions (Stage 3 includes facilitation moves).
                ptc = participant_turn_count(self.history)
                intervention = self._mod.should_intervene(
                    self.state, self.history, self._any_sim_stuck(), ptc,
                    structured=(self._tracker.state if self._tracker else None),
                )
                if intervention:
                    if intervention == "stall" and self._fresh_unanswered_question():
                        continue
                    _intervene(intervention)
                    # Route priority to the HOLDOUT (dissenter), never to the
                    # outlier itself. The outlier nudge exists to break a repeat
                    # loop; forcing the repeater to speak again recreates the loop.
                    if intervention == "stall" or intervention.startswith("outlier:"):
                        self._set_priority_to_holdout()

            else:
                # Hard ceiling hit.
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
            chat_path, eval_path = self._logger.paths
            total_in = setup_tokens_in + dialogue_in
            total_out = setup_tokens_out + dialogue_out
            print(f"\n[Tokens] setup={setup_tokens_in}/{setup_tokens_out} "
                  f"dialogue={dialogue_in}/{dialogue_out} total={total_in}/{total_out}")
            print(f"[Outcome] {self.outcome}")
            print(f"[Saved]   {chat_path} | {eval_path}")
