"""Dialogue runner for option-grounded multi-user simulation.

The runner separates four responsibilities:
1. environment setup (scenario + option board),
2. simulator state (private lean, agenda, tunable behavior),
3. routing policy (who speaks, to whom, with which dialogue act), and
4. visible transcript observation (votes/outcomes from public text only).

DialogueRunner is the orchestration loop. The three concern-specific mixins keep
the file readable: PolicyMixin (routing), ObserverMixin (parse + visible state),
ValidationMixin (turn validation + grounding + fallback).
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from datetime import datetime

import prompts
from aliases import short_alias_map
from builders import SetupBuilder, manual_environment
from config_loader import cfg
from consensus import ConsensusManager, participant_turn_count
from logger import DialogueLogger, token_summary_for
from llm_client import get_llm_client
from models import (
    ActType,
    DialogueAct,
    DialogueRunResult,
    DialogueState,
    MoveIntent,
    OptionCoverage,
    ParticipantRuntime,
    Persona,
    Phase,
    ResponseObligation,
    TurnRecord,
)
from parsing import OptionResolver
from simulator import mark_agenda_done
from style import strip_leading_name
from utils import clean_generated, normalise_lines
from observer import ObserverMixin
from policy import PolicyMixin
from validation import ValidationMixin


class DialogueRunner(PolicyMixin, ObserverMixin, ValidationMixin):
    def __init__(self, topic: str) -> None:
        # A manual environment carries its own topic; the CLI topic is unused then.
        manual_env = manual_environment()
        self.topic = str((manual_env or {}).get("topic") or topic).strip()
        if not self.topic:
            raise ValueError("topic must not be empty")
        seed = cfg.simulation.get("random_seed", None)
        if seed is not None:
            random.seed(int(seed))
        self._llm = get_llm_client()
        self._resolver: OptionResolver | None = None
        self._intervention_count = 0
        self._last_intervention_turn = -999

    def _mod(self, part: str) -> bool:
        """Whether the moderator voice performs a given structural job (issue 7).

        The controller policy is independent of these flags; they only gate the
        moderator's visible turns. An absent `moderator` config section keeps the
        fully-moderated default (every part on)."""
        mod = getattr(cfg, "moderator", None)
        if mod is None:
            return True
        if not bool(mod.get("enabled", True)):
            return False
        return bool(mod.get(part, True))

    def run(self) -> DialogueRunResult:
        n = cfg.participant_count()
        self._llm.reset_session()
        scenario, personas = SetupBuilder(self.topic).build(n)
        setup_tokens_in = self._llm.session_tokens_in
        setup_tokens_out = self._llm.session_tokens_out
        self._llm.reset_session()

        state = initialise_state(scenario, personas)
        state.setup_tokens_in = setup_tokens_in
        state.setup_tokens_out = setup_tokens_out
        self._resolver = OptionResolver(scenario.options)
        self._derive_pacing(state)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.logger = DialogueLogger(run_id, self.topic)

        # When the moderator opening is off, the board is still shown to the
        # reader as plain setup scaffolding (header + transcript "## Options"),
        # just not as a moderator turn.
        self._print_header(state, show_board=not self._mod("opening"))
        if self._mod("opening"):
            self._emit(self._append_moderator(state, prompts.moderator_opening(scenario), Phase.OPENING))

        self._opening_round(state)
        self._discussion_loop(state)
        self._decision_loop(state)

        outcome = ConsensusManager.finalize(state)
        state.outcome = outcome
        if self._mod("closing"):
            closing = self._moderator_say(prompts.moderator_closure_prompt(outcome, scenario, state), state)
            self._emit(self._append_moderator(state, closing, Phase.CLOSURE))
        self._mark_phase(state, Phase.CLOSURE, f"closed as {outcome.status}")

        state.dialogue_tokens_in = self._llm.session_tokens_in
        state.dialogue_tokens_out = self._llm.session_tokens_out
        paths = self.logger.finish(state, outcome)
        transcript = [f"{turn.speaker_name}: {turn.text}" for turn in state.turns]
        return DialogueRunResult(scenario, personas, transcript, outcome, paths, token_summary_for(state))

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _opening_round(self, state: DialogueState) -> None:
        state.phase = Phase.OPENING
        for persona in self._opening_order(state.personas):
            intent = MoveIntent(
                speaker_id=persona.id,
                act=ActType.OPENING,
                reason="state the current favorite and one grounded reason without making a final vote",
                option_focus=[persona.preferred_option],
            )
            self._emit(self._generate_and_append(state, intent))
        self._mark_phase(state, Phase.DISCUSSION, "all participants gave an opening view")

    def _discussion_loop(self, state: DialogueState) -> None:
        state.phase = Phase.DISCUSSION
        while True:
            if self._ready_for_vote(state):
                self._mark_phase(state, Phase.NARROWING, self._vote_reason(state))
                return
            maybe_nudge = self._maybe_moderator_nudge(state)
            if maybe_nudge:
                self._emit(maybe_nudge)
            intent = self._route_discussion_turn(state)
            self._emit(self._generate_and_append(state, intent))

    def _decision_loop(self, state: DialogueState) -> None:
        state.phase = Phase.NARROWING
        for round_index in range(int(cfg.conversation.max_vote_rounds)):
            # Clear any direct question still owed from the discussion before voting.
            self._resolve_pending_question(state)
            candidate = self._candidate_for_vote(state)
            state.candidate_option = candidate
            # Round 0 asks everyone; later rounds re-prompt only unclear/non-voters
            # so participants who already cast a clear vote are not asked again.
            order = self._vote_order(state, candidate)
            if round_index > 0:
                order = [p for p in order if not self._has_clear_vote(state, p.id)]
                if not order:
                    # Everyone has a clear vote but the round-end check above did
                    # not reach a majority/consensus, so the run has NOT closed:
                    # it falls through to the split-vote compromise pass. Record
                    # an intermediate narrowing marker, never a final closure
                    # (issue 6) — only a resolved outcome marks closure.
                    self._mark_phase(
                        state,
                        Phase.NARROWING,
                        "all participants voted but no majority; attempting split-vote compromise",
                    )
                    break
            reason = "let's test where everyone stands" if round_index == 0 else "let's hear from whoever hasn't given a clear vote"
            if self._mod("final_vote_call"):
                nudge_record, target_id = self._moderator_vote_nudge(state, candidate, reason)
                self._emit(nudge_record)
                if target_id:
                    order.sort(key=lambda p: p.id != target_id)
            for persona in order:
                self._emit(self._generate_and_append(state, self._vote_intent(state, persona, candidate)))
            provisional = ConsensusManager.finalize(state)
            if provisional.status in {"successful", "majority"}:
                if provisional.status == "majority":
                    # A majority should not end the chat mid-conversation: give
                    # the holdouts one visible beat before closing (issue #26).
                    self._minority_check(state, provisional.final_option)
                    provisional = ConsensusManager.finalize(state)
                state.outcome = provisional
                self._mark_phase(state, Phase.CLOSURE, f"{provisional.status} visible after vote round {round_index + 1}")
                return
        # No majority after the standard rounds: one bounded compromise attempt
        # (summarize the split, propose one option) before closing as unresolved.
        if self._maybe_split_vote_compromise(state):
            provisional = ConsensusManager.finalize(state)
            if provisional.status in {"successful", "majority"}:
                state.outcome = provisional
                self._mark_phase(state, Phase.CLOSURE, f"{provisional.status} after split-vote compromise")
                return
        self._mark_phase(state, Phase.CLOSURE, "vote rounds exhausted without visible consensus")

    def _minority_check(self, state: DialogueState, winner: str | None) -> None:
        """One bounded beat after a majority forms: the holdouts are acknowledged
        and each gets one visible turn — accept the majority option (with a
        bridge clause) if they can move, or briefly restate what holds them
        back. The most movable holdout first gets a two-turn reservation
        exchange (issue 4): state the concrete reservation, hear one supporter
        respond, then decide. May upgrade the outcome to unanimity; runs at
        most once and is skipped after the split-compromise pass (those
        dissenters were just asked)."""
        if state.minority_check_attempted or state.compromise_attempted:
            return
        if winner not in state.scenario.option_ids:
            return
        dissenters = [p for p in state.personas if state.runtimes[p.id].explicit_vote != winner]
        if not dissenters:
            return
        state.minority_check_attempted = True
        winner_name = state.scenario.option(winner).name
        aliases = short_alias_map(state.scenario.options)
        movers = [p for p in dissenters if self._can_shift_to(state, p, winner)]
        negotiator = (
            min(movers, key=lambda p: state.runtimes[p.id].commitment_strength) if movers else None
        )
        if self._mod("final_vote_call"):
            text = self._moderator_say(
                prompts.moderator_nudge_prompt(
                    state,
                    "a clear majority has formed but a few chose differently",
                    winner_name,
                    requested_action=(
                        f"acknowledge the majority for {winner_name} in a friendly line and ask those who "
                        "chose differently whether they can live with it or what still holds them back — "
                        "do not reopen the full debate"
                    ),
                    focus_options=[winner],
                ),
                state,
            )
            self._emit(self._append_moderator(state, text, Phase.NARROWING))
        elif negotiator is not None:
            # No moderator voice: a majority supporter owns the probe instead.
            self._emit_peer_holdout_probe(state, negotiator, winner)
        if negotiator is not None:
            self._reservation_exchange(state, negotiator, winner)
        for persona in dissenters:
            can_move = self._can_shift_to(state, persona, winner)
            current = state.runtimes[persona.id].current_preference or persona.preferred_option
            current_name = aliases.get(current, current)
            intent = MoveIntent(
                speaker_id=persona.id,
                act=ActType.VOTE,
                reason=(
                    "most of the group has landed on the majority option; either accept it with a "
                    f"direct commitment and one clause on what makes it workable despite preferring "
                    f"{current_name}, or briefly state what still holds you back and restate your pick"
                    if can_move
                    else "most of the group has landed on the majority option; briefly restate your "
                    "pick and your reservation in one line — you are not switching"
                ),
                option_focus=[winner, current] if current != winner else [winner],
                length_hint="short",
                allow_vote_change=can_move,
            )
            self._emit(self._generate_and_append(state, intent))

    def _reservation_exchange(self, state: DialogueState, holdout: Persona, candidate: str) -> None:
        """Bounded reservation micro-negotiation (issue 4), exactly two turns:
        the holdout states one concrete reservation about the candidate (no vote
        yet), and one supporter of the candidate responds to it honestly. The
        holdout's actual decision comes in its regular closing beat afterwards.
        Runs at most once per run."""
        if state.reservation_exchange_done:
            return
        state.reservation_exchange_done = True
        aliases = short_alias_map(state.scenario.options)
        reservation = MoveIntent(
            speaker_id=holdout.id,
            act=ActType.ANSWER,
            reason=(
                f"say concretely what still makes you hesitate about {aliases[candidate]} — one specific "
                "reservation or condition, grounded in the option facts or what they leave unknown; "
                "do not cast a vote yet"
            ),
            option_focus=[candidate],
            length_hint="short",
        )
        record = self._generate_and_append(state, reservation)
        self._emit(record)
        supporters = [
            p for p in state.personas
            if p.id != holdout.id and state.runtimes[p.id].explicit_vote == candidate
        ]
        if not supporters:
            return
        responder = max(supporters, key=lambda p: p.sim_params.responsiveness + 0.3 * p.sim_params.engagement)
        response = MoveIntent(
            speaker_id=responder.id,
            act=ActType.ANSWER,
            reason=(
                f"respond to {holdout.name}'s reservation about {aliases[candidate]} honestly: use only "
                "the option facts, concede what the board cannot prove, and point to what still helps "
                "their concern — no pressure to switch"
            ),
            addressee_id=holdout.id,
            option_focus=[candidate],
            respond_to_turn=record.index,
            length_hint="short",
        )
        self._emit(self._generate_and_append(state, response))

    def _emit_peer_holdout_probe(self, state: DialogueState, holdout: Persona, candidate: str) -> None:
        """With the moderator voice off, a supporter of the candidate asks the
        holdout what still blocks agreement (participant-owned procedure)."""
        supporters = [
            p for p in state.personas
            if p.id != holdout.id and state.runtimes[p.id].explicit_vote == candidate
        ]
        if not supporters:
            return
        aliases = short_alias_map(state.scenario.options)
        asker = max(supporters, key=lambda p: p.sim_params.initiative + 0.3 * p.sim_params.engagement)
        intent = MoveIntent(
            speaker_id=asker.id,
            act=ActType.ASK,
            reason=(
                f"most of the group has landed on {aliases[candidate]}; ask {holdout.name} in a friendly, "
                "genuine way what still holds them back or what they would need — no pressure"
            ),
            addressee_id=holdout.id,
            option_focus=[candidate],
            length_hint="short",
        )
        self._emit(self._generate_and_append(state, intent))

    def _maybe_split_vote_compromise(self, state: DialogueState) -> bool:
        """One bounded compromise pass when votes are split with no majority.

        The moderator summarizes the split and names one candidate; participants
        who can move are invited to switch to it or restate their vote. Runs at
        most once per run and only if some participant could plausibly move.
        """
        # The hard turn cap forces the *vote*; it does not forbid this bounded
        # closing pass — starving it just manufactures unresolved runs (issue 4).
        if state.compromise_attempted:
            return False
        votes = [rt.explicit_vote for rt in state.runtimes.values() if rt.explicit_vote in state.scenario.option_ids]
        if len(set(votes)) < 2:
            return False
        probe = self._split_probe_candidate(state, votes)
        if probe is None:
            return False
        leader, dissenters, movers = probe
        state.compromise_attempted = True
        state.candidate_option = leader
        leader_name = state.scenario.option(leader).name
        if self._mod("final_vote_call"):
            # Only claim a lead when the candidate is a strict plurality; on a
            # pure tie (e.g. 1-1-1) it merely has as much support as the rest, so
            # the moderator must not assert it "has the most support" (todo §5).
            counts = Counter(votes)
            top = counts[leader]
            strict_leader = sum(1 for c in counts.values() if c == top) == 1
            standing = (
                f"note that {leader_name} currently has the most support"
                if strict_leader
                else f"note that the vote is evenly split with no option ahead, and float {leader_name} as one option to rally around"
            )
            text = self._moderator_say(
                prompts.moderator_nudge_prompt(
                    state,
                    "the votes are split with no majority",
                    leader_name,
                    requested_action=(
                        f"summarize the split in one line, {standing}, and ask those who chose "
                        "differently whether they could genuinely live with it or would rather stay "
                        "with their own pick — make clear both answers are fine; don't push, and "
                        f"don't call {leader_name} a middle ground"
                    ),
                    focus_options=[leader],
                ),
                state,
            )
            self._emit(self._append_moderator(state, text, Phase.NARROWING))
        # Reservation micro-negotiation (issue 4): before the closing beats, the
        # most movable dissenter states what blocks agreement and one supporter
        # responds — two bounded turns, once per run.
        negotiator = min(movers, key=lambda p: state.runtimes[p.id].commitment_strength)
        self._reservation_exchange(state, negotiator, leader)
        # Everyone who is not on the leader gets one closing beat: movers may
        # switch, the rest briefly restate — so a failed compromise does not
        # end the chat one turn after the split summary (issue #22).
        aliases = short_alias_map(state.scenario.options)
        for persona in dissenters:
            can_move = persona in movers
            current = state.runtimes[persona.id].current_preference or persona.preferred_option
            current_name = aliases.get(current, current)
            intent = MoveIntent(
                speaker_id=persona.id,
                act=ActType.VOTE,
                reason=(
                    "the group is split; if you can accept the proposed compromise, switch to it "
                    f"with a direct commitment AND say in one clause what makes it acceptable to you "
                    f"despite preferring {current_name} (what you give up or what it still delivers); "
                    "otherwise clearly restate the option you still choose and why"
                    if can_move
                    else "the group is split; briefly restate the option you still choose and react "
                    "to the split in one line — you are not switching"
                ),
                option_focus=[leader, current] if current != leader else [leader],
                length_hint="short",
                allow_vote_change=can_move,
            )
            self._emit(self._generate_and_append(state, intent))
        return True

    def _generate_and_append(self, state: DialogueState, intent: MoveIntent) -> TurnRecord:
        self._apply_style_flags(state, intent)
        persona = state.persona_by_id(intent.speaker_id)
        min_words, max_words = self._word_bounds(intent, persona)
        recent_lines = self._recent_lines(state)
        focus_options = [state.scenario.option(i) for i in intent.option_focus if i in state.scenario.option_ids]
        addressee_name = state.name_for(intent.addressee_id) if intent.addressee_id else None
        prompt = prompts.sim_utterance(
            persona=persona,
            state=state,
            intent=intent,
            recent_lines=recent_lines,
            focus_options=focus_options,
            addressee_name=addressee_name,
            max_words=max_words,
            min_words=min_words,
        )
        self.logger.write_prompt(prompt, f"{state.turn_index + 1:03d}_{persona.id}_{intent.act.value}")
        text, tokens_in, tokens_out = self._call_participant(prompt, persona.name, max_words)
        if intent.suppress_name_prefix:
            text = strip_leading_name(text, [p.name for p in state.personas])
        act = self._parse_act(state, persona, text, intent)
        report, gti, gto = self._collect_report(text, state, persona, intent, act, focus_options)
        tokens_in += gti
        tokens_out += gto
        repaired = False
        trigger_codes = list(report.issues)
        attempts = int(cfg.simulation.max_repairs_per_turn)
        while report.issues and attempts > 0:
            attempts -= 1
            repaired = True
            repair_prompt = prompts.repair_utterance(
                original_text=text,
                issue_codes=report.issues,
                persona=persona,
                state=state,
                recent_lines=recent_lines,
                intent=intent,
                max_words=max_words,
            )
            self.logger.write_prompt(repair_prompt, f"{state.turn_index + 1:03d}_{persona.id}_repair")
            candidate_text, ti, to = self._call_participant(repair_prompt, persona.name, max_words)
            tokens_in += ti
            tokens_out += to
            candidate_act = self._parse_act(state, persona, candidate_text, intent)
            candidate_report, gti, gto = self._collect_report(candidate_text, state, persona, intent, candidate_act, focus_options)
            tokens_in += gti
            tokens_out += gto
            if not candidate_report.issues or len(candidate_report.issues) <= len(report.issues):
                text, act, report = candidate_text, candidate_act, candidate_report

        block = report.block_state_mutation or self._semantic_block(persona, intent, act)
        used_fallback = False
        if block:
            # Blocking issues survived generation + repair: the LLM text must not
            # reach the transcript (issue I1). Replace it with a deterministic
            # fallback for this intent, re-parse, and re-validate before appending.
            text = self._safe_fallback_text(state, persona, intent, report)
            act = self._parse_act(state, persona, text, intent)
            report = self._validate_turn_text(text, state, persona, intent, act)
            used_fallback = True
            state.fallback_turn_count += 1
            block = report.block_state_mutation or self._semantic_block(persona, intent, act)
            if block:
                # Should be unreachable: the fallback is built to parse cleanly.
                state.invalid_printed_turn_count += 1
        record = self._append_participant(
            state,
            persona,
            text,
            act,
            intent,
            tokens_in,
            tokens_out,
            report.issues,
            repaired,
            trigger_codes,
            block,
        )
        record.used_fallback = used_fallback
        if not block:
            self._apply_semantics(state, record)
            mark_agenda_done(persona, intent.agenda_index)
        else:
            state.no_progress_count += 1
        return record

    def _call_participant(self, prompt: str, speaker_name: str, max_words: int) -> tuple[str, int, int]:
        raw = self._llm.generate(prompt, profile="dialogue")
        text = clean_generated(raw, speaker_name, max_words)
        return text, self._llm.last_tokens_in, self._llm.last_tokens_out

    def _append_participant(
        self,
        state: DialogueState,
        persona: Persona,
        text: str,
        act: DialogueAct,
        intent: MoveIntent,
        tokens_in: int,
        tokens_out: int,
        issues: list[str],
        repaired: bool,
        trigger_codes: list[str],
        block: bool,
    ) -> TurnRecord:
        state.turn_index += 1
        rt = state.runtimes[persona.id]
        rt.turn_count += 1
        rt.last_spoke_turn = state.turn_index
        rt.already_said.append(text)
        record = TurnRecord(
            index=state.turn_index,
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            phase=state.phase,
            act=act,
            intent=intent,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            validation_issues=issues,
            repaired=repaired,
            repair_trigger_codes=trigger_codes,
            state_mutation_blocked=block,
        )
        state.turns.append(record)
        return record

    def _append_moderator(self, state: DialogueState, text: str, phase: Phase) -> TurnRecord:
        state.turn_index += 1
        text = normalise_lines(text)
        act = DialogueAct(speaker_id="moderator", text=text, act_type=ActType.REACT)
        record = TurnRecord(index=state.turn_index, speaker_id="moderator", speaker_name="Moderator", text=text, phase=phase, act=act)
        state.turns.append(record)
        return record

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def _maybe_moderator_nudge(self, state: DialogueState) -> TurnRecord | None:
        if not self._mod("mid_discussion_nudges"):
            return None
        if self._intervention_count >= int(cfg.conversation.moderator_max_interventions):
            return None
        if participant_turn_count(state) < max(state.min_discussion_turns + 2, state.force_narrow_turns - 1):
            return None
        if state.no_progress_count < int(cfg.conversation.moderator_stall_window):
            return None
        if state.turn_index - self._last_intervention_turn < int(cfg.conversation.moderator_cooldown_turns):
            return None
        candidate = self._visible_candidate(state) or self._latent_leading_option(state)
        candidate_name = state.scenario.option(candidate).name if candidate else None
        target_id, requested_action, focus = self._moderator_intervention_details(state, candidate)
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state,
                "discussion seems to be circling",
                candidate_name,
                target_name=state.name_for(target_id) if target_id else None,
                requested_action=requested_action,
                focus_options=focus,
            ),
            state,
        )
        self._intervention_count += 1
        self._last_intervention_turn = state.turn_index
        state.no_progress_count = 0
        record = self._append_moderator(state, text, Phase.DISCUSSION)
        if target_id:
            self._set_obligation(
                state,
                target_id=target_id,
                source_id="moderator",
                text=text,
                expected_act=ActType.ANSWER,
                option_focus=focus,
            )
        return record

    def _moderator_vote_nudge(self, state: DialogueState, candidate: str, reason: str) -> tuple[TurnRecord, str | None]:
        target_id, requested_action, focus = self._moderator_intervention_details(state, candidate, voting=True)
        # Vote calls are option-neutral: never hand the prompt a candidate name
        # it could merge into the question (issue I10).
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state,
                reason,
                None,
                target_name=state.name_for(target_id) if target_id else None,
                requested_action=requested_action,
                focus_options=focus,
            ),
            state,
        )
        record = self._append_moderator(state, text, Phase.NARROWING)
        if target_id:
            self._set_obligation(
                state,
                target_id=target_id,
                source_id="moderator",
                text=text,
                expected_act=ActType.VOTE,
                option_focus=focus,
            )
        return record, target_id

    def _moderator_say(self, prompt: str, state: DialogueState) -> str:
        raw = self._llm.generate(prompt, profile="dialogue")
        text = clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.moderator))
        if self._resolver and self._resolver.invalid_option_refs(text):
            raw = self._llm.generate(prompt + "\nOnly use the exact option names already listed.", profile="repair")
            text = clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.moderator))
        return text or "Let’s make this concrete: what is the strongest remaining concern before we choose?"

    def _moderator_intervention_details(
        self,
        state: DialogueState,
        candidate: str | None,
        *,
        voting: bool = False,
    ) -> tuple[str | None, str, list[str]]:
        """Return (target_id, requested_action, focus_option_ids) for the moderator."""
        gap = self._coverage_gap_option(state)
        if gap is not None and not voting:
            return (
                None,
                f"ask the group to briefly compare Option {gap} before narrowing",
                [gap],
            )

        open_question = self._next_answerable_question(state)
        if open_question is not None:
            return (
                open_question.target_id,
                "ask for a direct answer to the pending question",
                open_question.option_focus,
            )

        # During finalization only unclear/non-voters should be prompted again.
        # Vote calls stay option-neutral: the moderator invites picks without
        # naming any option, so the candidate can never leak into the question
        # ("which Space Station option are you going with", I10).
        unresolved = [p for p in state.personas if not self._has_clear_vote(state, p.id)]
        if voting and len(unresolved) == 1:
            return (
                unresolved[0].id,
                "ask them casually which option they'll actually go with — one definite pick, no conditions; "
                "do not name or suggest any option yourself, and don't ask what they're 'leaning' toward",
                [],
            )
        if voting:
            return (
                None,
                "invite everyone to give their final pick now — each person names the one option they're "
                "going with, definite wording; do not name or suggest any option yourself, and never use "
                "the word 'leaning'",
                [],
            )

        aliases = short_alias_map(state.scenario.options)
        if candidate:
            # An unresolved visible blocker on the likely candidate is the most
            # useful thing to surface; ask that person directly, once.
            probe_key = f"mod:{candidate}"
            blockers = [p for p in state.personas if candidate in state.runtimes[p.id].hard_rejections]
            if blockers and probe_key not in state.blocker_probes:
                state.blocker_probes.add(probe_key)
                return (
                    blockers[0].id,
                    f"ask them what would need to change about {aliases[candidate]} for it to work for them, "
                    "or what they could support instead — one genuine question, no pressure",
                    [candidate],
                )
        # A visible split deserves a head-to-head request before any narrowing.
        supported = sorted(
            (oid for oid in state.scenario.option_ids if self._visible_support_count(state, oid) >= 1),
            key=lambda oid: -self._visible_support_count(state, oid),
        )
        if len(supported) >= 2:
            return (
                None,
                f"ask the group to weigh {aliases[supported[0]]} against {aliases[supported[1]]} on the "
                "trade-off that actually divides them — no verdict, just the comparison",
                supported[:2],
            )
        if candidate:
            dissenters = [p for p in state.personas if state.runtimes[p.id].current_preference != candidate]
            if len(dissenters) == 1:
                return (
                    dissenters[0].id,
                    "ask what remaining concern would need to be resolved to move",
                    [candidate, state.runtimes[dissenters[0].id].current_preference or dissenters[0].preferred_option],
                )
        return (None, "ask for the strongest remaining concern before choosing", [candidate] if candidate else [])

    @staticmethod
    def _has_clear_vote(state: DialogueState, persona_id: str) -> bool:
        return bool(state.runtimes[persona_id].explicit_vote)

    def _derive_pacing(self, state: DialogueState) -> None:
        n = len(state.personas)
        prefs = [p.preferred_option for p in state.personas]
        distinct = len(set(prefs))
        avg_compromise = sum(1.0 - p.sim_params.compromise_threshold for p in state.personas) / max(1, n)
        min_turns = math.ceil(float(cfg.conversation.min_discussion_turns_per_participant) * n)
        target = math.ceil(float(cfg.conversation.target_discussion_turns_per_participant) * n)
        hard = math.ceil(float(cfg.conversation.max_discussion_turns_per_participant) * n)
        if distinct > 1:
            target += int(cfg.conversation.contention_extra_turns)
            hard += int(cfg.conversation.contention_extra_turns)
        if avg_compromise < 0.45:
            target += int(cfg.conversation.low_compromise_extra_turns)
            hard += int(cfg.conversation.low_compromise_extra_turns)
        state.min_discussion_turns = max(n, min_turns)
        vote_buffer = max(1, math.ceil(n / 2))
        state.force_narrow_turns = max(state.min_discussion_turns + vote_buffer, target)
        state.hard_max_turns = max(state.force_narrow_turns + vote_buffer, hard)
        state.phase_history.append(
            f"pacing: min={state.min_discussion_turns}, force={state.force_narrow_turns}, hard={state.hard_max_turns}, distinct_initial_prefs={distinct}, avg_compromise={avg_compromise:.2f}"
        )

    def _mark_phase(self, state: DialogueState, phase: Phase, reason: str) -> None:
        state.phase = phase
        state.phase_history.append(f"turn {state.turn_index}: {phase.value} — {reason}")

    @staticmethod
    def _opening_order(personas: list[Persona]) -> list[Persona]:
        return sorted(personas, key=lambda p: (p.sim_params.initiative, random.random()), reverse=True)

    def _resolve_pending_question(self, state: DialogueState) -> None:
        """Let a directly-asked participant answer before a vote round starts."""
        obligation = self._active_obligation(state)
        if obligation is None or obligation.expected_act != ActType.ANSWER:
            return
        self._emit(self._generate_and_append(state, self._obligation_intent(state, obligation)))

    def _recent_lines(self, state: DialogueState) -> list[str]:
        limit = int(cfg.utterances.recent_turns_in_prompt)
        return [f"{turn.speaker_name}: {turn.text}" for turn in state.turns[-limit:]]

    @staticmethod
    def _print_header(state: DialogueState, show_board: bool = False) -> None:
        print("\n" + "=" * 72)
        print(f"Topic: {state.scenario.topic}")
        print("Participants: " + ", ".join(p.name for p in state.personas))
        if show_board:
            # No moderator opening turn: still show the reader the option board
            # as plain setup scaffolding so the console isn't missing the world.
            print(prompts.moderator_opening(state.scenario))
        print("=" * 72)

    @staticmethod
    def _emit(record: TurnRecord) -> None:
        print(f"{record.speaker_name}: {record.text}")


def initialise_state(scenario, personas: list[Persona]) -> DialogueState:
    state = DialogueState(scenario=scenario, personas=personas)
    # Initial commitment to the starting favorite scales with stubbornness, so a
    # flexible sim starts movable and a stubborn one starts dug in (issue 2).
    state.runtimes = {
        p.id: ParticipantRuntime(
            persona_id=p.id,
            current_preference=p.preferred_option,
            commitment_strength=0.45 + 0.40 * p.sim_params.stubbornness,
            commitment_min=0.45 + 0.40 * p.sim_params.stubbornness,
        )
        for p in personas
    }
    state.coverage = {option.id: OptionCoverage() for option in scenario.options}
    for persona in personas:
        if persona.rejection:
            state.runtimes[persona.id].hard_rejections[persona.rejection] = persona.rejection_reason
    return state
