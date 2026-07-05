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
        self._record_token_usage(state, "setup", setup_tokens_in, setup_tokens_out)
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
            else:
                procedural = self._maybe_participant_procedural(state)
                if procedural is not None:
                    self._emit(self._generate_and_append(state, procedural))
                    continue
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
            elif round_index == 0 and not state.peer_vote_call_done:
                # Participant-owned vote call (issue 5): with the moderator's
                # vote-call voice off, the highest-initiative sim casually asks
                # for final picks — option-neutral, then votes like everyone.
                state.peer_vote_call_done = True
                caller = max(state.personas, key=lambda p: p.sim_params.initiative + 0.3 * p.sim_params.engagement)
                call_intent = MoveIntent(
                    speaker_id=caller.id,
                    act=ActType.CALL_VOTE,
                    reason=(
                        "you feel the group has weighed enough — casually suggest everyone give their "
                        "definite final pick now, in your own words; do not name or suggest any option "
                        "in this line"
                    ),
                    length_hint="short",
                )
                state.procedural_move_count += 1
                self._emit(self._generate_and_append(state, call_intent))
                order.sort(key=lambda p: p.id != caller.id)
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
            act=ActType.PROBE_HOLDOUT,
            reason=(
                f"most of the group has landed on {aliases[candidate]}; ask {holdout.name} in a friendly, "
                "genuine way what still holds them back or what they would need — no pressure"
            ),
            addressee_id=holdout.id,
            option_focus=[candidate],
            length_hint="short",
        )
        state.procedural_move_count += 1
        self._emit(self._generate_and_append(state, intent))

    def _maybe_split_vote_compromise(self, state: DialogueState) -> bool:
        """Bounded narrowing when final votes are split with no majority.

        This is controller-owned: the split summary is deterministic when the
        moderator is on, and the sequence is fixed enough that a 3-way or 1-1
        split cannot collapse straight into an unresolved close. The LLM still
        renders participant reservations/responses, but the controller chooses
        the candidate, the dissenters, and the stop point.
        """
        if state.compromise_attempted:
            return False
        votes_by_id = {
            p.id: state.runtimes[p.id].explicit_vote
            for p in state.personas
            if state.runtimes[p.id].explicit_vote in state.scenario.option_ids
        }
        votes = list(votes_by_id.values())
        if len(set(votes)) < 2:
            return False
        state.compromise_attempted = True
        if len(state.personas) == 2 and len(votes_by_id) == 2 and len(set(votes)) == 2:
            return self._two_person_deadlock_protocol(state, votes_by_id)
        tested: set[str] = set()
        for attempt_index in range(2):
            votes_by_id = {
                p.id: state.runtimes[p.id].explicit_vote
                for p in state.personas
                if state.runtimes[p.id].explicit_vote in state.scenario.option_ids
            }
            votes = list(votes_by_id.values())
            ranked = self._rank_split_candidates(state, votes, exclude=tested)
            if not ranked:
                # No plausible mover: still emit one honest split summary so the
                # unresolved outcome is socially explained, not an abrupt stop.
                if attempt_index == 0:
                    self._emit_split_summary(state, None, votes_by_id)
                return True
            leader, dissenters, movers, meta = ranked[0]
            tested.add(leader)
            state.candidate_option = leader
            self._emit_split_summary(state, leader, votes_by_id, attempt_index=attempt_index, meta=meta)

            # Ask concrete reservations before final switch/stay beats. Bound the
            # cost: first candidate may get two reservation/supporter pairs; the
            # optional second candidate gets one. Include hard but relevant
            # dissenters only for the final decision beat, not for a fake
            # reservation exchange they cannot move from.
            reservation_limit = 2 if attempt_index == 0 else 1
            ordered_holdouts = sorted(
                [p for p in dissenters if not self._hard_blocks_candidate(state, p, leader)],
                key=lambda p: self._candidate_resistance(state, p, leader),
            )
            for holdout in ordered_holdouts[:reservation_limit]:
                self._split_reservation_exchange(state, holdout, leader)

            for persona in dissenters:
                can_move = persona in movers
                alternative = self._holdout_alternative_candidate(state, persona, leader, tested=tested)
                intent = self._post_reservation_decision_intent(
                    state,
                    persona,
                    leader,
                    can_move=can_move,
                    alternative=alternative if attempt_index == 0 else None,
                )
                self._emit(
                    self._append_post_reservation_decision(
                        state,
                        persona,
                        intent,
                        candidate=leader,
                        can_move=can_move,
                        alternative=alternative if attempt_index == 0 else None,
                    )
                )

            provisional = ConsensusManager.finalize(state)
            if provisional.status in {"successful", "majority"}:
                return True
            # A second candidate is only useful if there is still a split. If all
            # visible votes collapsed to one option, finalization above would have
            # returned; otherwise the next loop tests the best remaining candidate.
        return True

    def _post_reservation_decision_intent(
        self,
        state: DialogueState,
        persona: Persona,
        candidate: str,
        *,
        can_move: bool,
        alternative: str | None = None,
    ) -> MoveIntent:
        aliases = short_alias_map(state.scenario.options)
        current = state.runtimes[persona.id].current_preference or persona.preferred_option
        current_name = aliases.get(current, current)
        candidate_name = aliases.get(candidate, candidate)
        focus = [candidate]
        if current in state.scenario.option_ids and current != candidate:
            focus.append(current)
        if alternative and alternative in state.scenario.option_ids and alternative not in focus:
            focus.append(alternative)
        if can_move:
            alt_clause = (
                f", or name {aliases[alternative]} as the concrete alternative compromise"
                if alternative and alternative != current
                else ""
            )
            reason = (
                f"post-reservation decision step: choose exactly one visible outcome — switch to {candidate_name}, "
                f"stay with {current_name}{alt_clause}. If you switch, commit clearly and bridge from your earlier pick; "
                "if you stay or propose the alternative, name the one blocker that remains. No vague re-vote."
            )
        else:
            alt_clause = (
                f" or, only if it is genuinely acceptable, name {aliases[alternative]} as the alternative compromise"
                if alternative and alternative != current
                else ""
            )
            reason = (
                f"post-reservation decision step: do not accept {candidate_name}; clearly stay with {current_name}"
                f"{alt_clause}, and state the concrete blocker that remains. No new option facts."
            )
        return MoveIntent(
            speaker_id=persona.id,
            act=ActType.POST_RESERVATION_DECISION,
            reason=reason,
            option_focus=focus,
            length_hint="short",
            allow_vote_change=can_move or bool(alternative),
        )

    def _append_post_reservation_decision(
        self,
        state: DialogueState,
        persona: Persona,
        intent: MoveIntent,
        *,
        candidate: str,
        can_move: bool,
        alternative: str | None = None,
    ) -> TurnRecord:
        """Append an explicit switch/stay/alternative decision after a reservation exchange.

        This beat is controller-shaped on purpose: previous logs showed that a
        generic LLM vote prompt often produced another vague re-vote. The text is
        still grounded in the participant's visible stance and traits because the
        controller decides whether a switch is plausible from commitment,
        stubbornness, compromise tendency, and current vote pressure. Keeping the
        wording deterministic also removes one utterance/grounding call per
        holdout in the split-narrowing phase.
        """
        aliases = short_alias_map(state.scenario.options)
        rt = state.runtimes[persona.id]
        current = rt.explicit_vote or rt.current_preference or persona.preferred_option
        if current not in state.scenario.option_ids:
            current = persona.preferred_option

        target = current
        outcome = "stay"
        if can_move and self._should_switch_after_reservation(state, persona, candidate):
            target = candidate
            outcome = "switch_candidate"
        elif (
            alternative
            and alternative in state.scenario.option_ids
            and alternative != candidate
            and alternative != current
            and self._can_shift_to(state, persona, alternative)
            and self._should_offer_alternative_after_reservation(state, persona, candidate, alternative)
        ):
            target = alternative
            outcome = "switch_alternative"

        current_name = aliases.get(current, current)
        candidate_name = aliases.get(candidate, candidate)
        target_name = aliases.get(target, target)
        # Rotate phrasings so several holdouts deciding in a row don't print the
        # same template line (R9). Every variant keeps a parser-safe commitment
        # family aimed at the intended option.
        beat_index = sum(
            1 for t in state.turns
            if t.intent is not None and t.intent.act == ActType.POST_RESERVATION_DECISION
        )
        if outcome == "switch_candidate":
            variants = [
                f"I still like {current_name}, but I'll switch to {candidate_name} — it's the clearest common ground now.",
                f"Okay, {candidate_name} works for me; {current_name} clearly isn't getting the group there.",
                f"Count me in for {candidate_name} then — I'm letting {current_name} go.",
                f"I'll go with {candidate_name} because I'd rather land this than keep circling; I can live with it over {current_name}.",
            ]
        elif outcome == "switch_alternative":
            variants = [
                f"I still prefer {current_name}, but I'd go with {target_name} as the concrete compromise — {candidate_name} doesn't solve my concern.",
                f"{target_name} works for me as the middle ground; {candidate_name} still leaves my concern open.",
            ]
        else:
            variants = [
                f"My vote goes to {current_name} — {candidate_name} still does not solve my concern.",
                f"I'm sticking with {current_name}; {candidate_name} still doesn't address what bothers me.",
                f"{current_name} still gets my vote — {candidate_name} hasn't fixed my main concern.",
                f"My vote goes to {current_name}; {candidate_name} still leaves my main concern open.",
            ]
        text = variants[beat_index % len(variants)]

        act = self._parse_act(state, persona, text, intent)
        report = self._validate_turn_text(text, state, persona, intent, act)
        if report.block_state_mutation:
            text = self._safe_fallback_text(state, persona, intent, report)
            act = self._parse_act(state, persona, text, intent)
            report = self._validate_turn_text(text, state, persona, intent, act)
        record = self._append_participant(
            state,
            persona,
            text,
            act,
            intent,
            0,
            0,
            report.issues,
            False,
            list(report.issues),
            report.block_state_mutation,
        )
        if not report.block_state_mutation:
            self._apply_semantics(state, record)
        else:
            state.no_progress_count += 1
        return record

    def _should_switch_after_reservation(self, state: DialogueState, persona: Persona, candidate: str) -> bool:
        if not self._can_shift_to(state, persona, candidate):
            return False
        votes = [state.runtimes[p.id].explicit_vote for p in state.personas]
        candidate_votes = sum(1 for vote in votes if vote == candidate)
        rt = state.runtimes[persona.id]
        current = rt.explicit_vote or rt.current_preference or persona.preferred_option
        own_votes = sum(1 for vote in votes if vote == current)
        # Visible majority pressure is the candidate's *net* advantage over the
        # sim's own camp (P6): a 1-1 deadlock or a 2-2 split is symmetric and
        # exerts none — there a switch must come from traits and eroded
        # commitment, not from a phantom majority.
        advantage = max(0, candidate_votes - own_votes) / max(1, len(state.personas))
        resistance = self._candidate_resistance(state, persona, candidate)
        pressure = (
            0.18
            + 0.55 * advantage
            + 0.20 * (1.0 - persona.sim_params.compromise_threshold)
            + 0.12 * (1.0 - persona.sim_params.stubbornness)
            + 0.08 * persona.sim_params.responsiveness
            - 0.16 * rt.commitment_strength
            - 0.12 * min(resistance, 1.5)
        )
        # Only a strict visible plurality earns the extra push; ties don't.
        counts = Counter(v for v in votes if v in state.scenario.option_ids)
        max_votes = max(counts.values(), default=0)
        strict_plurality = candidate_votes == max_votes and sum(1 for c in counts.values() if c == max_votes) == 1
        plurality_bonus = 0.08 if strict_plurality else 0.0
        return pressure + plurality_bonus >= 0.32

    def _should_offer_alternative_after_reservation(
        self,
        state: DialogueState,
        persona: Persona,
        candidate: str,
        alternative: str,
    ) -> bool:
        # Alternatives are useful only when they are less resistant than the
        # tested candidate. This prevents cycling through arbitrary one-vote
        # options while still making a concrete counter-proposal visible.
        if alternative == candidate or alternative not in state.scenario.option_ids:
            return False
        cand_res = self._candidate_resistance(state, persona, candidate)
        alt_res = self._candidate_resistance(state, persona, alternative)
        return alt_res + 0.10 < cand_res

    def _holdout_alternative_candidate(
        self,
        state: DialogueState,
        persona: Persona,
        candidate: str,
        *,
        tested: set[str],
    ) -> str | None:
        """Best concrete alternative a holdout may name without opening a loop."""
        counts = Counter(
            rt.explicit_vote
            for rt in state.runtimes.values()
            if rt.explicit_vote in state.scenario.option_ids
        )
        current = state.runtimes[persona.id].current_preference or persona.preferred_option
        ordered = [oid for oid, _count in counts.most_common()]
        ordered.extend(persona.preferred_options)
        ordered.extend(state.scenario.option_ids)
        for option_id in ordered:
            if option_id == candidate or option_id in tested:
                continue
            if option_id not in state.scenario.option_ids:
                continue
            if self._hard_blocks_candidate(state, persona, option_id):
                continue
            if option_id == current or self._candidate_resistance(state, persona, option_id) <= 0.65:
                return option_id
        return current if current in state.scenario.option_ids and current != candidate else None

    def _emit_split_summary(
        self,
        state: DialogueState,
        candidate: str | None,
        votes_by_id: dict[str, str],
        *,
        attempt_index: int = 0,
        meta: dict | None = None,
    ) -> None:
        """Visible, non-malformed split summary for moderator or peer modes."""
        aliases = short_alias_map(state.scenario.options)
        counts = Counter(votes_by_id.values())
        split = ", ".join(f"{aliases[oid]} ({count})" for oid, count in counts.most_common())
        prefix = "Second narrowing attempt. " if attempt_index else ""
        if candidate:
            candidate_name = aliases[candidate]
            dissenters = [p.name for p in state.personas if votes_by_id.get(p.id) != candidate]
            dissenter_text = ", ".join(dissenters) or "the others"
            meta_text = ""
            if meta and meta.get("votes", 0) > 1:
                max_votes = max(counts.values(), default=0)
                tied_for_lead = sum(1 for count in counts.values() if count == max_votes) > 1
                if meta.get("votes") == max_votes and not tied_for_lead:
                    meta_text = " It has the visible lead, so we test it first."
                elif meta.get("votes") == max_votes and tied_for_lead:
                    meta_text = " It is tied for the lead, so we test the least-blocked candidate first."
                else:
                    meta_text = " It is the next concrete alternative, so we test it once."
            text = (
                f"{prefix}We are split: {split}. Let's test {candidate_name} as the compromise; "
                f"{dissenter_text}, what would still block that for you?"
                f"{meta_text}"
            )
        else:
            text = (
                f"{prefix}We are split: {split}. No option has a workable path yet, so let's name the blockers plainly."
            )
        if self._mod("final_vote_call"):
            self._emit(self._append_moderator(state, text, Phase.NARROWING))
            return
        caller = self._procedural_speaker(state)
        state.procedural_move_count += 1
        self._emit(self._append_peer_procedure(
            state,
            caller,
            text,
            ActType.SUMMARIZE_SPLIT,
            [candidate] if candidate else list(counts.keys())[:2],
        ))

    def _split_reservation_exchange(self, state: DialogueState, holdout: Persona, candidate: str) -> None:
        """One targeted reservation + one supporter answer inside split narrowing."""
        aliases = short_alias_map(state.scenario.options)
        state.reservation_exchange_done = True
        state.split_reservation_exchanges += 1
        candidate_name = aliases[candidate]
        reservation = MoveIntent(
            speaker_id=holdout.id,
            act=ActType.ANSWER,
            reason=(
                f"answer the split prompt: name the single strongest reservation or condition that still makes {candidate_name} "
                "hard for you. Refer only to this candidate's own listed facts or to unknowns; do not vote yet, "
                "do not borrow tradeoffs from another option, and do not invent facts"
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
                f"respond directly to {holdout.name}'s reservation about {candidate_name}; concede what the option board cannot prove, "
                "then point to the candidate's listed fact or trade-off that still makes it workable — no pressure, "
                "and do not import facts from any other option"
            ),
            addressee_id=holdout.id,
            option_focus=[candidate],
            respond_to_turn=record.index,
            length_hint="short",
        )
        self._emit(self._generate_and_append(state, response))

    def _two_person_deadlock_protocol(self, state: DialogueState, votes_by_id: dict[str, str]) -> bool:
        """Special bounded protocol for a 1-1 vote split."""
        state.two_person_deadlock_attempted = True
        personas = [p for p in state.personas if p.id in votes_by_id]
        if len(personas) != 2:
            return True
        aliases = short_alias_map(state.scenario.options)
        p1, p2 = personas
        v1, v2 = votes_by_id[p1.id], votes_by_id[p2.id]
        if self._mod("final_vote_call"):
            text = (
                f"We are one-one: {p1.name} is on {aliases[v1]}, {p2.name} is on {aliases[v2]}. "
                "Each of you name the one thing that would have to change for the other option to work."
            )
            self._emit(self._append_moderator(state, text, Phase.NARROWING))
        else:
            caller = self._procedural_speaker(state)
            state.procedural_move_count += 1
            intent = MoveIntent(
                speaker_id=caller.id,
                act=ActType.SUMMARIZE_SPLIT,
                reason=(
                    f"say the vote is one-one between {aliases[v1]} and {aliases[v2]}, then suggest each person names "
                    "the one blocker that would have to change before switching"
                ),
                option_focus=[v1, v2],
                length_hint="short",
            )
            self._emit(self._generate_and_append(state, intent))
        for speaker, other, own_vote, other_vote in ((p1, p2, v1, v2), (p2, p1, v2, v1)):
            intent = MoveIntent(
                speaker_id=speaker.id,
                act=ActType.ANSWER,
                reason=(
                    f"deadlock blocker/concession step: use exactly two short clauses. First, name your strongest "
                    f"blocker to switching from {aliases[own_vote]} to {aliases[other_vote]}. Second, name the "
                    "one condition or concession that would have to be true before you could move. Do not vote yet."
                ),
                addressee_id=other.id,
                option_focus=[other_vote, own_vote],
                length_hint="short",
            )
            self._emit(self._generate_and_append(state, intent))
        for speaker, other_vote in ((p1, v2), (p2, v1)):
            own = state.runtimes[speaker.id].explicit_vote or state.runtimes[speaker.id].current_preference or speaker.preferred_option
            can_move = self._can_shift_to(state, speaker, other_vote)
            intent = MoveIntent(
                speaker_id=speaker.id,
                act=ActType.POST_RESERVATION_DECISION,
                reason=(
                    f"deadlock final decision: choose exactly one visible outcome — switch to {aliases[other_vote]} "
                    f"or stay with {aliases.get(own, own)} and name the remaining blocker. No vague re-vote."
                ),
                option_focus=[other_vote, own] if own != other_vote else [other_vote],
                length_hint="short",
                allow_vote_change=can_move,
            )
            self._emit(
                self._append_post_reservation_decision(
                    state,
                    speaker,
                    intent,
                    candidate=other_vote,
                    can_move=can_move,
                    alternative=None,
                )
            )
        return True

    def _procedural_speaker(self, state: DialogueState) -> Persona:
        last = self._last_participant_id(state)
        candidates = [p for p in state.personas if p.id != last] or state.personas[:]
        return max(candidates, key=lambda p: p.sim_params.initiative + 0.5 * p.sim_params.engagement)

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
        self._record_token_usage(state, "utterance", tokens_in, tokens_out)
        if intent.suppress_name_prefix:
            text = strip_leading_name(text, [p.name for p in state.personas])
        act = self._parse_act(state, persona, text, intent)
        report, gti, gto = self._collect_report(text, state, persona, intent, act, focus_options)
        self._record_token_usage(state, "grounding", gti, gto)
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
            self._record_token_usage(state, "repair", ti, to)
            tokens_in += ti
            tokens_out += to
            candidate_act = self._parse_act(state, persona, candidate_text, intent)
            candidate_report, gti, gto = self._collect_report(candidate_text, state, persona, intent, candidate_act, focus_options)
            self._record_token_usage(state, "grounding", gti, gto)
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

    @staticmethod
    def _record_token_usage(state: DialogueState, kind: str, tokens_in: int, tokens_out: int) -> None:
        if tokens_in == 0 and tokens_out == 0:
            return
        bucket = state.token_usage_by_call_type.setdefault(kind, {"in": 0, "out": 0, "calls": 0})
        bucket["in"] += int(tokens_in)
        bucket["out"] += int(tokens_out)
        bucket["calls"] += 1

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

    def _append_peer_procedure(
        self,
        state: DialogueState,
        persona: Persona,
        text: str,
        act_type: ActType,
        option_focus: list[str],
    ) -> TurnRecord:
        """Append deterministic participant-owned procedure text.

        Split summaries are controller facts: vote counts and the candidate being
        tested must not drift in an LLM paraphrase. Keeping this line
        deterministic also removes one utterance+grounding call in no-moderator
        split cases while still making the procedural move visibly owned by a
        participant.
        """
        state.turn_index += 1
        text = normalise_lines(text)
        rt = state.runtimes[persona.id]
        rt.turn_count += 1
        rt.last_spoke_turn = state.turn_index
        rt.already_said.append(text)
        act = DialogueAct(
            speaker_id=persona.id,
            text=text,
            act_type=act_type,
            option_refs=[oid for oid in option_focus if oid in state.scenario.option_ids],
        )
        record = TurnRecord(
            index=state.turn_index,
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            phase=Phase.NARROWING,
            act=act,
        )
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

    def _maybe_participant_procedural(self, state: DialogueState) -> MoveIntent | None:
        """Participant-owned structure beat (issue 5), active when the moderator's
        mid-discussion voice is off: a high-initiative sim summarizes the visible
        split and suggests narrowing, proposes dropping an untouched option, or
        suggests moving toward a decision. Same stall conditions as the moderator
        nudge, bounded to two per run."""
        if self._mod("mid_discussion_nudges"):
            return None
        if state.procedural_move_count >= 2:
            return None
        if participant_turn_count(state) < max(state.min_discussion_turns + 2, state.force_narrow_turns - 1):
            return None
        if state.no_progress_count < int(cfg.conversation.moderator_stall_window):
            return None
        state.procedural_move_count += 1
        state.no_progress_count = 0
        last = self._last_participant_id(state)
        candidates = [p for p in state.personas if p.id != last] or state.personas[:]
        speaker = max(candidates, key=lambda p: p.sim_params.initiative + 0.5 * p.sim_params.engagement)
        aliases = short_alias_map(state.scenario.options)
        camps = sorted({
            rt.current_preference for rt in state.runtimes.values()
            if rt.current_preference in state.scenario.option_ids
        })
        untouched = [oid for oid, cov in state.coverage.items() if cov.mentions <= 1 and oid not in camps]
        if len(camps) >= 2:
            names = " and ".join(aliases[c] for c in camps[:2])
            reason = (
                f"you feel the group is circling — sum up in your own words that it comes down to {names}, "
                "and suggest the group focus on that trade-off or start moving toward a pick; don't push "
                "your own option in this line"
            )
            focus = camps[:2]
            act = ActType.SUMMARIZE_SPLIT
        elif untouched:
            reason = (
                f"suggest the group set {aliases[untouched[0]]} aside since nobody has made a case for it, "
                "so the discussion can focus on the real contenders"
            )
            focus = [untouched[0]]
            act = ActType.SUGGEST_NARROWING
        else:
            reason = (
                "you feel the group has compared enough — suggest in your own casual words that it's time "
                "to move toward a decision, and ask if anything important is still unresolved"
            )
            focus = []
            act = ActType.SUGGEST_NARROWING
        return MoveIntent(
            speaker_id=speaker.id,
            act=act,
            reason=reason,
            option_focus=focus,
            length_hint="short",
        )

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
        self._record_token_usage(state, "moderator", self._llm.last_tokens_in, self._llm.last_tokens_out)
        text = clean_generated(raw, "Moderator", int(cfg.utterances.word_budgets.moderator))
        if self._resolver and self._resolver.invalid_option_refs(text):
            raw = self._llm.generate(prompt + "\nOnly use the exact option names already listed.", profile="repair")
            self._record_token_usage(state, "moderator_repair", self._llm.last_tokens_in, self._llm.last_tokens_out)
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
