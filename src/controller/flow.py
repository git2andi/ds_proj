"""Flow layer: phase transitions, narrowing/vote readiness, repair machine.

FlowMixin owns global progress: the opening round, the discussion loop's
stop condition, bounded narrowing, formal vote collection, and the single
repair state machine. It is mixed into DialogueRunner and drives generation
through the shared pipeline (`self._generate_and_append`); it never renders
text itself and mutates only phase/repair/candidate flow state.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import prompts
from aliases import short_alias_map
from config_loader import cfg
from consensus import ConsensusManager, participant_turn_count, public_support, visible_votes_from_transcript
from controller import threads
from models import (
    ActType,
    DialogueState,
    MoveIntent,
    Persona,
    Phase,
    RepairState,
    ThreadStatus,
    ThreadType,
    TurnRecord,
)


class FlowMixin:
    def _opening_round(self, state: DialogueState) -> None:
        state.phase = Phase.OPENING
        for persona in self._opening_order(state.personas):
            intent = MoveIntent(
                speaker_id=persona.id,
                act=ActType.OPENING,
                reason="state the current favorite and one grounded reason without making a final vote",
                route_source="opening",
                option_focus=[persona.preferred_option],
            )
            self._emit(self._generate_and_append(state, intent))
        self._mark_phase(state, Phase.DISCUSSION, "all participants gave an opening view")

    def _discussion_loop(self, state: DialogueState) -> None:
        while True:
            if self._ready_to_narrow(state):
                # Comparisons live inside the discussion (6.4): leaving for
                # narrowing settles any open head-to-head.
                threads.resolve_comparison_threads(state, reason="left discussion for narrowing")
                self._mark_phase(state, Phase.NARROWING, self._narrow_reason(state))
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

    def _narrowing_phase(self, state: DialogueState) -> None:
        """Bounded narrowing (12.3): test the candidate/top pair, then vote.

        One summary beat (participant-led when possible, moderator-led when the
        discussion was circling or forced) plus one holdout reaction beat. If
        the tested candidate visibly collapses, fall back to discussion at most
        once while turn budget remains; otherwise proceed to voting.
        """
        allow_return = bool(cfg.narrowing.get("allow_return_to_discussion_once", True))
        while True:
            candidate = self._candidate_for_vote(state)
            state.candidate_option = candidate
            pair = self._current_top_pair(state) or ([candidate] if candidate else [])
            moderator_led = self._mod("final_vote_call") and (
                state.no_progress_count >= int(cfg.conversation.moderator_stall_window)
                or participant_turn_count(state) >= state.force_narrow_turns
            )
            if moderator_led:
                self._emit_moderator_narrowing(state, candidate, pair)
            else:
                self._emit_participant_narrowing(state, candidate, pair)
            self._emit_narrowing_reaction(state, candidate)

            collapsed = self._public_candidate(state) is None or (
                candidate is not None
                and threads.hot_blocking_thread_against(state, [candidate]) is not None
            )
            if (
                collapsed
                and allow_return
                and not state.narrowing_returned
                and participant_turn_count(state) < state.hard_max_turns
            ):
                state.narrowing_returned = True
                self._mark_phase(
                    state, Phase.DISCUSSION,
                    "candidate collapsed during narrowing; returning to discussion once",
                )
                self._discussion_loop(state)
                continue
            self._mark_phase(state, Phase.VOTING, "narrowing complete; collecting formal votes")
            return

    def _emit_participant_narrowing(self, state: DialogueState, candidate: str | None, pair: list[str]) -> None:
        """Participant-led narrowing: someone visibly summarizes the emerging pick."""
        aliases = short_alias_map(state.scenario.options)
        speaker = self._procedural_speaker(state)
        focus = [oid for oid in pair if oid in state.scenario.option_ids][:2]
        if candidate and len(focus) <= 1:
            reason = (
                f"you feel the group has mostly landed on {aliases.get(candidate, candidate)} — sum that up "
                "in your own words and suggest checking whether it actually works for everyone"
            )
            focus = [candidate]
        elif len(focus) == 2:
            reason = (
                f"sum up that it has come down to {aliases[focus[0]]} versus {aliases[focus[1]]}, and suggest "
                "the group settle that trade-off now instead of circling"
            )
        else:
            reason = (
                "you feel the group has compared enough — suggest in your own casual words that it's time "
                "to move toward a decision, and ask if anything important is still unresolved"
            )
        intent = MoveIntent(
            speaker_id=speaker.id,
            act=ActType.PROCESS,
            reason=reason,
            route_source="participant_narrowing",
            option_focus=focus,
            length_hint="short",
        )
        self._emit(self._generate_and_append(state, intent))

    def _emit_moderator_narrowing(self, state: DialogueState, candidate: str | None, pair: list[str]) -> None:
        """Moderator-led narrowing summary when circling or forced (12.3)."""
        aliases = short_alias_map(state.scenario.options)
        if candidate:
            requested = (
                f"note that {aliases.get(candidate, candidate)} has emerged as the likely pick and ask whether "
                "anything concrete still blocks it before the group decides — no verdict, one focusing question"
            )
            focus = [candidate]
        elif len(pair) >= 2:
            requested = (
                f"ask the group to settle {aliases[pair[0]]} versus {aliases[pair[1]]} now — which one they "
                "could actually commit to, not another round of comparison"
            )
            focus = pair[:2]
        else:
            requested = "ask for the strongest remaining concern before choosing"
            focus = []
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state,
                "the discussion is ready to narrow",
                aliases.get(candidate) if candidate else None,
                requested_action=requested,
                focus_options=focus,
            ),
            state,
        )
        self._emit(self._append_moderator(state, text, state.phase))

    def _emit_narrowing_reaction(self, state: DialogueState, candidate: str | None) -> None:
        """One bounded reaction beat: the most relevant holdout tests the candidate."""
        if candidate not in state.scenario.option_ids:
            return
        holdouts = [
            p for p in state.personas
            if state.runtimes[p.id].top_option() != candidate
            and candidate not in state.runtimes[p.id].rejected_options()
        ]
        if not holdouts:
            return
        speaker = min(holdouts, key=lambda p: self._candidate_resistance(state, p, candidate))
        aliases = short_alias_map(state.scenario.options)
        intent = MoveIntent(
            speaker_id=speaker.id,
            act=ActType.SUPPORT,
            reason=(
                f"the group is testing {aliases.get(candidate, candidate)}: say honestly whether you could "
                "live with it, or name the one concrete thing that still blocks it — this is not a final vote"
            ),
            route_source="participant_narrowing",
            option_focus=[candidate],
            length_hint="short",
        )
        self._emit(self._generate_and_append(state, intent))

    def _decision_loop(self, state: DialogueState) -> None:
        """Formal vote collection plus the single bounded repair state machine (13.7).

        One flow handles successful, majority, holdout, split, blocker,
        unclear-vote, and two-person cases: collect everyone's formal vote,
        then repeatedly classify at most one repair objective, run it bounded,
        and re-tally. Outcome definitions stay exact (consensus.py).
        """
        # Clear any owed answer, then collect one formal vote from everyone.
        self._resolve_pending_question(state)
        candidate = self._candidate_for_vote(state)
        state.candidate_option = candidate
        order = self._vote_order(state, candidate)
        if self._mod("final_vote_call"):
            nudge_record, target_id = self._moderator_vote_nudge(state, candidate, "let's test where everyone stands")
            self._emit(nudge_record)
            if target_id:
                order.sort(key=lambda p: p.id != target_id)
        for persona in order:
            self._emit(self._generate_and_append(state, self._vote_intent(state, persona, candidate)))

        while True:
            provisional = ConsensusManager.finalize(state)
            if provisional.status == "successful":
                state.outcome = provisional
                self._mark_phase(state, Phase.CLOSING, "successful: all formal votes converged")
                return
            repair = self._classify_repair(state, provisional)
            if repair is None:
                state.outcome = provisional
                reason = (
                    "majority stands after bounded repair"
                    if provisional.status == "majority"
                    else "no visible majority after bounded repair; unresolved"
                )
                self._mark_phase(state, Phase.CLOSING, reason)
                return
            self._run_repair(state, repair)

    # ------------------------------------------------------------------
    # Repair state machine (13.7): one bounded objective at a time
    # ------------------------------------------------------------------

    def _classify_repair(self, state: DialogueState, provisional) -> RepairState | None:
        """Select at most one repair objective, in the 13.7 priority order.

        1. unclear formal vote  2. (hard blockers are handled inside the
        holdout/split flows via `_valid_holdout_against`/candidate ranking)
        3. majority holdout  4. split/no-majority  5. two-person deadlock.
        Each reason runs at most once per run.
        """
        ran = {r.repair_reason for r in state.repair_history}
        formal = visible_votes_from_transcript(state)
        unclear = [p for p in state.personas if p.id not in formal]
        if unclear and "unclear_vote" not in ran and int(cfg.conversation.max_vote_rounds) > 1:
            return RepairState(
                repair_reason="unclear_vote",
                candidate_or_pair=[state.candidate_option] if state.candidate_option in state.scenario.option_ids else [],
                participants_involved=[p.id for p in unclear],
                max_attempts=1,
            )
        if provisional.status == "majority":
            if ran & {"majority_holdout", "split_vote", "two_person_deadlock"}:
                return None
            winner = provisional.final_option
            dissenters = [p.id for p in state.personas if formal.get(p.id) != winner]
            if winner not in state.scenario.option_ids or not dissenters:
                return None
            return RepairState(
                repair_reason="majority_holdout",
                candidate_or_pair=[winner],
                participants_involved=dissenters,
                max_attempts=1,
            )
        distinct = sorted({v for v in formal.values()})
        if len(distinct) >= 2:
            if len(state.personas) == 2 and len(formal) == 2:
                if "two_person_deadlock" in ran:
                    return None
                return RepairState(
                    repair_reason="two_person_deadlock",
                    candidate_or_pair=distinct[:2],
                    participants_involved=sorted(formal),
                    max_attempts=1,
                )
            if "split_vote" in ran:
                return None
            return RepairState(
                repair_reason="split_vote",
                candidate_or_pair=distinct[:2],
                participants_involved=[p.id for p in state.personas],
                max_attempts=2,
            )
        return None

    def _run_repair(self, state: DialogueState, repair: RepairState) -> None:
        """Run one bounded repair objective; only one may be active at once."""
        state.active_repair = repair
        if repair.repair_reason != "unclear_vote" and state.phase is not Phase.COMPROMISE_REPAIR:
            self._mark_phase(
                state, Phase.COMPROMISE_REPAIR, f"running {repair.repair_reason} repair"
            )
        handlers = {
            "unclear_vote": self._repair_unclear_vote,
            "majority_holdout": self._repair_majority_holdout,
            "split_vote": self._repair_split_vote,
            "two_person_deadlock": self._repair_two_person_deadlock,
        }
        completed = handlers[repair.repair_reason](state, repair)
        repair.status = "resolved" if completed else "exhausted"
        state.repair_history.append(repair)
        state.active_repair = None
        state.controller_trace.append({
            "type": "repair",
            "turn_index": state.turn_index,
            "repair_reason": repair.repair_reason,
            "candidate_or_pair": list(repair.candidate_or_pair),
            "attempt_count": repair.attempt_count,
            "status": repair.status,
        })

    def _repair_unclear_vote(self, state: DialogueState, repair: RepairState) -> bool:
        """Bounded clarification round: re-prompt only the unclear voters (13.2)."""
        repair.attempt_count += 1
        candidate = state.candidate_option or self._candidate_for_vote(state)
        unclear = [
            state.persona_by_id(pid)
            for pid in repair.participants_involved
            if not self._has_clear_vote(state, pid)
        ]
        if not unclear:
            return True
        if self._mod("final_vote_call"):
            nudge_record, target_id = self._moderator_vote_nudge(
                state, candidate, "let's hear from whoever hasn't given a clear vote"
            )
            self._emit(nudge_record)
            if target_id:
                unclear.sort(key=lambda p: p.id != target_id)
        for persona in unclear:
            intent = self._vote_intent(state, persona, candidate)
            intent.route_source = "vote_clarification"
            self._emit(self._generate_and_append(state, intent))
        return all(self._has_clear_vote(state, p.id) for p in unclear)

    def _repair_majority_holdout(self, state: DialogueState, repair: RepairState) -> bool:
        """Bounded holdout repair after a provisional majority (13.4).

        The holdouts are acknowledged; the most movable one gets a two-turn
        reservation exchange; each holdout then gives one visible stay/switch
        commitment. May upgrade the outcome to unanimity, never pressures a
        valid holdout or hard blocker into fake agreement. Runs at most once.
        """
        repair.attempt_count += 1
        winner = repair.candidate_or_pair[0] if repair.candidate_or_pair else None
        if winner not in state.scenario.option_ids:
            return False
        dissenters = [
            state.persona_by_id(pid)
            for pid in repair.participants_involved
            if pid in state.runtimes
        ]
        if not dissenters:
            return True
        winner_name = state.scenario.option(winner).name
        movers = [
            p for p in dissenters
            if self._can_shift_to(state, p, winner, final_decision=True) and not self._valid_holdout_against(state, p, winner)
        ]
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
            self._emit(self._append_moderator(state, text, state.phase))
        elif negotiator is not None:
            # No moderator voice: a majority supporter owns the probe instead.
            self._emit_peer_holdout_probe(state, negotiator, winner)
        if negotiator is not None:
            self._reservation_exchange(state, negotiator, winner)
        for persona in dissenters:
            can_move = self._can_shift_to(state, persona, winner, final_decision=True) and not self._valid_holdout_against(state, persona, winner)
            self._emit(
                self._append_final_decision(
                    state,
                    persona,
                    candidate=winner,
                    can_move=can_move,
                    route_source="majority_holdout_repair",
                )
            )
        return all(self._has_clear_vote(state, p.id) for p in dissenters)

    @staticmethod
    def _candidate_supporter(state: DialogueState, candidate: str, *, exclude: str) -> Persona | None:
        """Most engaged formal supporter of the candidate (shared repair op)."""
        formal = visible_votes_from_transcript(state)
        supporters = [
            p for p in state.personas
            if p.id != exclude and formal.get(p.id) == candidate
        ]
        if not supporters:
            return None
        return max(supporters, key=lambda p: (p.sim_params.engagement, random.random()))

    def _reservation_exchange(
        self, state: DialogueState, holdout: Persona, candidate: str, *, split: bool = False
    ) -> None:
        """Bounded reservation micro-negotiation (issue 4), exactly two turns:
        the holdout states one concrete reservation about the candidate (no vote
        yet), and one formal supporter of the candidate responds to it honestly.
        Used by both the majority-holdout and split-vote repairs; only the route
        source and instruction wording differ. Bounded by the calling repair
        objective; the holdout's actual decision comes in its later beat."""
        state.reservation_exchanges += 1
        aliases = short_alias_map(state.scenario.options)
        candidate_name = aliases[candidate]
        route_source = "split_vote_repair" if split else "majority_holdout_repair"
        if split:
            reservation_reason = (
                f"answer the split prompt: name the single strongest reservation or condition that still makes {candidate_name} "
                "hard for you. Refer only to this candidate's own listed facts or to unknowns; do not vote yet, "
                "do not borrow tradeoffs from another option, and do not invent facts"
            )
        else:
            reservation_reason = (
                f"say concretely what still makes you hesitate about {candidate_name} — one specific "
                "reservation or condition, grounded in the option facts or what they leave unknown; "
                "do not cast a vote yet"
            )
        reservation = MoveIntent(
            speaker_id=holdout.id,
            act=ActType.ANSWER,
            reason=reservation_reason,
            route_source=route_source,
            option_focus=[candidate],
            length_hint="short",
        )
        record = self._generate_and_append(state, reservation)
        self._emit(record)
        if not record.text.strip():
            return
        responder = self._candidate_supporter(state, candidate, exclude=holdout.id)
        if responder is None:
            return
        if split:
            response_reason = (
                f"respond directly to {holdout.name}'s reservation about {candidate_name}; concede what the option board cannot prove, "
                "then point to the candidate's listed fact or trade-off that still makes it workable — no pressure, "
                "and do not import facts from any other option"
            )
        else:
            response_reason = (
                f"respond to {holdout.name}'s reservation about {candidate_name} honestly: use only "
                "the option facts, concede what the board cannot prove, and point to what still helps "
                "their concern — no pressure to switch"
            )
        response = MoveIntent(
            speaker_id=responder.id,
            act=ActType.ANSWER,
            reason=response_reason,
            route_source=route_source,
            addressee_id=holdout.id,
            option_focus=[candidate],
            respond_to_turn=record.index,
            length_hint="short",
        )
        self._emit(self._generate_and_append(state, response))

    def _emit_peer_holdout_probe(self, state: DialogueState, holdout: Persona, candidate: str) -> None:
        """With the moderator voice off, a supporter of the candidate asks the
        holdout what still blocks agreement (participant-owned procedure)."""
        asker = self._candidate_supporter(state, candidate, exclude=holdout.id)
        if asker is None:
            return
        aliases = short_alias_map(state.scenario.options)
        intent = MoveIntent(
            speaker_id=asker.id,
            act=ActType.PROCESS,
            reason=(
                f"most of the group has landed on {aliases[candidate]}; ask {holdout.name} in a friendly, "
                "genuine way what still holds them back or what they would need — no pressure"
            ),
            route_source="majority_holdout_repair",
            addressee_id=holdout.id,
            option_focus=[candidate],
            length_hint="short",
        )
        self._emit(self._generate_and_append(state, intent))
        state.procedural_move_count += 1

    def _repair_split_vote(self, state: DialogueState, repair: RepairState) -> bool:
        """Bounded split-vote repair when formal votes have no majority (13.5).

        Controller-owned: the split summary is deterministic when the moderator
        is on, and the controller chooses the candidate, the dissenters, and
        the stop point. Bounded by repair.max_attempts candidate tests; each
        attempt is one exchange plus one visible stay/switch round.
        """
        tested: set[str] = set()
        while repair.attempt_count < repair.max_attempts:
            attempt_index = repair.attempt_count
            repair.attempt_count += 1
            votes_by_id = {
                pid: vote
                for pid, vote in visible_votes_from_transcript(state).items()
                if vote in state.scenario.option_ids
            }
            ranked = self._rank_split_candidates(state, votes_by_id, exclude=tested)
            if not ranked:
                # No plausible mover: still emit one honest split summary so the
                # unresolved outcome is socially explained, not an abrupt stop.
                if attempt_index == 0:
                    self._emit_split_summary(state, None, votes_by_id)
                return False
            leader, dissenters, movers, meta = ranked[0]
            tested.add(leader)
            state.candidate_option = leader
            repair.candidate_or_pair = [leader]
            caller_id = self._emit_split_summary(state, leader, votes_by_id, attempt_index=attempt_index, meta=meta)

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
            if caller_id is not None:
                # Avoid a participant immediately answering their own split prompt.
                ordered_holdouts.sort(key=lambda p: p.id == caller_id)
            for holdout in ordered_holdouts[:reservation_limit]:
                self._reservation_exchange(state, holdout, leader, split=True)

            for persona in dissenters:
                can_move = persona in movers
                alternative = self._holdout_alternative_candidate(state, persona, leader, tested=tested)
                self._emit(
                    self._append_final_decision(
                        state,
                        persona,
                        candidate=leader,
                        can_move=can_move,
                        route_source="split_vote_repair",
                        alternative=alternative if attempt_index == 0 else None,
                    )
                )

            provisional = ConsensusManager.finalize(state)
            if provisional.status in {"successful", "majority"}:
                return True
            # A second candidate is only useful if there is still a split;
            # the next attempt tests the best remaining candidate.
        return False

    def _append_final_decision(
        self,
        state: DialogueState,
        persona: Persona,
        *,
        candidate: str,
        can_move: bool,
        route_source: str,
        alternative: str | None = None,
    ) -> TurnRecord:
        """Append a visible switch/stay/alternative decision after reservations.

        The single place a repair's final beat is computed: target, expected
        outcome, grounded reason, and generation intent are derived once here
        (used by the majority-holdout, split-vote, and deadlock repairs). The
        controller decides the allowed outcome so consensus logic stays stable,
        but the actual line is generated by the LLM and validated against the
        required target. If the LLM drifts to a different option, validation
        repairs or falls back to a parser-safe line.
        """
        aliases = short_alias_map(state.scenario.options)
        rt = state.runtimes[persona.id]
        current = rt.explicit_vote or rt.top_option() or persona.preferred_option
        if current not in state.scenario.option_ids:
            current = persona.preferred_option

        target = current
        outcome = "stay"
        valid_holdout = self._valid_holdout_against(state, persona, candidate)
        if valid_holdout:
            can_move = False
        if can_move and self._should_switch_after_reservation(state, persona, candidate):
            target = candidate
            outcome = "switch_candidate"
        elif (
            alternative
            and alternative in state.scenario.option_ids
            and alternative != candidate
            and alternative != current
            and self._can_shift_to(state, persona, alternative, final_decision=True)
            and self._should_offer_alternative_after_reservation(state, persona, candidate, alternative)
        ):
            target = alternative
            outcome = "switch_alternative"

        current_name = aliases.get(current, current)
        candidate_name = aliases.get(candidate, candidate)
        target_name = aliases.get(target, target)
        allowed_reason = self._allowed_decision_reason(state, persona, target, current=current, candidate=candidate, outcome=outcome)
        focus = [target]
        for oid in (candidate, current, alternative):
            if oid and oid in state.scenario.option_ids and oid not in focus:
                focus.append(oid)
        if outcome == "switch_candidate":
            reason = (
                f"final decision: controller outcome is a switch from {current_name} to {candidate_name}. "
                f"Use exactly this grounded reason, in your own words: {allowed_reason}. Do not add new facts or pressure language"
            )
        elif outcome == "switch_alternative":
            reason = (
                f"final decision: controller outcome is a concrete alternative switch from {current_name} to {target_name}. "
                f"Use exactly this grounded reason, in your own words: {allowed_reason}. Do not add new facts"
            )
        else:
            reason = (
                f"final decision: controller outcome is staying with {current_name}; {candidate_name} still does not solve the concern. "
                f"Use exactly this grounded reason, in your own words: {allowed_reason}. Do not accept {candidate_name}"
            )
        generated_intent = MoveIntent(
            speaker_id=persona.id,
            act=ActType.VOTE,
            reason=reason,
            route_source=route_source,
            option_focus=focus,
            length_hint="short",
            allow_vote_change=target != current,
            required_vote=target,
            old_preference=(current if target != current else None),
            allowed_reason=allowed_reason,
        )
        return self._generate_and_append(state, generated_intent)


    def _allowed_decision_reason(
        self,
        state: DialogueState,
        persona: Persona,
        target: str,
        *,
        current: str | None,
        candidate: str | None,
        outcome: str,
    ) -> str:
        """One grounded reason fragment for a controller-selected final move.

        The generator may paraphrase this, but it should not invent a new factual
        justification. Keeping this reason in the intent makes LLM-rendered
        switches varied while still parseable and grounded.
        """
        if target not in state.scenario.option_ids:
            return "it is the clearest option left in the visible discussion"
        rt = state.runtimes[persona.id]
        personal_for = rt.reason_for(target)
        if outcome != "stay" and personal_for:
            return personal_for
        card = state.scenario.option(target)
        if outcome == "stay":
            if candidate and candidate in state.scenario.option_ids:
                cand = state.scenario.option(candidate)
                if cand.concern:
                    return f"listed concern remains: {cand.concern}"
            personal_against = rt.reason_against(candidate) if candidate else ""
            if personal_against:
                return personal_against
            if card.upside:
                return card.upside
            return "this is still the more defensible option from the listed facts"
        if card.upside:
            return card.upside
        if card.attrs:
            key, value = next(iter(card.attrs.items()))
            return f"{key.replace('_', ' ')}: {value}"
        return "it has the broadest visible support now"

    def _should_switch_after_reservation(self, state: DialogueState, persona: Persona, candidate: str) -> bool:
        if not self._can_shift_to(state, persona, candidate, final_decision=True) or self._valid_holdout_against(state, persona, candidate):
            return False
        formal = visible_votes_from_transcript(state)
        votes = list(formal.values())
        candidate_votes = sum(1 for vote in votes if vote == candidate)
        rt = state.runtimes[persona.id]
        current = formal.get(persona.id) or rt.explicit_vote or rt.top_option() or persona.preferred_option
        own_votes = sum(1 for vote in votes if vote == current)
        # Never "compromise" downhill: switching to a smaller visible camp breaks a
        # forming majority and makes flexible sims ping-pong between candidates.
        if candidate_votes < own_votes:
            return False
        advantage = max(0, candidate_votes - own_votes) / max(1, len(state.personas))
        resistance = self._candidate_resistance(state, persona, candidate)
        counts = Counter(v for v in votes if v in state.scenario.option_ids)
        max_votes = max(counts.values(), default=0)
        strict_plurality = candidate_votes == max_votes and sum(1 for c in counts.values() if c == max_votes) == 1
        tied_leader = candidate_votes == own_votes and candidate_votes == max_votes and len(counts) > 1
        # Final switching is switch_resistance territory (Section 14).
        flexibility = 1.0 - persona.sim_params.switch_resistance
        pressure = (
            0.22
            + 0.58 * advantage
            + 0.48 * flexibility
            - 0.12 * rt.commitment_strength
            - 0.08 * min(resistance, 1.5)
        )
        plurality_bonus = 0.10 if strict_plurality else 0.0
        tie_compromise_bonus = 0.08 if (tied_leader and flexibility >= 0.30) else 0.0
        threshold = 0.39
        if persona.sim_params.switch_resistance >= 0.70:
            threshold += 0.12
        return pressure + plurality_bonus + tie_compromise_bonus >= threshold

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
            vote for vote in visible_votes_from_transcript(state).values()
            if vote in state.scenario.option_ids
        )
        current = state.runtimes[persona.id].top_option() or persona.preferred_option
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
    ) -> str | None:
        """Visible, non-malformed split summary.

        The moderator owns exact procedural wording (vote counts, tested
        candidate). When no moderator vote-call exists, a participant owns the
        move instead and must sound like a group member (P1): no vote-count
        dumps, no candidate-testing vocabulary, and never addressing themself.
        Returns None for moderator wording and the peer caller id otherwise.
        """
        aliases = short_alias_map(state.scenario.options)
        counts = Counter(votes_by_id.values())
        if self._mod("final_vote_call"):
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
            self._emit(self._append_moderator(state, text, state.phase))
            return None
        caller = self._procedural_speaker(state)
        contested = [aliases[oid] for oid, _count in counts.most_common(3)]
        if len(contested) > 2:
            split_text = ", ".join(contested[:-1]) + " and " + contested[-1]
        else:
            split_text = " and ".join(contested)
        if candidate:
            candidate_name = aliases[candidate]
            holdouts = [
                p.name for p in state.personas
                if votes_by_id.get(p.id) != candidate and p.id != caller.id
            ]
            if holdouts:
                names = ", ".join(holdouts)
                if attempt_index:
                    text = f"Okay, other way around then: would {candidate_name} work? {names}, what still bothers you about it?"
                else:
                    text = (
                        f"Looks like we're still split between {split_text}. "
                        f"Could {candidate_name} work for everyone? {names}, what still bothers you about it?"
                    )
            else:
                # The caller is the only visible holdout: voice it, don't poll it.
                if attempt_index:
                    text = f"Okay, other way around then: {candidate_name}. I'm the one still hesitating, so let me say what bothers me."
                else:
                    text = f"Seems like {candidate_name} has the most support and I'm the holdout — let me say what still bothers me about it."
            focus = [candidate]
        else:
            text = (
                f"We keep going back and forth between {split_text}. "
                "Maybe everyone just says the one thing that's really blocking them."
            )
            focus = [oid for oid, _count in counts.most_common(2)]
        self._emit(self._append_peer_procedure(
            state,
            caller,
            text,
            ActType.PROCESS,
            focus,
        ))
        state.procedural_move_count += 1
        return caller.id

    def _emit_unresolved_acknowledgement(self, state: DialogueState, outcome) -> None:
        """Add one participant line that socially owns an unresolved outcome.

        The final outcome is already decided; this line is appended as a closure
        reaction and does not update votes. That keeps unresolved endings honest
        while avoiding the abrupt "last vote -> moderator close" feel.
        """
        aliases = short_alias_map(state.scenario.options)
        votes = outcome.metadata.get("visible_votes", {})
        counts = Counter(v for v in votes.values() if v in state.scenario.option_ids)
        contested = [oid for oid, _count in counts.most_common(3)]
        if len(contested) < 2:
            latent = Counter(
                rt.top_option() for rt in state.runtimes.values()
                if rt.top_option() in state.scenario.option_ids
            )
            for oid, _count in latent.most_common():
                if oid not in contested:
                    contested.append(oid)
                if len(contested) == 2:
                    break
        caller = self._procedural_speaker(state)
        if len(contested) >= 3 and len(set(counts.values())) == 1:
            a, b, c = (aliases[oid] for oid in contested[:3])
            text = f"I think we're genuinely stuck — {a}, {b} and {c} all still have support."
        elif len(contested) >= 2:
            a, b = aliases[contested[0]], aliases[contested[1]]
            text = f"I think we're genuinely stuck between {a} and {b}."
        elif contested:
            text = f"{aliases[contested[0]]} is the closest we got, but not everyone is sold on it."
        else:
            text = "We've circled long enough without anyone landing on a pick."
        self._emit(self._append_peer_procedure(
            state,
            caller,
            text,
            ActType.SUPPORT,
            contested[:3],
            phase=Phase.CLOSING,
        ))

    def _emit_peer_closing(self, state: DialogueState, outcome) -> None:
        """One participant-owned wrap-up line when moderator closing is off (P10).

        Deterministic: the outcome facts must not drift in a paraphrase, and
        the decision is already made — this line is social closure, not a new
        move. Natural group-member wording, no vote counts (P1 rules apply).
        """
        aliases = short_alias_map(state.scenario.options)
        caller = self._procedural_speaker(state)
        final = outcome.final_option
        if outcome.status == "successful" and final:
            text = f"Okay, {aliases[final]} it is — glad we landed on the same thing."
        elif outcome.status == "majority" and final:
            votes = outcome.metadata.get("visible_votes", {})
            caller_vote = votes.get(caller.id)
            holdouts = [
                p.name for p in state.personas
                if votes.get(p.id) != final and p.id != caller.id
            ]
            if caller_vote and caller_vote != final:
                mine = aliases.get(caller_vote, caller_vote)
                text = f"Alright, {aliases[final]} has the majority — I'm still on {mine}, but so be it."
            elif holdouts:
                text = f"So {aliases[final]} wins for most of us, with {', '.join(holdouts)} still not sold."
            else:
                text = f"Okay, then {aliases[final]} has the majority."
        else:
            text = random.choice([
                "Looks like we're not landing this one today.",
                "Okay, let's leave it here for now.",
                "Better to leave it open than force it.",
            ])
        self._emit(self._append_peer_procedure(
            state,
            caller,
            text,
            ActType.SUPPORT,
            [final] if final else [],
            phase=Phase.CLOSING,
        ))

    def _repair_two_person_deadlock(self, state: DialogueState, repair: RepairState) -> bool:
        """Bounded 1-1 deadlock repair (13.6): each side names its blocker and
        condition, then gives one final visible stay/switch commitment."""
        repair.attempt_count += 1
        formal = visible_votes_from_transcript(state)
        votes_by_id = {
            pid: formal[pid]
            for pid in repair.participants_involved
            if pid in state.runtimes and formal.get(pid) in state.scenario.option_ids
        }
        personas = [p for p in state.personas if p.id in votes_by_id]
        if len(personas) != 2 or len(set(votes_by_id.values())) != 2:
            return False
        aliases = short_alias_map(state.scenario.options)
        p1, p2 = personas
        v1, v2 = votes_by_id[p1.id], votes_by_id[p2.id]
        if self._mod("final_vote_call"):
            text = (
                f"We are one-one: {p1.name} is on {aliases[v1]}, {p2.name} is on {aliases[v2]}. "
                "Each of you name the one thing that would have to change for the other option to work."
            )
            self._emit(self._append_moderator(state, text, state.phase))
        else:
            caller = self._procedural_speaker(state)
            intent = MoveIntent(
                speaker_id=caller.id,
                act=ActType.PROCESS,
                reason=(
                    f"say the vote is one-one between {aliases[v1]} and {aliases[v2]}, then suggest each person names "
                    "the one blocker that would have to change before switching"
                ),
                route_source="two_person_deadlock_repair",
                option_focus=[v1, v2],
                length_hint="short",
            )
            self._emit(self._generate_and_append(state, intent))
            state.procedural_move_count += 1
        for speaker, other, own_vote, other_vote in ((p1, p2, v1, v2), (p2, p1, v2, v1)):
            intent = MoveIntent(
                speaker_id=speaker.id,
                act=ActType.ANSWER,
                reason=(
                    f"deadlock blocker/concession step: use exactly two short clauses. First, name your strongest "
                    f"blocker to switching from {aliases[own_vote]} to {aliases[other_vote]}. Second, name the "
                    "one condition or concession that would have to be true before you could move. Do not vote yet."
                ),
                route_source="two_person_deadlock_repair",
                addressee_id=other.id,
                option_focus=[other_vote, own_vote],
                length_hint="short",
            )
            self._emit(self._generate_and_append(state, intent))
        for speaker, other_vote in ((p1, v2), (p2, v1)):
            can_move = self._can_shift_to(state, speaker, other_vote, final_decision=True)
            self._emit(
                self._append_final_decision(
                    state,
                    speaker,
                    candidate=other_vote,
                    can_move=can_move,
                    route_source="two_person_deadlock_repair",
                )
            )
        # Resolved only if the deadlock actually broke into a shared pick.
        final = visible_votes_from_transcript(state)
        return len({final.get(p1.id), final.get(p2.id)} - {None}) == 1

    def _procedural_speaker(self, state: DialogueState) -> Persona:
        last = self._last_participant_id(state)
        candidates = [p for p in state.personas if p.id != last] or state.personas[:]
        return max(candidates, key=lambda p: (p.sim_params.engagement, p.id))

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
        candidate = self._public_candidate(state) or self._latent_leading_option(state)
        candidate_name = state.scenario.option(candidate).name if candidate else None
        target_id, requested_action, focus, probe_key = self._moderator_intervention_details(state, candidate)
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
        if probe_key and probe_key in state.threads:
            # Charged only now that the probing moderator turn actually exists.
            state.threads[probe_key].probe_count += 1
        if target_id and target_id in state.runtimes:
            # A targeted moderator question is a direct question thread: the
            # named participant owes the next answer.
            threads.open_thread(
                state,
                thread_type=ThreadType.QUESTION,
                focus_options=[o for o in focus if o in state.scenario.option_ids][:2],
                issue_key=threads.normalize_issue_key(
                    text, state.scenario, [p.name for p in state.personas], focus_options=focus
                ),
                started_by="moderator",
                source_turn_index=record.index,
                required_respondent=target_id,
                question_scope="direct",
            )
        return record

    def _maybe_participant_procedural(self, state: DialogueState) -> MoveIntent | None:
        """Participant-owned structure beat (issue 5), active when the moderator's
        mid-discussion voice is off: an engaged sim summarizes the visible
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
        # procedural_move_count / no_progress_count are charged post-turn via
        # _post_turn_route_accounting (route_source="participant_narrowing").
        last = self._last_participant_id(state)
        candidates = [p for p in state.personas if p.id != last] or state.personas[:]
        speaker = max(candidates, key=lambda p: (p.sim_params.engagement, p.id))
        aliases = short_alias_map(state.scenario.options)
        camps = sorted({
            rt.top_option() for rt in state.runtimes.values()
            if rt.top_option() in state.scenario.option_ids
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
            act = ActType.PROCESS
        elif untouched:
            reason = (
                f"suggest the group set {aliases[untouched[0]]} aside since nobody has made a case for it, "
                "so the discussion can focus on the real contenders"
            )
            focus = [untouched[0]]
            act = ActType.PROCESS
        else:
            reason = (
                "you feel the group has compared enough — suggest in your own casual words that it's time "
                "to move toward a decision, and ask if anything important is still unresolved"
            )
            focus = []
            act = ActType.PROCESS
        return MoveIntent(
            speaker_id=speaker.id,
            act=act,
            reason=reason,
            route_source="participant_narrowing",
            option_focus=focus,
            length_hint="short",
        )

    def _moderator_vote_nudge(self, state: DialogueState, candidate: str, reason: str) -> tuple[TurnRecord, str | None]:
        target_id, requested_action, focus, _probe_key = self._moderator_intervention_details(state, candidate, voting=True)
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
        record = self._append_moderator(state, text, state.phase)
        # No obligation state: the decision loop already asks the targeted
        # participant first (order sort in _decision_loop).
        return record, target_id

    def _moderator_intervention_details(
        self,
        state: DialogueState,
        candidate: str | None,
        *,
        voting: bool = False,
    ) -> tuple[str | None, str, list[str], str | None]:
        """Return (target_id, requested_action, focus_option_ids, blocker_probe_key).

        Read-only: the blocker-probe key is charged by the caller after the
        moderator turn is actually appended, never during selection.
        """
        gap = self._coverage_gap_option(state)
        if gap is not None and not voting:
            return (
                None,
                f"ask the group to briefly compare Option {gap} before narrowing",
                [gap],
                None,
            )

        answer_thread = self._required_answer_thread(state)
        if answer_thread is not None:
            return (
                answer_thread.required_respondent,
                "ask for a direct answer to the pending question",
                list(answer_thread.focus_options),
                None,
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
                None,
            )
        if voting:
            return (
                None,
                "invite everyone to give their final pick now — each person names the one option they're "
                "going with, definite wording; do not name or suggest any option yourself, and never use "
                "the word 'leaning'",
                [],
                None,
            )

        aliases = short_alias_map(state.scenario.options)
        if candidate:
            # An unprobed visible blocker THREAD on the likely candidate is the
            # most useful thing to surface; ask its raiser directly. The probe
            # is charged on that thread (shared with the reactive participant
            # probe), so one sim's probe never suppresses another sim's
            # separate blocker against the same option.
            unprobed = sorted(
                (
                    t for t in state.threads.values()
                    if t.thread_type is ThreadType.BLOCKER
                    and candidate in t.focus_options
                    and t.status in (ThreadStatus.HOT, ThreadStatus.COOLING)
                    and t.probe_count == 0
                    and t.started_by in state.runtimes
                ),
                key=lambda t: (t.created_turn, t.thread_id),
            )
            if unprobed:
                return (
                    unprobed[0].started_by,
                    f"ask them what would need to change about {aliases[candidate]} for it to work for them, "
                    "or what they could support instead — one genuine question, no pressure",
                    [candidate],
                    unprobed[0].thread_id,
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
                None,
            )
        if candidate:
            dissenters = [p for p in state.personas if state.runtimes[p.id].top_option() != candidate]
            if len(dissenters) == 1:
                return (
                    dissenters[0].id,
                    "ask what remaining concern would need to be resolved to move",
                    [candidate, state.runtimes[dissenters[0].id].top_option() or dissenters[0].preferred_option],
                    None,
                )
        return (None, "ask for the strongest remaining concern before choosing", [candidate] if candidate else [], None)

    @staticmethod
    def _has_clear_vote(state: DialogueState, persona_id: str) -> bool:
        """Clarity means a FORMAL commitment (13.1): a visible vote/acceptance
        during voting or compromise repair, never a discussion-phase lean."""
        return persona_id in visible_votes_from_transcript(state)

    def _derive_pacing(self, state: DialogueState) -> None:
        n = len(state.personas)
        prefs = [p.preferred_option for p in state.personas]
        distinct = len(set(prefs))
        avg_flexibility = sum(1.0 - p.sim_params.stubbornness for p in state.personas) / max(1, n)
        min_turns = math.ceil(float(cfg.conversation.min_discussion_turns_per_participant) * n)
        target = math.ceil(float(cfg.conversation.target_discussion_turns_per_participant) * n)
        hard = math.ceil(float(cfg.conversation.max_discussion_turns_per_participant) * n)
        if distinct > 1:
            target += int(cfg.conversation.contention_extra_turns)
            hard += int(cfg.conversation.contention_extra_turns)
        if avg_flexibility < 0.45:
            target += int(cfg.conversation.low_compromise_extra_turns)
            hard += int(cfg.conversation.low_compromise_extra_turns)
        state.min_discussion_turns = max(n, min_turns)
        vote_buffer = max(1, math.ceil(n / 2))
        state.force_narrow_turns = max(state.min_discussion_turns + vote_buffer, target)
        state.hard_max_turns = max(state.force_narrow_turns + vote_buffer, hard)
        state.phase_history.append(
            f"pacing: min={state.min_discussion_turns}, force={state.force_narrow_turns}, hard={state.hard_max_turns}, distinct_initial_prefs={distinct}, avg_flexibility={avg_flexibility:.2f}"
        )

    # The complete allowed phase graph (5.1). Anything else is a controller bug.
    _ALLOWED_PHASE_TRANSITIONS = {
        (Phase.OPENING, Phase.DISCUSSION),
        (Phase.DISCUSSION, Phase.NARROWING),
        (Phase.NARROWING, Phase.VOTING),
        (Phase.NARROWING, Phase.DISCUSSION),          # at most once, on candidate collapse
        (Phase.VOTING, Phase.CLOSING),
        (Phase.VOTING, Phase.COMPROMISE_REPAIR),
        (Phase.COMPROMISE_REPAIR, Phase.VOTING),      # one bounded re-vote round
        (Phase.COMPROMISE_REPAIR, Phase.CLOSING),
    }

    def _mark_phase(self, state: DialogueState, phase: Phase, reason: str) -> None:
        previous = state.phase
        if phase is previous:
            # Same-phase note (e.g. an intermediate marker); no transition.
            state.phase_history.append(f"turn {state.turn_index}: {phase.value} — {reason}")
            return
        if (previous, phase) not in self._ALLOWED_PHASE_TRANSITIONS:
            raise ValueError(f"illegal phase transition {previous.value} -> {phase.value} ({reason})")
        state.phase = phase
        state.phase_history.append(f"turn {state.turn_index}: {phase.value} — {reason}")
        state.controller_trace.append({
            "type": "phase_transition",
            "turn_index": state.turn_index,
            "from_phase": previous.value,
            "to_phase": phase.value,
            "reason": reason,
        })

    @staticmethod
    def _opening_order(personas: list[Persona]) -> list[Persona]:
        # Engagement plus light randomness: engaged sims tend to open first,
        # but the order is not a fixed ranking.
        return sorted(personas, key=lambda p: p.sim_params.engagement + random.uniform(0.0, 0.5), reverse=True)

    def _resolve_pending_question(self, state: DialogueState) -> None:
        """Let the owed question-thread respondent answer before a vote round starts."""
        thread = self._required_answer_thread(state)
        if thread is None:
            return
        self._emit(self._generate_and_append(state, self._answer_intent_for_thread(state, thread)))

    def _ready_to_narrow(self, state: DialogueState) -> bool:
        """Exact discussion -> narrowing gate (12.1 mandatory + 12.2 triggers).

        The hard-cap override may relax minimum-turn/coverage/support evidence,
        but it never erases a direct answer obligation, an active repair, the
        hot-hard-blocker gate, or the need for at least one viable candidate.
        """
        participant_turns = participant_turn_count(state)
        hard_cap = participant_turns >= state.hard_max_turns
        candidate = self._public_candidate(state)
        pair = self._current_top_pair(state)

        # --- Mandatory conditions (12.1) ---
        if self._required_answer_thread(state) is not None:
            return False
        if state.active_repair is not None:
            return False
        if (
            bool(cfg.narrowing.get("require_no_hot_blocking_thread", True))
            and candidate is not None
            and threads.hot_blocking_thread_against(state, [candidate]) is not None
        ):
            # Absolute even at the hard cap; blocker staleness bounds the stall.
            return False
        if not hard_cap:
            if participant_turns < state.min_discussion_turns:
                return False
            if self._coverage_gap_option(state) is not None:
                return False
            if bool(cfg.narrowing.get("require_discussion_support", True)) and not self._discussion_support_options(state):
                return False
            # The strongest options should have met head-to-head at least once
            # before pre-force narrowing: a realized comparison thread is the
            # visible evidence.
            compared = any(t.thread_type is ThreadType.COMPARISON for t in state.threads.values())
            if len(state.scenario.option_ids) >= 2 and not compared and participant_turns < state.force_narrow_turns:
                return False

        # --- Readiness triggers (12.2): at least one must hold ---
        if hard_cap:
            # Cannot fabricate support: still needs one viable candidate.
            return candidate is not None or self._latent_leading_option(state) is not None
        early_gate = state.min_discussion_turns + int(cfg.conversation.early_vote_extra_turns)
        if candidate is not None and participant_turns >= early_gate:
            support = self._visible_support_count(state, candidate)
            cluster = 2 if len(state.personas) >= 3 else 1
            if support >= cluster:
                return True
            if support >= 1 and self._visibly_proposed(state, candidate):
                return True
        if self._stable_top_pair(state):
            return True
        if participant_turns >= state.force_narrow_turns and (candidate or pair):
            return True
        stall = int(cfg.conversation.moderator_stall_window)
        if state.no_progress_count >= stall and (candidate or pair):
            return True
        return False

    @staticmethod
    def _discussion_support_options(state: DialogueState) -> set[str]:
        """Options with visible support evidenced in DISCUSSION-phase turns.

        Opening leans do not count (12.1): the evidence must be an accepted
        discussion turn that visibly accepts, votes for, or supports (realized
        act) the option. Feeds narrowing readiness only, never consensus.
        """
        support = public_support(state, phase=Phase.DISCUSSION, include_support_acts=True)
        return {oid for oid, backers in support.items() if backers}

    def _stable_top_pair(self, state: DialogueState) -> bool:
        """True when the visible top pair persisted for the configured window."""
        window = int(cfg.narrowing.get("stable_top_pair_window", 2))
        history = state.top_pair_history
        if window <= 0 or len(history) < window:
            return False
        recent = history[-window:]
        first = recent[0]
        return bool(first) and len(first) == 2 and all(entry == first for entry in recent)

    def _narrow_reason(self, state: DialogueState) -> str:
        participant_turns = participant_turn_count(state)
        if participant_turns >= state.hard_max_turns:
            return "hard cap reached; forcing narrowing instead of closing early"
        if self._stable_top_pair(state):
            return "stable visible top pair persisted"
        if participant_turns >= state.force_narrow_turns:
            return "target discussion length reached"
        if state.no_progress_count >= int(cfg.conversation.moderator_stall_window):
            return "no-progress threshold reached with a candidate present"
        return "visible support for one option held after enough back-and-forth"

