"""Flow layer: phases, protocols, bidding orchestration, and the repair machine.

FlowMixin owns global progress: the opening round, the discussion loop's open-
floor bidding, bounded narrowing, formal vote collection, and the single repair
state machine. It schedules protocol obligations and open-floor bid rounds, but
every participant behavioral choice (act, target, focus, reason, vote target,
switch decision, reservation content) comes from the simulator policy — the
framework never authors it. FlowMixin drives generation through the shared
pipeline (`self._generate_and_append`); it never renders participant text itself
and mutates only phase/repair/candidate flow state.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import prompts
import simulator as sim_policy
from aliases import short_alias_map
from config_loader import cfg
from consensus import ConsensusManager, participant_turn_count, public_participant_ledger, public_support, visible_votes_from_transcript
from controller import threads
from models import (
    ActType,
    DialogueState,
    DiscussionStimulus,
    MoveIntent,
    Persona,
    Phase,
    RepairState,
    ThreadStatus,
    ThreadType,
    TurnObligation,
    TurnRecord,
)

# Authority-source label recorded for each obligation kind (trace item 21).
_AUTHORITY_BY_OBLIGATION = {
    "opening": "opening_protocol",
    "direct_answer": "direct_obligation",
    "vote": "vote_protocol",
    "final_decision": "repair_protocol",
    "reservation": "repair_protocol",
    "majority_concern": "repair_protocol",
    "narrowing_reaction": "narrowing_protocol",
    "narrowing_answer": "narrowing_protocol",
    "reservation_response": "repair_protocol",
}

# Consecutive no-bid discussion rounds before the framework progresses toward
# narrowing rather than inventing a participant stance (todo 15).
_MAX_EMPTY_ROUNDS = 2
_MAX_GENERATION_FAILURE_ROUNDS = 3
_PROTOCOL_REALIZATION_ATTEMPTS = 3


class FlowMixin:
    # ------------------------------------------------------------------
    # Turn execution: open-floor bidding and protocol obligations
    # ------------------------------------------------------------------

    def _run_open_floor_turn(
        self, state: DialogueState, stimulus: DiscussionStimulus
    ) -> TurnRecord | None:
        """Collect one bid per eligible simulator, arbitrate, and realize the
        winning bid. On generation failure the same intent is retried inside the
        pipeline; if it still drops, the next-best valid bid is used (todo 10).
        The floor never rewrites a bid's act, focus, target, reason, or vote."""
        state.bid_round_count += 1
        bids = self._collect_bids(state, stimulus)
        ranked = self._ranked_valid_bids(state, bids)
        self._last_floor_bids = bids
        self._last_authority = "self_selection"
        if not ranked:
            state.no_bid_round_count += 1
            return None
        for bid in ranked:
            state.valid_bid_attempt_count += 1
            record = self._generate_and_append(state, bid.intent)
            if not record.state_mutation_blocked and record.text.strip():
                self._emit(record)
                return record
            # Bounded generation failure on the winner: fall through to the
            # next-best previously submitted valid bid, unchanged.
        state.generation_failure_round_count += 1
        state.final_dropped_intent_count += 1
        return None

    def _run_obligation_turn(self, state: DialogueState, ob: TurnObligation) -> TurnRecord:
        """Realize one protocol-required turn. The framework fixes speaker+act;
        the simulator policy chose the substance in its bid."""
        last_record: TurnRecord | None = None
        bid = sim_policy.decide_simulator_bid(state, ob.participant_id, obligation=ob)
        for attempt in range(_PROTOCOL_REALIZATION_ATTEMPTS):
            # Preserve the complete simulator-owned decision for bounded wording
            # retries. Only the final attempt asks the same simulator to choose a
            # fresh intent under the unchanged protocol obligation.
            if attempt == _PROTOCOL_REALIZATION_ATTEMPTS - 1:
                bid = sim_policy.decide_simulator_bid(state, ob.participant_id, obligation=ob)
            self._last_floor_bids = [bid]
            self._last_authority = _AUTHORITY_BY_OBLIGATION.get(ob.kind, "self_selection")
            assert bid.intent is not None
            state.valid_bid_attempt_count += 1
            last_record = self._generate_and_append(state, bid.intent)
            if not last_record.state_mutation_blocked and last_record.text.strip():
                self._emit(last_record)
                return last_record
        state.protocol_obligation_failures += 1
        state.final_dropped_intent_count += 1
        raise RuntimeError(
            f"protocol obligation {ob.kind!r} for {ob.participant_id!r} "
            f"could not be realized after {_PROTOCOL_REALIZATION_ATTEMPTS} attempts"
        )

    def _discussion_stimulus(self, state: DialogueState, *, kind: str = "normal") -> DiscussionStimulus:
        candidate = self._public_candidate(state)
        group_q = next(
            (
                t.thread_id for t in state.threads.values()
                if t.thread_type is ThreadType.QUESTION
                and t.status is ThreadStatus.HOT
                and t.question_scope == "group"
            ),
            None,
        )
        return DiscussionStimulus(
            kind=kind,
            candidate=candidate,
            top_pair=self._current_top_pair(state),
            coverage_gap=self._coverage_gap_option(state),
            group_question_thread_id=group_q,
        )

    def _pending_answer_obligation(self, state: DialogueState) -> TurnObligation | None:
        """A direct question owed by a named respondent (mandatory adjacency
        pair). Group questions carry no required respondent and never create an
        obligation — they are stimuli for self-selection instead (todo 11/12)."""
        thread = self._required_answer_thread(state)
        if thread is None or thread.required_respondent not in state.runtimes:
            return None
        asker = thread.started_by
        return TurnObligation(
            kind="direct_answer",
            participant_id=str(thread.required_respondent),
            act=ActType.ANSWER,
            respond_to_turn=thread.source_turn_index,
            thread_id=thread.thread_id,
            addressee_id=None if asker in {"moderator", ""} else asker,
            focus_options=list(thread.focus_options),
        )

    # ------------------------------------------------------------------
    # Opening round (protocol-required; simulator picks the opening position)
    # ------------------------------------------------------------------

    def _opening_round(self, state: DialogueState) -> None:
        state.phase = Phase.OPENING
        for persona in self._opening_order(state.personas):
            self._run_obligation_turn(
                state, TurnObligation(kind="opening", participant_id=persona.id, act=ActType.OPENING)
            )
        self._mark_phase(state, Phase.DISCUSSION, "all participants gave an opening view")

    # ------------------------------------------------------------------
    # Discussion loop: mandatory answers, moderator nudges, open-floor bids
    # ------------------------------------------------------------------

    def _discussion_loop(self, state: DialogueState) -> None:
        empty_rounds = 0
        generation_failures = 0
        while True:
            if self._ready_to_narrow(state):
                threads.resolve_comparison_threads(state, reason="left discussion for narrowing")
                self._mark_phase(state, Phase.NARROWING, self._narrow_reason(state))
                return

            obligation = self._pending_answer_obligation(state)
            if obligation is not None:
                self._run_obligation_turn(state, obligation)
                empty_rounds = 0
                continue

            nudge = self._maybe_moderator_nudge(state)
            if nudge is not None:
                self._emit(nudge)
                empty_rounds = 0
                continue

            before_no_bid = state.no_bid_round_count
            before_generation_failures = state.generation_failure_round_count
            record = self._run_open_floor_turn(state, self._discussion_stimulus(state))
            if record is not None:
                empty_rounds = 0
                generation_failures = 0
                continue
            if state.generation_failure_round_count > before_generation_failures:
                generation_failures += 1
                # Valid simulator intentions existed. Do not treat failed wording
                # as silence or advance silence-based narrowing.
                if generation_failures < _MAX_GENERATION_FAILURE_ROUNDS:
                    continue
                generation_failures = 0
                if self._handle_stall(state):
                    continue
                continue

            # No simulator claimed the floor. First create a new public
            # opportunity (moderator group question or a stronger public stall
            # stimulus). Silence remains a real simulator decision.
            state.no_progress_count += 1
            if self._handle_stall(state):
                empty_rounds = 0
                continue

            empty_rounds += 1
            turns = participant_turn_count(state)
            progress = self._discussion_progress(state)
            public_path = bool(self._public_candidate(state) or self._current_top_pair(state))

            # Empty rounds never bypass the configured minimum. They do count
            # toward the hard interaction budget so an all-silent cast cannot
            # loop forever.
            if turns < state.min_discussion_turns and progress < state.hard_max_turns:
                continue

            if progress >= state.hard_max_turns or (
                turns >= state.min_discussion_turns
                and empty_rounds >= _MAX_EMPTY_ROUNDS
                and public_path
            ):
                threads.resolve_comparison_threads(state, reason="stalled discussion; narrowing")
                reason = (
                    "hard interaction cap reached after repeated no-bid rounds"
                    if progress >= state.hard_max_turns
                    else "no further bids after public stall recovery"
                )
                self._mark_phase(state, Phase.NARROWING, reason)
                return

    @staticmethod
    def _discussion_progress(state: DialogueState) -> int:
        """Accepted participant turns plus explicit all-silent bid rounds.

        The minimum gate still uses accepted participant turns. Empty rounds only
        prevent an infinite loop by advancing the hard interaction budget.
        """
        return participant_turn_count(state) + state.no_bid_round_count

    def _handle_stall(self, state: DialogueState) -> bool:
        """Create a fresh public opportunity after an empty open-floor round."""
        gap = self._coverage_gap_option(state)
        if (
            self._mod("mid_discussion_nudges")
            and self._intervention_count < int(cfg.conversation.moderator_max_interventions)
            and state.turn_index - self._last_intervention_turn >= int(cfg.conversation.moderator_cooldown_turns)
        ):
            record = self._emit_stall_recovery_question(state, gap)
            self._emit(record)
            return True

        kind = "coverage" if gap else "stall"
        if gap and gap in state.coverage:
            state.coverage[gap].coverage_attempts += 1
        record = self._run_open_floor_turn(state, self._discussion_stimulus(state, kind=kind))
        return record is not None

    def _emit_stall_recovery_question(
        self, state: DialogueState, gap: str | None
    ) -> TurnRecord:
        candidate = self._public_candidate(state)
        target_id, requested_action, focus, probe_key = self._moderator_intervention_details(
            state, candidate
        )
        if gap in state.scenario.option_ids:
            aliases = short_alias_map(state.scenario.options)
            target_id = None
            focus = [gap]
            requested_action = (
                f"ask the whole group one open question about {aliases[gap]}: what concrete reason, concern, "
                "or trade-off should determine whether it stays in consideration; name nobody"
            )
            state.coverage[gap].coverage_attempts += 1
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state, "the open floor produced no contribution",
                state.scenario.option(candidate).name if candidate else None,
                target_name=state.name_for(target_id) if target_id else None,
                requested_action=requested_action, focus_options=focus,
            ),
            state,
        )
        self._intervention_count += 1
        self._last_intervention_turn = state.turn_index
        state.no_progress_count = 0
        record = self._append_moderator(state, text, Phase.DISCUSSION)
        scope = "direct" if target_id else "group"
        threads.open_thread(
            state, thread_type=ThreadType.QUESTION,
            focus_options=[o for o in focus if o in state.scenario.option_ids][:2],
            issue_key=threads.normalize_issue_key(
                text, state.scenario, [p.name for p in state.personas], focus_options=focus
            ),
            started_by="moderator", source_turn_index=record.index,
            required_respondent=target_id, question_scope=scope,
        )
        if probe_key and probe_key in state.threads:
            state.threads[probe_key].probe_count += 1
        return record

    # ------------------------------------------------------------------
    # Narrowing (framework decides readiness; simulators react)
    # ------------------------------------------------------------------

    def _narrowing_phase(self, state: DialogueState) -> None:
        allow_return = bool(cfg.narrowing.get("allow_return_to_discussion_once", True))
        while True:
            candidate = self._candidate_for_vote(state)
            state.candidate_option = candidate
            pair = self._current_top_pair(state) or ([candidate] if candidate else [])

            if self._mod("final_vote_call"):
                self._emit_moderator_narrowing(state, candidate, pair)

            # Narrowing is a public stimulus, not a hidden holdout assignment.
            # Any relevant simulator may support, object, answer, compromise, or
            # stay silent; the floor selects the complete winning bid unchanged.
            self._run_open_floor_turn(
                state,
                DiscussionStimulus(kind="narrowing", candidate=candidate, top_pair=pair),
            )

            collapsed = self._public_candidate(state) is None or (
                candidate is not None
                and threads.hot_blocking_thread_against(state, [candidate]) is not None
            )
            if (
                collapsed
                and allow_return
                and not state.narrowing_returned
                and self._discussion_progress(state) < state.hard_max_turns
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

    def _emit_moderator_narrowing(
        self, state: DialogueState, candidate: str | None, pair: list[str]
    ) -> TurnRecord:
        aliases = short_alias_map(state.scenario.options)
        if candidate:
            requested = (
                f"note that {aliases.get(candidate, candidate)} has emerged from the visible discussion and ask "
                "the whole group whether one concrete concern still blocks it; one open question, no verdict"
            )
            focus = [candidate]
        elif len(pair) >= 2:
            requested = (
                f"ask the whole group to settle the visible trade-off between {aliases[pair[0]]} and "
                f"{aliases[pair[1]]}: which concern or benefit should decide it; name nobody"
            )
            focus = pair[:2]
        else:
            requested = "ask the whole group for the strongest remaining concern before the final vote; name nobody"
            focus = []
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state, "the discussion is ready to narrow",
                aliases.get(candidate) if candidate else None,
                requested_action=requested, focus_options=focus,
            ),
            state,
        )
        record = self._append_moderator(state, text, state.phase)
        threads.open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=focus,
            issue_key=threads.normalize_issue_key(
                text, state.scenario, [p.name for p in state.personas], focus_options=focus
            ),
            started_by="moderator", source_turn_index=record.index,
            required_respondent=None, question_scope="group",
        )
        self._emit(record)
        return record

    # ------------------------------------------------------------------
    # Voting + repair machine
    # ------------------------------------------------------------------

    def _decision_loop(self, state: DialogueState) -> None:
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
            self._run_obligation_turn(state, self._vote_obligation(persona, candidate, kind="vote"))

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

    @staticmethod
    def _vote_obligation(persona: Persona, candidate: str | None, *, kind: str) -> TurnObligation:
        return TurnObligation(
            kind=kind, participant_id=persona.id, act=ActType.VOTE,
            candidate=candidate if candidate else None,
        )

    def _classify_repair(self, state: DialogueState, provisional) -> RepairState | None:
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
            winner_votes = sum(1 for vote in formal.values() if vote == winner)
            other_votes = len(formal) - winner_votes
            if (
                winner not in state.scenario.option_ids
                or not dissenters
                or winner_votes != other_votes + 1
            ):
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
                p1, p2 = state.personas
                v1, v2 = formal.get(p1.id), formal.get(p2.id)
                if (
                    v1 in state.scenario.option_ids and v2 in state.scenario.option_ids
                    and self._hard_blocks_candidate(state, p1, v2)
                    and self._hard_blocks_candidate(state, p2, v1)
                ):
                    return None
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
                max_attempts=1,
            )
        return None

    def _run_repair(self, state: DialogueState, repair: RepairState) -> None:
        state.active_repair = repair
        if repair.repair_reason != "unclear_vote" and state.phase is not Phase.COMPROMISE_REPAIR:
            self._mark_phase(state, Phase.COMPROMISE_REPAIR, f"running {repair.repair_reason} repair")
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
        repair.attempt_count += 1
        candidate = state.candidate_option or self._candidate_for_vote(state)
        unclear = [
            state.persona_by_id(pid) for pid in repair.participants_involved
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
            self._run_obligation_turn(state, self._vote_obligation(persona, candidate, kind="vote"))
        return all(self._has_clear_vote(state, p.id) for p in unclear)

    def _repair_majority_holdout(self, state: DialogueState, repair: RepairState) -> bool:
        repair.attempt_count += 1
        winner = repair.candidate_or_pair[0] if repair.candidate_or_pair else None
        if winner not in state.scenario.option_ids:
            return False
        dissenters = [
            state.persona_by_id(pid) for pid in repair.participants_involved if pid in state.runtimes
        ]
        if not dissenters:
            return True
        aliases = short_alias_map(state.scenario.options)
        winner_name = aliases[winner]

        if self._mod("final_vote_call"):
            names = ", ".join(p.name for p in dissenters)
            text = self._moderator_say(
                prompts.moderator_nudge_prompt(
                    state, "a narrow majority has formed", winner_name,
                    requested_action=(
                        f"address {names} together: ask for each person's main concern about "
                        f"{winner_name} and whether they could reasonably move for the group, "
                        "with one reason; do not reopen the full debate"
                    ),
                    focus_options=[winner],
                ),
                state,
            )
            self._emit(self._append_moderator(state, text, state.phase))
        concern_records: list[tuple[Persona, TurnRecord]] = []
        for persona in dissenters:
            record = self._run_obligation_turn(state, TurnObligation(
                kind="majority_concern", participant_id=persona.id, act=ActType.ANSWER,
                candidate=winner, focus_options=[winner],
            ))
            if record.text.strip():
                concern_records.append((persona, record))

        for holdout, record in concern_records[:2]:
            responder = self._candidate_supporter(state, winner, exclude=holdout.id)
            if responder is None:
                continue
            self._emit_reservation_response(state, responder, holdout, winner, record.index)

        for persona in dissenters:
            self._run_obligation_turn(state, self._final_decision_obligation(persona, winner))
        return all(self._has_clear_vote(state, p.id) for p in dissenters)

    @staticmethod
    def _candidate_supporter(state: DialogueState, candidate: str, *, exclude: str) -> Persona | None:
        formal = visible_votes_from_transcript(state)
        supporters = [p for p in state.personas if p.id != exclude and formal.get(p.id) == candidate]
        if not supporters:
            return None
        return min(supporters, key=lambda p: (state.runtimes[p.id].turn_count, p.id))

    def _emit_reservation_response(
        self, state: DialogueState, responder: Persona, holdout: Persona, candidate: str, source_index: int,
        *, split: bool = False,
    ) -> None:
        """Schedule a public response obligation; the responder policy owns the answer."""
        self._run_obligation_turn(state, TurnObligation(
            kind="reservation_response",
            participant_id=responder.id,
            act=ActType.ANSWER,
            candidate=candidate,
            focus_options=[candidate],
            respond_to_turn=source_index,
            addressee_id=holdout.id,
            note="split" if split else "majority",
        ))

    def _reservation_exchange(
        self, state: DialogueState, holdout: Persona, candidate: str,
        *, split: bool = False, allow_response: bool = True,
    ) -> None:
        """Two beats: the holdout states its own reservation (simulator-owned),
        then a supporter responds (framework-scheduled response beat)."""
        state.reservation_exchanges += 1
        record = self._run_obligation_turn(state, TurnObligation(
            kind="reservation", participant_id=holdout.id, act=ActType.ANSWER,
            candidate=candidate, focus_options=[candidate],
        ))
        if not record.text.strip() or not allow_response:
            return
        responder = self._candidate_supporter(state, candidate, exclude=holdout.id)
        if responder is None:
            return
        self._emit_reservation_response(state, responder, holdout, candidate, record.index, split=split)

    def _repair_split_vote(self, state: DialogueState, repair: RepairState) -> bool:
        if repair.attempt_count >= 1:
            return False
        repair.attempt_count += 1
        votes_by_id = {
            pid: vote for pid, vote in visible_votes_from_transcript(state).items()
            if vote in state.scenario.option_ids
        }
        ranked = self._rank_split_candidates(state, votes_by_id)
        if not ranked:
            self._emit_split_summary(state, None, votes_by_id)
            return False

        leader, _dissenters, movers, meta = ranked[0]
        state.candidate_option = leader
        repair.candidate_or_pair = [leader]
        caller_id = self._emit_split_summary(state, leader, votes_by_id, attempt_index=0, meta=meta)

        ordered_movers = sorted(movers, key=lambda p: (state.runtimes[p.id].turn_count, p.id))
        if caller_id is not None:
            ordered_movers.sort(key=lambda p: p.id == caller_id)
        for index, mover in enumerate(ordered_movers):
            self._reservation_exchange(state, mover, leader, split=True, allow_response=index < 2)

        for mover in movers:
            self._run_obligation_turn(state, self._final_decision_obligation(mover, leader))

        provisional = ConsensusManager.finalize(state)
        return provisional.status in {"successful", "majority"}

    @staticmethod
    def _final_decision_obligation(persona: Persona, candidate: str) -> TurnObligation:
        """A repair re-vote: the simulator decides stay vs switch to the tested
        candidate. The framework never computes can_move or the switch itself."""
        return TurnObligation(
            kind="final_decision", participant_id=persona.id, act=ActType.VOTE, candidate=candidate,
        )

    def _repair_two_person_deadlock(self, state: DialogueState, repair: RepairState) -> bool:
        repair.attempt_count += 1
        formal = visible_votes_from_transcript(state)
        votes_by_id = {
            pid: formal[pid] for pid in repair.participants_involved
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
        # Each side names its own blocker/condition (simulator-owned reservation).
        for speaker, own_vote in ((p1, v1), (p2, v2)):
            self._run_obligation_turn(state, TurnObligation(
                kind="reservation", participant_id=speaker.id, act=ActType.ANSWER,
                candidate=(v2 if speaker is p1 else v1), focus_options=[v1, v2],
            ))
        # Each side gives its own final stay/switch decision.
        for speaker, other_vote in ((p1, v2), (p2, v1)):
            self._run_obligation_turn(state, self._final_decision_obligation(speaker, other_vote))
        final = visible_votes_from_transcript(state)
        return len({final.get(p1.id), final.get(p2.id)} - {None}) == 1

    # ------------------------------------------------------------------
    # Split / unresolved / closing summaries (deterministic framework text)
    # ------------------------------------------------------------------

    def _emit_split_summary(
        self, state: DialogueState, candidate: str | None, votes_by_id: dict[str, str],
        *, attempt_index: int = 0, meta: dict | None = None,
    ) -> str | None:
        aliases = short_alias_map(state.scenario.options)
        counts = Counter(votes_by_id.values())
        if self._mod("final_vote_call"):
            split = ", ".join(f"{aliases[oid]} ({count})" for oid, count in counts.most_common())
            prefix = "Second narrowing attempt. " if attempt_index else ""
            if candidate:
                candidate_name = aliases[candidate]
                selected_ids = set((meta or {}).get("selected_mover_ids", []))
                dissenters = [
                    p.name for p in state.personas if p.id in selected_ids
                ] or [p.name for p in state.personas if votes_by_id.get(p.id) != candidate]
                dissenter_text = ", ".join(dissenters) or "the others"
                meta_text = ""
                if meta:
                    max_votes = max(counts.values(), default=0)
                    tied_for_lead = sum(1 for count in counts.values() if count == max_votes) > 1
                    if meta.get("votes") == max_votes and tied_for_lead:
                        if int(meta.get("positive_mentions", 0) or 0) > 0:
                            meta_text = (
                                " It is tied for the lead and had the most positive "
                                "discussion support, so we test it once."
                            )
                        else:
                            meta_text = " It is tied for the lead, so we test one concrete option once."
                    elif meta.get("votes") == max_votes:
                        meta_text = " It has the visible lead, so we test it first."
                    else:
                        meta_text = " It is the next concrete alternative, so we test it once."
                text = (
                    f"{prefix}We are split: {split}. Let's test {candidate_name} as the compromise; "
                    f"{dissenter_text}, what would still block that for you?{meta_text}"
                )
            else:
                text = (
                    f"{prefix}We are split: {split}. No option has a workable path yet, so let's name the blockers plainly."
                )
            self._emit(self._append_moderator(state, text, state.phase))
            return None
        # Without a visible moderator, framework vote-count narration is logged
        # but not attributed to a participant.
        return None


    # ------------------------------------------------------------------
    # Moderator nudges (framework voice; never assigns participant behavior)
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
        candidate = self._public_candidate(state)
        candidate_name = state.scenario.option(candidate).name if candidate else None
        target_id, requested_action, focus, probe_key = self._moderator_intervention_details(state, candidate)
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state, "discussion seems to be circling", candidate_name,
                target_name=state.name_for(target_id) if target_id else None,
                requested_action=requested_action, focus_options=focus,
            ),
            state,
        )
        self._intervention_count += 1
        self._last_intervention_turn = state.turn_index
        state.no_progress_count = 0
        record = self._append_moderator(state, text, Phase.DISCUSSION)
        if probe_key and probe_key in state.threads:
            state.threads[probe_key].probe_count += 1
        if target_id and target_id in state.runtimes:
            threads.open_thread(
                state, thread_type=ThreadType.QUESTION,
                focus_options=[o for o in focus if o in state.scenario.option_ids][:2],
                issue_key=threads.normalize_issue_key(
                    text, state.scenario, [p.name for p in state.personas], focus_options=focus
                ),
                started_by="moderator", source_turn_index=record.index,
                required_respondent=target_id, question_scope="direct",
            )
        return record

    def _moderator_vote_nudge(self, state: DialogueState, candidate: str, reason: str) -> tuple[TurnRecord, str | None]:
        target_id, requested_action, focus, _probe_key = self._moderator_intervention_details(state, candidate, voting=True)
        text = self._moderator_say(
            prompts.moderator_nudge_prompt(
                state, reason, None,
                target_name=state.name_for(target_id) if target_id else None,
                requested_action=requested_action, focus_options=focus,
            ),
            state,
        )
        record = self._append_moderator(state, text, state.phase)
        return record, target_id

    def _moderator_intervention_details(
        self, state: DialogueState, candidate: str | None, *, voting: bool = False,
    ) -> tuple[str | None, str, list[str], str | None]:
        """Return (target_id, requested_action, focus_option_ids, blocker_probe_key).

        A coverage gap becomes a group question about the ignored option (no
        assigned respondent, todo 14). A pending direct question, a visible
        split, or a lone holdout may name a target for a direct question."""
        gap = self._coverage_gap_option(state)
        if gap is not None and not voting:
            aliases = short_alias_map(state.scenario.options)
            return (
                None,
                f"ask the group to briefly bring {aliases.get(gap, gap)} into the discussion before narrowing — "
                "an open question to everyone, naming no one",
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
            ledger = public_participant_ledger(state)
            visible_dissenters = [
                item for item in ledger.values()
                if item.public_position in state.scenario.option_ids
                and item.public_position != candidate
                and candidate in item.concerned_options
            ]
            if len(visible_dissenters) == 1:
                item = visible_dissenters[0]
                focus = [candidate]
                if item.public_position:
                    focus.append(item.public_position)
                return (
                    item.participant_id,
                    "ask what publicly raised concern would need to be resolved before moving",
                    focus,
                    None,
                )
        return (None, "ask for the strongest remaining concern before choosing", [candidate] if candidate else [], None)

    @staticmethod
    def _has_clear_vote(state: DialogueState, persona_id: str) -> bool:
        return persona_id in visible_votes_from_transcript(state)

    # ------------------------------------------------------------------
    # Pacing, phase graph, narrowing readiness
    # ------------------------------------------------------------------

    def _resolve_pending_question(self, state: DialogueState) -> None:
        obligation = self._pending_answer_obligation(state)
        if obligation is not None:
            self._run_obligation_turn(state, obligation)

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
            f"pacing: min={state.min_discussion_turns}, force={state.force_narrow_turns}, "
            f"hard={state.hard_max_turns}, distinct_initial_prefs={distinct}, avg_flexibility={avg_flexibility:.2f}"
        )

    _ALLOWED_PHASE_TRANSITIONS = {
        (Phase.OPENING, Phase.DISCUSSION),
        (Phase.DISCUSSION, Phase.NARROWING),
        (Phase.NARROWING, Phase.VOTING),
        (Phase.NARROWING, Phase.DISCUSSION),
        (Phase.VOTING, Phase.CLOSING),
        (Phase.VOTING, Phase.COMPROMISE_REPAIR),
        (Phase.COMPROMISE_REPAIR, Phase.VOTING),
        (Phase.COMPROMISE_REPAIR, Phase.CLOSING),
    }

    def _mark_phase(self, state: DialogueState, phase: Phase, reason: str) -> None:
        previous = state.phase
        if phase is previous:
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
        return sorted(personas, key=lambda p: p.sim_params.engagement + random.uniform(0.0, 0.5), reverse=True)

    def _ready_to_narrow(self, state: DialogueState) -> bool:
        participant_turns = participant_turn_count(state)
        hard_cap = self._discussion_progress(state) >= state.hard_max_turns
        candidate = self._public_candidate(state)
        pair = self._current_top_pair(state)

        if self._required_answer_thread(state) is not None:
            return False
        if state.active_repair is not None:
            return False
        if (
            bool(cfg.narrowing.get("require_no_hot_blocking_thread", True))
            and candidate is not None
            and threads.hot_blocking_thread_against(state, [candidate]) is not None
        ):
            return False
        if not hard_cap:
            if participant_turns < state.min_discussion_turns:
                return False
            if self._coverage_gap_option(state) is not None:
                return False
            if bool(cfg.narrowing.get("require_discussion_support", True)) and not self._discussion_support_options(state):
                return False
            compared = any(t.thread_type is ThreadType.COMPARISON for t in state.threads.values())
            if len(state.scenario.option_ids) >= 2 and not compared and participant_turns < state.force_narrow_turns:
                return False

        if hard_cap:
            return True
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
        support = public_support(state, phase=Phase.DISCUSSION, include_support_acts=True)
        return {oid for oid, backers in support.items() if backers}

    def _stable_top_pair(self, state: DialogueState) -> bool:
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
