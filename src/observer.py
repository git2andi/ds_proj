"""Post-turn observation: the single semantic state-update entry point.

``_apply_semantics`` is the only place a final accepted participant turn
changes semantic dialogue state (votes, stance ranks, coverage, threads,
progress). Thread lifecycle transitions are delegated to
the thread engine (threads.py); the observer never assigns thread statuses
directly. The remaining mutation paths outside this module are pipeline
bookkeeping only: turn/trace appends and bounded route-attempt accounting in
dialogue.py, and phase/repair flow transitions owned by the flow code.
"""

from __future__ import annotations

import random

from models import (
    ActType,
    BlockingStrength,
    DialogueState,
    MoveIntent,
    ParticipantRuntime,
    Persona,
    Phase,
    STANCE_ACCEPTABLE,
    STANCE_NEUTRAL,
    STANCE_REJECTED,
    ThreadState,
    ThreadStatus,
    ThreadType,
    TurnRecord,
    VisibleEvidence,
)
from controller import threads as threads_engine
from simulator import expected_turn_share
from utils import weighted_choice


class ObserverMixin:
    def _apply_semantics(self, state: DialogueState, record: TurnRecord) -> None:
        # The observer consumes the accepted evidence contract and controller
        # metadata only — it never reinterprets natural language (item 5).
        if record.evidence is None:
            raise ValueError(
                f"turn {record.index} reached the observer without accepted evidence"
            )
        rt = state.runtimes[record.speaker_id]
        persona = state.persona_by_id(record.speaker_id)
        evidence = record.evidence
        before = self._snapshot_progress(state)
        prior_vote = rt.explicit_vote
        prior_pref = rt.top_option() or persona.preferred_option

        # Question threads (6.1): first fold this turn into existing threads
        # (a valid answer moves hot -> cooling), then open a thread for any new
        # visible question, then age all threads post-turn. Questions inside
        # voting/repair/closing belong to the bounded decision flow that asked
        # them — they never open ordinary discussion question threads, so a
        # finished repair exchange cannot leave a false hot question behind.
        self._observe_question_answers(state, record, evidence)
        genuine_questions = [q for q in evidence.questions if q.scope in ("direct", "group")]
        if genuine_questions and record.phase in (Phase.OPENING, Phase.DISCUSSION, Phase.NARROWING):
            self._register_question_thread(state, record, genuine_questions[0], evidence)

        # Coverage from option-SPECIFIC accepted evidence (item 12): support
        # for A gives no reason credit to other mentioned options, a concern
        # about A never attaches to B, and a comparison credits exactly its
        # pair. A bare mention counts only as a mention.
        mentioned = [
            m.option_id for m in evidence.mentions if m.option_id in state.scenario.option_ids
        ]
        soft_concerns = {
            c.option_id: (c.span.text or record.text)
            for c in evidence.concerns if c.severity == "ordinary"
        }
        hard_concerns = {
            c.option_id: (c.span.text or record.text)
            for c in evidence.concerns if c.severity == "hard"
        }
        for blocker in evidence.blockers:
            if blocker.action == "raised":
                hard_concerns.setdefault(blocker.option_id, blocker.span.text or record.text)
        challenged = set(soft_concerns) | set(hard_concerns)
        accepts = [c.option_id for c in evidence.commitments if c.kind == "accept"]
        commitment = evidence.sole_commitment()
        committed = {c.option_id for c in evidence.commitments}
        supported = {s.option_id for s in evidence.supports}
        compared_ids = {oid for comp in evidence.comparisons for oid in comp.option_ids}
        proposal_ids = {p.option_id for p in evidence.proposals if p.option_id}
        for option_id in dict.fromkeys(mentioned):
            cov = state.coverage[option_id]
            cov.mentions += 1
            if option_id in challenged:
                cov.objections += 1
            elif option_id in (committed | supported | compared_ids | proposal_ids):
                cov.reasons += 1

        # Concern/blocker threads (6.2/6.3): first fold this turn into open
        # threads (a semantically relevant reply moves hot -> cooling; a
        # raiser's visible acceptance resolves), then open option-specific
        # threads for objections raised here.
        self._observe_concern_responses(state, record, evidence)
        # Comparison threads (6.4): a realized head-to-head gets one normalized
        # pair thread; a relevant pair response cools it. Short-lived by design.
        self._observe_comparison_responses(state, record, evidence)
        self._register_comparison_thread(state, record, evidence)
        # Concern/blocker threads open only from accepted visible objections —
        # a routed CONCERN intent whose line shows no visible objection
        # registers nothing (accepted evidence is the semantic authority).
        for option_id, reason in {**soft_concerns, **hard_concerns}.items():
            self._register_concern_thread(
                state, record, option_id, hard=option_id in hard_concerns, reason=reason
            )

        # A visible resolution can reopen a blocker for THIS sim only, and it
        # must run before votes so "that fixes my concern; I can live with X" counts.
        resolved_blockers = {b.option_id for b in evidence.blockers if b.action == "resolved"}
        for option_id in resolved_blockers:
            if rt.rank(option_id) == STANCE_REJECTED:
                rt.set_rank(option_id, STANCE_NEUTRAL)

        allow_change = bool(record.intent and record.intent.allow_vote_change)
        switch_targets = {s.target for s in evidence.switches}
        if commitment and commitment.option_id not in rt.rejected_options():
            self._set_vote(
                rt, commitment.option_id, record.text,
                force=allow_change,
                stance=commitment.kind,
                visible_switch=commitment.option_id in switch_targets,
            )
        for option_id in accepts:
            if option_id in rt.rejected_options():
                continue  # an actively blocked option needs a visible resolution first
            rt.mark_acceptable(option_id, reason_for=record.text)
            state.coverage[option_id].acceptances += 1
        for option_id, reason in soft_concerns.items():
            rt.mark_disliked(option_id, reason_against=reason)
        for option_id, reason in hard_concerns.items():
            # An artifact must not turn the speaker's own current favorite into a hard blocker.
            if option_id == (rt.top_option() or persona.preferred_option):
                continue
            rt.mark_rejected(option_id, reason_against=reason)

        # Record visible vote movement (first vote away from the initial
        # preference, or a change of an earlier vote) from the accepted switch
        # evidence only: reason/bridge signals come from the validated switch
        # entry, never from reparsing the text.
        if rt.explicit_vote and rt.explicit_vote != prior_vote:
            baseline = prior_vote or persona.preferred_option
            if rt.explicit_vote != baseline:
                switch = next((s for s in evidence.switches if s.target == rt.explicit_vote), None)
                bridged = (
                    prior_pref == rt.explicit_vote
                    or (switch is not None and (switch.reason_span or switch.source))
                )
                has_reason = switch is not None and switch.reason_span is not None
                rt.switch_events.append({
                    "from": baseline,
                    "to": rt.explicit_vote,
                    "has_reason": bool(has_reason),
                    "has_bridge": bool(bridged),
                })
                rt.concessions_made += 1

        self._apply_lean_movement(state, record, rt, persona, evidence)

        # One counting point for genuine discussion-phase lean movement
        # (todo_prompt item 5): ANY visible evidence path that changed this
        # speaker's top option during DISCUSSION counts — softening, compromise,
        # acceptance, or a visible commitment — never controller intent alone
        # (blocked/dropped turns skip _apply_semantics entirely).
        if record.phase is Phase.DISCUSSION and rt.top_option() != prior_pref:
            state.discussion_lean_shifts += 1
            state.discussion_lean_shift_turns.append(record.index)

        # Track the visible top pair per accepted turn for the stability
        # trigger (12.2). Tuples, so equality comparison is exact.
        state.top_pair_history.append(tuple(self._current_top_pair(state)))

        after = self._snapshot_progress(state)
        state.no_progress_count = 0 if after != before else state.no_progress_count + 1

        # Thread aging runs after the progress snapshot: decaying to stale is
        # the absence of progress and must not reset the stagnation counter.
        threads_engine.age_threads(state)

    # ------------------------------------------------------------------
    # Concern threads and lean movement
    # ------------------------------------------------------------------

    def _apply_lean_movement(
        self,
        state: DialogueState,
        record: TurnRecord,
        rt: ParticipantRuntime,
        persona: Persona,
        evidence: VisibleEvidence,
    ) -> None:
        """Move the latent lean only on the speaker's own accepted visible
        evidence: an opening that backs an option, visible softening, a
        visible compromise proposal, or explicit conditional support. Never
        from routing intent alone (votes/acceptances go through ``_set_vote``)."""
        mentioned = [
            m.option_id for m in evidence.mentions if m.option_id in state.scenario.option_ids
        ]
        challenged = {c.option_id for c in evidence.concerns} | {
            b.option_id for b in evidence.blockers if b.action == "raised"
        }
        softened = next((s.option_id for s in evidence.softenings if s.option_id), None)
        if record.intent and record.intent.act == ActType.OPENING:
            # The opening lean follows the option the line visibly BACKS: an
            # option the same line objects to is never promoted, even when it
            # was the routed favorite ("A seems too expensive, while B fits us
            # much better" leans B). Preference order: the routed favorite when
            # named positively, else a unique positively named option;
            # ambiguous or all-negative naming moves nothing.
            positive = [oid for oid in dict.fromkeys(mentioned) if oid not in challenged]
            focus = record.intent.option_focus[0] if record.intent.option_focus else None
            target = focus if focus in positive else (positive[0] if len(positive) == 1 else None)
            if target:
                rt.promote_to_preferred(target)
        elif softened and softened != rt.top_option() and self._can_shift_to(state, persona, softened):
            # Explicit visible softening ("B is starting to make more sense to
            # me", issue 3): the internal lean follows the sim's own words —
            # withholding the shift would make the state dishonest.
            rt.promote_to_preferred(softened, reason_for=record.text)
            rt.concessions_made += 1
        else:
            proposal = next((p.option_id for p in evidence.proposals if p.option_id), None)
            conditional = next(
                (s.option_id for s in evidence.supports if s.strength == "conditional"), None
            )
            signal = proposal or conditional
            if signal and signal != rt.top_option() and self._can_shift_to(state, persona, signal):
                # The sim visibly floated this option itself; whether the
                # internal lean follows is discussion-phase concession
                # territory: stubbornness gates it, and an already-acceptable
                # target moves more easily than an untested one (no hidden
                # commitment float — ranks and traits only).
                resist = persona.sim_params.stubbornness * (
                    0.5 if rt.rank(signal) >= STANCE_ACCEPTABLE else 0.8
                )
                if random.random() > resist:
                    rt.promote_to_preferred(signal, reason_for=record.text)
                    rt.concessions_made += 1

    def _register_concern_thread(
        self, state: DialogueState, record: TurnRecord, option_id: str, *, hard: bool, reason: str = ""
    ) -> None:
        """Open an option-specific concern/blocker thread from accepted
        concern/blocker evidence (6.2/6.3).

        Repeats of a resolved issue are suppressed by the thread engine, so
        repeated generic objections open nothing while a fresh issue still
        registers as a challenge on the option's advocates.
        """
        if option_id not in state.scenario.option_ids:
            return
        is_hard = hard
        issue_key = threads_engine.normalize_issue_key(
            record.text,
            state.scenario,
            [p.name for p in state.personas],
            focus_options=[option_id],
        )
        thread = threads_engine.open_thread(
            state,
            thread_type=ThreadType.BLOCKER if is_hard else ThreadType.CONCERN,
            focus_options=[option_id],
            issue_key=issue_key,
            started_by=record.speaker_id,
            source_turn_index=record.index,
            blocking_strength=BlockingStrength.HARD if is_hard else BlockingStrength.SOFT,
        )
        if thread is None or thread.created_turn != record.index:
            return  # suppressed repeat or reinforcement of an existing thread
        for persona in state.personas:
            rt = state.runtimes[persona.id]
            if persona.id != record.speaker_id and rt.top_option() == option_id:
                rt.challenges_received += 1

    def _observe_concern_responses(
        self, state: DialogueState, record: TurnRecord, evidence: VisibleEvidence
    ) -> None:
        """Fold this final turn into open concern/blocker threads (6.2/6.3).

        Merely mentioning the concerned option is not enough: a response counts
        only when it references the option AND responds to the issue — routed
        as a reply to the source turn, matching the issue key, visibly
        resolving the blocker, or visibly accepting the option. The raiser's
        own visible acceptance/softening/resolution resolves the thread.
        """
        refs = {m.option_id for m in evidence.mentions}
        committed = {c.option_id for c in evidence.commitments}
        accepted_ids = {c.option_id for c in evidence.commitments if c.kind == "accept"}
        softened_ids = {s.option_id for s in evidence.softenings if s.option_id}
        resolved_ids = {b.option_id for b in evidence.blockers if b.action == "resolved"}
        for thread in list(state.threads.values()):
            if thread.thread_type not in (ThreadType.CONCERN, ThreadType.BLOCKER):
                continue
            if thread.status in (ThreadStatus.RESOLVED, ThreadStatus.STALE):
                continue
            focus = set(thread.focus_options)
            if not focus & refs and not focus & resolved_ids:
                continue
            if record.speaker_id == thread.started_by:
                accepted = bool(focus & (committed | softened_ids | resolved_ids))
                if accepted:
                    threads_engine.resolve_thread(
                        state, thread, reason="raiser visibly accepted/softened/resolved"
                    )
                else:
                    # Restating the own concern keeps the thread warm.
                    threads_engine.touch_thread(
                        state, thread, turn_index=record.index, participant_id=record.speaker_id
                    )
                continue
            # Relevance from controller metadata and accepted evidence only:
            # a routed reply to this thread (unless the validator judged it
            # off-topic), a visible resolution/acceptance of the focus option,
            # or semantic engagement with the focus option in the evidence.
            routed_reply = bool(
                record.intent
                and (
                    record.intent.respond_to_turn == thread.source_turn_index
                    or record.intent.thread_id == thread.thread_id
                )
                and evidence.thread_relevant is not False
            )
            relevant = (
                routed_reply
                or bool(focus & resolved_ids)
                or bool(focus & accepted_ids)
                or self._evidence_engages(evidence, focus)
            )
            if relevant and thread.status is ThreadStatus.HOT:
                threads_engine.mark_response(
                    state, thread, responder_id=record.speaker_id, turn_index=record.index
                )
            elif relevant:
                threads_engine.touch_thread(
                    state, thread, turn_index=record.index, participant_id=record.speaker_id
                )

    def _register_comparison_thread(
        self, state: DialogueState, record: TurnRecord, evidence: VisibleEvidence
    ) -> None:
        """Open/update one normalized option-pair thread for realized
        comparison evidence.

        The accepted comparison evidence is the single authority; a routed
        compare whose line never contrasts anything registers nothing, while a
        comparative question realizes ASK *and* a comparison (multi-label).
        Pair identity ignores the issue key: one pair, one thread (6.4).
        """
        comparison = next(
            (
                c for c in evidence.comparisons
                if len({o for o in c.option_ids if o in state.scenario.option_ids}) >= 2
            ),
            None,
        )
        if comparison is None:
            return
        refs = [oid for oid in comparison.option_ids if oid in state.scenario.option_ids]
        pair = threads_engine.normalize_pair(refs)[:2]
        threads_engine.open_thread(
            state,
            thread_type=ThreadType.COMPARISON,
            focus_options=pair,
            issue_key="pair",  # pair-only identity: same pair -> same thread
            started_by=record.speaker_id,
            source_turn_index=record.index,
        )

    def _observe_comparison_responses(
        self, state: DialogueState, record: TurnRecord, evidence: VisibleEvidence
    ) -> None:
        """Move a comparison thread hot -> cooling on a relevant pair response:
        another participant engaging the same pair (both options named) or
        replying directly to the comparison turn."""
        refs = {m.option_id for m in evidence.mentions}
        for thread in state.threads.values():
            if thread.thread_type is not ThreadType.COMPARISON:
                continue
            if thread.status is not ThreadStatus.HOT:
                continue
            if record.speaker_id == thread.started_by:
                if set(thread.focus_options) <= refs:
                    threads_engine.touch_thread(
                        state, thread, turn_index=record.index, participant_id=record.speaker_id
                    )
                continue
            routed_reply = bool(
                record.intent and record.intent.respond_to_turn == thread.source_turn_index
            )
            if routed_reply or set(thread.focus_options) <= refs:
                threads_engine.mark_response(
                    state, thread, responder_id=record.speaker_id, turn_index=record.index
                )

    @staticmethod
    def _evidence_engages(evidence: VisibleEvidence, focus: set[str]) -> bool:
        """Semantic engagement with a focus option in the accepted evidence —
        support, concern, comparison, proposal, blocker, commitment, or
        softening bound to it. A bare mention is not engagement."""
        if not focus:
            return False
        bound = (
            {s.option_id for s in evidence.supports}
            | {c.option_id for c in evidence.concerns}
            | {oid for comp in evidence.comparisons for oid in comp.option_ids}
            | {p.option_id for p in evidence.proposals if p.option_id}
            | {b.option_id for b in evidence.blockers}
            | {c.option_id for c in evidence.commitments}
            | {s.option_id for s in evidence.softenings if s.option_id}
        )
        return bool(focus & bound)

    @staticmethod
    def _set_vote(
        rt: ParticipantRuntime,
        option_id: str,
        text: str,
        *,
        force: bool = False,
        stance: str = "vote",
        visible_switch: bool = False,
    ) -> None:
        """Record a clear vote, protecting an existing one from silent overwrite.

        A participant who already cast a clear vote keeps it unless their turn
        carries visible switch evidence (``visible_switch``) or ``force`` is
        set (the explicit split-vote compromise step). Exception (issue #23):
        a formal direct vote replaces a commitment that was only an acceptance
        earlier in discussion — otherwise a casual "X works for me" locks out
        the actual vote round and manufactures a phantom split. A direct vote
        never silently replaces another direct vote (issue #5).
        """
        if (
            rt.explicit_vote
            and rt.explicit_vote != option_id
            and not force
            and not visible_switch
        ):
            if not (stance == "vote" and rt.vote_stance == "accept"):
                return
        rt.explicit_vote = option_id
        rt.vote_stance = stance
        if stance == "accept":
            rt.mark_acceptable(option_id, reason_for=text)
        else:
            rt.promote_to_preferred(option_id, reason_for=text)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _register_question_thread(
        self, state: DialogueState, record: TurnRecord, question, evidence: VisibleEvidence
    ) -> None:
        """Open/update a question thread from accepted question evidence (6.1).

        Scope comes from the validated visible evidence; the respondent for a
        group question is a controller decision made here, using relevance and
        turn balance — never the interpreter's.
        """
        mentioned = [
            m.option_id for m in evidence.mentions if m.option_id in state.scenario.option_ids
        ]
        scope = question.scope
        question_focus = [o for o in question.option_ids if o in state.scenario.option_ids]
        if scope == "direct":
            respondent = question.addressee_id
            if not respondent or respondent == record.speaker_id or respondent not in state.runtimes:
                return
        else:
            respondent = self._pick_group_respondent(
                state, record.speaker_id, (question_focus or mentioned)[:2]
            )
            if respondent is None:
                return
        focus = (question_focus or mentioned)[:2]
        issue_key = threads_engine.normalize_issue_key(
            record.text,
            state.scenario,
            [p.name for p in state.personas],
            focus_options=focus,
        )
        threads_engine.open_thread(
            state,
            thread_type=ThreadType.QUESTION,
            focus_options=focus,
            issue_key=issue_key,
            started_by=record.speaker_id,
            source_turn_index=record.index,
            required_respondent=respondent,
            question_scope=scope,
        )

    def _observe_question_answers(
        self, state: DialogueState, record: TurnRecord, evidence: VisibleEvidence
    ) -> None:
        """Move question threads hot -> cooling when this final turn answers them.

        A response counts only when the required respondent speaks AND the
        accepted evidence visibly relates to the question — its focused option
        or its normalized issue key. The routed act alone is never sufficient
        (closeout 2): an accepted but unrelated statement closes nothing, and
        a fallback line resolves a question only when its accepted answer
        evidence says it addressed the target (item 12).
        """
        answered_target = any(a.addresses_target for a in evidence.answers)
        mentioned = {m.option_id for m in evidence.mentions}
        for thread in list(state.threads.values()):
            if thread.thread_type is not ThreadType.QUESTION or thread.status is not ThreadStatus.HOT:
                continue
            if thread.required_respondent != record.speaker_id:
                continue
            if record.index <= thread.source_turn_index:
                continue
            if record.used_fallback and not answered_target:
                continue
            routed_reply = bool(
                record.intent and record.intent.respond_to_turn == thread.source_turn_index
            )
            relevant = (
                bool(set(thread.focus_options) & mentioned)
                or (routed_reply and answered_target)
            )
            if relevant:
                threads_engine.mark_response(
                    state, thread, responder_id=record.speaker_id, turn_index=record.index
                )

    def _required_answer_thread(self, state: DialogueState) -> "ThreadState | None":
        """The question thread whose answer is owed next: direct before group,
        then earliest creation. Read-only; deterministic."""
        candidates = [
            t for t in state.threads.values()
            if t.thread_type is ThreadType.QUESTION
            and t.status is ThreadStatus.HOT
            and t.required_respondent in state.runtimes
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda t: (t.question_scope != "direct", t.created_turn, t.thread_id),
        )

    def _answer_intent_for_thread(self, state: DialogueState, thread: "ThreadState") -> MoveIntent:
        asker = thread.started_by
        return MoveIntent(
            speaker_id=str(thread.required_respondent),
            act=ActType.ANSWER,
            reason="answer the direct question you were just asked, then add one implication for the decision",
            route_source="answer_required",
            thread_id=thread.thread_id,
            addressee_id=None if asker in {"moderator", ""} else asker,
            option_focus=list(thread.focus_options) or self._focus_from_recent(state),
            # Without this the prompt never shows WHICH question is owed, and
            # group-directed questions get pivoted around instead of answered.
            respond_to_turn=thread.source_turn_index,
        )

    def _pick_group_respondent(
        self, state: DialogueState, asker_id: str, option_focus: list[str] | None = None
    ) -> str | None:
        """Respondent for a group-directed question, chosen by a weighted score:
        relevance to the question's option focus, engagement, expected-share
        deficit relative to the sim's own target, a recent-speaker penalty, and
        the sampler's randomness. Not simply the quietest person."""
        others = [p for p in state.personas if p.id != asker_id]
        if not others:
            return None
        expected = expected_turn_share(state.personas)
        total = sum(rt.turn_count for rt in state.runtimes.values()) or 1
        weights = []
        for p in others:
            rt = state.runtimes[p.id]
            deficit = expected[p.id] - rt.turn_count / total
            relevance = sum(
                0.35 for oid in (option_focus or []) if rt.rank(oid) != STANCE_NEUTRAL
            )
            weight = 0.30 + p.sim_params.engagement + relevance + max(0.0, 3.0 * deficit)
            if rt.last_spoke_turn is not None and state.turn_index - rt.last_spoke_turn <= 1:
                weight *= 0.5
            weights.append(weight)
        return weighted_choice(others, weights).id

    @staticmethod
    def _focus_from_recent(state: DialogueState) -> list[str]:
        for turn in reversed(state.turns):
            mentioned = turn.mentioned_options()
            if mentioned:
                return mentioned[:2]
        return []

    def _snapshot_progress(self, state: DialogueState) -> tuple:
        """Deterministic progress signature (Section 11).

        no_progress_count resets only when this changes meaningfully: a thread
        changes status (question answered, concern/blocker moved, comparison
        settled), the visible support/vote picture shifts, someone's lean
        moves, an option becomes covered, the candidate changes, a phase turns,
        or a repair objective advances. A generic comment or repeated wording
        changes none of these and therefore never resets progress.
        """
        leans = tuple(sorted((pid, rt.top_option()) for pid, rt in state.runtimes.items()))
        votes = tuple(sorted((pid, rt.explicit_vote) for pid, rt in state.runtimes.items() if rt.explicit_vote))
        support = tuple(
            (oid, self._visible_support_count(state, oid)) for oid in state.scenario.option_ids
        )
        thread_statuses = tuple(
            sorted((tid, t.status.value) for tid, t in state.threads.items())
        )
        coverage = tuple(
            (oid, cov.mentions > 0, cov.coverage_attempts) for oid, cov in sorted(state.coverage.items())
        )
        candidate = (state.candidate_option, self._public_candidate(state))
        repair = (
            (state.active_repair.repair_reason, state.active_repair.status, state.active_repair.attempt_count)
            if state.active_repair
            else None
        )
        return (state.phase.value, leans, votes, support, thread_statuses, coverage, candidate, repair)
