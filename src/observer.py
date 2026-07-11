"""Post-turn observation: the single semantic state-update entry point.

``_apply_semantics`` is the only place a final accepted participant turn
changes semantic dialogue state (votes, stance ranks, coverage, threads,
commitment dynamics, progress). Thread lifecycle transitions are delegated to
the thread engine (threads.py); the observer never assigns thread statuses
directly. The remaining mutation paths outside this module are pipeline
bookkeeping only: turn/trace appends and bounded route-attempt accounting in
dialogue.py, and phase/repair flow transitions owned by the flow code.
"""

from __future__ import annotations

import random
import re

from models import (
    ActType,
    BlockingStrength,
    DialogueAct,
    DialogueState,
    MoveIntent,
    ParticipantRuntime,
    Persona,
    Phase,
    STANCE_NEUTRAL,
    STANCE_REJECTED,
    ThreadState,
    ThreadStatus,
    ThreadType,
    TurnRecord,
)
from controller import threads as threads_engine
from parsing import (
    commitment_has_reason,
    has_support_claim,
    parse_dialogue_act,
    realized_comparison,
    switch_bridge_ok,
)
from simulator import expected_turn_share
from utils import weighted_choice

_VOTE_CHANGE = re.compile(
    r"\b(?:i\s+changed\s+my\s+mind|change\s+my\s+vote|switch(?:ing)?\s+(?:to|my\s+vote)|"
    r"actually\s+i\s+(?:vote|choose|support|pick|prefer)|on\s+second\s+thought|"
    r"i'?ll\s+switch|then\s+i\s+switch|let'?s\s+switch)\b",
    re.I,
)

class ObserverMixin:
    def _parse_act(self, state: DialogueState, persona: Persona, text: str, intent: MoveIntent) -> DialogueAct:
        assert self._resolver is not None
        previous = self._last_participant_id(state)
        names = {p.id: p.name for p in state.personas}
        return parse_dialogue_act(
            speaker_id=persona.id,
            speaker_name=persona.name,
            text=text,
            resolver=self._resolver,
            participant_names=names,
            intent=intent,
            previous_speaker_id=previous,
        )

    def _apply_semantics(self, state: DialogueState, record: TurnRecord) -> None:
        rt = state.runtimes[record.speaker_id]
        persona = state.persona_by_id(record.speaker_id)
        act = record.act
        before = self._snapshot_progress(state)
        prior_vote = rt.explicit_vote
        prior_pref = rt.top_option() or persona.preferred_option

        # Question threads (6.1): first fold this turn into existing threads
        # (a valid answer moves hot -> cooling), then open a thread for any new
        # visible question, then age all threads post-turn. Questions inside
        # voting/repair/closing belong to the bounded decision flow that asked
        # them — they never open ordinary discussion question threads, so a
        # finished repair exchange cannot leave a false hot question behind.
        self._observe_question_answers(state, record)
        if act.question_scope and record.phase in (Phase.OPENING, Phase.DISCUSSION, Phase.NARROWING):
            self._register_question_thread(state, record)

        # Coverage from independent per-option semantic evidence (closeout 5):
        # a multi-function line can carry mentions, positive support, objection,
        # and comparison evidence at once — the dominant act label stays for
        # routing/reporting but is not the only coverage signal. An objected
        # option gets an objection, not a reason; a positively engaged option
        # in a supportive/comparative/committing line gets a reason.
        challenged = set(act.soft_rejects) | set(act.hard_rejects)
        committed = set(act.accepts) | ({act.explicit_vote} if act.explicit_vote else set())
        compared = self._resolver is not None and realized_comparison(record.text, self._resolver)
        supportive_line = (
            compared
            or act.act_type in {ActType.SUPPORT, ActType.COMPROMISE, ActType.OPENING}
            or has_support_claim(record.text)
        )
        for option_id in act.option_refs:
            cov = state.coverage[option_id]
            cov.mentions += 1
            if option_id in challenged:
                cov.objections += 1
            elif option_id in committed or supportive_line:
                cov.reasons += 1

        # Concern/blocker threads (6.2/6.3): first fold this turn into open
        # threads (a semantically relevant reply moves hot -> cooling; a
        # raiser's visible acceptance resolves), then open option-specific
        # threads for objections raised here, eroding advocates' commitment.
        self._observe_concern_responses(state, record)
        # Comparison threads (6.4): a realized head-to-head gets one normalized
        # pair thread; a relevant pair response cools it. Short-lived by design.
        self._observe_comparison_responses(state, record)
        self._register_comparison_thread(state, record)
        # Concern/blocker threads open only from parsed objections in the final
        # accepted text — a routed CONCERN intent whose line shows no visible
        # objection registers nothing (accepted text is the semantic authority).
        for option_id in challenged:
            self._register_concern_thread(state, record, option_id)
        # Visible support is social pressure too: a commitment/acceptance for one
        # option slightly erodes other sims' hold on different favorites.
        supported = set(act.accepts) | ({act.explicit_vote} if act.explicit_vote else set())
        for option_id in supported:
            self._apply_support_pressure(state, record.speaker_id, option_id)
        # Speaking in support of the own favorite rebuilds commitment a little.
        own = rt.top_option()
        if (
            own
            and own in act.option_refs
            and act.act_type in {ActType.OPENING, ActType.SUPPORT, ActType.COMPARE}
            and own not in challenged
        ):
            self._set_commitment(rt, rt.commitment_strength + 0.04)

        # A visible resolution can reopen a parsed blocker for THIS sim only, and it
        # must run before votes so "that fixes my concern; I can live with X" counts.
        if act.resolves_blocker:
            # Re-open a previously blocked option only when the speaker visibly
            # resolves their own blocker; rank moves back to neutral/acceptable later.
            if rt.rank(act.resolves_blocker) == STANCE_REJECTED:
                rt.set_rank(act.resolves_blocker, STANCE_NEUTRAL)

        allow_change = bool(record.intent and record.intent.allow_vote_change)
        if act.explicit_vote and act.explicit_vote not in rt.rejected_options():
            vote_stance = "accept" if act.explicit_vote in act.accepts else "vote"
            self._set_vote(rt, act.explicit_vote, act.text, force=allow_change, stance=vote_stance)
        for option_id in act.accepts:
            if option_id in rt.rejected_options():
                continue  # an actively blocked option needs a visible resolution first
            rt.mark_acceptable(option_id, reason_for=act.text)
            self._set_vote(rt, option_id, act.text, force=allow_change, stance="accept")
            state.coverage[option_id].acceptances += 1
        for option_id, reason in act.soft_rejects.items():
            rt.mark_disliked(option_id, reason_against=reason)
        for option_id, reason in act.hard_rejects.items():
            # A parse artifact must not turn the speaker's own current favorite into a hard blocker.
            if option_id == (rt.top_option() or persona.preferred_option):
                continue
            rt.mark_rejected(option_id, reason_against=reason)

        # Record visible vote movement (first vote away from the initial
        # preference, or a change of an earlier vote). `has_reason` is the weak
        # signal (any reason clause); `has_bridge` is the issue-5 signal — the
        # switch visibly links the sim's pre-turn lean to the new pick with a
        # reason, which is what the validator enforces on a switch away from lean.
        if rt.explicit_vote and rt.explicit_vote != prior_vote:
            baseline = prior_vote or persona.preferred_option
            if rt.explicit_vote != baseline:
                bridged = (
                    prior_pref == rt.explicit_vote
                    or (self._resolver is not None
                        and switch_bridge_ok(act.text, prior_pref, self._resolver))
                )
                rt.switch_events.append({
                    "from": baseline,
                    "to": rt.explicit_vote,
                    "has_reason": commitment_has_reason(act.text),
                    "has_bridge": bool(bridged),
                })
                rt.concessions_made += 1

        self._apply_lean_movement(state, record, rt, persona)

        # Track the visible top pair per accepted turn for the stability
        # trigger (12.2). Tuples, so equality comparison is exact.
        state.top_pair_history.append(tuple(self._current_top_pair(state)))

        after = self._snapshot_progress(state)
        state.no_progress_count = 0 if after != before else state.no_progress_count + 1

        # Thread aging runs after the progress snapshot: decaying to stale is
        # the absence of progress and must not reset the stagnation counter.
        # A concern that dies unanswered weighs on the option's advocates more
        # than a defended one (issue 2/3 — unrebutted points erode).
        unanswered_hot = {
            tid for tid, t in state.threads.items()
            if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
            and t.status is ThreadStatus.HOT
        }
        threads_engine.age_threads(state)
        for tid in unanswered_hot:
            thread = state.threads[tid]
            if thread.status is not ThreadStatus.STALE:
                continue
            for option_id in thread.focus_options:
                for persona in state.personas:
                    prt = state.runtimes[persona.id]
                    if persona.id != thread.started_by and prt.top_option() == option_id:
                        self._set_commitment(
                            prt, prt.commitment_strength - 0.10 * (1.0 - 0.7 * persona.sim_params.stubbornness)
                        )

    # ------------------------------------------------------------------
    # Concern threads and commitment dynamics (issue 2)
    # ------------------------------------------------------------------

    def _apply_lean_movement(
        self, state: DialogueState, record: TurnRecord, rt: ParticipantRuntime, persona: Persona
    ) -> None:
        """Move the latent lean only on a visible signal in the parsed text:
        an opening that names an option, visible softening wording, a visible
        compromise offer, or explicit conditional support. Never from routing
        intent alone (votes/acceptances are handled by ``_set_vote``)."""
        act = record.act
        if record.intent and record.intent.act == ActType.OPENING:
            # The opening lean follows the option the line visibly BACKS: an
            # option the same line soft/hard-rejects is never promoted, even
            # when it was the routed favorite ("A seems too expensive, while B
            # fits us much better" leans B). Preference order: the routed
            # favorite when named positively, else a unique positively named
            # option; ambiguous or all-negative naming moves nothing.
            challenged = set(act.soft_rejects) | set(act.hard_rejects)
            positive = [
                oid for oid in act.option_refs
                if oid in state.scenario.option_ids and oid not in challenged
            ]
            focus = record.intent.option_focus[0] if record.intent.option_focus else None
            target = focus if focus in positive else (positive[0] if len(set(positive)) == 1 else None)
            if target:
                rt.promote_to_preferred(target)
        elif (softened := act.softens_toward) and softened != rt.top_option() and self._can_shift_to(state, persona, softened):
            # Explicit visible softening ("B is starting to make more sense to
            # me", issue 3): the internal lean follows the sim's own words —
            # withholding the shift would make the state dishonest.
            rt.promote_to_preferred(softened, reason_for=act.text)
            rt.concessions_made += 1
            self._set_commitment(rt, max(rt.commitment_strength, 0.30) + 0.10)
            if record.phase == Phase.DISCUSSION:
                state.discussion_lean_shifts += 1
        else:
            signal = act.offers_compromise or act.conditional_support
            if signal and signal != rt.top_option() and self._can_shift_to(state, persona, signal):
                # Movability scales with the tracked commitment (issue 2): a sim
                # whose favorite took unanswered challenges/pressure moves more
                # easily than one that has been defending it.
                effective = persona.sim_params.stubbornness * (0.4 + 0.9 * rt.commitment_strength)
                if random.random() > effective:
                    rt.promote_to_preferred(signal, reason_for=act.text)
                    rt.concessions_made += 1
                    self._set_commitment(rt, max(rt.commitment_strength, 0.35) + 0.10)
                    if record.phase == Phase.DISCUSSION:
                        state.discussion_lean_shifts += 1

    @staticmethod
    def _set_commitment(rt: ParticipantRuntime, value: float) -> None:
        rt.commitment_strength = max(0.05, min(0.95, value))
        rt.commitment_min = min(rt.commitment_min, rt.commitment_strength)

    def _register_concern_thread(self, state: DialogueState, record: TurnRecord, option_id: str) -> None:
        """Open an option-specific concern/blocker thread from a final turn (6.2/6.3).

        Repeats of a resolved issue are suppressed by the thread engine; only a
        genuinely new thread erodes advocates' commitment, so repeated generic
        objections stop applying pressure while a fresh issue still bites.
        """
        if option_id not in state.scenario.option_ids:
            return
        is_hard = option_id in record.act.hard_rejects
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
                erosion = 0.12 * (1.0 - 0.7 * persona.sim_params.stubbornness)
                self._set_commitment(rt, rt.commitment_strength - erosion)

    def _observe_concern_responses(self, state: DialogueState, record: TurnRecord) -> None:
        """Fold this final turn into open concern/blocker threads (6.2/6.3).

        Merely mentioning the concerned option is not enough: a response counts
        only when it references the option AND responds to the issue — routed
        as a reply to the source turn, matching the issue key, visibly
        resolving the blocker, or visibly accepting the option. The raiser's
        own visible acceptance/softening/resolution resolves the thread.
        """
        act = record.act
        refs = set(act.option_refs)
        for thread in list(state.threads.values()):
            if thread.thread_type not in (ThreadType.CONCERN, ThreadType.BLOCKER):
                continue
            if thread.status in (ThreadStatus.RESOLVED, ThreadStatus.STALE):
                continue
            focus = set(thread.focus_options)
            if not focus & refs and act.resolves_blocker not in focus:
                continue
            if record.speaker_id == thread.started_by:
                accepted = bool(
                    focus & set(act.accepts)
                    or (act.explicit_vote in focus)
                    or (act.softens_toward in focus)
                    or (act.resolves_blocker in focus)
                )
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
            routed_reply = bool(
                record.intent and record.intent.respond_to_turn == thread.source_turn_index
            )
            relevant = (
                routed_reply
                or (act.resolves_blocker in focus)
                or bool(focus & set(act.accepts))
                or self._issue_relevant(state, record.text, thread)
            )
            if relevant and thread.status is ThreadStatus.HOT:
                threads_engine.mark_response(
                    state, thread, responder_id=record.speaker_id, turn_index=record.index
                )
            elif relevant:
                threads_engine.touch_thread(
                    state, thread, turn_index=record.index, participant_id=record.speaker_id
                )

    def _register_comparison_thread(self, state: DialogueState, record: TurnRecord) -> None:
        """Open/update one normalized option-pair thread for a realized comparison.

        A comparison is realized when the final text names at least two valid
        options with visibly comparative wording — the parsed text is the
        single authority; a routed compare whose line never contrasts anything
        registers nothing, while a comparative question realizes ASK *and* a
        comparison. Pair identity ignores the issue key: one pair, one thread
        ("one normalized option-pair thread", 6.4).
        """
        assert self._resolver is not None
        if not realized_comparison(record.text, self._resolver):
            return
        refs = [oid for oid in record.act.option_refs if oid in state.scenario.option_ids]
        if len(set(refs)) < 2:
            return
        pair = threads_engine.normalize_pair(refs)[:2]
        threads_engine.open_thread(
            state,
            thread_type=ThreadType.COMPARISON,
            focus_options=pair,
            issue_key="pair",  # pair-only identity: same pair -> same thread
            started_by=record.speaker_id,
            source_turn_index=record.index,
        )

    def _observe_comparison_responses(self, state: DialogueState, record: TurnRecord) -> None:
        """Move a comparison thread hot -> cooling on a relevant pair response:
        another participant engaging the same pair (both options named) or
        replying directly to the comparison turn."""
        refs = set(record.act.option_refs)
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
    def _issue_relevant(state: DialogueState, text: str, thread) -> bool:
        """Deterministic issue-level relevance between a response and a thread."""
        key = threads_engine.normalize_issue_key(
            text,
            state.scenario,
            [p.name for p in state.personas],
            focus_options=list(thread.focus_options),
        )
        if key == thread.issue_key:
            return True
        if thread.issue_key.startswith("sig:") and key.startswith("sig:"):
            thread_tokens = set(thread.issue_key[4:].split("-"))
            response_tokens = set(key[4:].split("-"))
            return bool(thread_tokens & response_tokens)
        return False

    def _apply_support_pressure(self, state: DialogueState, supporter_id: str, option_id: str) -> None:
        """Visible backing for one option erodes rival advocates' commitment."""
        if option_id not in state.scenario.option_ids:
            return
        for persona in state.personas:
            rt = state.runtimes[persona.id]
            if persona.id != supporter_id and rt.top_option() and rt.top_option() != option_id:
                erosion = 0.05 * (1.0 - 0.7 * persona.sim_params.stubbornness)
                self._set_commitment(rt, rt.commitment_strength - erosion)

    @staticmethod
    def _set_vote(rt: ParticipantRuntime, option_id: str, text: str, *, force: bool = False, stance: str = "vote") -> None:
        """Record a clear vote, protecting an existing one from silent overwrite.

        A participant who already cast a clear vote keeps it unless their visible
        text explicitly signals a change (e.g. 'actually I vote for', 'switch to'),
        or ``force`` is set (used in the explicit split-vote compromise step).
        Exception (issue #23): a formal direct vote replaces a commitment that
        was only an acceptance earlier in discussion — otherwise a casual
        "X works for me" locks out the actual vote round and manufactures a
        phantom split. A direct vote never silently replaces another direct
        vote (issue #5).
        """
        if rt.explicit_vote and rt.explicit_vote != option_id and not force and not _VOTE_CHANGE.search(text or ""):
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

    def _register_question_thread(self, state: DialogueState, record: TurnRecord) -> None:
        """Open/update a question thread from a final public question (6.1).

        Scope comes from the parser's visible-text reading; the respondent for
        a group question is a controller decision made here, using relevance
        and turn balance — never the parser's.
        """
        act = record.act
        scope = act.question_scope
        if scope == "direct":
            respondent = act.question_target_id
            if not respondent or respondent == record.speaker_id or respondent not in state.runtimes:
                return
        else:
            respondent = self._pick_group_respondent(state, record.speaker_id, act.option_refs[:2])
            if respondent is None:
                return
        focus = [oid for oid in act.option_refs[:2] if oid in state.scenario.option_ids]
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

    def _observe_question_answers(self, state: DialogueState, record: TurnRecord) -> None:
        """Move question threads hot -> cooling when this final turn answers them.

        A response counts only when the required respondent speaks AND the
        accepted text visibly relates to the question — its focused option or
        its normalized issue key. The routed act alone is never sufficient
        (closeout 2): an accepted but unrelated statement closes nothing, and
        a deterministic fallback line never realizes an answer (Section 15).
        """
        for thread in list(state.threads.values()):
            if thread.thread_type is not ThreadType.QUESTION or thread.status is not ThreadStatus.HOT:
                continue
            if thread.required_respondent != record.speaker_id:
                continue
            if record.index <= thread.source_turn_index:
                continue
            if record.used_fallback:
                continue
            relevant = (
                bool(set(thread.focus_options) & set(record.act.option_refs))
                or self._issue_relevant(state, record.text, thread)
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
            if turn.act.option_refs:
                return turn.act.option_refs[:2]
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
