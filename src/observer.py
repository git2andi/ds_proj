"""Visible-state observation for the dialogue runner (issue 8 extraction).

ObserverMixin turns a generated line into parsed structure and folds it into the
visible state: it parses the dialogue act, applies semantics (votes, acceptances,
blockers, latent-lean movement, switch events), and manages response obligations
and open questions. Mixed into DialogueRunner; all state lives on ``self``.
"""

from __future__ import annotations

import random
import re

from config_loader import cfg
from models import (
    ActType,
    AgendaStatus,
    Concern,
    DialogueAct,
    DialogueState,
    MoveIntent,
    OpenQuestion,
    ParticipantRuntime,
    Persona,
    Phase,
    ResponseObligation,
    STANCE_NEUTRAL,
    STANCE_REJECTED,
    TurnRecord,
)
from parsing import commitment_has_reason, parse_dialogue_act, switch_bridge_ok
from simulator import expected_turn_share
from utils import weighted_choice

_VOTE_CHANGE = re.compile(
    r"\b(?:i\s+changed\s+my\s+mind|change\s+my\s+vote|switch(?:ing)?\s+(?:to|my\s+vote)|"
    r"actually\s+i\s+(?:vote|choose|support|pick|prefer)|on\s+second\s+thought|"
    r"i'?ll\s+switch|then\s+i\s+switch|let'?s\s+switch)\b",
    re.I,
)

# Practical-logistics issue lexicon for the issue ledger (P7). An issue is
# recorded only when the turn treats it as unknown or asks about it — a fact
# actually stated on the option board never trips the uncertainty gate.
_LEDGER_ISSUES: dict[str, re.Pattern] = {
    "parking": re.compile(r"\bparking\b", re.I),
    "booking/reservations": re.compile(r"\breserv\w+\b|\bbook(?:ing|ed|s)?\b", re.I),
    "weather": re.compile(r"\bweather\b|\brain\w*\b|\bforecast\w*\b", re.I),
    "seating/space": re.compile(r"\bseating\b|\bseats?\b|\benough\s+(?:space|room)\b|\bfit\s+(?:all|everyone)\b", re.I),
    "availability/scheduling": re.compile(r"\bavailab\w+\b|\bfree\s+that\s+(?:day|evening|afternoon|night)\b|\bopen\s+(?:on|that|late)\b", re.I),
    "prep/setup time": re.compile(r"\bprep\s+time\b|\bset-?up\s+time\b", re.I),
    "crowds/queues": re.compile(r"\bcrowd\w*\b|\bqueue\w*\b|\bhow\s+busy\b", re.I),
}


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

    def _update_issue_ledger(self, state: DialogueState, record: TurnRecord) -> None:
        """Track practical unknowns so the discussion stops reopening them (P7).

        A ledger entry opens when a turn raises a logistics issue as unknown or
        as a question. ``_UNCERTAINTY`` comes from ValidationMixin — both mixins
        are composed into the same DialogueRunner.
        """
        text = record.text
        uncertain = bool(self._UNCERTAINTY.search(text)) or "?" in text
        if not uncertain:
            return
        for issue, pattern in _LEDGER_ISSUES.items():
            if not pattern.search(text):
                continue
            entry = state.issue_ledger.setdefault(issue, {"mentions": 0, "options": []})
            entry["mentions"] += 1
            for option_id in record.act.option_refs:
                if option_id not in entry["options"]:
                    entry["options"].append(option_id)

    def _apply_semantics(self, state: DialogueState, record: TurnRecord) -> None:
        rt = state.runtimes[record.speaker_id]
        persona = state.persona_by_id(record.speaker_id)
        act = record.act
        before = self._snapshot_progress(state)
        prior_vote = rt.explicit_vote
        prior_pref = rt.top_option() or persona.preferred_option

        if act.question_target_id:
            self._register_question(state, record)
        if record.intent and record.intent.act == ActType.ANSWER:
            self._close_answered_questions(state, record.speaker_id)

        for option_id in act.option_refs:
            cov = state.coverage[option_id]
            cov.mentions += 1
            if act.act_type in {ActType.SUPPORT, ActType.SUPPORT, ActType.COMPARE, ActType.COMPROMISE, ActType.SOFTEN_TOWARD, ActType.OPENING}:
                cov.reasons += 1
            if act.act_type in {ActType.CONCERN, ActType.CONCERN} or option_id in act.soft_rejects or option_id in act.hard_rejects:
                cov.objections += 1

        self._update_issue_ledger(state, record)

        # Stateful concern threads (issue 2): first fold this turn into any open
        # threads (a reply about the option addresses it; unanswered threads
        # age out after a few turns), then open new threads for objections and
        # challenges raised here, applying social pressure to the option's
        # advocates.
        self._update_concern_threads(state, record)
        challenged = set(act.soft_rejects) | set(act.hard_rejects)
        if record.intent and record.intent.act == ActType.CONCERN:
            # Register challenge concerns against rivals, not the speaker's own current pick.
            own = state.runtimes[record.speaker_id].top_option() if record.speaker_id in state.runtimes else None
            rival_refs = [oid for oid in act.option_refs if oid != own]
            challenged.update(rival_refs[:1] or act.option_refs[:1])
        for option_id in challenged:
            self._register_concern(state, record, option_id)
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
            and act.act_type in {ActType.OPENING, ActType.SUPPORT, ActType.SUPPORT, ActType.COMPARE}
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

        # Latent lean may move only on a visible signal in the parsed text:
        # a vote/acceptance (handled by _set_vote), a visible compromise offer
        # or proposal, or explicit conditional support. Never from routing
        # intent alone (issue I4).
        if record.intent and record.intent.act == ActType.OPENING and record.intent.option_focus:
            rt.promote_to_preferred(record.intent.option_focus[0])
        elif (softened := self._softening_signal(state, record, rt)) and softened != rt.top_option() and self._can_shift_to(state, persona, softened):
            # Explicit visible softening ("B is starting to make more sense to
            # me", issue 3): the internal lean follows the sim's own words —
            # withholding the shift would make the state dishonest.
            rt.promote_to_preferred(softened, reason_for=act.text)
            rt.concessions_made += 1
            self._set_commitment(rt, max(rt.commitment_strength, 0.30) + 0.10)
            if record.phase == Phase.DISCUSSION:
                state.discussion_lean_shifts += 1
        else:
            signal = act.offers_compromise or act.proposes_option or act.conditional_support
            if signal and signal != rt.top_option() and self._can_shift_to(state, persona, signal):
                # Movability scales with the tracked commitment (issue 2): a sim
                # whose favorite took unanswered challenges/pressure moves more
                # easily than one that has been defending it.
                effective = persona.sim_params.compromise_threshold * (0.4 + 0.9 * rt.commitment_strength)
                if random.random() > effective:
                    rt.promote_to_preferred(signal, reason_for=act.text)
                    rt.concessions_made += 1
                    self._set_commitment(rt, max(rt.commitment_strength, 0.35) + 0.10)
                    if record.phase == Phase.DISCUSSION:
                        state.discussion_lean_shifts += 1

        self._update_discussion_agenda(state, record)

        after = self._snapshot_progress(state)
        state.no_progress_count = 0 if after != before else state.no_progress_count + 1

    def _update_discussion_agenda(self, state: DialogueState, record: TurnRecord) -> None:
        """Mark chat-level checklist items completed by visible transcript evidence."""
        act = record.act
        for item in state.discussion_agenda:
            if item.status != AgendaStatus.PENDING:
                continue
            if item.key.startswith("cover_option:") and item.option in state.coverage:
                coverage = state.coverage[item.option]
                if coverage.mentions > 0 or item.option in act.option_refs:
                    item.status = AgendaStatus.DONE
            elif item.key == "compare_top_options":
                compared = [oid for oid in act.option_refs if oid in state.scenario.option_ids]
                if act.act_type == ActType.COMPARE and len(set(compared)) >= 2:
                    item.status = AgendaStatus.DONE
            elif item.key == "candidate_concern_check":
                candidate = state.candidate_option or self._visible_candidate(state) or self._latent_leading_option(state)
                if candidate and (candidate in act.soft_rejects or candidate in act.hard_rejects):
                    item.status = AgendaStatus.DONE
                elif candidate and state.coverage[candidate].objections > 0:
                    item.status = AgendaStatus.DONE

        # Avoid controller loops: if a checklist-routed turn failed to satisfy
        # its own item after validation/parsing, do not route the same item
        # forever. Coverage and vote readiness can still force progress.
        routed_key = record.intent.agenda_key if record.intent else None
        if routed_key:
            for item in state.discussion_agenda:
                if item.key == routed_key and item.status == AgendaStatus.PENDING:
                    item.status = AgendaStatus.SKIPPED
                    break

    # ------------------------------------------------------------------
    # Concern threads and commitment dynamics (issue 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _softening_signal(state: DialogueState, record: TurnRecord, rt: ParticipantRuntime) -> str | None:
        """Option this turn visibly warms to: the parsed softening phrase, or —
        on a routed softening beat — the attractor, provided the visible text
        actually engages it (names it, doesn't reject it). Wordings the regex
        misses ('genuinely clicks with me now') still move the lean then."""
        if record.act.softens_toward:
            return record.act.softens_toward
        intent = record.intent
        if intent is None or not intent.soften_toward:
            return None
        target = intent.soften_toward
        if (
            target in record.act.option_refs
            and target not in record.act.soft_rejects
            and target not in record.act.hard_rejects
        ):
            return target
        return None

    @staticmethod
    def _set_commitment(rt: ParticipantRuntime, value: float) -> None:
        rt.commitment_strength = max(0.05, min(0.95, value))
        rt.commitment_min = min(rt.commitment_min, rt.commitment_strength)

    def _register_concern(self, state: DialogueState, record: TurnRecord, option_id: str) -> None:
        """Open a concern thread and erode advocates' commitment (issue 2)."""
        if option_id not in state.scenario.option_ids:
            return
        state.concerns_raised_total += 1
        state.open_concerns.append(
            Concern(turn_id=record.index, raised_by=record.speaker_id, option_id=option_id, text=record.text)
        )
        state.open_concerns = state.open_concerns[-3:]
        for persona in state.personas:
            rt = state.runtimes[persona.id]
            if persona.id != record.speaker_id and rt.top_option() == option_id:
                rt.challenges_received += 1
                erosion = 0.12 * (1.0 - 0.7 * persona.sim_params.stubbornness)
                self._set_commitment(rt, rt.commitment_strength - erosion)

    def _update_concern_threads(self, state: DialogueState, record: TurnRecord) -> None:
        """Age open concerns; a reply about the option by someone else closes it."""
        live: list[Concern] = []
        for concern in state.open_concerns:
            if concern.turn_id == record.index:
                live.append(concern)
                continue
            if (
                concern.addressed_by is None
                and record.speaker_id != concern.raised_by
                and concern.option_id in record.act.option_refs
            ):
                concern.addressed_by = record.speaker_id
                state.concerns_addressed_total += 1
                continue  # thread completed, drop it
            concern.age += 1
            if concern.addressed_by is None and concern.age < 3:
                live.append(concern)
            elif concern.addressed_by is None:
                # The concern died unanswered: that weighs on the advocates more
                # than a defended one (issue 2/3 — unrebutted points erode).
                for persona in state.personas:
                    rt = state.runtimes[persona.id]
                    if persona.id != concern.raised_by and rt.top_option() == concern.option_id:
                        self._set_commitment(
                            rt, rt.commitment_strength - 0.10 * (1.0 - 0.7 * persona.sim_params.stubbornness)
                        )
        state.open_concerns = live

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

    def _next_answerable_question(self, state: DialogueState) -> OpenQuestion | None:
        live = []
        for question in state.open_questions:
            if any(t.speaker_id == question.target_id and t.index > question.turn_id for t in state.turns):
                continue
            if question.target_id in state.runtimes:
                live.append(question)
        state.open_questions = live[-4:]
        return state.open_questions[0] if state.open_questions else None

    def _register_question(self, state: DialogueState, record: TurnRecord) -> None:
        # A direct question is detected from visible text (parser sets
        # question_target_id), not from the routed act label: a question embedded
        # in a challenge/compare turn still creates a response obligation.
        target = record.act.question_target_id
        if not target or target == record.speaker_id or target not in state.runtimes:
            return
        # A group-directed question (no name, no "you") should not always fall to
        # the previous speaker — that chains one sim into an interview loop
        # (issue 1). Re-target by responsiveness and turn-share deficit.
        if not self._question_explicitly_addressed(state, record, target):
            target = self._pick_group_respondent(state, record.speaker_id)
            if target is None:
                return
            record.act.question_target_id = target
        state.open_questions.append(OpenQuestion(turn_id=record.index, asked_by=record.speaker_id, target_id=target, text=record.text, option_focus=record.act.option_refs[:2]))
        state.open_questions = state.open_questions[-4:]
        self._set_obligation(
            state,
            target_id=target,
            source_id=record.speaker_id,
            text=record.text,
            expected_act=ActType.ANSWER,
            option_focus=record.act.option_refs[:2],
        )

    @staticmethod
    def _question_explicitly_addressed(state: DialogueState, record: TurnRecord, target: str) -> bool:
        """True when the question visibly addresses the target (name or 'you')."""
        name = state.name_for(target)
        if name and re.search(rf"\b{re.escape(name.lower())}\b", record.text.lower()):
            return True
        return bool(re.search(r"\byou(?:r|rs)?\b", record.text, re.I))

    def _pick_group_respondent(self, state: DialogueState, asker_id: str) -> str | None:
        """Respondent for a group-directed question: responsive sims behind on
        their trait share answer first, so one sim never becomes the room's
        default interviewee."""
        others = [p for p in state.personas if p.id != asker_id]
        if not others:
            return None
        expected = expected_turn_share(state.personas)
        total = sum(rt.turn_count for rt in state.runtimes.values()) or 1
        weights = []
        for p in others:
            deficit = expected[p.id] - state.runtimes[p.id].turn_count / total
            weights.append(0.30 + p.sim_params.responsiveness + max(0.0, 3.0 * deficit))
        return weighted_choice(others, weights).id

    def _set_obligation(
        self,
        state: DialogueState,
        *,
        target_id: str,
        source_id: str,
        text: str,
        expected_act: ActType,
        option_focus: list[str],
    ) -> None:
        if target_id not in state.runtimes or target_id == source_id:
            return
        window = max(1, int(cfg.conversation.get("response_obligation_turns", 2)))
        state.obligations_created += 1
        state.response_obligation = ResponseObligation(
            target_id=target_id,
            source_id=source_id,
            question_text=text,
            expected_act=expected_act,
            created_turn=state.turn_index,
            expires_after=state.turn_index + 2 * window,  # turn_index counts moderator turns too
            option_focus=[o for o in option_focus if o in state.scenario.option_ids][:2],
        )

    def _active_obligation(self, state: DialogueState) -> ResponseObligation | None:
        ob = state.response_obligation
        if ob is None:
            return None
        if ob.target_id not in state.runtimes:
            state.response_obligation = None
            return None
        # Satisfied: the target has taken a turn since the obligation was created.
        if any(t.speaker_id == ob.target_id and t.index > ob.created_turn for t in state.turns):
            state.response_obligation = None
            return None
        # Lapsed: too many turns passed without the target answering.
        if state.turn_index > ob.expires_after:
            state.unanswered_obligations += 1
            state.response_obligation = None
            return None
        return ob

    def _obligation_intent(self, state: DialogueState, obligation: ResponseObligation) -> MoveIntent:
        focus = obligation.option_focus or self._focus_from_recent(state)
        if obligation.expected_act == ActType.VOTE:
            return self._vote_intent(state, state.persona_by_id(obligation.target_id), state.candidate_option or (focus[0] if focus else state.personas[0].preferred_option))
        return MoveIntent(
            speaker_id=obligation.target_id,
            act=ActType.ANSWER,
            reason="answer the direct question you were just asked, then add one implication for the decision",
            addressee_id=None if obligation.source_id == "moderator" else obligation.source_id,
            option_focus=focus,
            # Without this the prompt never shows WHICH question is owed, and
            # group-directed questions get pivoted around instead of answered.
            respond_to_turn=obligation.created_turn,
        )

    @staticmethod
    def _close_answered_questions(state: DialogueState, speaker_id: str) -> None:
        state.open_questions = [q for q in state.open_questions if q.target_id != speaker_id]

    @staticmethod
    def _focus_from_recent(state: DialogueState) -> list[str]:
        for turn in reversed(state.turns):
            if turn.act.option_refs:
                return turn.act.option_refs[:2]
        return []

    @staticmethod
    def _snapshot_progress(state: DialogueState) -> tuple:
        leans = tuple(sorted((pid, rt.top_option()) for pid, rt in state.runtimes.items()))
        votes = tuple(sorted((pid, rt.explicit_vote) for pid, rt in state.runtimes.items() if rt.explicit_vote))
        questions = tuple((q.turn_id, q.target_id) for q in state.open_questions)
        return leans, votes, questions
