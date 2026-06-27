"""Turn-taking, speaker selection, addressee selection, and local move routing.

The router follows a practical 3W decomposition for multi-party chat: who should
speak, when they should speak, and whom/what they should respond to.  It returns
MoveIntent objects rather than bare speaker IDs.
"""

from __future__ import annotations

import random

from config_loader import cfg
from models import ActType, DialogueState, MoveIntent, Phase, Persona, TurnRecord
from scoring import leading_option, option_support
from utils import weighted_choice


class TurnRouter:
    def next_intent(self, state: DialogueState) -> MoveIntent:
        if state.phase == Phase.OPENING:
            return self._opening_intent(state)

        if state.open_questions:
            q = state.open_questions[0]
            target = self._best_answerer(state, q)
            last = self._last_participant_turn(state)
            # Don't route the same speaker twice in a row (e.g. opening → immediate answer).
            # The question stays queued and is picked up after someone else speaks.
            if not (last and last.speaker_id == target and len(state.personas) > 1):
                return MoveIntent(
                    speaker_id=target,
                    addressee_id=q.asked_by,
                    act=ActType.ANSWER,
                    option_focus=q.option_focus or self._focus_for_person(state, target),
                    reason="answer the direct question before the thread moves on",
                    length_hint="medium",
                    respond_to_turn=q.turn_id,
                )

        if state.phase == Phase.NARROWING:
            return self._vote_intent(state)

        if state.phase == Phase.CONFIRMATION:
            confirm = self._confirmation_intent(state)
            if confirm:
                return confirm
            return self._discussion_intent(state, forced_act=ActType.PROPOSE_COMPROMISE)

        return self._discussion_intent(state)

    # ------------------------------------------------------------------

    def _opening_intent(self, state: DialogueState) -> MoveIntent:
        last = self._last_participant_turn(state)
        last_pid = last.speaker_id if last else None
        candidates = [p for p in state.personas if state.runtimes[p.id].turn_count == 0]
        # Avoid the same speaker going twice in a row (e.g. after their greeting).
        reordered = [p for p in candidates if p.id != last_pid] + [p for p in candidates if p.id == last_pid]
        for persona in reordered:
            return MoveIntent(
                speaker_id=persona.id,
                act=ActType.OPENING,
                option_focus=[persona.preferred_option],
                reason="state an initial priority without voting yet",
                length_hint="short",
            )
        return self._discussion_intent(state)

    def _vote_intent(self, state: DialogueState) -> MoveIntent:
        last_turn = self._last_participant_turn(state)
        last_pid = last_turn.speaker_id if last_turn else None
        unvoted = [p for p in self._least_recent_personas(state) if not state.runtimes[p.id].explicit_vote]
        others = [p for p in unvoted if p.id != last_pid]
        # Hard-exclude the last speaker when someone else still needs to vote.
        # Only let them vote when they're the sole remaining voter — BUT if they just had a
        # VOTE-routed turn and still have no explicit_vote (UNCLEAR_VOTE / repair failed),
        # don't route them again immediately; advance to confirmation instead.
        if not others and last_turn and last_turn.intent and last_turn.intent.act == ActType.VOTE:
            state.candidate_option = leading_option(state) or state.candidate_option
            state.phase = Phase.CONFIRMATION
            return self._confirmation_intent(state) or self._discussion_intent(state, forced_act=ActType.PROPOSE_COMPROMISE)
        ordered = others if others else unvoted
        for persona in ordered:
            if persona.is_hard_blocker:
                focus = [persona.preferred_option]
            else:
                focus = [state.runtimes[persona.id].current_preference or persona.preferred_option]
            return MoveIntent(
                speaker_id=persona.id,
                act=ActType.VOTE,
                option_focus=focus,
                reason="the group has discussed enough, so name your current pick and why",
                length_hint="medium",
            )
        state.candidate_option = leading_option(state) or state.candidate_option
        state.phase = Phase.CONFIRMATION
        return self._confirmation_intent(state) or self._discussion_intent(state, forced_act=ActType.PROPOSE_COMPROMISE)

    def _confirmation_intent(self, state: DialogueState) -> MoveIntent | None:
        candidate = state.candidate_option or leading_option(state)
        if not candidate:
            return None
        state.candidate_option = candidate
        threshold = int(cfg.scenario.acceptance_score)
        last_turn = self._last_participant_turn(state)
        last_pid = last_turn.speaker_id if last_turn else None
        candidates: list[tuple[Persona, bool]] = []
        for persona in self._least_recent_personas(state):
            rt = state.runtimes[persona.id]
            if rt.explicit_vote == candidate or candidate in rt.accepted_options:
                continue
            if candidate in rt.hard_rejections or candidate in rt.soft_rejections:
                continue
            if persona.id == last_pid and len(state.personas) > 1:
                continue
            can_accept = (
                (candidate in persona.acceptable_options or candidate == persona.preferred_option
                 or persona.score_for(candidate) >= threshold)
                and candidate not in persona.hard_rejections
                and not (persona.is_hard_blocker and candidate != persona.preferred_option)
            )
            candidates.append((persona, can_accept))
        for persona, can_accept in candidates:
            return MoveIntent(
                speaker_id=persona.id,
                act=ActType.ACCEPT if can_accept else ActType.REJECT,
                option_focus=[candidate],
                reason=f"say whether Option {candidate} works for you; if you accept, give the one thing that makes it okay",
                length_hint="short",
            )
        return None

    def _unanswered_challenge(self, state: DialogueState) -> MoveIntent | None:
        """If a recent OBJECT/PUSH_BACK hasn't been answered by the natural respondent
        (champion of the targeted option or the explicit addressee), route them to
        respond before phase/coverage logic picks someone else."""
        last_turn = self._last_participant_turn(state)
        last_pid = last_turn.speaker_id if last_turn else None
        window = int(cfg.routing.recent_speaker_window)
        recent = [t for t in state.turns if t.speaker_id != "moderator"][-window:]
        for turn in reversed(recent):
            if turn.act.act_type not in {ActType.OBJECT, ActType.PUSH_BACK}:
                continue
            target_id = turn.act.addressee_id
            if not target_id and turn.act.option_refs:
                opt = turn.act.option_refs[0]
                for p in state.personas:
                    if p.id == turn.speaker_id:
                        continue
                    if p.preferred_option == opt or state.runtimes[p.id].current_preference == opt:
                        target_id = p.id
                        break
            if not target_id or target_id == last_pid:
                continue
            if any(t.speaker_id == target_id and t.index > turn.index for t in state.turns):
                continue
            persona = state.persona_by_id(target_id)
            return MoveIntent(
                speaker_id=target_id,
                addressee_id=turn.speaker_id,
                act=ActType.REACT,
                option_focus=turn.act.option_refs or self._focus_for_person(state, target_id),
                reason=f"respond to {state.name_for(turn.speaker_id)}'s objection",
                length_hint=self._length_hint(persona),
                respond_to_turn=turn.index,
            )
        return None

    def _discussion_intent(self, state: DialogueState, forced_act: ActType | None = None) -> MoveIntent:
        if forced_act is None:
            challenge_response = self._unanswered_challenge(state)
            if challenge_response is not None:
                return challenge_response

        gap = self._coverage_gap_option(state)
        if gap:
            speaker_id = self._speaker_for_gap(state, gap)
            act = forced_act or self._act_for_gap(state, speaker_id, gap)
            addressee = self._target_for_act(state, speaker_id, act, gap)
            focus = self._focus_for_act(state, speaker_id, act, gap)
            return MoveIntent(
                speaker_id=speaker_id,
                addressee_id=addressee,
                act=act,
                option_focus=focus,
                reason=f"develop under-discussed Option {gap} with a real trade-off",
                length_hint=self._length_hint(state.persona_by_id(speaker_id)),
            )

        speaker_id = self._select_speaker(state)
        if forced_act is None and state.phase == Phase.DISCUSSION:
            persuasion = self._persuasion_intent(state, speaker_id)
            if persuasion is not None:
                return persuasion
        act = forced_act or self._sample_discussion_act(state, speaker_id)
        if self._should_skip(state, speaker_id, act):
            speaker_id = self._select_speaker(state, exclude=speaker_id)
            act = forced_act or self._sample_discussion_act(state, speaker_id)
        focus = self._focus_for_act(state, speaker_id, act, None)
        addressee = self._target_for_act(state, speaker_id, act, focus[0] if focus else None)
        return MoveIntent(
            speaker_id=speaker_id,
            addressee_id=addressee,
            act=act,
            option_focus=focus,
            reason=self._reason_for_act(act),
            length_hint=self._length_hint(state.persona_by_id(speaker_id)),
        )

    def _should_skip(self, state: DialogueState, speaker_id: str, act: ActType) -> bool:
        if len(state.personas) < 4:
            return False
        if act in {ActType.ASK, ActType.OBJECT, ActType.PUSH_BACK, ActType.PROPOSE_COMPROMISE}:
            return False
        rt = state.runtimes[speaker_id]
        if rt.turn_count < 2:
            return False
        persona = state.persona_by_id(speaker_id)
        focus = rt.current_preference or persona.preferred_option
        covered = state.coverage.get(focus)
        if covered and covered.reasons >= 3 and act == ActType.SUPPORT:
            return True
        return False

    # ------------------------------------------------------------------

    def _pick_speaker(self, state: DialogueState, extra_score) -> str:
        ids = state.participant_ids()
        recent = self._recent_speaker_ids(state)
        last_turn = self._last_participant_turn(state)
        last = last_turn.speaker_id if last_turn else None
        min_turns = min((state.runtimes[pid].turn_count for pid in ids), default=0)
        n = len(ids)
        weights: list[float] = []
        for pid in ids:
            persona = state.persona_by_id(pid)
            rt = state.runtimes[pid]
            score = 1.0
            if rt.turn_count == min_turns and rt.turn_count == 0:
                score += float(cfg.routing.unspoken_boost)
            elif rt.turn_count == min_turns:
                score += float(cfg.routing.low_turn_count_boost) * (0.5 if n >= 5 else 1.0)
            score += persona.traits.initiative * float(cfg.routing.initiative_weight)
            score += persona.traits.extraversion * float(cfg.routing.extraversion_weight)
            score += extra_score(persona, rt, recent)
            weights.append(0.0 if (pid == last and len(ids) > 1) else max(0.01, score))
        return weighted_choice(ids, weights)

    def _select_speaker(self, state: DialogueState, exclude: str | None = None) -> str:
        def extra(persona: Persona, rt, recent: list[str]) -> float:
            score = 0.0
            if rt.turn_count == 0:
                score += float(cfg.routing.unspoken_boost)
            score += len(rt.soft_rejections) * float(cfg.routing.unresolved_objection_boost)
            score -= recent.count(persona.id) * float(cfg.routing.recent_speaker_penalty)
            if exclude and persona.id == exclude:
                score -= 10.0
            return score
        return self._pick_speaker(state, extra)

    def _speaker_for_gap(self, state: DialogueState, option_id: str) -> str:
        def extra(persona: Persona, rt, recent: list[str]) -> float:
            score = 0.0
            if option_id == persona.preferred_option:
                score += float(cfg.routing.preferred_option_gap_boost)
            if option_id in persona.acceptable_options:
                score += float(cfg.routing.acceptable_option_gap_boost)
            if option_id in persona.soft_rejections or option_id in persona.hard_rejections:
                score += float(cfg.routing.unresolved_objection_boost)
            return score
        return self._pick_speaker(state, extra)

    def _coverage_gap_option(self, state: DialogueState) -> str | None:
        # Surface a real alternative that hasn't come up yet — but only one that at least
        # one person actually prefers or could live with. Options nobody wants are left
        # unmentioned on purpose: forcing talk about a dead option ("let's also consider
        # D") is exactly the filler that doesn't sound like real group conversation.
        with_stake = {p.preferred_option for p in state.personas}
        with_stake.update(opt for p in state.personas for opt in p.acceptable_options)
        for option_id, coverage in state.coverage.items():
            if coverage.mentions == 0 and option_id in with_stake:
                return option_id
        return None

    def _act_for_gap(self, state: DialogueState, speaker_id: str, option_id: str) -> ActType:
        # Surface an under-discussed option in a way that stays true to the speaker's stance:
        #  - their own preferred pick -> SUPPORT (champion it)
        #  - merely acceptable (not preferred) -> COMPARE, so they weigh it against their pick
        #    and stay anchored, instead of advocating it as if it were their favourite
        #  - rejected -> OBJECT, which surfaces the option *and* gives an explicit reason to drop
        #    it (so an option isn't just silently forgotten — ANALYSIS #8)
        persona = state.persona_by_id(speaker_id)
        if option_id in persona.soft_rejections or option_id in persona.hard_rejections:
            return ActType.OBJECT
        if option_id == persona.preferred_option:
            return ActType.SUPPORT
        return ActType.COMPARE

    def _sample_discussion_act(self, state: DialogueState, speaker_id: str) -> ActType:
        if state.no_progress_count >= int(cfg.conversation.no_progress_window_turns):
            return weighted_choice(
                [ActType.PROPOSE_COMPROMISE, ActType.COMPARE],
                [float(cfg.routing.no_progress_compromise_probability), 1.0 - float(cfg.routing.no_progress_compromise_probability)],
            )
        probs = dict(cfg.routing.act_probabilities.items())
        if state.readiness_score >= float(cfg.conversation.concentration_to_narrow):
            probs[ActType.PROPOSE_COMPROMISE.value] = float(probs.get(ActType.PROPOSE_COMPROMISE.value, 0.0)) + float(cfg.routing.late_discussion_compromise_bonus)
        persona = state.persona_by_id(speaker_id)
        trait_mid = (int(cfg.personas.trait_min) + int(cfg.personas.trait_max)) / 2.0
        # openness: curious speakers ask more
        curiosity = persona.traits.openness / trait_mid
        probs[ActType.ASK.value] = float(probs.get(ActType.ASK.value, 0.0)) * curiosity
        # agreeableness: agreeable → react/support more, object/push-back less; disagreeable → reverse
        agree_mod = 1.0 + (persona.traits.agreeableness - trait_mid) * 0.15
        probs[ActType.REACT.value] = float(probs.get(ActType.REACT.value, 0.0)) * agree_mod
        probs[ActType.SUPPORT.value] = float(probs.get(ActType.SUPPORT.value, 0.0)) * agree_mod
        probs[ActType.OBJECT.value] = float(probs.get(ActType.OBJECT.value, 0.0)) * (2.0 - agree_mod)
        probs[ActType.PUSH_BACK.value] = float(probs.get(ActType.PUSH_BACK.value, 0.0)) * (2.0 - agree_mod)
        # conscientiousness: careful thinkers ask before reacting; low-consc speakers react first
        consc_mod = 1.0 + (persona.traits.conscientiousness - trait_mid) * 0.15
        probs[ActType.ASK.value] = float(probs.get(ActType.ASK.value, 0.0)) * consc_mod
        probs[ActType.REACT.value] = float(probs.get(ActType.REACT.value, 0.0)) * (2.0 - consc_mod)
        # When discussion stalls (2-3 turns without progress), boost questions and
        # comparisons to steer toward concrete details instead of restating preferences.
        if state.no_progress_count >= 2:
            probs[ActType.ASK.value] = float(probs.get(ActType.ASK.value, 0.0)) * 2.0
            probs[ActType.COMPARE.value] = float(probs.get(ActType.COMPARE.value, 0.0)) * 1.5
            probs[ActType.SUPPORT.value] = float(probs.get(ActType.SUPPORT.value, 0.0)) * 0.3
        window = int(cfg.routing.ask_quiet_window)
        recent = [turn for turn in reversed(state.turns) if turn.speaker_id != "moderator"][:window]
        if any("?" in turn.text for turn in recent):
            probs[ActType.ASK.value] *= float(cfg.routing.ask_after_question_damping)
        # Hard veto: never route back-to-back ASK turns; if the previous speaker asked a
        # question, force the next act to be a response/reaction instead.
        if recent and "?" in recent[0].text:
            probs[ActType.ASK.value] = 0.0
        if len(state.personas) >= 5:
            focus_opt = state.runtimes[speaker_id].current_preference or persona.preferred_option
            covered = state.coverage.get(focus_opt)
            if covered and len(covered.covered_slots) >= 3:
                probs[ActType.COMPARE.value] = float(probs.get(ActType.COMPARE.value, 0.0)) * 1.5
                probs[ActType.ASK.value] = float(probs.get(ActType.ASK.value, 0.0)) * 1.3
                probs[ActType.OBJECT.value] = float(probs.get(ActType.OBJECT.value, 0.0)) * 1.3
                probs[ActType.SUPPORT.value] = float(probs.get(ActType.SUPPORT.value, 0.0)) * 0.5
        keys = [ActType(key) for key in probs.keys()]
        weights = [float(probs[key.value]) for key in keys]
        return weighted_choice(keys, weights)

    def _persuasion_intent(self, state: DialogueState, speaker_id: str) -> MoveIntent | None:
        """If the group is rallying behind an option this speaker can genuinely live with
        (more than their current lean), give them a chance to be won over — a real change
        of mind voiced in the chat, not just a silent tally shift. Routed as SUPPORT of
        that option so the state tracker moves their leaning toward it."""
        persona = state.persona_by_id(speaker_id)
        if persona.is_hard_blocker:
            return None
        rt = state.runtimes[speaker_id]
        # Nobody folds before they've actually voiced and held their pick a couple of times —
        # instant capitulation in the first exchange is the main thing that read as fake.
        if rt.turn_count < int(cfg.routing.persuasion_min_speaker_turns):
            return None
        lean = rt.current_preference or persona.preferred_option
        threshold = int(cfg.scenario.acceptance_score)
        margin = float(cfg.routing.persuasion_support_margin)
        # A rival option has to clearly out-pull the current lean, not just edge it.
        best, best_support = None, option_support(state, lean) + margin
        for opt in persona.acceptable_options:
            if opt == lean or persona.score_for(opt) < threshold:
                continue
            support = option_support(state, opt)
            if support > best_support:
                best, best_support = opt, support
        if best is None:
            return None
        # Only fold when someone actually made the case in the chat (real expressed support),
        # not on latent utility alone — and address that person when voicing the change.
        supporter = self._supporter_of(state, best, speaker_id)
        if supporter is None:
            return None
        chance = persona.traits.compromise_willingness * float(cfg.routing.persuasion_probability_factor)
        if random.random() >= chance:
            return None
        return MoveIntent(
            speaker_id=speaker_id,
            addressee_id=supporter,
            act=ActType.SUPPORT,
            option_focus=[best],
            reason="a good case was made for an option you can live with; move toward it",
            length_hint=self._length_hint(persona),
            moves_lean=True,  # the only SUPPORT that legitimately shifts the speaker's lean
        )

    def _supporter_of(self, state: DialogueState, option_id: str, exclude: str) -> str | None:
        for turn in reversed(state.turns):
            if turn.speaker_id in {"moderator", exclude}:
                continue
            rt = state.runtimes.get(turn.speaker_id)
            if rt and (rt.explicit_vote == option_id or option_id in rt.accepted_options or rt.current_preference == option_id):
                return turn.speaker_id
        return None

    def _target_for_act(self, state: DialogueState, speaker_id: str, act: ActType, option_id: str | None) -> str | None:
        # Most lines in a real group chat are said to the room, not aimed at one person.
        # Only reply-style and conflict moves get an explicit addressee, and even those
        # only sometimes (direct_reply_probability) — over-addressing was what made every
        # turn read as second-person advice ("X works for you").
        if not state.turns:
            return None
        reply_prob = float(cfg.routing.direct_reply_probability)
        last_participant = self._last_participant_turn(state)
        has_other_last = bool(last_participant and last_participant.speaker_id != speaker_id)
        if act in {ActType.OBJECT, ActType.PUSH_BACK, ActType.COMPARE} and option_id:
            target = self._conflicting_person_for_option(state, speaker_id, option_id)
            if target and random.random() < reply_prob:
                return target
        if act in {ActType.ANSWER, ActType.REACT, ActType.PUSH_BACK, ActType.ASK} and has_other_last and random.random() < reply_prob:
            return last_participant.speaker_id
        return None

    def _conflicting_person_for_option(self, state: DialogueState, speaker_id: str, option_id: str) -> str | None:
        candidates: list[str] = []
        speaker = state.persona_by_id(speaker_id)
        speaker_likes = option_id == speaker.preferred_option or option_id in speaker.acceptable_options
        for persona in state.personas:
            if persona.id == speaker_id:
                continue
            other_likes = option_id == persona.preferred_option or option_id in persona.acceptable_options
            if other_likes != speaker_likes:
                candidates.append(persona.id)
        if not candidates:
            return None
        return weighted_choice(candidates, [1.0] * len(candidates))

    def _focus_for_act(self, state: DialogueState, speaker_id: str, act: ActType, option_id: str | None) -> list[str]:
        persona = state.persona_by_id(speaker_id)
        if option_id:
            focus = [option_id]
        elif state.candidate_option and act in {ActType.PROPOSE_COMPROMISE, ActType.ACCEPT, ActType.REJECT}:
            focus = [state.candidate_option]
        else:
            focus = [state.runtimes[speaker_id].current_preference or persona.preferred_option]
        if act in {ActType.COMPARE, ActType.PUSH_BACK}:
            pref = state.runtimes[speaker_id].current_preference or persona.preferred_option
            if pref not in focus:
                focus.append(pref)
        if act == ActType.PROPOSE_COMPROMISE:
            candidate = self._best_compromise_focus(state, speaker_id)
            if candidate and candidate not in focus:
                focus.insert(0, candidate)
        return focus[: int(cfg.utterances.max_focus_options_in_prompt)]

    def _best_answerer(self, state: DialogueState, q) -> str:
        """Pick the best person to answer an open question. If the question mentions
        an option, prefer whoever champions that option. Otherwise use the registered
        target (previous speaker or explicit addressee).

        Excludes q.asked_by and the askers of any other open questions — they are the
        ones who don't know, so routing them produces the "asker answers own question" bug."""
        other_askers = {oq.asked_by for oq in state.open_questions if oq.turn_id != q.turn_id}

        def _eligible(pid: str) -> bool:
            return pid != q.asked_by and pid not in other_askers

        if q.option_focus:
            opt = q.option_focus[0]
            for p in state.personas:
                if not _eligible(p.id):
                    continue
                rt = state.runtimes[p.id]
                if rt.current_preference == opt or p.preferred_option == opt:
                    return p.id
        if q.target_id and _eligible(q.target_id):
            return q.target_id
        for p in self._least_recent_personas(state):
            if _eligible(p.id):
                return p.id
        # Last resort: at least exclude this question's own asker
        for p in self._least_recent_personas(state):
            if p.id != q.asked_by:
                return p.id
        return q.target_id

    def _focus_for_person(self, state: DialogueState, speaker_id: str) -> list[str]:
        rt = state.runtimes[speaker_id]
        persona = state.persona_by_id(speaker_id)
        return [rt.current_preference or persona.preferred_option]

    def _best_compromise_focus(self, state: DialogueState, speaker_id: str) -> str | None:
        persona = state.persona_by_id(speaker_id)
        candidates = list(dict.fromkeys(persona.acceptable_options + ([state.candidate_option] if state.candidate_option else [])))
        if not candidates:
            return persona.preferred_option
        return max(candidates, key=lambda opt: option_support(state, opt))

    def _least_recent_personas(self, state: DialogueState) -> list[Persona]:
        return sorted(state.personas, key=lambda p: (state.runtimes[p.id].last_spoke_turn is not None, state.runtimes[p.id].last_spoke_turn or -1))

    def _recent_speaker_ids(self, state: DialogueState) -> list[str]:
        window = int(cfg.routing.recent_speaker_window)
        ids = [turn.speaker_id for turn in state.turns if turn.speaker_id != "moderator"]
        return ids[-window:]

    @staticmethod
    def _last_participant_turn(state: DialogueState) -> TurnRecord | None:
        for turn in reversed(state.turns):
            if turn.speaker_id != "moderator":
                return turn
        return None

    @staticmethod
    def _consecutive_same_speaker(state: DialogueState) -> int:
        count = 0
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                continue
            if count == 0:
                count = 1
                pid = turn.speaker_id
            elif turn.speaker_id == pid:
                count += 1
            else:
                break
        return count

    def _length_hint(self, persona: Persona) -> str:
        # Bias length toward the persona's verbosity so a terse speaker and a chatty one
        # read differently, instead of every turn drawing the same random length.
        base = cfg.routing.length_hint_probabilities
        trait_mid = (int(cfg.personas.trait_min) + int(cfg.personas.trait_max)) / 2.0
        slope = float(cfg.routing.length_trait_slope)
        rl = persona.traits.response_length
        w_short = float(base.get("short", 0.0)) * max(0.0, 1.0 + (trait_mid - rl) * slope)
        w_long = float(base.get("long", 0.0)) * max(0.0, 1.0 + (rl - trait_mid) * slope)
        w_medium = float(base.get("medium", 0.0))
        return weighted_choice(["short", "medium", "long"], [w_short, w_medium, w_long])

    @staticmethod
    def _reason_for_act(act: ActType) -> str:
        return {
            ActType.REACT: "react naturally to the last useful point",
            ActType.ASK: "ask one targeted question that helps the decision",
            ActType.COMPARE: "weigh the real trade-offs between options and say which side you land on",
            ActType.SUPPORT: "add a concrete reason for an option",
            ActType.OBJECT: "raise a soft objection without derailing the group",
            ActType.PUSH_BACK: "push back on a point while staying cooperative",
            ActType.PROPOSE_COMPROMISE: "suggest a workable compromise using an existing option",
        }.get(act, "make a useful local contribution")
