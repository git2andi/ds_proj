"""Routing policy for the dialogue runner (issue 8 extraction).

PolicyMixin owns the controller's decisions — who speaks, which dialogue act,
which thread to target, when a vote is ready, and which option is the vote
candidate — plus the trait-driven word budget and surface-style intent flags.
It holds no orchestration or I/O; every method operates on the shared
DialogueState via ``self`` and is mixed into DialogueRunner.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter

from aliases import short_alias_map
from config_loader import cfg
from consensus import participant_turn_count
from models import (
    ActType,
    DialogueState,
    MoveIntent,
    Persona,
    TurnRecord,
    _DECISION_ACTS,
    _DISCUSSION_ACTS,
)
from parsing import round_reason_snippets, used_commitment_phrases
from simulator import expected_turn_share, next_agenda_item
from style import (
    first_person_opening_fraction,
    name_prefix_fraction,
    option_opening_fraction,
    repeated_opening_token,
    repeated_pattern,
    surface_pattern,
    we_opening_fraction,
)
from utils import preset_dominance_weight, weighted_choice


class PolicyMixin:
    def _rank_split_candidates(
        self,
        state: DialogueState,
        votes: list[str],
        *,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, list[Persona], list[Persona], dict]]:
        """Rank concrete split-vote candidates from visible votes.

        Candidate choice is intentionally deterministic. The previous scorer mixed
        visible vote count with stochastic shift checks, which allowed a one-vote
        option to beat a visible plurality in some ``2-1-1`` splits. This ranking
        treats the vote structure as the first-order social signal: a strict
        plurality is tested first unless every relevant dissenter has a hard
        blocker. Ties are then broken by objection load and compromise fit.
        """
        exclude = exclude or set()
        counts = Counter(v for v in votes if v in state.scenario.option_ids)
        if not counts:
            return []

        ranked: list[tuple[tuple[float, float, float, float, str], str, list[Persona], list[Persona], dict]] = []
        for candidate, count in counts.items():
            if candidate in exclude:
                continue
            dissenters = [p for p in state.personas if state.runtimes[p.id].explicit_vote != candidate]
            hard_blockers = [p for p in dissenters if self._hard_blocks_candidate(state, p, candidate)]
            if dissenters and len(hard_blockers) == len(dissenters):
                # Nobody outside the candidate camp can even conditionally move.
                # Testing this first would be a performative dead end.
                continue
            movable = [p for p in dissenters if p not in hard_blockers]
            resistance = [self._candidate_resistance(state, p, candidate) for p in movable]
            avg_resistance = sum(resistance) / max(1, len(resistance))
            compromise_fit = sum(1.0 - p.sim_params.compromise_threshold for p in movable) / max(1, len(movable))
            support_quality = 0.25 * state.coverage[candidate].reasons + 0.10 * state.coverage[candidate].mentions
            meta = {
                "votes": count,
                "hard_blockers": len(hard_blockers),
                "avg_resistance": round(avg_resistance, 3),
                "compromise_fit": round(compromise_fit, 3),
            }
            # Sort key is inverted later. Vote count dominates; blockers and
            # resistance only break ties / select among equal leaders.
            key = (
                float(count),
                -float(len(hard_blockers)),
                -avg_resistance,
                compromise_fit + support_quality,
                candidate,
            )
            ranked.append((key, candidate, dissenters, movable, meta))

        ranked.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], -item[0][3], item[1]))
        return [(candidate, dissenters, movers, meta) for _key, candidate, dissenters, movers, meta in ranked]

    def _split_probe_candidate(
        self, state: DialogueState, votes: list[str]
    ) -> tuple[str, list[Persona], list[Persona]] | None:
        """(candidate, dissenters, movers) for the split-vote compromise probe.

        The probe candidate must be genuinely acceptable to test: never an
        option someone has a visible unresolved dealbreaker on, and at least
        one dissenter must actually be able to move to it (I18). If no
        vote-getter qualifies, the pass is skipped and the run closes honestly.
        """
        ranked = self._rank_split_candidates(state, votes)
        if ranked:
            candidate, dissenters, movers, _meta = ranked[0]
            return candidate, dissenters, movers
        return None

    @staticmethod
    def _hard_blocks_candidate(state: DialogueState, persona: Persona, candidate: str) -> bool:
        rt = state.runtimes[persona.id]
        return persona.rejection == candidate or candidate in rt.hard_rejections

    @staticmethod
    def _candidate_resistance(state: DialogueState, persona: Persona, candidate: str) -> float:
        """Lower means the dissenter is a better candidate for a switch/stay beat."""
        rt = state.runtimes[persona.id]
        if persona.rejection == candidate or candidate in rt.hard_rejections:
            return 99.0
        resistance = 0.45 * rt.commitment_strength
        resistance += 0.35 * persona.sim_params.compromise_threshold
        resistance += 0.20 * persona.sim_params.stubbornness
        if candidate in persona.preferred_options:
            resistance -= 0.25
        if candidate in rt.accepted_options:
            resistance -= 0.20
        if rt.current_preference == candidate:
            resistance -= 0.35
        return max(0.0, resistance)

    # ------------------------------------------------------------------
    # Routing policy: who / what / whom
    # ------------------------------------------------------------------

    def _route_discussion_turn(self, state: DialogueState) -> MoveIntent:
        obligation = self._active_obligation(state)
        if obligation is not None:
            # Responsiveness controls how promptly a directly-addressed sim
            # answers (issue 1): a low-responsiveness sim may sit out one beat
            # before replying, but only when the obligation window still has
            # room — hesitation alone never lets a direct question lapse.
            target = state.persona_by_id(obligation.target_id)
            answer_now = 0.45 + 0.55 * target.sim_params.responsiveness
            if (
                obligation.deferred
                or state.turn_index + 1 >= obligation.expires_after
                or random.random() < answer_now
            ):
                return self._obligation_intent(state, obligation)
            obligation.deferred = True

        coverage_gap = self._coverage_gap_option(state)
        if coverage_gap is not None:
            # One bounded coverage attempt per option: even if the realizer never
            # names it cleanly, we do not re-route the same option forever.
            state.coverage[coverage_gap].coverage_attempts += 1
            speaker = self._speaker_for_option_coverage(state, coverage_gap)
            current = state.runtimes[speaker.id].current_preference or speaker.preferred_option
            focus = [coverage_gap]
            if current in state.scenario.option_ids and current != coverage_gap:
                focus.append(current)
            return MoveIntent(
                speaker_id=speaker.id,
                act=ActType.COMPARE,
                reason="briefly bring in an option that has not yet been socially processed, then compare it with the current lean",
                option_focus=focus,
            )

        # Reactive adjacency-pair moves (defend a challenged pick, follow up an
        # answer, probe a blocker, compare a visible split) come before the
        # agenda: local context drives acts, the agenda only fills quiet moments.
        reactive = self._reactive_intent(state)
        if reactive is not None:
            return reactive

        # Rare same-speaker continuation (issue 6): a short, genuinely additive
        # follow-up to the speaker's own last message. Normal turns still hard-
        # exclude the last speaker in _choose_speaker.
        continuation = self._maybe_continuation_intent(state)
        if continuation is not None:
            return continuation

        speaker = self._choose_speaker(state)
        agenda_pair = next_agenda_item(speaker)
        if agenda_pair and random.random() < (0.25 + 0.25 * speaker.sim_params.initiative):
            agenda_index, agenda_item = agenda_pair
            act = agenda_item.act
            focus = [agenda_item.option] if agenda_item.option in state.scenario.option_ids else []
        else:
            agenda_index = None
            act = self._choose_discussion_act(state, speaker)
            focus = []

        target_turn = self._choose_target_turn(state, speaker, act)
        target_speaker = target_turn.speaker_id if target_turn and target_turn.speaker_id != "moderator" else None
        addressee = target_speaker if target_speaker and random.random() < float(cfg.routing.direct_address_probability) else None
        if not focus:
            focus = self._focus_options(state, speaker, act, target_turn)
        reason = self._reason_for_act(state, speaker, act, focus, target_turn)
        moves_lean = act in {ActType.AGREE, ActType.PROPOSE_COMPROMISE} and bool(focus)
        return MoveIntent(
            speaker_id=speaker.id,
            act=act,
            reason=reason,
            addressee_id=addressee,
            option_focus=focus,
            respond_to_turn=target_turn.index if target_turn else None,
            agenda_index=agenda_index,
        )

    def _reactive_intent(self, state: DialogueState) -> MoveIntent | None:
        """Adjacency-pair moves driven by what just visibly happened (issue I9).

        Checked in order, each with a probability gate so runs don't become a
        rigid script: a challenged option gets defended by an advocate, an
        answer gets a follow-up, an active blocker on the leading option gets
        probed once, and a visible split gets an explicit comparison.
        """
        participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
        if not participant_turns:
            return None
        last = participant_turns[-1]
        aliases = short_alias_map(state.scenario.options)

        # 1. Visible softening (issue 3): a sim whose hold on its favorite has
        #    been eroded (low tracked commitment) says so in the discussion —
        #    "X is starting to make more sense to me" — instead of silently
        #    flipping at the final vote. Once per sim, needs a visibly backed
        #    attractor the sim can actually move to, and never a final vote.
        if participant_turn_count(state) > len(state.personas) and random.random() < (0.75 if state.discussion_lean_shifts == 0 else 0.55):
            movers: list[tuple[Persona, str]] = []
            for p in state.personas:
                rt = state.runtimes[p.id]
                if p.id in state.softened_sims or p.id == last.speaker_id:
                    continue
                # Trigger: eroded commitment, OR sustained social pressure — the
                # favorite keeps taking challenges, nobody else visibly backs
                # it, and the sim is flexible enough to plausibly move.
                own = rt.current_preference or p.preferred_option
                pressured = (
                    rt.challenges_received >= 2
                    and self._visible_support_count(state, own, exclude=p.id) == 0
                    and p.sim_params.compromise_threshold <= 0.50
                )
                if rt.commitment_strength > 0.58 and not pressured:
                    continue
                attractor = self._softening_attractor(state, p)
                if attractor:
                    movers.append((p, attractor))
            if movers:
                persona, attractor = min(movers, key=lambda pair: state.runtimes[pair[0].id].commitment_strength)
                state.softened_sims.add(persona.id)
                current = state.runtimes[persona.id].current_preference or persona.preferred_option
                phrase = random.choice([
                    "is starting to make more sense to me",
                    "is starting to look better to me",
                    "is growing on me",
                ])
                return MoveIntent(
                    speaker_id=persona.id,
                    act=ActType.SOFTEN,
                    reason=(
                        f"the case others made for {aliases[attractor]} genuinely lands with you — say "
                        f"openly that it {phrase}, name the argument that moved you, and what you still "
                        f"like about {aliases.get(current, current)}; this is a shift in view, NOT a final "
                        "vote — do not fully commit"
                    ),
                    option_focus=[attractor] + ([current] if current in state.scenario.option_ids and current != attractor else []),
                    soften_toward=attractor,
                )

        # 2. Concern thread: an open concern about an option someone else backs
        #    gets a reaction from an advocate within a turn or two (issue 2) —
        #    the thread persists across a turn, so a concern is not lost the
        #    moment another sim speaks about something else. How the advocate
        #    reacts depends on their tracked commitment: firm advocates defend,
        #    a shaken advocate is told to concede honestly and may name what
        #    still matters to them.
        for concern in reversed(state.open_concerns):
            if concern.addressed_by is not None:
                continue
            advocates = [
                p for p in state.personas
                if p.id not in {concern.raised_by, last.speaker_id}
                and state.runtimes[p.id].current_preference == concern.option_id
            ]
            if not advocates or random.random() >= 0.80:
                continue
            speaker = max(advocates, key=lambda p: p.sim_params.engagement + 0.3 * p.sim_params.stubbornness)
            concern.addressed_by = speaker.id
            state.concerns_addressed_total += 1
            rt = state.runtimes[speaker.id]
            if rt.commitment_strength <= 0.35:
                reason = (
                    f"a concern was raised about {aliases[concern.option_id]}, which you back, and it lands — "
                    "concede the point honestly, say what still matters to you in this choice, and if another "
                    "option now genuinely looks stronger you may say your view is shifting (no final vote)"
                )
            else:
                reason = (
                    f"a concern was raised about {aliases[concern.option_id]}, which you back — "
                    "respond to it directly: defend it with a grounded reason or concede the point honestly"
                )
            return MoveIntent(
                speaker_id=speaker.id,
                act=ActType.BUILD,
                reason=reason,
                option_focus=[concern.option_id],
                respond_to_turn=concern.turn_id,
                addressee_id=concern.raised_by if random.random() < 0.5 else None,
            )

        # 3. Follow-up: an answer was just given; someone reacts to it instead
        #    of the topic silently jumping.
        if last.intent and last.intent.act == ActType.ANSWER and random.random() < 0.6:
            speaker = self._choose_speaker(state)
            if speaker.id != last.speaker_id:
                act = weighted_choice(
                    [ActType.AGREE, ActType.CHALLENGE, ActType.ASK],
                    [0.4 + (1.0 - speaker.sim_params.stubbornness) * 0.3,
                     0.3 + speaker.sim_params.stubbornness * 0.4,
                     0.25 + speaker.sim_params.initiative * 0.2],
                )
                focus = last.act.option_refs[:2]
                return MoveIntent(
                    speaker_id=speaker.id,
                    act=act,
                    reason="react to the answer just given: acknowledge what it settles, push back on what it doesn't, or ask the natural follow-up",
                    option_focus=focus,
                    respond_to_turn=last.index,
                )

        # 4. Blocker probe: the leading option has an unresolved visible blocker;
        #    a supporter asks once what would make it workable.
        leading = self._visible_candidate(state) or self._latent_leading_option(state)
        if leading and leading not in state.blocker_probes:
            blockers = [p for p in state.personas if leading in state.runtimes[p.id].hard_rejections]
            askers = [
                p for p in state.personas
                if p.id != last.speaker_id and p not in blockers
            ]
            if blockers and askers:
                state.blocker_probes.add(leading)
                blocker = blockers[0]
                speaker = max(askers, key=lambda p: p.sim_params.responsiveness)
                return MoveIntent(
                    speaker_id=speaker.id,
                    act=ActType.ASK,
                    reason=(
                        f"{blocker.name} clearly can't accept {aliases[leading]}; ask them one genuine "
                        "question about what would make it workable or what they'd need instead — no pressure"
                    ),
                    addressee_id=blocker.id,
                    option_focus=[leading],
                )

        # 5. Visible split: two options both have visible backing and nobody has
        #    compared them head-to-head recently.
        supported = [oid for oid in state.scenario.option_ids if self._visible_support_count(state, oid) >= 1]
        if len(supported) >= 2 and random.random() < 0.5:
            recent = participant_turns[-4:]
            recently_compared = any(
                t.intent and t.intent.act in {ActType.COMPARE, ActType.PROPOSE_COMPROMISE} for t in recent
            )
            if not recently_compared:
                speaker = self._choose_speaker(state)
                pair = sorted(supported, key=lambda oid: -self._visible_support_count(state, oid))[:2]
                act = (
                    ActType.PROPOSE_COMPROMISE
                    if speaker.sim_params.compromise_threshold <= 0.4 and random.random() < 0.5
                    else ActType.COMPARE
                )
                return MoveIntent(
                    speaker_id=speaker.id,
                    act=act,
                    reason=(
                        f"the group is visibly split between {aliases[pair[0]]} and {aliases[pair[1]]}; "
                        + ("test whether one of them could be common ground without claiming it's decided"
                           if act == ActType.PROPOSE_COMPROMISE
                           else "put them side by side on the trade-off that actually divides the group")
                    ),
                    option_focus=pair,
                )

        # 6. Stagnation rescue (once per run): several turns in a row re-assert
        #    positions with no question, acceptance, proposal, or compromise —
        #    the thread is circling (worst at n=2, issue I20). Force one
        #    criteria-level move that engages the other side's criterion
        #    instead of another restatement.
        window = participant_turns[-4:]
        if (
            not state.stagnation_break_done
            and len(window) == 4
            and len(participant_turns) >= len(state.personas) + 4
        ):
            moved = any(
                t.act.accepts or t.act.proposes_option or t.act.offers_compromise
                or t.act.resolves_blocker or "?" in t.text
                for t in window
            )
            camps = {rt.current_preference for rt in state.runtimes.values() if rt.current_preference}
            if not moved and len(camps) >= 2:
                state.stagnation_break_done = True
                speaker = self._choose_speaker(state)
                own = state.runtimes[speaker.id].current_preference or speaker.preferred_option
                other = next((c for c in sorted(camps) if c != own), None)
                pair = [o for o in (own, other) if o in state.scenario.option_ids]
                act = (
                    ActType.PROPOSE_COMPROMISE
                    if speaker.sim_params.compromise_threshold <= 0.5
                    else ActType.ASK
                )
                return MoveIntent(
                    speaker_id=speaker.id,
                    act=act,
                    reason=(
                        "the discussion is circling with both sides restating their pick; "
                        + ("name what you would give up and propose concretely which option could work for everyone"
                           if act == ActType.PROPOSE_COMPROMISE
                           else "ask the other side directly what would make your pick workable for them, or what single thing their pick does better")
                    ),
                    option_focus=pair,
                )
        return None

    @staticmethod
    def _silence_streak(state: DialogueState, persona_id: str) -> int:
        """Participant turns since this sim last spoke (all of them if never)."""
        streak = 0
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                continue
            if turn.speaker_id == persona_id:
                break
            streak += 1
        return streak

    def _maybe_continuation_intent(self, state: DialogueState) -> MoveIntent | None:
        """Rare same-speaker follow-up (issue 6): an afterthought, clarification,
        or small self-correction right after the sim's own turn. Only when it is
        a real continuation-type move — never a repeat of the same question or
        point (validated as CONTINUATION_REPEATS) — short, and chain-capped:
        no same-speaker run longer than 3 turns."""
        participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
        if len(participant_turns) < 2:
            return None
        last = participant_turns[-1]
        # Chain length of the last speaker's current run of consecutive turns.
        chain = 0
        for turn in reversed(participant_turns):
            if turn.speaker_id != last.speaker_id:
                break
            chain += 1
        if chain >= 3 or len(state.personas) < 2:
            return None
        persona = state.persona_by_id(last.speaker_id)
        probability = 0.03 + 0.07 * persona.sim_params.initiative
        if chain == 2:
            probability *= 0.5
        if random.random() >= probability:
            return None
        asked_question = "?" in last.text
        if asked_question:
            purpose = (
                "add a quick separate afterthought on a DIFFERENT aspect than your question — a practical "
                "point or a small aside; do not repeat or rephrase the question you just asked"
            )
        else:
            purpose = random.choice([
                "add one quick afterthought that occurred to you — a small practical point you forgot ('Oh, and…')",
                "clarify one thing from your last message in different words so it can't be misread ('Just to be clear…')",
                "soften or correct one small thing you just said ('Actually, …') without changing your overall point",
            ])
        return MoveIntent(
            speaker_id=persona.id,
            act=ActType.BUILD,
            reason=purpose,
            length_hint="short",
            continuation=True,
        )

    def _softening_attractor(self, state: DialogueState, persona: Persona) -> str | None:
        """The option a shaken sim would plausibly warm to (issue 3): visibly
        backed or argued-for by others, shiftable, and not the sim's current pick."""
        rt = state.runtimes[persona.id]
        current = rt.current_preference or persona.preferred_option
        best: tuple[float, str] | None = None
        latent_counts = Counter(
            rt.current_preference for pid, rt in state.runtimes.items()
            if pid != persona.id and rt.current_preference in state.scenario.option_ids
        )
        for option_id in state.scenario.option_ids:
            if option_id == current or not self._can_shift_to(state, persona, option_id):
                continue
            score = 2.0 * self._visible_support_count(state, option_id, exclude=persona.id)
            score += 0.75 * state.coverage[option_id].reasons
            score += 0.35 * state.coverage[option_id].mentions
            score += 0.80 * latent_counts[option_id]
            if option_id in persona.preferred_options:
                score += 1.0
            if self._visibly_proposed(state, option_id):
                score += 1.0
            if score > 0 and (best is None or score > best[0]):
                best = (score, option_id)
        # Require some momentum, not a single stray mention.
        return best[1] if best and best[0] >= 1.5 else None

    def _choose_speaker(self, state: DialogueState) -> Persona:
        recent_speakers = self._recent_participant_ids(state, 2)
        last_speaker = recent_speakers[0] if recent_speakers else None
        prior_speaker = recent_speakers[1] if len(recent_speakers) > 1 else None
        preset = getattr(cfg, "corpus_active", None)
        total_turns = sum(rt.turn_count for rt in state.runtimes.values())
        expected = expected_turn_share(state.personas)
        adaptation = float(cfg.routing.get("trait_share_adaptation", 3.5))
        overshoot = float(cfg.routing.get("max_share_overshoot", 0.12))
        silence_cap = int(cfg.routing.get("max_silence_rounds", 2)) * len(state.personas)
        dominant_id = None
        if preset:
            dominant_id = max(
                state.personas,
                key=lambda p: (p.sim_params.engagement + 0.5 * p.sim_params.initiative, p.id),
            ).id
        candidates: list[Persona] = []
        weights: list[float] = []
        for persona in state.personas:
            if len(state.personas) > 1 and persona.id == last_speaker:
                continue
            rt = state.runtimes[persona.id]
            p = persona.sim_params
            base = 0.35 + p.engagement + 0.35 * p.initiative
            if preset:
                base = preset_dominance_weight(
                    base, persona.id == dominant_id, rt.turn_count,
                    total_turns, len(state.personas), preset,
                    float(cfg.routing.quiet_speaker_boost),
                )
            else:
                # Trait-weighted participation (issue 1): pull each sim's actual
                # turn share toward its trait-derived target instead of strict
                # turn-count equalization. Behind target -> boosted, ahead ->
                # damped, so engagement visibly shapes who talks how much.
                share = rt.turn_count / total_turns if total_turns > 0 else expected[persona.id]
                base *= math.exp(adaptation * (expected[persona.id] - share))
                # Anti-monopoly: clearly past the target share means a hard damp,
                # not a soft one — high engagement may lead, never monologue.
                if total_turns >= len(state.personas) and share - expected[persona.id] > overshoot:
                    base *= 0.20
                # Minimum visibility: nobody disappears. A sim silent for two
                # full rounds gets pushed back in regardless of traits.
                if self._silence_streak(state, persona.id) >= silence_cap:
                    base += 2.0 * float(cfg.routing.quiet_speaker_boost)
            # Discourage two speakers from ping-ponging when others are available.
            if persona.id == prior_speaker and len(state.personas) > 2:
                base *= 0.5
            candidates.append(persona)
            weights.append(base)
        if not candidates:
            candidates = state.personas[:]
            weights = [1.0 for _ in candidates]
        return weighted_choice(candidates, weights)

    def _choose_discussion_act(self, state: DialogueState, speaker: Persona) -> ActType:
        raw = dict(cfg.routing.move_weights.items())
        p = speaker.sim_params
        raw["ask"] = raw.get("ask", 0.0) * (0.75 + p.initiative)
        raw["challenge"] = raw.get("challenge", 0.0) * (0.70 + p.stubbornness)
        raw["agree"] = raw.get("agree", 0.0) * (0.60 + (1.0 - p.stubbornness))
        raw["invite"] = raw.get("invite", 0.0) * (0.50 + p.responsiveness)
        if self._recent_question_count(state) >= 1:
            raw["ask"] *= 0.25
        if participant_turn_count(state) >= max(state.min_discussion_turns, state.force_narrow_turns - 2):
            raw["ask"] *= 0.25
            raw["invite"] *= 0.5
        if self._latent_leading_count(state) >= max(2, math.ceil(0.5 * len(state.personas))):
            raw["propose_compromise"] = raw.get("propose_compromise", 0.0) + 0.10
            raw["soften"] = raw.get("soften", 0.0) + 0.08
        # When recent turns are stuck in a concession/worry/trade-off-plus-objection
        # shape, bias act selection toward moves that break the template.
        recent_texts = self._recent_participant_texts(state, int(cfg.style.repeated_pattern_window))
        recent_patterns = [surface_pattern(t) for t in recent_texts]
        templated = sum(recent_patterns.count(p) for p in ("concede_but", "worry_but", "tradeoff_but"))
        if templated >= 2:
            raw["agree"] *= 0.4
            raw["build"] = raw.get("build", 0.0) * 0.6
            raw["challenge"] = raw.get("challenge", 0.0) * 0.7  # challenge tends to reuse the "but" shape too
            raw["compare"] = raw.get("compare", 0.0) * 1.3
            raw["ask"] = raw.get("ask", 0.0) * 1.4
            raw["propose_compromise"] = raw.get("propose_compromise", 0.0) * 1.5
        mapping = {
            "build": ActType.BUILD,
            "agree": ActType.AGREE,
            "challenge": ActType.CHALLENGE,
            "ask": ActType.ASK,
            "answer": ActType.ANSWER,
            "compare": ActType.COMPARE,
            "invite": ActType.INVITE,
            "propose_compromise": ActType.PROPOSE_COMPROMISE,
            "soften": ActType.SOFTEN,
        }
        names = [k for k in raw if k in mapping]
        act = mapping[weighted_choice(names, [float(raw[k]) for k in names])]
        if act == ActType.ANSWER and not self._next_answerable_question(state):
            return ActType.BUILD
        return act

    _last_target_speaker: str | None = None
    def _choose_target_turn(self, state: DialogueState, speaker: Persona, act: ActType) -> TurnRecord | None:
        """Score a pool of recent threads instead of always taking the last line.

        Open questions, objections/blockers, minority voices, and turns about the
        leading or under-discussed options outrank plain recency, so earlier
        unresolved points get revisited instead of dying after one reply (I8).
        """
        pool = [t for t in state.turns if t.speaker_id not in {"moderator", speaker.id}]
        if not pool:
            return None
        # An answer turn must target the pending question when one exists.
        if act == ActType.ANSWER:
            question = self._next_answerable_question(state)
            if question:
                for turn in pool:
                    if turn.index == question.turn_id:
                        return turn
            return pool[-1]
        window = pool[-int(cfg.routing.get("target_window", 6)):]
        leading = self._visible_candidate(state) or self._latent_leading_option(state)
        under = self._least_mentioned_option(state)
        open_turn_ids = {q.turn_id for q in state.open_questions}
        latest_index = window[-1].index
        weights: list[float] = []
        for turn in window:
            score = 1.0 / (1.0 + 0.6 * (latest_index - turn.index))
            if turn.index in open_turn_ids:
                score += 2.0
            elif "?" in turn.text:
                score += 0.5
            if turn.act.soft_rejects or turn.act.hard_rejects:
                score += 1.0
            if leading and leading in turn.act.option_refs:
                score += 0.6
            if under and under in turn.act.option_refs:
                score += 0.4
            rt = state.runtimes.get(turn.speaker_id)
            if leading and rt and rt.current_preference and rt.current_preference != leading:
                score += 0.6  # minority/holdout voices stay in play
            if turn.speaker_id == self._last_target_speaker:
                score *= 0.6  # don't keep targeting the same person
            weights.append(score)
        chosen = weighted_choice(window, weights)
        self._last_target_speaker = chosen.speaker_id
        return chosen

    @staticmethod
    def _least_mentioned_option(state: DialogueState) -> str | None:
        pairs = sorted(state.coverage.items(), key=lambda kv: (kv[1].mentions, kv[0]))
        return pairs[0][0] if pairs else None

    def _focus_options(self, state: DialogueState, speaker: Persona, act: ActType, target_turn: TurnRecord | None) -> list[str]:
        ids: list[str] = []
        if target_turn:
            ids.extend(target_turn.act.option_refs)
        current = state.runtimes[speaker.id].current_preference or speaker.preferred_option
        if current and current not in ids:
            ids.append(current)
        if act in {ActType.COMPARE, ActType.CHALLENGE, ActType.ASK, ActType.SOFTEN}:
            rival = self._rival_option(state, speaker, exclude=set(ids))
            if rival:
                ids.append(rival)
        leading = self._latent_leading_option(state)
        if act == ActType.PROPOSE_COMPROMISE and leading and leading not in ids:
            ids.insert(0, leading)
        return [x for x in ids if x in state.scenario.option_ids][:3]

    def _reason_for_act(self, state: DialogueState, speaker: Persona, act: ActType, focus: list[str], target_turn: TurnRecord | None) -> str:
        names = short_alias_map(state.scenario.options)
        focus_names = ", ".join(names[o] for o in focus) if focus else "the options"
        if act == ActType.BUILD:
            # Rotate the move purpose so back-to-back build turns don't all
            # become the same trade-off-statement shape (issue I12).
            return random.choice([
                f"add a new grounded reason about {focus_names}, connected to the current discussion",
                f"bring up a practical, everyday consideration about {focus_names} that hasn't come up yet",
                f"say plainly what matters most to you personally in this choice and how {focus_names} fits that",
            ])
        current = state.runtimes[speaker.id].current_preference or speaker.preferred_option
        current_name = names.get(current, current)
        if act == ActType.AGREE:
            return f"agree with the recent point where it genuinely fits your view, and add a small new angle about {focus_names}"
        if act == ActType.CHALLENGE:
            # Never route a sim into arguing against their own pick right before
            # voting for it (issue I9, restaurant run).
            rivals = [o for o in focus if o != current]
            if rivals:
                return (
                    f"push back on {names[rivals[0]]} with one concrete concern — "
                    f"you currently favor {current_name}, so aim at the rival, not your own pick"
                )
            return f"name a concern others might raise about {current_name}, and say why it still holds up for you"
        if act == ActType.ASK:
            return f"ask one concrete question that helps compare {focus_names}"
        if act == ActType.ANSWER:
            return f"answer the recent question using only the option facts, then move the decision forward"
        if act == ActType.COMPARE:
            return f"compare {focus_names} with one clear trade-off"
        if act == ActType.INVITE:
            quiet = self._quietest_other(state, speaker.id)
            return f"bring {quiet.name if quiet else 'someone quieter'} into the discussion with a useful prompt"
        if act == ActType.PROPOSE_COMPROMISE:
            return f"test whether {focus_names} could be common ground without claiming it is already decided"
        if act == ActType.SOFTEN:
            return f"non-finally acknowledge that {focus_names} is becoming more convincing, while still naming what you give up"
        if act == ActType.CALL_VOTE:
            return "casually ask everyone for a definite final pick now; do not suggest any option yourself"
        if act == ActType.SUMMARIZE_SPLIT:
            return f"summarize the visible split around {focus_names} and suggest a concrete narrowing step"
        if act == ActType.PROBE_HOLDOUT:
            return f"ask the holdout what still blocks {focus_names}, without pressuring them"
        if act == ActType.SUGGEST_NARROWING:
            return f"suggest narrowing the discussion to {focus_names} or the main trade-off"
        return "respond naturally and move the decision forward"

    def _vote_intent(self, state: DialogueState, persona: Persona, candidate: str) -> MoveIntent:
        rt = state.runtimes[persona.id]
        blocked = persona.rejection == candidate or candidate in rt.hard_rejections
        current = rt.current_preference or persona.preferred_option
        if blocked:
            # The alternative must actually be acceptable: never the blocked
            # candidate itself (even when the sim's lean is stuck on it) and
            # never another option this sim has vetoed.
            alternative = next(
                (
                    o for o in [current, *persona.preferred_options, *state.scenario.option_ids]
                    if o in state.scenario.option_ids
                    and o != candidate
                    and o != persona.rejection
                    and o not in rt.hard_rejections
                ),
                candidate,
            )
            return MoveIntent(
                speaker_id=persona.id,
                act=ActType.VOTE,
                reason="cast a clear visible vote for the best acceptable alternative and briefly mention why the candidate is blocked",
                option_focus=[alternative, candidate] if alternative != candidate else [alternative],
                length_hint="short",
            )
        if self._should_compromise_to_candidate(state, persona, candidate):
            switching = current != candidate
            return MoveIntent(
                speaker_id=persona.id,
                act=ActType.ACCEPT if switching else ActType.VOTE,
                reason=(
                    "others have visibly backed this option; commit to it clearly and add one short "
                    "clause on why you can accept it"
                    if switching
                    else "make a clear visible commitment to the option you have been backing"
                ),
                option_focus=[candidate],
                length_hint="short",
                allow_vote_change=switching,
            )
        return MoveIntent(
            speaker_id=persona.id,
            act=ActType.VOTE,
            reason="cast a clear visible vote for the option they actually choose now",
            option_focus=[current if current in state.scenario.option_ids else candidate],
            length_hint="short",
        )

    # ------------------------------------------------------------------
    # Generation, observation, and state mutation
    # ------------------------------------------------------------------

    def _ready_for_vote(self, state: DialogueState) -> bool:
        participant_turns = participant_turn_count(state)
        if participant_turns < state.min_discussion_turns:
            return False
        if participant_turns >= state.hard_max_turns:
            return True
        # Do not move to voting while a directly-addressed question is still owed.
        if self._active_obligation(state) is not None:
            return False
        if self._coverage_gap_option(state) is not None and participant_turns < state.hard_max_turns:
            return False
        if participant_turns >= state.force_narrow_turns:
            return True
        # Early narrowing needs visible transcript evidence, never latent
        # concentration (issue I5): a support cluster or a visibly proposed
        # compromise, with no open question or active blocker on the candidate.
        early_gate = state.min_discussion_turns + int(cfg.conversation.early_vote_extra_turns)
        if participant_turns < early_gate:
            return False
        candidate = self._visible_candidate(state)
        if candidate is None:
            return False
        if self._candidate_blocked(state, candidate) or self._open_question_about(state, candidate):
            return False
        support = self._visible_support_count(state, candidate)
        cluster = 2 if len(state.personas) >= 3 else 1
        if support >= cluster:
            return True
        return support >= 1 and self._visibly_proposed(state, candidate)

    def _visible_candidate(self, state: DialogueState) -> str | None:
        """Option with the most visible backing (votes + acceptances), if any."""
        counts = {oid: self._visible_support_count(state, oid) for oid in state.scenario.option_ids}
        best = max(counts.values())
        if best == 0:
            return None
        leaders = sorted(oid for oid, c in counts.items() if c == best)
        if len(leaders) > 1:
            latent = self._latent_leading_option(state)
            return latent if latent in leaders else leaders[0]
        return leaders[0]

    @staticmethod
    def _candidate_blocked(state: DialogueState, candidate: str) -> bool:
        return any(candidate in rt.hard_rejections for rt in state.runtimes.values())

    @staticmethod
    def _open_question_about(state: DialogueState, candidate: str) -> bool:
        return any(candidate in q.option_focus for q in state.open_questions)

    def _vote_reason(self, state: DialogueState) -> str:
        participant_turns = participant_turn_count(state)
        if participant_turns >= state.hard_max_turns:
            return "hard cap reached; forcing a visible vote instead of closing early"
        if participant_turns >= state.force_narrow_turns:
            return "target discussion length reached"
        return "visible support for one option held after enough back-and-forth"

    def _candidate_for_vote(self, state: DialogueState) -> str:
        """Vote candidate from visible evidence; latent lean only breaks ties.

        Visible votes and acceptances weigh double, visible compromise offers
        and proposals count once. With no visible evidence at all (a first vote
        round after low-commitment discussion), fall back to the latent leader —
        it only shapes whom the moderator asks about, never the outcome.
        """
        scores: Counter[str] = Counter()
        option_ids = set(state.scenario.option_ids)
        for rt in state.runtimes.values():
            if rt.explicit_vote in option_ids:
                scores[rt.explicit_vote] += 2
            for oid in rt.accepted_options:
                if oid in option_ids and oid != rt.explicit_vote:
                    scores[oid] += 1
        for turn in state.turns:
            if turn.speaker_id == "moderator":
                continue
            for oid in {turn.act.offers_compromise, turn.act.proposes_option}:
                if oid in option_ids:
                    scores[oid] += 1
        if scores:
            best = max(scores.values())
            leaders = sorted(oid for oid, s in scores.items() if s == best)
            latent = self._latent_leading_option(state)
            if latent in leaders:
                return latent
            return leaders[0]
        return self._latent_leading_option(state) or state.personas[0].preferred_option

    def _should_compromise_to_candidate(self, state: DialogueState, persona: Persona, candidate: str) -> bool:
        """Whether this sim's vote turn asks them to commit to the candidate.

        Requires *visible* pressure: at least one other participant has visibly
        voted for / accepted the candidate, or someone visibly proposed it as
        common ground. Latent lean concentration is not evidence (issue I4).
        """
        if not self._can_shift_to(state, persona, candidate):
            return False
        rt = state.runtimes[persona.id]
        if rt.current_preference == candidate:
            return True
        support = self._visible_support_count(state, candidate, exclude=persona.id)
        if support == 0 and not self._visibly_proposed(state, candidate):
            return False
        pressure = support / max(1, len(state.personas) - 1)
        probability = 0.10 + 0.65 * (1.0 - persona.sim_params.compromise_threshold) + 0.25 * pressure
        # Tracked stance state (issue 2): a sim whose hold on its favorite was
        # eroded by challenges/support pressure accepts the candidate more
        # readily; one that spent the discussion defending it resists longer.
        probability += 0.25 * (0.60 - rt.commitment_strength)
        if candidate in persona.preferred_options:
            probability += 0.15
        return random.random() < min(0.95, probability)

    @staticmethod
    def _visible_support_count(state: DialogueState, option_id: str, exclude: str | None = None) -> int:
        return sum(
            1
            for pid, rt in state.runtimes.items()
            if pid != exclude and (rt.explicit_vote == option_id or option_id in rt.accepted_options)
        )

    @staticmethod
    def _visibly_proposed(state: DialogueState, option_id: str) -> bool:
        return any(
            t.act.offers_compromise == option_id or t.act.proposes_option == option_id
            for t in state.turns
            if t.speaker_id != "moderator"
        )

    def _can_shift_to(self, state: DialogueState, persona: Persona, option_id: str) -> bool:
        if persona.rejection == option_id:
            return False
        # A visible, unresolved blocker binds exactly like a setup rejection.
        if option_id in state.runtimes[persona.id].hard_rejections:
            return False
        if persona.sim_params.stubbornness >= 0.85 and option_id != persona.preferred_option:
            return random.random() < 0.04
        return True

    # ------------------------------------------------------------------
    # Moderator helpers
    # ------------------------------------------------------------------

    def _coverage_gap_option(self, state: DialogueState) -> str | None:
        if not bool(cfg.conversation.get("require_option_coverage_before_vote", True)):
            return None
        # Opening turns may naturally leave some non-preferred options untouched.
        # After everyone has spoken once, the router gives each untouched option a
        # light social check before final voting. This is coverage, not forced support.
        if participant_turn_count(state) <= len(state.personas):
            return None
        gaps = [
            (option_id, coverage.mentions, coverage.reasons + coverage.objections)
            for option_id, coverage in state.coverage.items()
            if coverage.mentions == 0 and coverage.coverage_attempts == 0
        ]
        if not gaps:
            return None
        gaps.sort(key=lambda item: (item[1], item[2], item[0]))
        return gaps[0][0]

    def _speaker_for_option_coverage(self, state: DialogueState, option_id: str) -> Persona:
        last = self._last_participant_id(state)
        # Prefer a participant who can plausibly discuss the option: secondary
        # preference first, then high openness/initiative, while avoiding the
        # last speaker in ordinary discussion.
        candidates = [p for p in state.personas if p.id != last] or state.personas[:]
        def score(persona: Persona) -> tuple[float, float]:
            p = persona.sim_params
            preference_bonus = 1.0 if option_id in persona.preferred_options else 0.0
            return (preference_bonus + 0.45 * p.initiative + 0.35 * p.engagement + 0.20 * (1.0 - p.stubbornness), -state.runtimes[persona.id].turn_count)
        return max(candidates, key=score)

    def _vote_order(self, state: DialogueState, candidate: str) -> list[Persona]:
        return sorted(state.personas, key=lambda p: (state.runtimes[p.id].current_preference != candidate, state.runtimes[p.id].turn_count))

    @staticmethod
    def _recent_participant_texts(state: DialogueState, limit: int) -> list[str]:
        texts = [t.text for t in state.turns if t.speaker_id != "moderator"]
        return texts[-limit:]

    @staticmethod
    def _current_round_texts(state: DialogueState) -> list[str]:
        """Participant turns since the last moderator line — i.e. this vote/beat round."""
        texts: list[str] = []
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                break
            texts.append(turn.text)
        return texts

    def _apply_style_flags(self, state: DialogueState, intent: MoveIntent) -> None:
        """Set compact surface-style flags (no LLM call) to keep dialogue varied.

        Names are only suppressed for ordinary continuation turns: answering a
        direct question, inviting a quiet participant, or a deliberate addressee
        keep their functional name use.
        """
        names = [p.name for p in state.personas]
        # Only inviting a quiet participant truly needs a leading name; answers and
        # addressed turns can respond without opening on the name, so they are still
        # subject to suppression when name-prefix density is high.
        functional_naming = intent.act == ActType.INVITE
        window = int(cfg.style.name_prefix_window)
        recent = self._recent_participant_texts(state, window)
        if not functional_naming and recent:
            if name_prefix_fraction(recent, names) >= float(cfg.style.name_prefix_max_fraction):
                intent.suppress_name_prefix = True
        alias_values = list(short_alias_map(state.scenario.options).values())
        if recent and option_opening_fraction(recent, alias_values) >= float(cfg.style.option_opening_max_fraction):
            intent.suppress_option_opening = True
        # Decision turns are exempt: "I vote/I'd go with" is natural and parser-relevant there.
        if recent and intent.act not in {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}:
            if first_person_opening_fraction(recent) >= float(cfg.style.i_opening_max_fraction):
                intent.suppress_i_opening = True
            if we_opening_fraction(recent) >= float(cfg.style.we_opening_max_fraction):
                intent.suppress_we_opening = True
        opening_window = int(cfg.style.repeated_opening_window)
        if repeated_opening_token(self._recent_participant_texts(state, opening_window), opening_window):
            intent.vary_opening = True
        if intent.act in _DISCUSSION_ACTS:
            pattern_window = int(cfg.style.repeated_pattern_window)
            intent.avoid_pattern = repeated_pattern(
                self._recent_participant_texts(state, pattern_window), pattern_window
            )
        # Deterministic anti-chorus for decision beats: phrase families already
        # used in this round OR by this speaker in any earlier turn are
        # off-limits, so a re-asked voter never repeats their own line verbatim
        # across vote rounds (issues #25, I12, I19). Reason snippets stay
        # round-scoped: a persona restating their own justification is coherent.
        if intent.act in _DECISION_ACTS:
            round_texts = self._current_round_texts(state)
            own_texts = [t.text for t in state.turns if t.speaker_id == intent.speaker_id]
            if not intent.avoid_phrases:
                intent.avoid_phrases = used_commitment_phrases(round_texts + own_texts)
            if not intent.avoid_reasons:
                intent.avoid_reasons = round_reason_snippets(round_texts)

    def _recent_question_count(self, state: DialogueState) -> int:
        recent = [t for t in state.turns[-3:] if t.speaker_id != "moderator"]
        return sum(1 for t in recent if "?" in t.text)

    def _rival_option(self, state: DialogueState, speaker: Persona, exclude: set[str]) -> str | None:
        candidates = [o for o in state.scenario.option_ids if o not in exclude and self._can_shift_to(state, speaker, o)]
        if not candidates:
            return None
        leading = self._latent_leading_option(state)
        if leading in candidates:
            return leading
        return random.choice(candidates)

    @staticmethod
    def _quietest_other(state: DialogueState, speaker_id: str) -> Persona | None:
        others = [p for p in state.personas if p.id != speaker_id]
        return min(others, key=lambda p: state.runtimes[p.id].turn_count) if others else None

    @staticmethod
    def _last_participant_id(state: DialogueState) -> str | None:
        for turn in reversed(state.turns):
            if turn.speaker_id != "moderator":
                return turn.speaker_id
        return None

    @staticmethod
    def _recent_participant_ids(state: DialogueState, limit: int) -> list[str]:
        """Most recent distinct participant speaker ids, newest first."""
        out: list[str] = []
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator" or turn.speaker_id in out:
                continue
            out.append(turn.speaker_id)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _latent_leading_option(state: DialogueState) -> str | None:
        counts = Counter(rt.current_preference for rt in state.runtimes.values() if rt.current_preference)
        return counts.most_common(1)[0][0] if counts else None

    @staticmethod
    def _latent_leading_count(state: DialogueState) -> int:
        counts = Counter(rt.current_preference for rt in state.runtimes.values() if rt.current_preference)
        return counts.most_common(1)[0][1] if counts else 0

    @staticmethod
    def _word_bounds(intent: MoveIntent, persona: Persona) -> tuple[int, int]:
        """Trait-driven (min, max) word budget so verbosity/engagement are visible."""
        budgets = cfg.utterances.word_budgets
        if intent.act == ActType.OPENING:
            base = int(budgets.opening)
        elif intent.act == ActType.ASK:
            base = int(budgets.ask)
        elif intent.act == ActType.ANSWER:
            base = int(budgets.answer)
        elif intent.act in _DECISION_ACTS:
            base = int(budgets.vote)
            # A compromise switch needs room for the bridge clause ("water
            # matters to me, but ...") on top of the direct commitment.
            if intent.allow_vote_change:
                base += 8
        else:
            base = int(budgets.discussion)
        # A continuation is a quick add-on, not a second full turn (issue 6).
        # Keep enough headroom that a one-clause afterthought is not chopped.
        if intent.continuation:
            base = max(12, round(base * 0.6))
        p = persona.sim_params
        # A real spread, not a +/-4 nudge: terse sims stay short, chatty ones longer.
        factor = 0.45 + 0.70 * p.verbosity + 0.15 * p.engagement   # ~0.45..1.30
        # Per-turn jitter around the persona's average so consecutive turns by
        # the same sim vary naturally instead of hitting one fixed length (I12).
        factor *= random.uniform(0.90, 1.10)
        max_words = max(6, round(base * factor))
        min_words = max(3, round(max_words * (0.30 + 0.25 * p.verbosity)))
        return min_words, max_words
