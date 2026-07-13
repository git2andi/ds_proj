"""Floor manager: turn-access arbitration only.

FloorMixin does NOT decide participant behavior. It:
- imposes protocol obligations (opening, direct answers, votes) that fix only
  speaker + act while the simulator policy chooses the substance;
- collects one complete bid per eligible simulator from the simulator policy;
- validates bids structurally before any LLM call;
- adjusts floor access (recent-speaker penalty, anti-monopoly damping,
  minimum-visibility correction) — never engagement a second time;
- selects the highest-scoring valid claiming bid without rewriting it;
- exposes framework-level public-evidence readers (public candidate/top pair,
  visible support, coverage gaps) shared with flow and the simulator policy;
- keeps surface-style flags and verbosity word budgets (wording-only controls).

It is read-only over DialogueState: it selects, it never mutates dialogue
state. Phase transitions and the repair machine live in controller/flow.py; the
observer owns visible-evidence state updates.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import simulator as sim_policy
from aliases import short_alias_map
from config_loader import cfg
from consensus import discussion_positive_mentions, participant_turn_count, public_evidence
from models import (
    ActType,
    DialogueState,
    DiscussionStimulus,
    MoveIntent,
    Persona,
    Phase,
    SimulatorBid,
    STANCE_REJECTED,
    TurnObligation,
    _DECISION_ACTS,
    _DISCUSSION_ACTS,
)
from parsing import round_reason_snippets, used_commitment_phrases
from simulator import expected_turn_share
from style import (
    first_person_opening_fraction,
    name_prefix_fraction,
    option_opening_fraction,
    repeated_opening_token,
    repeated_pattern,
    we_opening_fraction,
)

# Acts a simulator may legally submit on an open floor. PROCESS is included so
# a stall-stimulus procedural suggestion is structurally admissible; the
# simulator only ever scores PROCESS under an explicit stall stimulus.
_OPEN_FLOOR_ACTS = frozenset({
    ActType.ANSWER, ActType.SUPPORT, ActType.CONCERN, ActType.ASK,
    ActType.COMPARE, ActType.COMMENT, ActType.COMPROMISE, ActType.PROCESS,
})
# Longest run of consecutive turns one participant may hold before the floor
# excludes them regardless of willingness.
_MAX_SPEAKER_CHAIN = 3


class FloorMixin:
    # ------------------------------------------------------------------
    # Open-floor bidding (todo 8): collect -> validate -> score -> select
    # ------------------------------------------------------------------

    def _collect_bids(
        self, state: DialogueState, stimulus: DiscussionStimulus
    ) -> list[SimulatorBid]:
        """Ask every eligible simulator policy for exactly one open-floor bid."""
        bids: list[SimulatorBid] = []
        eligible = self._eligible_ids(state)
        for persona in state.personas:
            bid = sim_policy.decide_simulator_bid(state, persona.id, stimulus=stimulus)
            if persona.id not in eligible and bid.wants_to_speak:
                bid.wants_to_speak = False
                bid.intent = None
                bid.rejected_reason = "ineligible: speaker-chain cap"
            bids.append(bid)
        return bids

    def _eligible_ids(self, state: DialogueState) -> set[str]:
        """Normally everyone, including the last speaker. Only a participant who
        has held the floor for a full chain of consecutive turns is excluded."""
        eligible = {p.id for p in state.personas}
        chain_id, chain = self._current_chain(state)
        if chain_id is not None and chain >= _MAX_SPEAKER_CHAIN:
            eligible.discard(chain_id)
        return eligible

    @staticmethod
    def _current_chain(state: DialogueState) -> tuple[str | None, int]:
        chain = 0
        chain_id: str | None = None
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                continue
            if chain_id is None:
                chain_id = turn.speaker_id
                chain = 1
            elif turn.speaker_id == chain_id:
                chain += 1
            else:
                break
        return chain_id, chain

    def _ranked_valid_bids(
        self, state: DialogueState, bids: list[SimulatorBid]
    ) -> list[SimulatorBid]:
        """Validate claiming bids, score floor access, return them best-first.

        The floor may reject or reorder complete bids; it never rewrites an
        act, focus, target, addressee, reason, or vote (todo 8/9).
        """
        valid: list[SimulatorBid] = []
        for bid in bids:
            if not bid.wants_to_speak or bid.intent is None:
                continue
            reason = self._validate_bid(state, bid, obligation=None)
            if reason:
                bid.rejected_reason = reason
                continue
            bid.floor_score = self._floor_score(state, bid)
            valid.append(bid)
        valid.sort(key=lambda b: (-b.floor_score, b.participant_id))
        return valid

    def _floor_score(self, state: DialogueState, bid: SimulatorBid) -> float:
        """Turn-access score: submitted willingness adjusted for floor mechanics
        only. Engagement is already inside willingness and is not re-applied."""
        score = bid.willingness
        recent = self._recent_participant_ids(state, 2)
        if recent and bid.participant_id == recent[0]:
            score *= 0.35                       # strong recent-speaker penalty
        elif len(recent) > 1 and bid.participant_id == recent[1] and len(state.personas) > 2:
            score *= 0.80                       # lighter ping-pong damp
        expected = expected_turn_share(state.personas)
        total = sum(rt.turn_count for rt in state.runtimes.values())
        rt = state.runtimes[bid.participant_id]
        if total > 0:
            share = rt.turn_count / total
            overshoot = share - expected[bid.participant_id]
            if overshoot > float(cfg.floor.get("max_share_overshoot", 0.16)):
                score *= 0.40                   # anti-monopoly damping
        silence_cap = int(cfg.floor.get("max_silence_rounds", 2)) * len(state.personas)
        if self._silence_streak(state, bid.participant_id) >= silence_cap:
            score += float(cfg.floor.get("quiet_speaker_boost", 1.25))  # min visibility
        return score

    def _validate_bid(
        self, state: DialogueState, bid: SimulatorBid, *, obligation: TurnObligation | None
    ) -> str:
        """Structural validation before any LLM call (todo 9). Returns a
        rejection reason, or "" when the bid is structurally admissible."""
        intent = bid.intent
        if intent is None:
            return "no intent"
        if intent.speaker_id != bid.participant_id:
            return "intent speaker mismatch"
        # Phase legality.
        if obligation is None:
            if intent.act not in _OPEN_FLOOR_ACTS:
                return f"act {intent.act.value} not legal on open floor"
            if state.phase not in (Phase.DISCUSSION, Phase.NARROWING, Phase.COMPROMISE_REPAIR):
                return f"open-floor bid illegal in phase {state.phase.value}"
        # Target turn must exist when referenced.
        if intent.respond_to_turn is not None and not any(
            t.index == intent.respond_to_turn for t in state.turns
        ):
            return "target turn does not exist"
        # Thread must exist when referenced.
        if intent.thread_id is not None and intent.thread_id not in state.threads:
            return "referenced thread does not exist"
        # Option focus validity.
        for oid in intent.option_focus:
            if oid not in state.scenario.option_ids:
                return f"invalid option in focus: {oid}"
        # Addressee validity.
        if intent.addressee_id is not None:
            if intent.addressee_id == intent.speaker_id:
                return "self-addressed"
            if intent.addressee_id not in state.runtimes and intent.addressee_id != "moderator":
                return "invalid addressee"
        # Comparison needs two distinct options.
        if intent.act is ActType.COMPARE and len({*intent.option_focus}) < 2:
            return "comparison with fewer than two options"
        # A hard blocker may never propose accepting/voting for a rejected option.
        rt = state.runtimes[bid.participant_id]
        if intent.act in (ActType.VOTE, ActType.COMPROMISE):
            targets = [intent.required_vote] if intent.required_vote else list(intent.option_focus)
            if any(t in rt.rejected_options() for t in targets if t):
                return "hard blocker targeting a rejected option"
        # A clear repetition with no new grounded contribution.
        if self._is_repeat_bid(state, bid, obligation=obligation):
            state.repeated_bid_rejections += 1
            return "repeats an accepted own contribution"
        return ""

    @staticmethod
    def _is_repeat_bid(
        state: DialogueState,
        bid: SimulatorBid,
        *,
        obligation: TurnObligation | None,
    ) -> bool:
        intent = bid.intent
        if intent is None or not intent.contribution_key:
            return False
        # Protocol replies are allowed to restate a position when the public
        # context itself is new. Ordinary open-floor contributions are unique
        # across the participant's complete accepted history.
        if obligation is not None or intent.act in (ActType.VOTE, ActType.ANSWER):
            return False
        for turn in state.turns:
            if (
                turn.speaker_id == bid.participant_id
                and not turn.state_mutation_blocked
                and turn.text.strip()
                and turn.intent is not None
                and turn.intent.contribution_key == intent.contribution_key
            ):
                return True
        return False

    def _select_winner(self, ranked: list[SimulatorBid]) -> SimulatorBid | None:
        return ranked[0] if ranked else None

    # ------------------------------------------------------------------
    # Framework public-evidence readers (shared with flow / simulator policy)
    # ------------------------------------------------------------------

    def _current_top_pair(self, state: DialogueState) -> list[str]:
        return list(public_evidence(state).top_pair)

    def _public_candidate(self, state: DialogueState) -> str | None:
        leaders = public_evidence(state).candidate_leaders
        return leaders[0] if leaders else None

    def _candidate_for_vote(self, state: DialogueState) -> str | None:
        """Public candidate to test, or None when the transcript has no leader.

        A vote may still proceed without a tested candidate: each simulator then
        chooses its own stance-consistent target. Hidden ranks/preferences never
        select the framework candidate.
        """
        return self._public_candidate(state)

    @staticmethod
    def _visible_support_count(state: DialogueState, option_id: str, exclude: str | None = None) -> int:
        backers = public_evidence(state).backing.get(option_id, set())
        return len(backers - {exclude} if exclude else backers)

    @staticmethod
    def _visibly_proposed(state: DialogueState, option_id: str) -> bool:
        return option_id in public_evidence(state).proposals


    @staticmethod
    def _least_mentioned_option(state: DialogueState) -> str | None:
        pairs = sorted(state.coverage.items(), key=lambda kv: (kv[1].mentions, kv[0]))
        return pairs[0][0] if pairs else None

    def _coverage_gap_option(self, state: DialogueState) -> str | None:
        """An option nobody has brought into the discussion yet (framework
        coverage detection). This is a stimulus/invitation signal only — it
        never forces a participant to speak (todo 14)."""
        if not bool(cfg.conversation.get("require_option_coverage_before_vote", True)):
            return None
        turns = participant_turn_count(state)
        if turns <= len(state.personas):
            return None
        if state.hard_max_turns and turns >= state.hard_max_turns - max(2, math.ceil(len(state.personas) / 2)):
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

    # ------------------------------------------------------------------
    # Split-vote candidate ranking (framework selects from VISIBLE votes)
    # ------------------------------------------------------------------

    def _rank_split_candidates(
        self,
        state: DialogueState,
        votes_by_id: dict[str, str],
        *,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, list[Persona], list[Persona], dict]]:
        """Rank compromise candidates from formal visible votes only.

        The framework may select which existing option to TEST (todo 18) from
        the visible vote structure and visible positive discussion mentions. It
        does not decide who moves — the simulators do that in their own re-vote.
        A strict visible plurality is tested first; formal ties break by
        positive visible mentions.
        """
        exclude = exclude or set()
        counts = Counter(v for v in votes_by_id.values() if v in state.scenario.option_ids)
        if not counts:
            return []
        positive_mentions = discussion_positive_mentions(state)
        ranked: list[tuple[tuple, str, list[Persona], list[Persona], dict]] = []
        for candidate, count in counts.items():
            if candidate in exclude:
                continue
            dissenters = [p for p in state.personas if votes_by_id.get(p.id) != candidate]
            if not dissenters:
                continue
            # Movers are the visible dissenters; whether each actually moves is
            # decided by that simulator's own re-vote policy.
            movers = list(dissenters)
            positive_count = int(positive_mentions.get(candidate, 0))
            objection_load = state.coverage[candidate].objections
            meta = {
                "votes": count,
                "positive_mentions": positive_count,
                "selected_mover_ids": [p.id for p in movers],
            }
            key = (float(count), float(positive_count), -float(objection_load), candidate)
            ranked.append((key, candidate, dissenters, movers, meta))
        ranked.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1]))
        return [(cand, diss, movers, meta) for _k, cand, diss, movers, meta in ranked]

    @staticmethod
    def _hard_blocks_candidate(state: DialogueState, persona: Persona, candidate: str) -> bool:
        return state.runtimes[persona.id].rank(candidate) <= STANCE_REJECTED

    # ------------------------------------------------------------------
    # Scheduling helpers (framework decides ORDER, not behavior)
    # ------------------------------------------------------------------

    def _vote_order(self, state: DialogueState, candidate: str | None) -> list[Persona]:
        """Schedule formal voters from public participation state only."""
        public_backers = public_evidence(state).backing.get(candidate, set()) if candidate else set()
        return sorted(
            state.personas,
            key=lambda p: (p.id not in public_backers, state.runtimes[p.id].turn_count, p.id),
        )

    @staticmethod
    def _last_participant_id(state: DialogueState) -> str | None:
        for turn in reversed(state.turns):
            if turn.speaker_id != "moderator":
                return turn.speaker_id
        return None

    @staticmethod
    def _recent_participant_ids(state: DialogueState, limit: int) -> list[str]:
        out: list[str] = []
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator" or turn.speaker_id in out:
                continue
            out.append(turn.speaker_id)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _silence_streak(state: DialogueState, persona_id: str) -> int:
        streak = 0
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                continue
            if turn.speaker_id == persona_id:
                break
            streak += 1
        return streak

    @staticmethod
    def _recent_participant_texts(state: DialogueState, limit: int) -> list[str]:
        texts = [t.text for t in state.turns if t.speaker_id != "moderator"]
        return texts[-limit:]

    @staticmethod
    def _current_round_texts(state: DialogueState) -> list[str]:
        texts: list[str] = []
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                break
            texts.append(turn.text)
        return texts

    # ------------------------------------------------------------------
    # Surface-style flags (wording-only; never affect who speaks or the act)
    # ------------------------------------------------------------------

    def _apply_style_flags(self, state: DialogueState, intent: MoveIntent) -> None:
        names = [p.name for p in state.personas]
        functional_naming = (
            intent.act == ActType.PROCESS
            or (intent.addressee_id is not None and intent.act in {ActType.ASK, ActType.CONCERN})
            or (intent.addressee_id is not None and len(state.personas) >= 4 and intent.act == ActType.ANSWER)
        )
        window = int(cfg.style.name_prefix_window)
        recent = self._recent_participant_texts(state, window)
        if (
            intent.act in {ActType.SUPPORT, ActType.ANSWER, ActType.COMPARE, ActType.COMMENT,
                           ActType.COMPROMISE, ActType.CONCERN}
            and sum(1 for t in recent if "?" in t) >= 2
        ):
            intent.suppress_tail_question = True
        alias_values = list(short_alias_map(state.scenario.options).values())
        pattern = None
        if intent.act in _DISCUSSION_ACTS:
            pattern_window = int(cfg.style.repeated_pattern_window)
            pattern = repeated_pattern(
                self._recent_participant_texts(state, pattern_window), pattern_window
            )
        opening_window = int(cfg.style.repeated_opening_window)
        exempt_from_opening_notes = intent.act in {ActType.VOTE, ActType.CONCERN}
        if pattern:
            intent.avoid_pattern = pattern
        elif repeated_opening_token(self._recent_participant_texts(state, opening_window), opening_window):
            intent.vary_opening = True
        elif recent and option_opening_fraction(recent, alias_values) >= float(cfg.style.option_opening_max_fraction):
            intent.suppress_option_opening = True
        elif (
            not functional_naming
            and recent
            and name_prefix_fraction(recent, names) >= float(cfg.style.name_prefix_max_fraction)
        ):
            intent.suppress_name_prefix = True
        elif (
            not exempt_from_opening_notes
            and recent
            and first_person_opening_fraction(recent) >= float(cfg.style.i_opening_max_fraction)
        ):
            intent.suppress_i_opening = True
        elif (
            not exempt_from_opening_notes
            and recent
            and we_opening_fraction(recent) >= float(cfg.style.we_opening_max_fraction)
        ):
            intent.suppress_we_opening = True
        if intent.act in _DECISION_ACTS:
            round_texts = self._current_round_texts(state)
            own_texts = [t.text for t in state.turns if t.speaker_id == intent.speaker_id]
            if not intent.avoid_phrases:
                intent.avoid_phrases = used_commitment_phrases(round_texts + own_texts)
            if not intent.avoid_reasons:
                intent.avoid_reasons = round_reason_snippets(round_texts)

    # ------------------------------------------------------------------
    # Verbosity word budget (verbosity only ever becomes a numeric range)
    # ------------------------------------------------------------------

    @staticmethod
    def _word_bounds(intent: MoveIntent, persona: Persona) -> tuple[int, int]:
        budgets = cfg.utterances.word_budgets
        if intent.act == ActType.OPENING:
            verbosity = persona.sim_params.verbosity
            if verbosity < 0.34:
                return 10, 18
            if verbosity < 0.67:
                return 18, 28
            return 25, 38
        elif intent.act == ActType.ASK:
            base = int(budgets.ask)
        elif intent.act == ActType.ANSWER:
            base = int(budgets.answer)
        elif intent.act in _DECISION_ACTS:
            base = int(budgets.vote)
            if intent.allow_vote_change:
                base += 6
        else:
            base = int(budgets.discussion)
        if intent.continuation:
            base = max(8, round(base * 0.55))
        elif intent.length_hint == "short":
            base = max(8, round(base * 0.75))
        p = persona.sim_params
        factor = 0.42 + 1.03 * p.verbosity
        short_beat_ok = (
            intent.act not in {ActType.OPENING, ActType.PROCESS}
            and not (intent.act in _DECISION_ACTS and intent.allow_vote_change)
            and not intent.continuation
        )
        short_probability = 0.02 + 0.36 * (1.0 - p.verbosity) ** 2
        if short_beat_ok and random.random() < short_probability:
            factor *= random.uniform(0.45, 0.60)
        else:
            factor *= random.uniform(0.97, 1.03)
        max_words = max(6, round(base * factor))
        min_words = max(3, round(max_words * (0.38 + 0.32 * p.verbosity)))
        return min_words, max_words
