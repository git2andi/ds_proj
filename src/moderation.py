"""
moderation.py
-------------
ModerationEngine -- intervention timing + LLM-generated moderator lines.

The orchestrator owns phase transitions, votes, consensus, closure.
This module owns when/how to intervene and which prompt to use.

Stage 3 facilitator refactor:
  - The blunt info-gap shutdown is replaced with `moderator_reframe_missing_detail`,
    which never ends the line of inquiry -- it reframes a missing option attribute
    into a judgement call grounded in what IS listed.
  - A new `facilitate_disagreement` move surfaces live disagreement by name and
    asks the involved sims to engage each other (Du 2023; Liang 2023).
  - A new `what_would_change_mind` move pulls directly on AgentBeliefs.would_reconsider_if.
  - During real exchange (challenges firing + responses arriving) the moderator
    stays quiet -- intervention only kicks in at stall / outlier / explicit
    info-chase signals.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from utils import OptionResolver, current_votes, last_n_turns_for
from verifier import verify_moderator_turn

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from simulator import Simulator
    from state import StructuredState


# (text, tokens_in, tokens_out)
StoreFn = Callable[[str, int, int], None]


def _strip_wrapping_quotes(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] in {'"', "'", "“", "‘"} and t[-1] in {'"', "'", "”", "’"}:
        return t[1:-1].strip()
    return t


# Sims chasing an option attribute the options DON'T contain (cost, time, price).
# This signal is NARROWER than the old detector -- "wish we had numbers" is the
# kind of substantive info-chase the facilitator should reframe. Mere "?" piles
# are now deliberation and left alone.
_INFO_CHASE = re.compile(
    r"(isn'?t|not)\s+specified|wish\s+we\s+(?:had|knew)|still\s+waiting|"
    r"don'?t\s+have\s+(?:the\s+)?(?:cost|price|detail|info|number|figure|exact)|"
    r"no\s+(?:cost|price|detail|info|number|figure)s?\b|"
    r"what'?s\s+the\s+(?:exact\s+)?(?:cost|price|time|fee|rate)",
    re.I,
)


def clean_moderator_line(text: str, participant_names: list[str]) -> str:
    """Drop the turn if any line starts with a participant name + colon."""
    t = _strip_wrapping_quotes(text)
    name_set = {n.lower() for n in participant_names}
    for line in t.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^([A-Za-z][a-zA-Z]*):\s+", stripped)
        if m and m.group(1).lower() in name_set:
            return ""
    return t


class ModerationEngine:

    def __init__(self, topic: str, options: list[str], sims: list["Simulator"],
                 resolver: OptionResolver) -> None:
        self.topic = topic
        self.options = list(options)
        self.sims = sims
        self._resolver = resolver
        self._llm = get_llm_client()

    # ------------------------------------------------------------------

    def escalation_level(self, state: "DialogueState") -> int:
        # Give the group n rounds of grace to collect votes before the patience
        # clock effectively starts, so larger groups aren't forced prematurely.
        r = state.post_narrowing_rounds
        grace = len(self.sims)
        if r < cfg.turns.escalation_level_1 + grace:
            return 0
        if r < cfg.turns.escalation_level_2 + grace:
            return 1
        if r < cfg.turns.escalation_level_3 + grace:
            return 2
        return 3

    # ------------------------------------------------------------------
    # should_narrow -- Stage 5: deliberation-gated. Floor/ceiling on turn count;
    # the actual trigger is the deliberation signal (positions with reasons +
    # at least one answered challenge + repetition pressure crossing exhaustion).
    # ------------------------------------------------------------------

    def should_narrow(
        self,
        state: "DialogueState",
        participant_turn_count: int,
        structured: Optional["StructuredState"] = None,
    ) -> bool:
        if state.has_asked_narrowing:
            return False
        n = len(self.sims)

        floor = n * cfg.turns.min_before_narrowing_per_participant
        if participant_turn_count < floor:
            return False

        ceiling = n * cfg.turns.narrow_after_per_participant
        if participant_turn_count >= ceiling:
            return True

        # Stall short-circuit (kept) -- a stuck loop should narrow even if
        # deliberation hasn't ticked all the boxes.
        if state.repetition_pressure >= cfg.repetition.stall_increment_threshold \
                and state.stall_rounds >= 1:
            return True

        if structured is None:
            return False

        # Deliberation gate.
        # Phase 2: challenge-response tracking is kept for evaluation/logging but
        # no longer gates narrowing. Requiring an answered challenge before sims
        # can vote caused artificial debate loops. Positions + exhaustion are enough.
        ratio = cfg.deliberation.positions_with_reason_required_ratio
        needed_positions = max(1, int(round(n * ratio)))
        positions = sum(
            1 for ps in structured.participants.values()
            if ps.position_with_reason_stated
        )

        gate_positions = positions >= needed_positions
        gate_exhausted = state.repetition_pressure >= cfg.deliberation.exhaustion_pressure_threshold

        return gate_positions and gate_exhausted

    # ------------------------------------------------------------------

    def _recent_participant_lines(self, history: list[str], n: int) -> list[str]:
        out: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in {"Moderator"}:
                continue
            out.append(msg.strip())
            if len(out) >= n:
                break
        return out

    def detect_info_chase(self, history: list[str]) -> bool:
        """True when the group is fishing for a SPECIFIC missing option attribute
        (cost, price, exact figure). Question pile-up alone is NOT a signal --
        a chat full of "?" is deliberation."""
        recent = self._recent_participant_lines(history, cfg.moderation.recent_window_clarify)
        if not recent:
            return False
        return any(_INFO_CHASE.search(line) for line in recent[:2])

    def detect_facilitatable_disagreement(
        self,
        structured: Optional["StructuredState"],
    ) -> Optional[str]:
        """Stage 3 facilitation: two sims have stated stances on the SAME option
        in opposite directions, and the more recent one wasn't followed by an
        answered challenge. Returns a one-line summary or None."""
        if structured is None:
            return None
        latest_on: dict[str, tuple[str, str]] = {}  # option -> (speaker, stance)
        contested: list[tuple[str, str, str]] = []  # (option, supporter, opposer)
        for (speaker, opt), su in structured.stance_table.current_items():
            stance = su.stance
            if stance in ("support", "conditional_support"):
                stance_kind = "support"
            elif stance in ("oppose", "blocker"):
                stance_kind = "oppose"
            else:
                continue
            if opt in latest_on:
                prior_speaker, prior_kind = latest_on[opt]
                if prior_kind != stance_kind and prior_speaker != speaker:
                    if stance_kind == "support":
                        contested.append((opt, speaker, prior_speaker))
                    else:
                        contested.append((opt, prior_speaker, speaker))
            latest_on[opt] = (speaker, stance_kind)
        if not contested:
            return None
        opt, supporter, opposer = contested[-1]
        return f"{supporter} supports Option {opt}; {opposer} resists it."

    # ------------------------------------------------------------------

    def should_intervene(
        self,
        state: "DialogueState",
        history: list[str],
        any_sim_stuck: bool,
        participant_turn_count: int,
        structured: Optional["StructuredState"] = None,
    ) -> Optional[str]:
        """Return the intervention reason, or None.

        Facilitator gate: when an unanswered challenge exists OR sims are
        actively engaging each other, stay quiet -- real exchange is happening.
        """
        if participant_turn_count < len(self.sims):
            return None

        if state.facilitate_cooldown > 0:
            # Don't fire two facilitation moves back-to-back.
            return self._stall_check(state, any_sim_stuck)

        outlier = self._detect_outlier(state, history)
        if outlier:
            return f"outlier:{outlier}"

        # Stage 3 facilitation: a contested option with no answered exchange
        # yet -> surface the disagreement.
        if structured is not None and state.phase in ("negotiation", "emergence"):
            contested = self.detect_facilitatable_disagreement(structured)
            answered = sum(
                1 for c in structured.discourse.challenges if c.answered_turn_id is not None
            )
            if contested and answered == 0:
                return f"facilitate:{contested}"

        return self._stall_check(state, any_sim_stuck)

    def _stall_check(self, state: "DialogueState", any_sim_stuck: bool) -> Optional[str]:
        if state.has_asked_narrowing and any_sim_stuck:
            return "stall"
        if (state.repetition_pressure >= cfg.repetition.stall_increment_threshold
                and state.stall_rounds >= 2):
            return "stall"
        return None

    # ------------------------------------------------------------------

    def run_intervention(
        self,
        reason: str,
        state: "DialogueState",
        history: list[str],
        store_fn: StoreFn,
        structured: Optional["StructuredState"] = None,
    ) -> None:
        names = [s.name for s in self.sims]
        recent = "\n".join(history[-cfg.moderation.recent_window_history:])
        level = self.escalation_level(state)

        if reason == "clarify":
            # Stage 3 -- the reframer, NOT the shutdown.
            prompt_text = prompts.moderator_reframe_missing_detail(
                topic=self.topic, participant_names=names,
                options=self.options, recent_dialogue=recent,
            )
            line = self._llm.generate(prompt_text).strip()

        elif reason.startswith("facilitate:"):
            contested_summary = reason.split(":", 1)[1]
            state.facilitate_cooldown = cfg.moderation.facilitate_cooldown_rounds
            line = self._llm.generate(
                prompts.moderator_facilitate_disagreement(
                    topic=self.topic, participant_names=names,
                    recent_dialogue=recent, contested_summary=contested_summary,
                )
            ).strip()

        elif reason.startswith("what_would_change:"):
            target, position_phrase = self._parse_change_arg(reason)
            line = self._llm.generate(
                prompts.moderator_what_would_change_mind(
                    topic=self.topic, participant_names=names,
                    target_name=target, target_position=position_phrase,
                    recent_dialogue=recent,
                )
            ).strip()

        elif reason.startswith("outlier:"):
            outlier_name = reason.split(":", 1)[1]
            state.nudged_participants.add(outlier_name)
            line = self._llm.generate(
                prompts.moderator_stall(
                    topic=self.topic, participant_names=names,
                    recent_dialogue=recent,
                    current_votes=current_votes(history, self._resolver),
                    escalation_level=level,
                )
            ).strip()

        else:  # stall
            candidate = getattr(state, "candidate_option", None) or state.current_leading_option
            min_level = cfg.moderation.emergence_min_level_for_stall
            if state.phase == "emergence" and level < min_level and candidate:
                line = self._llm.generate(
                    prompts.moderator_emergence(
                        topic=self.topic, participant_names=names,
                        recent_dialogue=recent, candidate_option=candidate,
                    )
                ).strip()
            else:
                line = self._llm.generate(
                    prompts.moderator_stall(
                        topic=self.topic, participant_names=names,
                        recent_dialogue=recent,
                        current_votes=current_votes(history, self._resolver),
                        escalation_level=level,
                    )
                ).strip()

        cleaned = clean_moderator_line(line, names)

        # Verify moderator line. If it introduces a new option or is empty,
        # discard it silently (the orchestrator will fall through to participant turns).
        if cleaned:
            v_result = verify_moderator_turn(
                text=cleaned, options=self.options, resolver=self._resolver,
                phase=state.phase,
            )
            if v_result.needs_repair:
                # Bad moderator line -- drop it. Deterministic templates are
                # used for phase-critical moderator turns (narrowing, force-close)
                # directly in the orchestrator; intervention lines can just be skipped.
                cleaned = ""

        if cleaned:
            store_fn(cleaned, self._llm.last_tokens_in, self._llm.last_tokens_out)
        del structured  # not needed beyond the dispatch above

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_change_arg(reason: str) -> tuple[str, str]:
        """`what_would_change:Alice|holding Option A` -> ('Alice', 'holding Option A')."""
        payload = reason.split(":", 1)[1]
        if "|" in payload:
            target, position = payload.split("|", 1)
        else:
            target, position = payload, "their current position"
        return target.strip(), position.strip()

    def _detect_outlier(self, state: "DialogueState", history: list[str]) -> Optional[str]:
        """A participant who repeats themselves verbatim after narrowing."""
        if not state.has_asked_narrowing:
            return None
        threshold = cfg.moderation.outlier_repeat_jaccard
        for sim in self.sims:
            turns = last_n_turns_for(sim.name, history, n=2)
            if len(turns) < 2:
                continue
            words0 = set(turns[0].split())
            if not words0:
                continue
            ratio = len(words0 & set(turns[1].split())) / len(words0)
            if ratio >= threshold:
                return sim.name
        return None
