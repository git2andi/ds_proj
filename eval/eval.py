"""Evaluation of simulator runs.

Beyond basic counters, this module answers the framework question "do the
simulators behave according to their configured parameters?" with per-run
numbers: participation inequality, realization errors (configured engagement/
verbosity vs realized turn share/length), obligation and question-answer
completion, lexical repetition, compromise success, and visible preference-
switch explanation rates. Everything is computed from state already collected
during the run; no LLM calls.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# This module's imports (aliases, config_loader, ...) live in src/, not on
# sys.path by default. main.py adds src/ before it ever triggers this import
# (via logger.py), so that path works from the app's entry point; but eval.py
# also needs to be importable on its own (e.g. `import eval`, or running this
# file directly), so ensure src/ is on sys.path here too.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aliases import short_alias_map
from config_loader import cfg
from consensus import visible_votes_from_transcript
from models import DialogueState, Persona, RunOutcome, ThreadStatus, ThreadType, TurnRecord, _DISCUSSION_ACTS
from simulator import expected_turn_share
from style import leading_first_person, leading_name, leading_option, leading_we, surface_pattern

_WORD = re.compile(r"[a-z0-9'-]+")
# Function words excluded from repetition comparison so it measures repeated
# content, not repeated grammar.
_COMMON = frozenset(
    "a an the and or but so if to of in on at for with by from as is are was be been "
    "i we you they it he she this that these those my our your their its not no yes "
    "do does did can could would should will just really very more most much".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _COMMON and len(w) > 2}


def _gini(values: list[int]) -> float:
    """Gini coefficient of the turn distribution (0 = perfectly equal)."""
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    cumulative = sum((index + 1) * value for index, value in enumerate(ordered))
    total = sum(ordered)
    return round((2.0 * cumulative) / (n * total) - (n + 1.0) / n, 3)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Per-run correlation between a configured parameter and realized behavior.

    None when there are fewer than 3 simulators or a side has ~zero variance —
    a correlation over two points or a constant is noise, not signal.
    """
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x < 1e-9 or var_y < 1e-9:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return round(cov / (var_x ** 0.5 * var_y ** 0.5), 3)


def _answered_by_target(state: DialogueState, question_turn: TurnRecord, within: int | None = None) -> bool:
    """Whether the addressee spoke after the question (optionally within a window).

    Turn indices are 1-based while list positions are 0-based, so eligibility
    is by ``turn.index > question_turn.index`` — slicing by index skipped the
    immediately following turn (closeout 6).
    """
    target = question_turn.question_target()
    for turn in state.turns:
        if turn.index <= question_turn.index:
            continue
        if turn.speaker_id == target:
            return within is None or (turn.index - question_turn.index) <= within
    return False


def _directed_questions(state: DialogueState) -> list[TurnRecord]:
    return [
        t for t in state.turns
        if t.question_target() and t.question_target() != t.speaker_id
        and t.question_target() in state.runtimes
    ]


def _question_threads(state: DialogueState) -> list:
    return [t for t in state.threads.values() if t.thread_type is ThreadType.QUESTION]


def _direct_response_rate(state: DialogueState) -> float | None:
    """Share of question threads that received a valid answer.

    A question thread only leaves HOT through a valid response (cooling, then
    resolved) — staleness or a still-hot thread at run end means unanswered.
    """
    questions = _question_threads(state)
    if not questions:
        return None
    answered = sum(
        1 for t in questions if t.status in (ThreadStatus.COOLING, ThreadStatus.RESOLVED)
    )
    return round(answered / len(questions), 3)


def _unanswered_question_threads(state: DialogueState) -> int:
    return sum(
        1 for t in _question_threads(state)
        if t.status in (ThreadStatus.HOT, ThreadStatus.STALE)
    )


def _concern_threads(state: DialogueState) -> list:
    return [
        t for t in state.threads.values()
        if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
    ]


def _concern_response_rate(state: DialogueState) -> float | None:
    """Share of concern/blocker threads that ever received a relevant response.

    End-of-run status alone undercounts: a concern that cooled mid-run and then
    aged to stale WAS responded. Anyone in participants_involved beyond the
    raiser only gets there through a relevant response or reinforcement.
    """
    threads = _concern_threads(state)
    if not threads:
        return None
    responded = sum(
        1 for t in threads
        if t.status in (ThreadStatus.COOLING, ThreadStatus.RESOLVED)
        or any(pid != t.started_by for pid in t.participants_involved)
    )
    return round(responded / len(threads), 3)


def _question_answer_completion(state: DialogueState) -> float | None:
    """Adjacency-pair completion: directed questions answered by the addressee
    within the analysis window (2 x question_answer_window_turns)."""
    questions = _directed_questions(state)
    if not questions:
        return None
    window = 2 * max(1, int(cfg.conversation.get("question_answer_window_turns", 2)))
    answered = sum(1 for q in questions if _answered_by_target(state, q, within=window))
    return round(answered / len(questions), 3)


def _open_questions_at_end(state: DialogueState) -> int:
    return sum(1 for q in _directed_questions(state) if not _answered_by_target(state, q))


def _repetition_score(state: DialogueState) -> float | None:
    """Lexical repetition across each persona's own turns.

    For every participant turn after a persona's first, take the maximum
    content-word Jaccard overlap with any of that persona's earlier turns;
    average over all such turns. 0 = every turn fresh, 1 = verbatim repeats.
    """
    scores: list[float] = []
    for persona in state.personas:
        own = [t for t in state.turns if t.speaker_id == persona.id]
        seen: list[set[str]] = []
        for turn in own:
            words = _content_words(turn.text)
            if seen and words:
                best = max(len(words & earlier) / len(words | earlier) for earlier in seen if words | earlier)
                scores.append(best)
            seen.append(words)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def _repair_reasons(state: DialogueState) -> list[str]:
    return [r.repair_reason for r in state.repair_history]


def _compromise_success(state: DialogueState, outcome: RunOutcome) -> float | None:
    """1.0/0.0 when a split/deadlock repair ran and the run did/did not
    resolve; None when no compromise repair was attempted. Averages to a
    success share across the runs CSV."""
    if not {"split_vote", "two_person_deadlock"} & set(_repair_reasons(state)):
        return None
    return 1.0 if outcome.status in ("successful", "majority") else 0.0


def _switch_stats(state: DialogueState) -> tuple[int, float | None, float | None]:
    """(#switches, reason rate, bridge rate). `bridge` is the issue-5 signal:
    the switch visibly links the old stance to the new pick, not just carries a
    loose reason clause. None when no switch occurred."""
    events = [event for rt in state.runtimes.values() for event in rt.switch_events]
    if not events:
        return 0, None, None
    explained = sum(1 for event in events if event.get("has_reason"))
    bridged = sum(1 for event in events if event.get("has_bridge"))
    return len(events), round(explained / len(events), 3), round(bridged / len(events), 3)


def _expected_words(persona: Persona) -> float:
    """The controller's own length target for a discussion turn (midpoint of
    the jitter band in policy._word_bounds), used as the verbosity
    expectation. Verbosity is the only persona parameter affecting length.
    Openings/votes use other budgets, so treat deviations as a band, not an
    exact target."""
    base = int(cfg.utterances.word_budgets.discussion)
    p = persona.sim_params
    expected = base * (0.45 + 0.85 * p.verbosity)
    # policy._word_bounds mixes in occasional short beats (factor ~0.52, more
    # often for terse sims); fold that expectation into the average target.
    short_beat_prob = 0.22 + 0.28 * (1.0 - p.verbosity)
    return expected * (1.0 - 0.48 * short_beat_prob)


def _route_source_distribution(state: DialogueState) -> dict[str, int]:
    """How often each route source produced a participant turn."""
    counts: dict[str, int] = {}
    for turn in state.turns:
        if turn.speaker_id == "moderator" or turn.intent is None:
            continue
        counts[turn.intent.route_source] = counts.get(turn.intent.route_source, 0) + 1
    return dict(sorted(counts.items()))


def _act_mismatch_rate(state: DialogueState) -> float | None:
    """DIAGNOSTIC ONLY: share of routed turns whose derived primary label
    differs from the selected act. A label difference is NOT a semantic
    failure (a comparative question realizes a requested comparison) —
    intended_function_realized_rate and evidence alignment are the
    correctness signals; this only flags drift worth eyeballing.
    """
    routed = [t for t in state.turns if t.speaker_id != "moderator" and t.intent is not None]
    if not routed:
        return None
    mismatched = sum(1 for t in routed if t.realized_act() != t.intent.act)
    return round(mismatched / len(routed), 3)


def _validation_path_stats(state: DialogueState) -> dict[str, Any]:
    """Per-run validation-path summary from the controller trace (item 14)."""
    turn_entries = [e for e in state.controller_trace if e.get("type") == "turn"]
    finals = [e["result"] for e in turn_entries if e["result"].get("validator_llm_used") is not None]
    fast = sum(1 for r in finals if r.get("validator_llm_used") is False)
    # Logical validation checks: turns that actually consulted the validator
    # role (not fast-pathed). API calls include the ≤1 bounded structured-output
    # retry, so they can exceed logical checks — item 10 keeps them separate.
    logical_checks = len(finals) - fast
    validator = state.token_usage_by_call_type.get("validator", {})
    api_calls = int(validator.get("calls", 0))
    accepted = sum(1 for t in state.turns if t.speaker_id != "moderator" and t.text.strip())
    total_in = int(state.setup_tokens_in + state.dialogue_tokens_in + validator.get("in", 0))
    return {
        # API calls (endpoint hits, retries included) — kept as the historical
        # `validator_calls` name for baseline comparability.
        "validator_calls": api_calls,
        "validator_logical_checks": logical_checks,
        "validator_api_retries": max(0, api_calls - logical_checks),
        "validator_calls_per_accepted_turn": round(api_calls / accepted, 3) if accepted else None,
        "validator_logical_checks_per_turn": round(logical_checks / accepted, 3) if accepted else None,
        "validation_fast_path_rate": round(fast / len(finals), 3) if finals else None,
        "validator_input_share": round(int(validator.get("in", 0)) / total_in, 3) if total_in else None,
    }


def _vote_state_consistency_failures(state: DialogueState) -> int:
    """Participants whose runtime vote disagrees with the transcript-derived
    formal vote — public evidence and observer state must never diverge."""
    formal = visible_votes_from_transcript(state)
    return sum(
        1 for pid, vote in formal.items()
        if state.runtimes[pid].explicit_vote != vote
    )


def parameter_realization(state: DialogueState) -> dict[str, Any]:
    """Configured-parameter vs realized-behavior comparison for this run."""
    participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
    total_turns = max(1, len(participant_turns))
    # The router's own engagement-derived share targets (simulator.expected_turn_share),
    # so realization error measures deviation from what the controller aimed for.
    expected_shares = expected_turn_share(state.personas)

    engagement_errors: dict[str, float] = {}
    verbosity_errors: dict[str, float] = {}
    engagement_cfg: list[float] = []
    turn_shares: list[float] = []
    verbosity_cfg: list[float] = []
    realized_words: list[float] = []
    for persona in state.personas:
        share = state.runtimes[persona.id].turn_count / total_turns
        engagement_errors[persona.name] = round(abs(expected_shares[persona.id] - share), 3)
        own = [t for t in participant_turns if t.speaker_id == persona.id]
        avg_words = sum(len(t.text.split()) for t in own) / max(1, len(own))
        expected = _expected_words(persona)
        verbosity_errors[persona.name] = round(abs(avg_words - expected) / expected, 3)
        engagement_cfg.append(persona.sim_params.engagement)
        turn_shares.append(share)
        verbosity_cfg.append(persona.sim_params.verbosity)
        realized_words.append(avg_words)

    def _mean(values: dict[str, float]) -> float:
        return round(sum(values.values()) / max(1, len(values)), 3)

    # Trait-shaped dominance should be judged on free discussion turns only
    # (P4): opening and vote rounds are intentionally near-uniform, so they
    # dilute the trait signal the router is actually allowed to express.
    free_turns = [
        t for t in participant_turns
        if t.intent is not None and t.intent.act in _DISCUSSION_ACTS
    ]
    free_total = max(1, len(free_turns))
    free_share_by_persona = {
        p.name: round(sum(1 for t in free_turns if t.speaker_id == p.id) / free_total, 3)
        for p in state.personas
    }
    free_shares = [free_share_by_persona[p.name] for p in state.personas]
    top_free_share = round(max(free_shares, default=0.0), 3)

    return {
        "engagement_realization_error": _mean(engagement_errors),
        "verbosity_realization_error": _mean(verbosity_errors),
        "engagement_error_by_persona": engagement_errors,
        "verbosity_error_by_persona": verbosity_errors,
        # Per-run trait->behavior coupling signal: positive = configured
        # parameter visibly shapes behavior in this run (None = too few sims
        # or no variance to judge).
        "engagement_behavior_correlation": _pearson(engagement_cfg, turn_shares),
        "verbosity_behavior_correlation": _pearson(verbosity_cfg, realized_words),
        "free_discussion_share": free_share_by_persona,
        "top_free_discussion_share": top_free_share,
        "free_discussion_engagement_correlation": _pearson(engagement_cfg, free_shares),
    }


def token_summary_for(state: DialogueState) -> dict[str, int]:
    # Dialogue-role and validator-role usage are separately visible (item 15):
    # the dialogue counters come from the dialogue client session, validator
    # usage from its per-call accounting.
    validator = state.token_usage_by_call_type.get("validator", {})
    validator_in = int(validator.get("in", 0))
    validator_out = int(validator.get("out", 0))
    return {
        "setup_tokens_in": int(state.setup_tokens_in),
        "setup_tokens_out": int(state.setup_tokens_out),
        "dialogue_tokens_in": int(state.dialogue_tokens_in),
        "dialogue_tokens_out": int(state.dialogue_tokens_out),
        "validator_tokens_in": validator_in,
        "validator_tokens_out": validator_out,
        "total_tokens_in": int(state.setup_tokens_in + state.dialogue_tokens_in + validator_in),
        "total_tokens_out": int(state.setup_tokens_out + state.dialogue_tokens_out + validator_out),
    }


def _token_usage_flat(state: DialogueState) -> dict[str, int]:
    """Token/call diagnostics by call type for cost regression checks."""
    out: dict[str, int] = {}
    kinds = [
        "setup",
        "utterance",
        "validator",
        "repair",
        "moderator",
        "moderator_repair",
    ]
    for kind in kinds:
        usage = state.token_usage_by_call_type.get(kind, {})
        out[f"tokens_{kind}_in"] = int(usage.get("in", 0))
        out[f"tokens_{kind}_out"] = int(usage.get("out", 0))
        out[f"calls_{kind}"] = int(usage.get("calls", 0))
    return out


def _assessment_rate(turns: list[TurnRecord], selector) -> float | None:
    """Share of assessed turns where ``selector(assessment)`` is True, over
    turns where it is not None (no contract for that act/turn = excluded)."""
    values = [
        selector(t.assessment) for t in turns
        if t.assessment is not None and selector(t.assessment) is not None
    ]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 3)


def _repair_success_rate(turns: list[TurnRecord]) -> float | None:
    """Share of semantically-repaired turns whose final line is issue-free.
    Operational validator retries are not repairs and never count here."""
    repaired = [t for t in turns if t.repaired]
    if not repaired:
        return None
    return round(sum(1 for t in repaired if not t.validation_issues) / len(repaired), 3)


def _fallback_by_family(turns: list[TurnRecord]) -> dict[str, int]:
    families: dict[str, int] = {}
    for t in turns:
        if t.used_fallback:
            families[t.fallback_family or "unknown"] = families.get(t.fallback_family or "unknown", 0) + 1
    return families


def _assessment_action_counts(turns: list[TurnRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in turns:
        if t.assessment is not None:
            counts[t.assessment.action.value] = counts.get(t.assessment.action.value, 0) + 1
    return counts


def metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    switch_count, switch_explained, switch_bridged = _switch_stats(state)
    participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
    moderator_turns = [t for t in state.turns if t.speaker_id == "moderator"]
    n_turns = max(1, len(participant_turns))
    turn_counts = {p.name: state.runtimes[p.id].turn_count for p in state.personas}
    avg_words_by_persona = {
        p.name: round(
            sum(len(t.text.split()) for t in participant_turns if t.speaker_id == p.id)
            / max(1, state.runtimes[p.id].turn_count),
            1,
        )
        for p in state.personas
    }
    words_by_act: dict[str, list[int]] = {}
    for t in participant_turns:
        act_name = t.intent.act.value if t.intent else "unknown"
        words_by_act.setdefault(act_name, []).append(len(t.text.split()))
    avg_words_by_act = {
        act_name: round(sum(counts) / len(counts), 1)
        for act_name, counts in sorted(words_by_act.items())
    }
    short_turn_count = sum(1 for t in participant_turns if len(t.text.split()) <= 10)
    tiny_turn_count = sum(1 for t in participant_turns if len(t.text.split()) <= 5)
    # P2 diagnostic: questions appearing on statement-type turns (the chaining
    # pattern), as opposed to intentional ask/invite acts.
    question_acts = {"ask", "process"}
    statement_turns = [
        t for t in participant_turns
        if t.intent and t.intent.act.value not in question_acts
    ]
    tail_question_count = sum(1 for t in statement_turns if t.text.rstrip().endswith("?"))
    visible_vote_ids = visible_votes_from_transcript(state)
    visible_votes = {
        p.name: visible_vote_ids[p.id]
        for p in state.personas
        if p.id in visible_vote_ids
    }
    top_turn_share = round(max(turn_counts.values(), default=0) / max(1, len(participant_turns)), 3)
    expected_engagement = {p.name: round(p.sim_params.engagement, 3) for p in state.personas}
    names = [p.name for p in state.personas]
    alias_values = list(short_alias_map(state.scenario.options).values())
    name_prefixed = sum(1 for t in participant_turns if leading_name(t.text, names))
    option_opened = sum(1 for t in participant_turns if leading_option(t.text, alias_values))
    patterns = [surface_pattern(t.text) for t in participant_turns]
    templated = {"concede_but", "worry_but", "tradeoff_but"}
    repeated_openings = sum(
        1 for i in range(1, len(patterns))
        if patterns[i] == patterns[i - 1] and patterns[i] in templated
    )
    return {
        "participant_turns": len(participant_turns),
        "moderator_turns": len(moderator_turns),
        "moderator_ratio": round(len(moderator_turns) / max(1, len(state.turns)), 3),
        "turn_counts": turn_counts,
        "top_speaker_share": top_turn_share,
        "avg_words_by_persona": avg_words_by_persona,
        "avg_words_by_act": avg_words_by_act,
        "short_turn_rate": round(short_turn_count / n_turns, 3),
        "tiny_turn_rate": round(tiny_turn_count / n_turns, 3),
        "question_density": round(sum(1 for t in participant_turns if "?" in t.text) / n_turns, 3),
        "tail_question_rate": round(tail_question_count / max(1, len(statement_turns)), 3),
        "avg_words_per_turn": round(sum(len(t.text.split()) for t in participant_turns) / n_turns, 1),
        "repaired_turns": sum(1 for t in participant_turns if t.repaired),
        "repair_rate": round(sum(1 for t in participant_turns if t.repaired) / n_turns, 3),
        "flagged_turns": sum(1 for t in participant_turns if t.validation_issues),
        "fallback_turns": int(state.fallback_turn_count),
        "invalid_printed_turn_count": int(state.invalid_printed_turn_count),
        "visible_vote_count": len(visible_votes),
        "visible_votes": visible_votes,
        "unanswered_direct_questions": _unanswered_question_threads(state),
        "question_threads": len(_question_threads(state)),
        # Concern/blocker-thread completion (6.2/6.3): how many visible
        # objections opened a thread and what share got a relevant response
        # (cooling/resolved) before aging out.
        "concern_threads": len(_concern_threads(state)),
        "concern_response_rate": _concern_response_rate(state),
        "thread_count_by_type": {
            kind.value: sum(1 for t in state.threads.values() if t.thread_type is kind)
            for kind in ThreadType
        },
        "thread_count_by_status": {
            status.value: sum(1 for t in state.threads.values() if t.status is status)
            for status in ThreadStatus
        },
        "participation_gini": _gini(list(turn_counts.values())),
        "direct_response_rate": _direct_response_rate(state),
        "question_answer_completion": _question_answer_completion(state),
        "open_questions_at_end": _open_questions_at_end(state),
        # P7: issue keys whose concern/blocker thread already played out — the
        # thread engine suppresses re-raises of these (single owner of issue
        # repetition; the old issue_ledger is gone).
        "settled_issue_keys": sorted({
            t.issue_key for t in state.threads.values()
            if t.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
            and t.status in (ThreadStatus.RESOLVED, ThreadStatus.STALE)
            and t.issue_key
        }),
        "repetition_score": _repetition_score(state),
        "compromise_success_rate": _compromise_success(state, outcome),
        # Repair state machine (13.7): which objectives ran and how they ended.
        "repairs_run": _repair_reasons(state),
        "repair_statuses": {r.repair_reason: r.status for r in state.repair_history},
        "unclear_vote_repairs": _repair_reasons(state).count("unclear_vote"),
        "reservation_exchange": state.reservation_exchanges > 0,
        "participant_procedural_moves": int(state.procedural_move_count),
        "two_person_deadlock_attempted": "two_person_deadlock" in _repair_reasons(state),
        "split_reservation_exchanges": int(state.reservation_exchanges),
        # Same-speaker follow-up turns (issue 6) — rare by design.
        "continuation_turns": sum(
            1 for t in participant_turns if t.intent is not None and t.intent.continuation
        ),
        "switch_event_count": switch_count,
        "switch_explanation_rate": switch_explained,
        "switch_bridge_rate": switch_bridged,
        # Latent-favorite movements during the discussion phase (issue 3) —
        # visible softening or compromise signals, distinct from final votes.
        "discussion_lean_shifts": int(state.discussion_lean_shifts),
        "discussion_lean_shift_turns": list(state.discussion_lean_shift_turns),
        # Public evidence and observer state must agree on every formal vote.
        "vote_state_consistency_failures": _vote_state_consistency_failures(state),
        "name_prefix_rate": round(name_prefixed / n_turns, 3),
        "option_opening_rate": round(option_opened / n_turns, 3),
        "i_opening_rate": round(sum(1 for t in participant_turns if leading_first_person(t.text)) / n_turns, 3),
        "we_opening_rate": round(sum(1 for t in participant_turns if leading_we(t.text)) / n_turns, 3),
        "name_or_option_opening_rate": round((name_prefixed + option_opened) / n_turns, 3),
        "repeated_opening_patterns": repeated_openings,
        "unsupported_fact_flags": sum(
            1 for t in participant_turns
            if any(
                code.startswith("UNSUPPORTED_CLAIM")
                for code in list(t.validation_issues) + list(t.repair_trigger_codes)
            )
        ),
        # Printed turns whose FINAL ACCEPTED claims are unsupported (item 15):
        # zero here means the accepted claims were actually verified against
        # the fact table, not merely that no issue code happened to remain.
        "unsupported_printed_turns": sum(
            1 for t in participant_turns
            if t.evidence is not None and any(c.supported is False for c in t.evidence.claims)
        ),
        # --- semantic-correctness metrics for the evidence contract (item 15) ---
        # Realized-function rate vs exact-label agreement: a comparative
        # question realizing a requested COMPARE counts as realized even though
        # its primary label is ASK.
        "intended_function_realized_rate": _assessment_rate(
            participant_turns, lambda a: a.intended_act_realized
        ),
        "intended_focus_agreement_rate": _assessment_rate(
            participant_turns, lambda a: a.intended_focus_realized
        ),
        "ambiguous_reference_rate": round(
            sum(
                1 for t in participant_turns
                if t.evidence is not None and t.evidence.ambiguous_references
            ) / n_turns, 3,
        ),
        "validator_failure_turns": sum(
            1 for t in participant_turns
            if "VALIDATOR_UNAVAILABLE" in (list(t.validation_issues) + list(t.repair_trigger_codes))
        ),
        "repair_success_rate": _repair_success_rate(participant_turns),
        "fallback_by_family": _fallback_by_family(participant_turns),
        "dropped_turn_count": sum(
            1 for e in state.controller_trace
            if e.get("type") == "turn" and not e.get("result", {}).get("appended", True)
        ),
        "vote_clarity_failures": sum(
            1 for t in participant_turns
            if "UNCLEAR_VISIBLE_COMMITMENT" in (list(t.validation_issues) + list(t.repair_trigger_codes))
        ),
        "assessment_action_counts": _assessment_action_counts(participant_turns),
        # P11: a hard blocker visibly counted as supporting their rejected
        # option in the final tally — must always be 0.
        "final_blocker_violations": sum(
            1 for p in state.personas
            if outcome.final_option in state.runtimes[p.id].rejected_options()
            and visible_vote_ids.get(p.id) == outcome.final_option
        ),
        "final_support_fraction": _final_support_fraction(state, outcome),
        # Ranks are 1 (rejected) .. 5 (preferred); the earlier range(5) silently
        # dropped rank 5 and reported a meaningless rank 0.
        "stance_rank_distribution": {
            str(rank): sum(1 for rt in state.runtimes.values() for value in rt.option_ranks.values() if value == rank)
            for rank in range(1, 6)
        },
        "runtime_preferred_by_rank": {
            p.name: state.runtimes[p.id].top_option() for p in state.personas
        },
        "option_coverage": {
            opt: {
                "mentions": c.mentions,
                "reasons": c.reasons,
                "objections": c.objections,
                "acceptances": c.acceptances,
            }
            for opt, c in state.coverage.items()
        },
        # Selected vs realized coverage (16.3): routing a coverage turn is not
        # the same as the final text visibly processing the option.
        "coverage_routes_selected": sum(c.coverage_attempts for c in state.coverage.values()),
        "coverage_turns_realized": _coverage_turns_realized(state),
        # Controller-trace metrics (16.2): why turns were routed and whether the
        # final text realized the selected act. act_mismatch_rate is a
        # diagnostic label-drift signal only — never a semantic failure count.
        "route_source_distribution": _route_source_distribution(state),
        "act_mismatch_rate": _act_mismatch_rate(state),
        "expected_engagement": expected_engagement,
        "expected_switch_resistance": {
            p.name: round(p.sim_params.switch_resistance, 3) for p in state.personas
        },
        "expected_turn_share": {
            p.name: round(expected_turn_share(state.personas)[p.id], 3) for p in state.personas
        },
        "realized_turn_share": {
            p.name: round(state.runtimes[p.id].turn_count / n_turns, 3) for p in state.personas
        },
        "outcome_status": outcome.status,
        "final_option": outcome.final_option,
        "corpus_preset": (getattr(cfg, "corpus_active", None) or {}).get("name", ""),
        "min_discussion_turns": state.min_discussion_turns,
        "force_narrow_turns": state.force_narrow_turns,
        "hard_max_turns": state.hard_max_turns,
        "phase_history": list(state.phase_history),
    } | parameter_realization(state) | token_summary_for(state) | _token_usage_flat(state) \
        | _validation_path_stats(state)


def flat_metrics_for(run_id: str, state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    metrics = metrics_for(state, outcome)
    scalar = {k: v for k, v in metrics.items() if not isinstance(v, dict) and not isinstance(v, list)}
    dialogue_provider = str(cfg.llm.get("dialogue", "")).lower()
    validator_provider = str(cfg.llm.get("validator", "")).lower()
    return {
        "run_id": run_id,
        "topic": state.scenario.topic,
        "environment_type": state.scenario.environment_type,
        # Provider/model per role per row so runs from different backends can
        # be compared in the CSV (issue 9). llm_provider/llm_model keep their
        # historical column names and describe the dialogue role.
        "llm_provider": dialogue_provider,
        "llm_model": str(cfg.llm.models.get(dialogue_provider, "unknown")),
        "llm_validator_provider": validator_provider,
        "llm_validator_model": str(cfg.llm.models.get(validator_provider, "unknown")),
        "num_participants": len(state.personas),
        "hard_blocker_present": any(state.runtimes[p.id].rejected_options() for p in state.personas),
        **scalar,
    }


def _coverage_turns_realized(state: DialogueState) -> int:
    """Coverage-routed turns whose final accepted text visibly names the option.

    Distinct from coverage routes selected: a routed coverage intent whose
    generated turn was blocked, dropped, or never mentioned the target option
    did not realize coverage.
    """
    realized = 0
    for turn in state.turns:
        if turn.speaker_id == "moderator" or turn.state_mutation_blocked or turn.intent is None:
            continue
        if turn.intent.route_source != "coverage" or not turn.intent.option_focus:
            continue
        if turn.intent.option_focus[0] in turn.mentioned_options() and turn.text.strip():
            realized += 1
    return realized


def _final_support_fraction(state: DialogueState, outcome: RunOutcome) -> float:
    if not outcome.final_option:
        return 0.0
    final = outcome.final_option
    backers = sum(
        1
        for p in state.personas
        if state.runtimes[p.id].explicit_vote == final or final in state.runtimes[p.id].acceptable_options()
    )
    return round(backers / max(1, len(state.personas)), 3)
