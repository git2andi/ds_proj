"""Concise deterministic evaluation of one simulator run.

The summary keeps only defensible structural, participation, trait, interaction,
decision, validation, grounding-intervention, and token metrics. Detailed turn
and issue diagnostics remain in ``run.json``. No evaluation LLM is used.
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

from consensus import visible_votes_from_transcript
from models import ActType, DialogueState, RunOutcome, ThreadStatus, ThreadType
from simulator import expected_turn_share

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


def _vote_state_consistency_failures(state: DialogueState) -> int:
    """Participants whose runtime vote disagrees with the transcript-derived
    formal vote — public evidence and observer state must never diverge."""
    formal = visible_votes_from_transcript(state)
    return sum(
        1 for pid, vote in formal.items()
        if state.runtimes[pid].explicit_vote != vote
    )


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


def metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    """Concise, defensible run metrics.

    Detailed evidence, issue codes, and controller decisions remain in run.json.
    Rates are None when their denominator is zero.
    """
    participant_turns = [t for t in state.turns if t.speaker_id != "moderator" and t.text.strip()]
    moderator_turns = [t for t in state.turns if t.speaker_id == "moderator" and t.text.strip()]
    n_participant = len(participant_turns)
    n_all = n_participant + len(moderator_turns)
    expected_shares = expected_turn_share(state.personas)

    turns_by = {p.name: sum(1 for t in participant_turns if t.speaker_id == p.id) for p in state.personas}
    words_by = {}
    expected_engagement = {}
    expected_turn_share_by = {}
    realized_share = {}
    expected_verbosity = {}
    assigned_avg_word_budget = {}
    realized_words = {}
    word_budget_adherence = {}
    switch_resistance = {}
    switch_opportunities = {p.name: 0 for p in state.personas}
    for p in state.personas:
        own = [t for t in participant_turns if t.speaker_id == p.id]
        avg = sum(len(t.text.split()) for t in own) / len(own) if own else 0.0
        words_by[p.name] = round(avg, 3)
        expected_engagement[p.name] = round(p.sim_params.engagement, 3)
        expected_turn_share_by[p.name] = round(expected_shares[p.id], 3)
        realized_share[p.name] = round(len(own) / n_participant, 3) if n_participant else None
        expected_verbosity[p.name] = round(p.sim_params.verbosity, 3)
        budgeted = [t for t in own if t.assigned_max_words > 0]
        assigned = (
            sum((t.assigned_min_words + t.assigned_max_words) / 2 for t in budgeted) / len(budgeted)
            if budgeted else None
        )
        assigned_avg_word_budget[p.name] = round(assigned, 3) if assigned is not None else None
        realized_words[p.name] = round(avg, 3)
        word_budget_adherence[p.name] = (
            round(
                sum(
                    1 for t in budgeted
                    if t.assigned_min_words <= len(t.text.split()) <= t.assigned_max_words + 2
                ) / len(budgeted),
                3,
            )
            if budgeted else None
        )
        switch_resistance[p.name] = round(p.sim_params.switch_resistance, 3)
    for turn in participant_turns:
        if turn.intent and turn.intent.route_source in {
            "majority_holdout_repair", "split_vote_repair", "two_person_deadlock_repair"
        } and turn.intent.act is ActType.VOTE:
            switch_opportunities[state.name_for(turn.speaker_id)] += 1

    question_turns = [t for t in participant_turns if t.evidence and t.evidence.questions]
    q_threads = _question_threads(state)
    c_threads = _concern_threads(state)
    thread_status = {status.value: 0 for status in ThreadStatus}
    for thread in state.threads.values():
        thread_status[thread.status.value] = thread_status.get(thread.status.value, 0) + 1

    repaired = sum(1 for t in participant_turns if t.repaired)
    fallback = sum(1 for t in participant_turns if t.used_fallback)
    dropped = sum(
        1 for e in state.controller_trace
        if e.get("type") == "turn" and not e.get("result", {}).get("appended", True)
    )
    attempts = n_participant + dropped
    visible_votes = visible_votes_from_transcript(state)
    switch_count, _, _ = _switch_stats(state)
    compromise_attempts = sum(1 for r in state.repair_history if r.repair_reason == "split_vote")
    split_repair_switch = any(
        event.get("route_source") == "split_vote_repair"
        for rt in state.runtimes.values()
        for event in rt.switch_events
    )
    # A compromise succeeds only when the no-majority repair caused at least
    # one visible formal switch and the resulting tally resolved. Merely
    # running repair logic or ending with the same votes is not success.
    compromise_successes = int(
        compromise_attempts > 0
        and split_repair_switch
        and outcome.status in {"successful", "majority"}
    )
    active_blockers = {
        p.name: sorted(state.runtimes[p.id].rejected_options())
        for p in state.personas if state.runtimes[p.id].rejected_options()
    }
    grounding_codes = {
        "NUMERIC_CONTRADICTION", "ATTRIBUTE_CONTRADICTION",
        "CROSS_OPTION_VALUE", "UNLISTED_NUMERIC_DETAIL",
        "UNLISTED_FEATURE_DETAIL",
    }
    turn_traces = [
        entry for entry in state.controller_trace if entry.get("type") == "turn"
    ]
    if turn_traces:
        critical_grounding_interventions = sum(
            1 for entry in turn_traces
            if grounding_codes & set(
                (entry.get("result", {}) or {}).get("validation_repair_trigger_codes", [])
            )
        )
    else:
        # Synthetic/unit states may not carry controller traces.
        critical_grounding_interventions = sum(
            1 for turn in participant_turns
            if grounding_codes & set(list(turn.validation_issues) + list(turn.repair_trigger_codes))
        )

    token_usage = {}
    mapping = {
        "setup": ("setup",),
        "participant_generation": ("utterance",),
        "moderator_generation": ("moderator", "moderator_repair"),
        "repair_generation": ("repair",),
        "runtime_validation": ("validator",),
    }
    for label, kinds in mapping.items():
        token_usage[label] = {
            "input_tokens": sum(int(state.token_usage_by_call_type.get(k, {}).get("in", 0)) for k in kinds),
            "output_tokens": sum(int(state.token_usage_by_call_type.get(k, {}).get("out", 0)) for k in kinds),
            "api_calls": sum(int(state.token_usage_by_call_type.get(k, {}).get("calls", 0)) for k in kinds),
        }
    token_usage["total"] = {
        "input_tokens": sum(v["input_tokens"] for v in token_usage.values()),
        "output_tokens": sum(v["output_tokens"] for v in token_usage.values()),
        "api_calls": sum(v["api_calls"] for v in token_usage.values()),
    }

    return {
        "metric_schema_version": "2.1",
        "run_structure": {
            "participant_turn_count": n_participant,
            "participant_turn_count_by_persona": turns_by,
            "moderator_turns": len(moderator_turns),
            "moderator_ratio": round(len(moderator_turns) / n_all, 3) if n_all else None,
            "avg_words_per_participant_turn": round(
                sum(len(t.text.split()) for t in participant_turns) / n_participant, 3
            ) if n_participant else None,
            "avg_words_by_persona": words_by,
            "question_density": round(len(question_turns) / n_participant, 3) if n_participant else None,
        },
        "participation": {
            "expected_engagement": expected_engagement,
            "expected_turn_share": expected_turn_share_by,
            "realized_turn_count": turns_by,
            "realized_turn_share": realized_share,
            "participation_gini": _gini(list(turns_by.values())),
            "engagement_behavior_correlation": _pearson(
                [p.sim_params.engagement for p in state.personas],
                [realized_share[p.name] or 0.0 for p in state.personas],
            ),
        },
        "traits": {
            "expected_verbosity": expected_verbosity,
            "assigned_avg_word_budget": assigned_avg_word_budget,
            "realized_avg_words_per_turn": realized_words,
            "word_budget_adherence": word_budget_adherence,
            "verbosity_budget_correlation": _pearson(
                [p.sim_params.verbosity for p in state.personas],
                [assigned_avg_word_budget[p.name] or 0.0 for p in state.personas],
            ),
            "verbosity_behavior_correlation": _pearson(
                [assigned_avg_word_budget[p.name] or 0.0 for p in state.personas],
                [realized_words[p.name] for p in state.personas],
            ),
            "expected_switch_resistance": switch_resistance,
            "switch_opportunities": switch_opportunities,
            "visible_switches_by_persona": {
                p.name: len(state.runtimes[p.id].switch_events) for p in state.personas
            },
        },
        "interaction": {
            "question_threads": len(q_threads),
            "concern_threads": len(c_threads),
            "thread_count_by_status": thread_status,
            "question_completion_rate": _direct_response_rate(state),
            "concern_response_rate": _concern_response_rate(state),
            "repetition_score": _repetition_score(state),
        },
        "decision_behavior": {
            "visible_votes": visible_votes,
            "outcome_status": outcome.status,
            "final_option": outcome.final_option,
            "switch_event_count": switch_count,
            "discussion_lean_shifts": int(state.discussion_lean_shifts),
            "runtime_preferred_by_rank": {
                p.name: state.runtimes[p.id].top_option() for p in state.personas
            },
            "option_coverage": {
                oid: {
                    "mentions": c.mentions,
                    "reasons": c.reasons,
                    "objections": c.objections,
                    "acceptances": c.acceptances,
                } for oid, c in state.coverage.items()
            },
            "compromise_attempt_count": compromise_attempts,
            "compromise_success_count": compromise_successes,
            "compromise_success_rate": (
                round(compromise_successes / compromise_attempts, 3)
                if compromise_attempts else None
            ),
            "vote_state_consistency_failures": _vote_state_consistency_failures(state),
            "active_blockers_at_close": active_blockers,
        },
        "validation_grounding": {
            "repaired_turns": repaired,
            "repair_rate": round(repaired / attempts, 3) if attempts else None,
            "fallback_turns": fallback,
            "fallback_rate": round(fallback / attempts, 3) if attempts else None,
            "dropped_turns": dropped,
            "drop_rate": round(dropped / attempts, 3) if attempts else None,
            "critical_grounding_interventions": critical_grounding_interventions,
            "runtime_validator_calls": token_usage["runtime_validation"]["api_calls"],
        },
        "token_usage": token_usage,
    }


def flat_metrics_for(run_id: str, state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    m = metrics_for(state, outcome)
    rs = m["run_structure"]
    inter = m["interaction"]
    dec = m["decision_behavior"]
    val = m["validation_grounding"]
    tok = m["token_usage"]["total"]
    return {
        "metric_schema_version": m["metric_schema_version"],
        "run_id": run_id,
        "topic": state.scenario.topic,
        "num_participants": len(state.personas),
        "participant_turn_count": rs["participant_turn_count"],
        "moderator_turns": rs["moderator_turns"],
        "moderator_ratio": rs["moderator_ratio"],
        "avg_words_per_participant_turn": rs["avg_words_per_participant_turn"],
        "question_density": rs["question_density"],
        "question_threads": inter["question_threads"],
        "concern_threads": inter["concern_threads"],
        "question_completion_rate": inter["question_completion_rate"],
        "concern_response_rate": inter["concern_response_rate"],
        "repetition_score": inter["repetition_score"],
        "outcome_status": dec["outcome_status"],
        "final_option": dec["final_option"],
        "visible_vote_count": len(dec["visible_votes"]),
        "switch_event_count": dec["switch_event_count"],
        "compromise_attempt_count": dec["compromise_attempt_count"],
        "compromise_success_rate": dec["compromise_success_rate"],
        "vote_state_consistency_failures": dec["vote_state_consistency_failures"],
        "repaired_turns": val["repaired_turns"],
        "repair_rate": val["repair_rate"],
        "fallback_turns": val["fallback_turns"],
        "fallback_rate": val["fallback_rate"],
        "dropped_turns": val["dropped_turns"],
        "drop_rate": val["drop_rate"],
        "critical_grounding_interventions": val["critical_grounding_interventions"],
        "runtime_validator_calls": val["runtime_validator_calls"],
        "total_input_tokens": tok["input_tokens"],
        "total_output_tokens": tok["output_tokens"],
        "total_api_calls": tok["api_calls"],
    }


