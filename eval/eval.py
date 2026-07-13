"""Concise deterministic evaluation of one simulator run.

The summary keeps only defensible structural, participation, trait, interaction,
decision, validation, grounding-intervention, and token metrics. Detailed turn
and issue diagnostics remain in ``run.json``. No evaluation LLM is used.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
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
from models import ActType, DialogueState, Phase, RunOutcome, ThreadStatus, ThreadType
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


def _floor_autonomy_metrics(state: DialogueState) -> dict[str, Any]:
    participants = {p.id for p in state.personas}
    """Authority-split / floor metrics (todo 22): where each participant turn's
    authority came from, and how simulator bids competed for the floor."""
    from collections import Counter

    turn_traces = [
        e for e in state.controller_trace
        if e.get("type") == "turn" and e.get("result", {}).get("appended")
    ]
    names = {p.id: p.name for p in state.personas}
    authority_counts: Counter = Counter()
    submitted_acts: Counter = Counter()
    invalid_by_reason: Counter = Counter()
    claims = {p.name: 0 for p in state.personas}
    bids_seen = {p.name: 0 for p in state.personas}
    willingness_sum = {p.name: 0.0 for p in state.personas}
    floor_wins = {p.name: 0 for p in state.personas}
    act_match = act_total = 0
    next_best_substitutions = 0

    for entry in turn_traces:
        pre = entry.get("pre", {})
        result = entry.get("result", {})
        authority = pre.get("authority_source", pre.get("route_source", "self_selection"))
        authority_counts[authority] += 1
        bids = pre.get("bids", [])
        winner = pre.get("speaker_id")
        winner_score = None
        best_score = None
        for b in bids:
            pid = b.get("participant_id")
            name = names.get(pid, pid)
            if name in bids_seen:
                bids_seen[name] += 1
                willingness_sum[name] += float(b.get("willingness", 0.0))
                if b.get("wants_to_speak"):
                    claims[name] += 1
            if b.get("wants_to_speak"):
                submitted_acts[b.get("proposed_act")] += 1
            if b.get("rejected_reason"):
                invalid_by_reason[b["rejected_reason"]] += 1
            if b.get("wants_to_speak") and not b.get("rejected_reason"):
                if best_score is None or float(b.get("floor_score", 0)) > best_score:
                    best_score = float(b.get("floor_score", 0))
            if b.get("participant_id") == winner:
                winner_score = float(b.get("floor_score", 0))
        if authority == "self_selection" and winner in names:
            floor_wins[names[winner]] += 1
        # A next-best substitution: the realized winner was not the top-scoring
        # valid bid (its bid dropped in generation and the floor used the next).
        if best_score is not None and winner_score is not None and winner_score < best_score - 1e-9:
            next_best_substitutions += 1
        if authority == "self_selection":
            act_total += 1
            if not result.get("act_mismatch", False):
                act_match += 1

    # Speaker-chain maximum (longest run of consecutive participant turns).
    max_chain = chain = 0
    last = None
    for turn in state.turns:
        if turn.speaker_id == "moderator" or not turn.text.strip():
            continue
        chain = chain + 1 if turn.speaker_id == last else 1
        last = turn.speaker_id
        max_chain = max(max_chain, chain)

    appended = len(turn_traces)
    self_selected = authority_counts.get("self_selection", 0)
    avg_willingness = {
        name: round(willingness_sum[name] / bids_seen[name], 3) if bids_seen[name] else None
        for name in claims
    }
    claim_rate = {
        name: round(claims[name] / bids_seen[name], 3) if bids_seen[name] else None
        for name in claims
    }

    # Realization telemetry is computed from every generation trace, including
    # candidates that were repaired or finally dropped. This keeps simulator
    # policy quality separate from wording/validation reliability.
    attempts_by_act: Counter[str] = Counter()
    accepted_by_act: Counter[str] = Counter()
    realized_by_act: Counter[str] = Counter()
    dropped_by_act: Counter[str] = Counter()
    for item in state.controller_trace:
        if item.get("type") != "turn":
            continue
        pre = item.get("pre", {})
        result = item.get("result", {})
        act = pre.get("selected_act")
        if not act:
            continue
        attempts_by_act[act] += 1
        accepted = bool(result.get("appended")) and not bool(result.get("state_mutation_blocked"))
        if accepted:
            accepted_by_act[act] += 1
        if result.get("intended_act_realized") is True:
            realized_by_act[act] += 1
        if not accepted:
            dropped_by_act[act] += 1

    realization_rate_by_act = {
        act: round(realized_by_act[act] / count, 3) if count else None
        for act, count in attempts_by_act.items()
    }
    acceptance_rate_by_act = {
        act: round(accepted_by_act[act] / count, 3) if count else None
        for act, count in attempts_by_act.items()
    }

    return {
        "authority_source_distribution": dict(authority_counts),
        "self_selected_turns": self_selected,
        "protocol_forced_turns": appended - self_selected - authority_counts.get("direct_obligation", 0),
        "direct_answer_turns": authority_counts.get("direct_obligation", 0),
        "self_selected_ratio": round(self_selected / appended, 3) if appended else None,
        "bid_rounds": state.bid_round_count,
        "no_bid_rounds": state.no_bid_round_count,
        "true_no_claim_rounds": state.no_bid_round_count,
        "generation_failure_rounds": state.generation_failure_round_count,
        "valid_bid_attempts": state.valid_bid_attempt_count,
        "final_dropped_intents": state.final_dropped_intent_count,
        "protocol_obligation_failures": state.protocol_obligation_failures,
        "repeated_bid_rejections": state.repeated_bid_rejections,
        "discussion_conditional_acceptances": state.accepted_conditional_acceptances,
        "accepted_openings": sum(
            1 for turn in state.turns
            if turn.speaker_id in participants and turn.phase is Phase.OPENING
            and turn.text.strip() and not turn.state_mutation_blocked
        ),
        "expected_openings": len(state.personas),
        "accepted_formal_votes": sum(
            1 for turn in state.turns
            if turn.speaker_id in participants and turn.is_formal_commitment_turn()
            and turn.text.strip() and not turn.state_mutation_blocked
        ),
        "expected_formal_votes": len(state.personas),
        "claim_rate_by_persona": claim_rate,
        "avg_willingness_by_persona": avg_willingness,
        "floor_wins_by_persona": floor_wins,
        "submitted_act_distribution": dict(submitted_acts),
        "intended_vs_realized_act_match_rate": round(act_match / act_total, 3) if act_total else None,
        "realization_attempts_by_intended_act": dict(attempts_by_act),
        "accepted_realizations_by_intended_act": dict(accepted_by_act),
        "realization_rate_by_intended_act": realization_rate_by_act,
        "acceptance_rate_by_intended_act": acceptance_rate_by_act,
        "final_drops_by_intended_act": dict(dropped_by_act),
        "invalid_bid_count_by_reason": dict(invalid_by_reason),
        "next_best_bid_substitutions": next_best_substitutions,
        "speaker_chain_max": max_chain,
        "engagement_vs_floor_win_correlation": _pearson(
            [p.sim_params.engagement for p in state.personas],
            [floor_wins[p.name] for p in state.personas],
        ),
    }



def _social_interaction_metrics(state: DialogueState) -> dict[str, Any]:
    """Public, transcript-derived social interaction signals.

    Name mentions are reported separately from functional directed exchanges so
    incidental name use is never treated as a quality target.
    """
    participants = {p.id: p.name for p in state.personas}
    accepted = [
        turn for turn in state.turns
        if turn.speaker_id in participants and turn.text.strip() and not turn.state_mutation_blocked
    ]
    functional_pairs: set[tuple[str, str]] = set()
    referenced_pairs: set[tuple[str, str]] = set()
    reference_turns = 0
    functional_direct_turns = 0

    for turn in accepted:
        speaker = turn.speaker_id
        referenced = {
            pid for pid, name in participants.items()
            if pid != speaker and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", turn.text, re.IGNORECASE)
        }
        if referenced:
            reference_turns += 1
            referenced_pairs.update((speaker, pid) for pid in referenced)

        direct_targets = {
            q.addressee_id for q in (turn.evidence.questions if turn.evidence else [])
            if q.scope == "direct" and q.addressee_id in participants and q.addressee_id != speaker
        }
        functional_targets = set(direct_targets)
        if turn.realized_act() not in {ActType.OPENING, ActType.VOTE, ActType.PROCESS, ActType.CLOSING}:
            functional_targets.update(referenced)
        if functional_targets:
            functional_direct_turns += 1
            functional_pairs.update((speaker, pid) for pid in functional_targets)

    questions = _question_threads(state)
    direct = [thread for thread in questions if thread.question_scope == "direct"]
    group = [thread for thread in questions if thread.question_scope == "group"]

    def completion(threads: list) -> float | None:
        if not threads:
            return None
        answered = sum(
            1 for thread in threads
            if thread.status in (ThreadStatus.COOLING, ThreadStatus.RESOLVED)
            or any(pid != thread.started_by for pid in thread.participants_involved)
        )
        return round(answered / len(threads), 3)

    possible_pairs = len(participants) * max(0, len(participants) - 1)
    self_selected_acts: Counter = Counter()
    for turn in accepted:
        if turn.intent and turn.intent.route_source == "self_selection":
            self_selected_acts[turn.realized_act().value] += 1

    discussion_compromises = sum(
        1 for turn in accepted
        if turn.phase is Phase.DISCUSSION
        and (turn.realized_act() is ActType.COMPROMISE or bool(turn.evidence and turn.evidence.proposals))
    )
    repair_vote_turns = [
        turn for turn in accepted
        if turn.phase is Phase.COMPROMISE_REPAIR
        and turn.intent is not None
        and turn.intent.act is ActType.VOTE
        and turn.intent.route_source == "repair_protocol"
    ]
    repair_switches = sum(
        1 for runtime in state.runtimes.values()
        for event in runtime.switch_events
        if event.get("route_source") == "repair_protocol"
    )
    repair_attempts = len(repair_vote_turns)

    return {
        "direct_address_turn_count": functional_direct_turns,
        "direct_address_turn_rate": round(functional_direct_turns / len(accepted), 3) if accepted else None,
        "unique_directed_participant_pairs": len(functional_pairs),
        "pairwise_interaction_density": round(len(functional_pairs) / possible_pairs, 3) if possible_pairs else None,
        "direct_question_count": len(direct),
        "direct_question_response_success": completion(direct),
        "group_question_count": len(group),
        "group_question_response_success": completion(group),
        "participant_reference_turn_count": reference_turns,
        "participant_reference_turn_rate": round(reference_turns / len(accepted), 3) if accepted else None,
        "unique_reference_pairs": len(referenced_pairs),
        "self_selected_act_distribution": dict(self_selected_acts),
        "discussion_phase_compromise_count": discussion_compromises,
        "discussion_phase_stance_movement_count": int(state.discussion_lean_shifts),
        "repair_switch_attempts": repair_attempts,
        "repair_successful_switches": repair_switches,
        "repair_holdouts": max(0, repair_attempts - repair_switches),
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
        # A repair-phase vote is a simulator switch opportunity (its target and
        # any switch are the simulator's own decision under the new authority
        # split: route_source "repair_protocol").
        if turn.intent and turn.intent.route_source == "repair_protocol" and turn.intent.act is ActType.VOTE:
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
        event.get("route_source") == "repair_protocol"
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
        "metric_schema_version": "3.1",
        "floor_autonomy": _floor_autonomy_metrics(state),
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
            **_social_interaction_metrics(state),
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
    floor = m["floor_autonomy"]
    return {
        "metric_schema_version": m["metric_schema_version"],
        "run_id": run_id,
        "topic": state.scenario.topic,
        "num_participants": len(state.personas),
        "self_selected_ratio": floor["self_selected_ratio"],
        "no_bid_rounds": floor["no_bid_rounds"],
        "next_best_bid_substitutions": floor["next_best_bid_substitutions"],
        "speaker_chain_max": floor["speaker_chain_max"],
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
        "direct_address_turn_rate": inter["direct_address_turn_rate"],
        "pairwise_interaction_density": inter["pairwise_interaction_density"],
        "direct_question_response_success": inter["direct_question_response_success"],
        "group_question_response_success": inter["group_question_response_success"],
        "participant_reference_turn_rate": inter["participant_reference_turn_rate"],
        "discussion_phase_compromise_count": inter["discussion_phase_compromise_count"],
        "discussion_phase_stance_movement_count": inter["discussion_phase_stance_movement_count"],
        "repair_switch_attempts": inter["repair_switch_attempts"],
        "repair_successful_switches": inter["repair_successful_switches"],
        "repair_holdouts": inter["repair_holdouts"],
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


