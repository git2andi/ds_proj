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
from models import DialogueState, Persona, RunOutcome, TurnRecord, _DISCUSSION_ACTS
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
    target = question_turn.act.question_target_id
    for turn in state.turns[question_turn.index + 1:]:
        if turn.speaker_id == target:
            return within is None or (turn.index - question_turn.index) <= within
    return False


def _directed_questions(state: DialogueState) -> list[TurnRecord]:
    return [
        t for t in state.turns
        if t.act.question_target_id and t.act.question_target_id != t.speaker_id
        and t.act.question_target_id in state.runtimes
    ]


def _direct_response_rate(state: DialogueState) -> float | None:
    """Share of response obligations that did not lapse unanswered.

    Denominator is every obligation the router created; an obligation
    overwritten by a newer one before lapsing counts as handled, matching the
    single-slot obligation semantics.
    """
    created = int(state.obligations_created)
    if created == 0:
        return None
    unanswered = int(state.unanswered_obligations)
    ob = state.response_obligation
    if ob is not None and not any(
        t.speaker_id == ob.target_id and t.index > ob.created_turn for t in state.turns
    ):
        unanswered += 1  # still open at run end and never answered
    return round(max(0, created - unanswered) / created, 3)


def _question_answer_completion(state: DialogueState) -> float | None:
    """Adjacency-pair completion: directed questions answered by the addressee
    within the obligation window (2 x response_obligation_turns, as in the
    router's lapse rule)."""
    questions = _directed_questions(state)
    if not questions:
        return None
    window = 2 * max(1, int(cfg.conversation.get("response_obligation_turns", 2)))
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


def _compromise_success(state: DialogueState, outcome: RunOutcome) -> float | None:
    """1.0/0.0 when a split-vote compromise step ran and the run did/did not
    resolve; None when no compromise was attempted. Averages to a success
    share across the runs CSV."""
    if not state.compromise_attempted:
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
    the jitter band in dialogue._word_bounds), used as the verbosity
    expectation. Openings/votes use other budgets, so treat deviations as a
    band, not an exact target."""
    base = int(cfg.utterances.word_budgets.discussion)
    p = persona.sim_params
    expected = base * (0.45 + 0.70 * p.verbosity + 0.15 * p.engagement)
    # policy._word_bounds mixes in occasional short beats (factor ~0.52, more
    # often for terse sims); fold that expectation into the average target.
    short_beat_prob = 0.22 + 0.28 * (1.0 - p.verbosity)
    return expected * (1.0 - 0.48 * short_beat_prob)


def parameter_realization(state: DialogueState) -> dict[str, Any]:
    """Configured-parameter vs realized-behavior comparison for this run."""
    participant_turns = [t for t in state.turns if t.speaker_id != "moderator"]
    total_turns = max(1, len(participant_turns))
    # The router's own trait-derived share targets (simulator.expected_turn_share),
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
    return {
        "setup_tokens_in": int(state.setup_tokens_in),
        "setup_tokens_out": int(state.setup_tokens_out),
        "dialogue_tokens_in": int(state.dialogue_tokens_in),
        "dialogue_tokens_out": int(state.dialogue_tokens_out),
        "total_tokens_in": int(state.setup_tokens_in + state.dialogue_tokens_in),
        "total_tokens_out": int(state.setup_tokens_out + state.dialogue_tokens_out),
    }


def _token_usage_flat(state: DialogueState) -> dict[str, int]:
    """Token/call diagnostics by call type for cost regression checks."""
    out: dict[str, int] = {}
    kinds = [
        "setup",
        "utterance",
        "grounding",
        "repair",
        "grounding_repair",
        "moderator",
        "moderator_repair",
    ]
    for kind in kinds:
        usage = state.token_usage_by_call_type.get(kind, {})
        out[f"tokens_{kind}_in"] = int(usage.get("in", 0))
        out[f"tokens_{kind}_out"] = int(usage.get("out", 0))
        out[f"calls_{kind}"] = int(usage.get("calls", 0))
    return out


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
        "unanswered_direct_questions": int(state.unanswered_obligations),
        # Concern-thread completion (issue 2): how many visible objections opened
        # a thread and what share got a visible reaction before aging out.
        "concern_threads": int(state.concerns_raised_total),
        "concern_response_rate": (
            round(state.concerns_addressed_total / state.concerns_raised_total, 3)
            if state.concerns_raised_total
            else None
        ),
        "participation_gini": _gini(list(turn_counts.values())),
        "direct_response_rate": _direct_response_rate(state),
        "question_answer_completion": _question_answer_completion(state),
        "open_questions_at_end": _open_questions_at_end(state),
        # P7: mentions of a practical unknown beyond its allowed raise+answer
        # pair — the "we still don't know about parking" loop signal.
        "repeated_unknown_mentions": sum(
            max(0, int(v.get("mentions", 0)) - 2) for v in state.issue_ledger.values()
        ),
        "issue_ledger": {k: dict(v) for k, v in sorted(state.issue_ledger.items())},
        "repetition_score": _repetition_score(state),
        "compromise_success_rate": _compromise_success(state, outcome),
        "reservation_exchange": bool(state.reservation_exchange_done),
        "participant_procedural_moves": int(state.procedural_move_count),
        "two_person_deadlock_attempted": bool(state.two_person_deadlock_attempted),
        "split_reservation_exchanges": int(state.split_reservation_exchanges),
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
        "name_prefix_rate": round(name_prefixed / n_turns, 3),
        "option_opening_rate": round(option_opened / n_turns, 3),
        "i_opening_rate": round(sum(1 for t in participant_turns if leading_first_person(t.text)) / n_turns, 3),
        "we_opening_rate": round(sum(1 for t in participant_turns if leading_we(t.text)) / n_turns, 3),
        "name_or_option_opening_rate": round((name_prefixed + option_opened) / n_turns, 3),
        "repeated_opening_patterns": repeated_openings,
        "unsupported_fact_flags": sum(
            1 for t in participant_turns
            if "UNSUPPORTED_FACT" in (list(t.validation_issues) + list(t.repair_trigger_codes))
        ),
        # Lines that reached the transcript still carrying the flag (issue 7):
        # the number that matters for grounding quality; flags caught and
        # repaired away are the pipeline working as intended.
        "unsupported_printed_turns": sum(
            1 for t in participant_turns if "UNSUPPORTED_FACT" in t.validation_issues
        ),
        # P11: a hard blocker visibly counted as supporting their rejected
        # option in the final tally — must always be 0.
        "final_blocker_violations": sum(
            1 for p in state.personas
            if outcome.final_option in state.runtimes[p.id].rejected_options()
            and visible_vote_ids.get(p.id) == outcome.final_option
        ),
        "final_support_fraction": _final_support_fraction(state, outcome),
        "stance_rank_distribution": {
            str(rank): sum(1 for rt in state.runtimes.values() for value in rt.option_ranks.values() if value == rank)
            for rank in range(5)
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
        "expected_engagement": expected_engagement,
        "expected_turn_share": {
            p.name: round(expected_turn_share(state.personas)[p.id], 3) for p in state.personas
        },
        "realized_turn_share": {
            p.name: round(state.runtimes[p.id].turn_count / n_turns, 3) for p in state.personas
        },
        "agenda_status": _agenda_status_counts(state),
        "outcome_status": outcome.status,
        "final_option": outcome.final_option,
        "corpus_preset": (getattr(cfg, "corpus_active", None) or {}).get("name", ""),
        "min_discussion_turns": state.min_discussion_turns,
        "force_narrow_turns": state.force_narrow_turns,
        "hard_max_turns": state.hard_max_turns,
        "phase_history": list(state.phase_history),
    } | parameter_realization(state) | token_summary_for(state) | _token_usage_flat(state)


def flat_metrics_for(run_id: str, state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    metrics = metrics_for(state, outcome)
    scalar = {k: v for k, v in metrics.items() if not isinstance(v, dict) and not isinstance(v, list)}
    provider = str(cfg.llm.provider).lower()
    return {
        "run_id": run_id,
        "topic": state.scenario.topic,
        "environment_type": state.scenario.environment_type,
        # Provider/model per row so runs from different backends can be
        # compared in the CSV (issue 9).
        "llm_provider": provider,
        "llm_model": str(cfg.llm.models.get(provider, "unknown")),
        "num_participants": len(state.personas),
        "hard_blocker_present": any(state.runtimes[p.id].rejected_options() for p in state.personas),
        **scalar,
    }


def _agenda_status_counts(state: DialogueState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for persona in state.personas:
        for item in persona.agenda:
            key = item.status.value if hasattr(item.status, "value") else str(item.status)
            counts[key] = counts.get(key, 0) + 1
    return counts


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
