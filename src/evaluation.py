"""Evaluation scaffold for simulator runs.

The current metrics are deliberately lightweight. The module provides a stable
place to add deeper evaluation later: participation inequality, grounding,
question-answer completion, engagement realization, repetition, and visible
preference-shift validity.
"""

from __future__ import annotations

from typing import Any

from models import DialogueState, RunOutcome
from style import leading_name, surface_pattern


def token_summary_for(state: DialogueState) -> dict[str, int]:
    return {
        "setup_tokens_in": int(state.setup_tokens_in),
        "setup_tokens_out": int(state.setup_tokens_out),
        "dialogue_tokens_in": int(state.dialogue_tokens_in),
        "dialogue_tokens_out": int(state.dialogue_tokens_out),
        "total_tokens_in": int(state.setup_tokens_in + state.dialogue_tokens_in),
        "total_tokens_out": int(state.setup_tokens_out + state.dialogue_tokens_out),
    }


def metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
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
    visible_votes = {
        p.name: state.runtimes[p.id].explicit_vote
        for p in state.personas
        if state.runtimes[p.id].explicit_vote
    }
    top_turn_share = round(max(turn_counts.values(), default=0) / max(1, len(participant_turns)), 3)
    expected_engagement = {p.name: round(p.sim_params.engagement, 3) for p in state.personas}
    names = [p.name for p in state.personas]
    name_prefixed = sum(1 for t in participant_turns if leading_name(t.text, names))
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
        "question_density": round(sum(1 for t in participant_turns if "?" in t.text) / n_turns, 3),
        "avg_words_per_turn": round(sum(len(t.text.split()) for t in participant_turns) / n_turns, 1),
        "repaired_turns": sum(1 for t in participant_turns if t.repaired),
        "repair_rate": round(sum(1 for t in participant_turns if t.repaired) / n_turns, 3),
        "flagged_turns": sum(1 for t in participant_turns if t.validation_issues),
        "visible_vote_count": len(visible_votes),
        "visible_votes": visible_votes,
        "unanswered_direct_questions": int(state.unanswered_obligations),
        "name_prefix_rate": round(name_prefixed / n_turns, 3),
        "repeated_opening_patterns": repeated_openings,
        "unsupported_fact_flags": sum(
            1 for t in participant_turns
            if "UNSUPPORTED_FACT" in (list(t.validation_issues) + list(t.repair_trigger_codes))
        ),
        "final_support_fraction": _final_support_fraction(state, outcome),
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
        "outcome_status": outcome.status,
        "final_option": outcome.final_option,
        "min_discussion_turns": state.min_discussion_turns,
        "force_narrow_turns": state.force_narrow_turns,
        "hard_max_turns": state.hard_max_turns,
        "phase_history": list(state.phase_history),
    } | token_summary_for(state)


def flat_metrics_for(run_id: str, state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    metrics = metrics_for(state, outcome)
    scalar = {k: v for k, v in metrics.items() if not isinstance(v, dict) and not isinstance(v, list)}
    return {
        "run_id": run_id,
        "topic": state.scenario.topic,
        "environment_type": state.scenario.environment_type,
        "num_participants": len(state.personas),
        "hard_blocker_present": any(p.rejection for p in state.personas),
        **scalar,
    }


def _final_support_fraction(state: DialogueState, outcome: RunOutcome) -> float:
    if not outcome.final_option:
        return 0.0
    final = outcome.final_option
    backers = sum(
        1
        for p in state.personas
        if state.runtimes[p.id].explicit_vote == final or final in state.runtimes[p.id].accepted_options
    )
    return round(backers / max(1, len(state.personas)), 3)
