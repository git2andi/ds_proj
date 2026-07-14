"""Compact human-readable and structured run logging."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from config_loader import cfg
from models import ActionType, DialogueState, IssueKind, IssueStatus, RunOutcome, VoteStatus


class DialogueLogger:
    def __init__(self, topic: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic.strip()).strip("_")[:45] or "manual"
        self.run_id = f"{stamp}_{slug}"
        root = Path(cfg.root) / str(cfg.output.log_dir)
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_prompt(self, prompt: str, kind: str) -> str:
        if not bool(cfg.output.get("write_prompts", False)):
            return ""
        path = self.run_dir / str(cfg.output.prompt_file)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, "prompt": prompt}, ensure_ascii=False) + "\n")
        return str(path)

    def write_run(self, state: DialogueState, outcome: RunOutcome, *, seed: int) -> dict[str, str]:
        transcript = self.run_dir / str(cfg.output.transcript_file)
        json_path = self.run_dir / str(cfg.output.json_file)
        metrics_path = Path(cfg.root) / str(cfg.output.log_dir) / str(cfg.output.metrics_csv)
        metrics = metrics_for(state, outcome)
        transcript.write_text("\n".join(self._transcript_lines(state, outcome, seed, metrics)), encoding="utf-8")
        payload = {
            "run_id": self.run_id,
            "seed": seed,
            "scenario": _jsonable(state.scenario),
            "personas": [_jsonable(persona) for persona in state.personas],
            "runtimes": _jsonable(state.runtimes),
            "phase_history": list(state.phase_history),
            "active_issue": _jsonable(state.active_issue),
            "issue_history": _jsonable(state.issue_history),
            "group_stimulus": _jsonable(state.group_stimulus),
            "turns": [_turn_payload(turn) for turn in state.turns],
            "generation_attempts": _jsonable(state.generation_attempts),
            "validation_failures": dict(state.validation_failures),
            "distinct_supporters": _jsonable(state.public_supporters),
            "distinct_concern_raisers": _jsonable(state.public_concern_raisers),
            "switch_decisions": _jsonable(state.switch_decisions),
            "first_round_votes": dict(state.first_round_votes),
            "vote_records": _jsonable(state.vote_records),
            "votes": dict(state.votes),
            "outcome": _jsonable(outcome),
            "metrics": metrics,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_metrics(metrics_path, metrics, outcome)
        return {
            "dir": str(self.run_dir),
            "transcript": str(transcript),
            "json": str(json_path),
            "metrics_csv": str(metrics_path),
        }

    def _append_metrics(self, path: Path, metrics: dict[str, Any], outcome: RunOutcome) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "run_id": self.run_id,
            "outcome": outcome.status,
            "final_option": outcome.final_option or "",
            "participant_turns": metrics["turns"]["participant_turns"],
            "voluntary_turns": metrics["turns"]["voluntary_turns"],
            "moderator_turns": metrics["turns"]["moderator_turns"],
            "repairs": metrics["generation"]["repairs"],
            "dropped_turns": metrics["generation"]["dropped_turns"],
            "visible_switches": metrics["stances"]["visible_switches"],
            "public_acceptances": metrics["stances"]["public_acceptance_count"],
            "issues_opened": metrics["issues"]["opened"],
            "issues_resolved": metrics["issues"]["resolved"],
            "issues_stale": metrics["issues"]["stale"],
            "vote_failures": metrics["votes"]["non_valid_final_statuses"],
            "vote_protocol_degraded": metrics["votes"]["protocol_degraded"],
            "suppressed_repetitions": metrics["generation"]["suppressed_repetitions"],
            "llm_calls": metrics["tokens"]["llm_calls"],
            "tokens_in": metrics["tokens"]["input_tokens"],
            "tokens_out": metrics["tokens"]["output_tokens"],
        }
        fieldnames = list(row)
        mode = "a"
        write_header = not path.exists()
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                existing = handle.readline().strip().split(",")
            if existing != fieldnames:
                mode = "w"
                write_header = True
        with path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _transcript_lines(
        self,
        state: DialogueState,
        outcome: RunOutcome,
        seed: int,
        metrics: dict[str, Any],
    ) -> list[str]:
        lines = [
            f"# Dialogue run {self.run_id}",
            "",
            f"Topic: {state.scenario.topic}",
            f"Dialogue LLM: {cfg.llm.dialogue} ({cfg.llm.models[cfg.llm.dialogue]})",
            f"Random seed: {seed}",
            f"Moderator enabled: {bool((cfg.get('moderator', None) or {}).get('enabled', True))}",
            "Runtime: authoritative structured simulator actions; deterministic minimal validation",
            "",
            "## Options",
            "",
        ]
        for fact in state.scenario.shared_context:
            lines.append(f"- Shared: {fact}")
        for option in state.scenario.options:
            lines.append(f"- {option.public_line()}")
        lines += ["", "## Simulated users", ""]
        for persona in state.personas:
            params = persona.sim_params
            lines += [
                f"### {persona.name}",
                f"traits: engagement={params.engagement} verbosity={params.verbosity} directness={params.directness} stubbornness={params.stubbornness}",
                f"age/speech_style: {persona.age} — {persona.speech_style}",
                f"background: {persona.background}",
                f"private goal: {persona.private_goal}",
                f"initial preference: {persona.preferred_option}",
                f"hard blocker: {persona.hard_blocker}",
                "",
            ]
        lines += ["## Transcript", ""]
        for turn in state.turns:
            lines.append(f"**{turn.speaker_name}:** {turn.text}")
        lines += [
            "",
            "## Outcome",
            "",
            f"Status: {outcome.status}",
            f"Final option: {outcome.final_option}",
            f"Votes: {outcome.votes}",
            f"Reason: {outcome.reason}",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(_compact_transcript_metrics(metrics), ensure_ascii=False, indent=2),
            "```",
        ]
        return lines



def _compact_transcript_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the human transcript diagnostic without duplicating full run.json traces."""
    return {
        "turns": metrics["turns"],
        "traits": metrics["traits"],
        "generation": metrics["generation"],
        "issues": {
            key: value for key, value in metrics["issues"].items() if key != "provenance"
        },
        "stances": {
            key: value for key, value in metrics["stances"].items() if key != "switch_decisions"
        },
        "public_evidence": metrics["public_evidence"],
        "realization": {
            "within_target_rate": metrics["realization"]["within_target_rate"],
        },
        "narrowing": metrics["narrowing"],
        "votes": metrics["votes"],
        "coverage": metrics["coverage"],
        "tokens": metrics["tokens"],
    }

def metrics_for(state: DialogueState, outcome: RunOutcome) -> dict[str, Any]:
    participant_turns = state.participant_turns
    by_participant = Counter(turn.speaker_id for turn in participant_turns)
    voluntary_by_participant = Counter(turn.speaker_id for turn in participant_turns if turn.voluntary)
    average_words: dict[str, float] = {}
    average_voluntary_words: dict[str, float] = {}
    average_comparable_voluntary_words: dict[str, float] = {}
    comparable_voluntary_counts: dict[str, int] = {}
    word_budget_by_id: dict[str, list[dict[str, Any]]] = {}
    configured_traits: dict[str, Any] = {}
    comparable_actions = {
        ActionType.SUPPORT, ActionType.CONCERN, ActionType.ANSWER,
        ActionType.COMPARE, ActionType.COMMENT, ActionType.COMPROMISE,
    }
    for persona in state.personas:
        words = [turn.word_count for turn in participant_turns if turn.speaker_id == persona.id]
        voluntary_words = [
            turn.word_count for turn in participant_turns
            if turn.speaker_id == persona.id and turn.voluntary
        ]
        comparable_turns = [
            turn for turn in participant_turns
            if turn.speaker_id == persona.id and turn.voluntary
            and turn.action is not None and turn.action.act in comparable_actions
        ]
        comparable_words = [turn.word_count for turn in comparable_turns]
        average_words[persona.id] = round(sum(words) / len(words), 2) if words else 0.0
        average_voluntary_words[persona.id] = (
            round(sum(voluntary_words) / len(voluntary_words), 2) if voluntary_words else 0.0
        )
        average_comparable_voluntary_words[persona.id] = (
            round(sum(comparable_words) / len(comparable_words), 2) if comparable_words else 0.0
        )
        comparable_voluntary_counts[persona.id] = len(comparable_turns)
        word_budget_by_id[persona.id] = [
            {
                "turn": turn.index,
                "act": turn.action.act.value if turn.action else None,
                "intended_min": turn.intended_word_min,
                "intended_max": turn.intended_word_max,
                "realized": turn.word_count,
                "within_target": turn.intended_word_min <= turn.word_count <= turn.intended_word_max,
            }
            for turn in participant_turns if turn.speaker_id == persona.id
        ]
        params = persona.sim_params
        runtime = state.runtimes[persona.id]
        nonpreferred_acceptances = (
            len(runtime.public_acceptances - {persona.preferred_option}) if persona.hard_blocker else 0
        )
        nonpreferred_votes = int(
            persona.hard_blocker
            and state.votes.get(persona.id) not in {None, persona.preferred_option}
        )
        configured_traits[persona.id] = {
            "engagement": params.engagement,
            "verbosity": params.verbosity,
            "directness": params.directness,
            "stubbornness": params.stubbornness,
            "hard_blocker": persona.hard_blocker,
            "switch_opportunities": runtime.switch_opportunities,
            "visible_switches": runtime.visible_switches,
            "switch_rate_per_opportunity": round(
                runtime.visible_switches / runtime.switch_opportunities, 3
            ) if runtime.switch_opportunities else 0.0,
            "hard_blocker_nonpreferred_acceptances": nonpreferred_acceptances,
            "hard_blocker_nonpreferred_votes": nonpreferred_votes,
        }

    all_issues = [*state.issue_history]
    if state.active_issue:
        all_issues.append(state.active_issue)
    statuses = Counter(issue.status.value for issue in all_issues)
    issue_kinds = Counter(issue.kind.value for issue in all_issues)
    issue_outcomes = Counter(issue.outcome or "none" for issue in all_issues)
    question_issues = [issue for issue in all_issues if issue.kind is IssueKind.QUESTION]
    concern_issues = [issue for issue in all_issues if issue.kind is IssueKind.CONCERN]

    visible_switches = sum(runtime.visible_switches for runtime in state.runtimes.values())
    public_acceptance_count = sum(len(runtime.public_acceptances) for runtime in state.runtimes.values())
    repairs = sum(turn.repair_count for turn in participant_turns)
    action_counts = Counter(
        turn.action.act.value for turn in participant_turns if turn.action is not None
    )
    narrowing_turns = [
        turn for turn in participant_turns
        if turn.phase.value == "NARROWING" and turn.action is not None
    ]
    focused_narrowing = sum(
        bool(set(turn.action.option_focus) & set(turn.narrowing_options))
        or bool(turn.action.issue_id)
        or turn.action.act is ActionType.ANSWER
        or (
            turn.action.act is ActionType.CONCERN
            and state.runtimes[turn.speaker_id].public_preference in turn.action.option_focus
        )
        for turn in narrowing_turns
    )

    vote_statuses_by_round: dict[str, dict[str, str]] = {}
    for round_number, records in state.vote_records.items():
        vote_statuses_by_round[str(round_number)] = {
            participant_id: record.status.value for participant_id, record in records.items()
        }
    final_records = state.vote_records.get(state.vote_round, {})
    non_valid_final = sum(record.status is not VoteStatus.VALID for record in final_records.values())
    vote_switch_attempts = [
        attempt for attempt in state.generation_attempts
        if attempt.action.act is ActionType.VOTE and attempt.action.stance_update is not None
    ]
    vote_switch_accepted = sum(attempt.final_status == "accepted" for attempt in vote_switch_attempts)
    repetition_repairs = sum(
        attempt.repair_text is not None
        and any("repetition" in error.casefold() for error in attempt.validation_errors)
        for attempt in state.generation_attempts
    )

    return {
        "turns": {
            "participant_turns": len(participant_turns),
            "voluntary_turns": sum(turn.voluntary for turn in participant_turns),
            "mandatory_answers": sum(
                turn.mandatory and turn.action and turn.action.act.value == "answer"
                for turn in participant_turns
            ),
            "openings": sum(turn.action and turn.action.act.value == "opening" for turn in participant_turns),
            "votes": sum(turn.action and turn.action.act.value == "vote" for turn in participant_turns),
            "moderator_turns": state.stats.moderator_turns,
            "participant_turns_by_id": dict(by_participant),
            "voluntary_turns_by_id": dict(voluntary_by_participant),
            "average_words_by_id": average_words,
            "average_voluntary_words_by_id": average_voluntary_words,
            "comparable_voluntary_turns_by_id": comparable_voluntary_counts,
            "average_comparable_voluntary_words_by_id": average_comparable_voluntary_words,
            "action_counts": dict(action_counts),
        },
        "traits": configured_traits,
        "generation": {
            "repairs": repairs,
            "dropped_turns": state.stats.dropped_turns,
            "liveness_forced_turns": state.stats.liveness_forced_turns,
            "suppressed_repetitions": state.stats.suppressed_repetitions,
            "attempts": len(state.generation_attempts),
            "validation_failures": dict(state.validation_failures),
            "repair_rate": round(repairs / max(1, len(participant_turns)), 3),
            "drop_rate": round(state.stats.dropped_turns / max(1, len(state.generation_attempts)), 3),
            "repetition_repairs": repetition_repairs,
            "vote_switch_attempts": len(vote_switch_attempts),
            "vote_switch_accepted": vote_switch_accepted,
            "vote_switch_failures": len(vote_switch_attempts) - vote_switch_accepted,
        },
        "issues": {
            "opened": len(all_issues),
            "resolved": statuses[IssueStatus.RESOLVED.value],
            "stale": statuses[IssueStatus.STALE.value],
            "open": statuses[IssueStatus.OPEN.value],
            "follow_ups": sum(issue.follow_up_count for issue in all_issues),
            "by_kind": dict(issue_kinds),
            "outcomes": dict(issue_outcomes),
            "questions_answered": sum(issue.answered for issue in question_issues),
            "questions_resolved": sum(issue.status is IssueStatus.RESOLVED for issue in question_issues),
            "concerns_resolved": sum(issue.status is IssueStatus.RESOLVED for issue in concern_issues),
            "concerns_maintained": sum(issue.outcome == "maintained" for issue in concern_issues),
            "concerns_partially_addressed": sum(issue.outcome == "partially_addressed" for issue in concern_issues),
            "relevant_concern_responders": sum(len(issue.relevant_responder_ids) for issue in concern_issues),
            "same_attribute_mitigations": sum(issue.same_attribute_mitigation for issue in concern_issues),
            "provenance": [
                {
                    "id": issue.id,
                    "issue_key": list(issue.issue_key) if issue.issue_key else None,
                    "relevant_responders": sorted(issue.relevant_responder_ids),
                    "response_kinds": dict(issue.relevant_response_kinds),
                }
                for issue in concern_issues
            ],
        },
        "stances": {
            "visible_switches": visible_switches,
            "public_acceptance_count": public_acceptance_count,
            "public_preferences": {pid: runtime.public_preference for pid, runtime in state.runtimes.items()},
            "public_acceptances": {pid: sorted(runtime.public_acceptances) for pid, runtime in state.runtimes.items()},
            "switch_decisions": _jsonable(state.switch_decisions),
        },
        "public_evidence": {
            "distinct_supporters": {
                option_id: sorted(ids) for option_id, ids in state.public_supporters.items()
            },
            "distinct_concern_raisers": {
                option_id: sorted(ids) for option_id, ids in state.public_concern_raisers.items()
            },
        },
        "realization": {
            "word_budget_by_id": word_budget_by_id,
            "within_target_rate": round(
                sum(item["within_target"] for rows in word_budget_by_id.values() for item in rows)
                / max(1, sum(len(rows) for rows in word_budget_by_id.values())),
                3,
            ),
        },
        "narrowing": {
            "participant_turns": len(narrowing_turns),
            "focused_turns": focused_narrowing,
            "focus_adherence": round(focused_narrowing / len(narrowing_turns), 3) if narrowing_turns else 1.0,
        },
        "votes": {
            "first_round": dict(state.first_round_votes),
            "final": dict(state.votes),
            "records_by_round": vote_statuses_by_round,
            "non_valid_final_statuses": non_valid_final,
            "protocol_degraded": state.vote_protocol_degraded,
            "protocol_errors": list(state.vote_protocol_errors),
            "outcome": outcome.status,
            "final_option": outcome.final_option,
        },
        "coverage": {
            option_id: {
                "substantive_count": coverage.substantive_count,
                "participants": sorted(coverage.participant_ids),
                "actions": dict(coverage.action_types),
            }
            for option_id, coverage in state.coverage.items()
        },
        "tokens": {
            "llm_calls": state.stats.llm_calls + state.stats.setup_llm_calls,
            "runtime_llm_calls": state.stats.llm_calls,
            "setup_llm_calls": state.stats.setup_llm_calls,
            "repair_calls": state.stats.repair_calls,
            "input_tokens": state.stats.input_tokens,
            "output_tokens": state.stats.output_tokens,
        },
    }

def _turn_payload(turn: Any) -> dict[str, Any]:
    payload = _jsonable(turn)
    if not isinstance(payload, dict):
        return {"turn": payload}
    action = payload.get("action")
    # Every accepted participant bid is already a complete candidate and the
    # floor passes it through unchanged. Expose both labels in structured logs
    # so that authority can be audited without introducing duplicate runtime
    # state objects.
    payload["candidate_action"] = action
    payload["selected_action"] = action
    return payload


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Counter):
        return dict(value)
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
