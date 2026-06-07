"""
logger.py
---------
Writes readable transcripts and compact evaluation JSON for each dialogue.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from persona import Persona
    from simulator import Simulator
    from state import DialogueMemory


def _gini(values: list[int]) -> float:
    if not values or len(values) <= 1:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    numer = sum((2 * (i + 1) - n - 1) * value for i, value in enumerate(sorted_vals))
    return round(numer / (n * total), 4)


def _persona_block(personas: list["Persona"]) -> str:
    lines: list[str] = ["Participants", "-" * 50]
    for persona in personas:
        primary_tag = " (primary)" if persona.is_primary else ""
        lines.append(f"{persona.name}{primary_tag} -- {persona.role}")
        lines.append(
            f"  traits: open={persona.openness} consc={persona.conscientiousness} "
            f"extra={persona.extraversion} agree={persona.agreeableness} "
            f"neuro={persona.neuroticism} length={persona.response_length}"
        )
        if persona.goal:
            lines.append(f"  goal: {persona.goal}")
        if persona.backstory:
            lines.append(f"  backstory: {persona.backstory}")
        if persona.beliefs:
            beliefs = persona.beliefs
            accepts = [x for x in beliefs.acceptable if x != beliefs.preferred]
            accept_text = f", accepts {accepts}" if accepts else ""
            reject_text = f", rejects {beliefs.rejected}" if beliefs.rejected else ""
            lines.append(f"  prefers Option {beliefs.preferred}{accept_text}{reject_text}")
            lines.append(f"  concern: {beliefs.key_concern}")
            if beliefs.reasons:
                lines.append(f"  reasons: {' | '.join(beliefs.reasons)}")
            if beliefs.reservation:
                lines.append(f"  reservation: {beliefs.reservation}")
            if beliefs.would_reconsider_if:
                lines.append(f"  would reconsider if: {beliefs.would_reconsider_if}")
    lines.append("-" * 50)
    return "\n".join(lines)


def _transcript_quality(turn_records: list[dict[str, Any]]) -> dict[str, Any]:
    participant_records = [r for r in turn_records if not r.get("is_moderator", False)]
    total = max(1, len(participant_records))
    forbidden = [str(x).lower() for x in getattr(cfg.option_generation, "forbidden_terms", [])]
    question_turns = sum(1 for r in participant_records if "?" in r.get("text", ""))
    forbidden_turns = sum(
        1 for r in participant_records
        if any(term and term in r.get("text", "").lower() for term in forbidden)
    )
    structured_turns = sum(
        1 for r in participant_records
        if r.get("selected_reason") in {"vote", "targeted_holdout"}
    )
    repeated_phrase_turns = sum(
        1 for r in participant_records
        if any(code in r.get("verification_issues", []) for code in {"SELF_REPETITION", "SEMANTIC_POINT_REPEAT"})
    )
    return {
        "question_density": round(question_turns / total, 3),
        "forbidden_live_fact_turns": forbidden_turns,
        "structured_control_turns": structured_turns,
        "repeated_phrase_turns": repeated_phrase_turns,
    }


def _evaluation_summary(
    outcome: str,
    final_option: Optional[str],
    speaker_targets: dict[str, int],
    turn_records: list[dict[str, Any]],
) -> dict[str, Any]:
    participant_records = [r for r in turn_records if not r.get("is_moderator", False)]
    total_turns = max(1, len(participant_records))

    word_counts: dict[str, list[int]] = {}
    for record in participant_records:
        speaker = record["speaker"]
        text = record.get("text", "")
        count = len(re.sub(r"\s+", " ", text).split()) if text.strip() else 0
        word_counts.setdefault(speaker, []).append(count)

    avg_words = {
        speaker: round(sum(counts) / len(counts), 1)
        for speaker, counts in word_counts.items()
        if counts
    }
    adherence = {
        speaker: round(
            max(0.0, 1.0 - abs(avg - speaker_targets.get(speaker, 24)) / max(1, speaker_targets.get(speaker, 24))),
            2,
        )
        for speaker, avg in avg_words.items()
    }
    repairs = sum(1 for r in participant_records if r.get("repair_attempted", False))
    turn_counts = {speaker: len(counts) for speaker, counts in word_counts.items()}
    total_participant_turns = sum(turn_counts.values()) or 1
    return {
        "outcome_valid": outcome in {"success", "compromise_success", "force_close", "failed_no_viable_compromise"}
        and (final_option in {"A", "B", "C", "D"} or outcome == "failed_no_viable_compromise"),
        "avg_words_per_turn": avg_words,
        "target_words_per_turn": {speaker: speaker_targets.get(speaker, 24) for speaker in avg_words},
        "length_adherence_per_speaker": adherence,
        "repair_rate": round(repairs / total_turns, 3),
        "participation_ratio": {
            speaker: round(count / total_participant_turns, 3)
            for speaker, count in turn_counts.items()
        },
    }


class DialogueLogger:
    def __init__(self, dialogue_id: str, topic: str) -> None:
        self.dialogue_id = dialogue_id
        self.topic = topic
        log_dir = cfg.output.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.chat_file = os.path.join(log_dir, f"{dialogue_id}.txt")
        self.eval_file = os.path.join(log_dir, f"{dialogue_id}.eval.json")
        self._turn_records: list[dict[str, Any]] = []
        self._speaker_turn_counts: dict[str, int] = {}
        self._phase_turn_counts: dict[str, int] = {}

    def write_header(
        self,
        participant_names: list[str],
        personas: list["Persona"],
        opening_lines: list[str],
    ) -> None:
        header = (
            f"Dialogue ID : {self.dialogue_id}\n"
            f"Participants: {', '.join(participant_names)}\n"
            f"Topic       : {self.topic}\n"
            + "=" * 50 + "\n\n"
            + _persona_block(personas) + "\n\n"
            + "=" * 50 + "\n"
        )
        with open(self.chat_file, "w", encoding="utf-8") as file:
            file.write(header)
            for line in opening_lines:
                file.write(f"{line}\n")
            file.write("\n")

    def append_chat_line(self, line: str) -> None:
        with open(self.chat_file, "a", encoding="utf-8") as file:
            file.write(f"{line}\n\n")

    def buffer(
        self,
        line: str,
        selected_reason: str,
        state: "DialogueState",
        tokens_in: int = 0,
        tokens_out: int = 0,
        verification_result: Optional[dict[str, Any]] = None,
    ) -> None:
        if ":" not in line:
            return
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        is_moderator = speaker in cfg.EXCLUDED_SPEAKERS
        record: dict[str, Any] = {
            "turn_index": state.turn_index,
            "phase": state.phase,
            "speaker": speaker,
            "is_moderator": is_moderator,
            "text": text.strip(),
            "selected_reason": selected_reason,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        if verification_result is not None:
            record["verification_issues"] = [
                issue["code"] for issue in verification_result.get("issues", [])
                if issue["severity"] == "repair"
            ]
            record["repair_attempted"] = verification_result.get("repair_attempted", False)
            record["repair_succeeded"] = verification_result.get("repair_succeeded", False)
            if "structured_control" in verification_result:
                record["structured_control"] = verification_result["structured_control"]

        self._turn_records.append(record)
        if not is_moderator:
            self._speaker_turn_counts[speaker] = self._speaker_turn_counts.get(speaker, 0) + 1
            self._phase_turn_counts[state.phase] = self._phase_turn_counts.get(state.phase, 0) + 1

    def flush(
        self,
        outcome: str,
        sims: list["Simulator"],
        state: "DialogueState",
        memory: Optional["DialogueMemory"],
        setup_tokens_in: int,
        setup_tokens_out: int,
        dialogue_tokens_in: int,
        dialogue_tokens_out: int,
    ) -> None:
        total_in = setup_tokens_in + dialogue_tokens_in
        total_out = setup_tokens_out + dialogue_tokens_out

        with open(self.chat_file, "a", encoding="utf-8") as file:
            file.write(
                f"\n--- Outcome: {outcome} ---\n"
                f"--- Tokens : setup={setup_tokens_in}/{setup_tokens_out}  "
                f"dialogue={dialogue_tokens_in}/{dialogue_tokens_out}  "
                f"total={total_in}/{total_out} (in/out) ---\n"
            )

        final_candidate = state.preferred_option or state.candidate_option
        word_budgets = cfg.response_length.word_budgets
        speaker_targets = {
            sim.name: word_budgets[max(0, min(len(word_budgets) - 1, sim.persona.response_length - 1))]
            for sim in sims
        }
        evaluation = _evaluation_summary(
            outcome=outcome,
            final_option=final_candidate,
            speaker_targets=speaker_targets,
            turn_records=self._turn_records,
        )
        evaluation["transcript_quality"] = _transcript_quality(self._turn_records)
        evaluation["outcome_valid"] = self._outcome_valid(outcome, final_candidate, sims, state)

        repair_attempts = sum(1 for t in self._turn_records if t.get("repair_attempted", False))
        failed_repairs = sum(
            1 for t in self._turn_records
            if t.get("repair_attempted", False) and not t.get("repair_succeeded", True)
        )

        meta: dict[str, Any] = {
            "dialogue_id": self.dialogue_id,
            "topic": self.topic,
            "outcome": outcome,
            "tokens": {
                "setup_in": setup_tokens_in,
                "setup_out": setup_tokens_out,
                "dialogue_in": dialogue_tokens_in,
                "dialogue_out": dialogue_tokens_out,
                "total_in": total_in,
                "total_out": total_out,
            },
            "participation": {
                "gini": _gini(list(self._speaker_turn_counts.values())),
                "speaker_turn_counts": self._speaker_turn_counts,
                "phase_turn_counts": self._phase_turn_counts,
            },
            "dynamics": {
                "vote_flips_per_speaker": dict(state.vote_changes),
                "confirmation_rejection_count": state.confirmation_rejection_count,
                "rejected_options_by_speaker": dict(state.rejected_options_by_speaker),
                "explicit_votes": dict(getattr(state, "explicit_votes", {})),
                "explicit_accepts": {
                    name: sorted(options)
                    for name, options in getattr(state, "explicit_accepts", {}).items()
                },
                "explicit_rejects": getattr(state, "explicit_rejects", {}),
                "confirmation_rejected_options": sorted(getattr(state, "confirmation_rejected_options", set())),
                "final_candidate_option": state.candidate_option,
                "final_preferred_option": state.preferred_option,
                "outcome_reason": getattr(state, "outcome_reason", ""),
            },
            "personas": [sim.persona.as_dict() for sim in sims],
            "memory": self._memory_summary(memory),
            "turns": self._turn_records,
            "verification": {
                "repair_attempts": repair_attempts,
                "failed_repairs": failed_repairs,
            },
            "evaluation": evaluation,
        }

        with open(self.eval_file, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2, default=str)

    def _outcome_valid(
        self,
        outcome: str,
        final_candidate: Optional[str],
        sims: list["Simulator"],
        state: "DialogueState",
    ) -> bool:
        valid_letters = {"A", "B", "C", "D"}
        if outcome in {"success", "compromise_success"} and final_candidate in valid_letters:
            return all(
                getattr(state, "explicit_votes", {}).get(sim.name) == final_candidate
                or final_candidate in getattr(state, "explicit_accepts", {}).get(sim.name, set())
                for sim in sims
            )
        if outcome == "force_close":
            return final_candidate in valid_letters
        return outcome == "failed_no_viable_compromise"

    def _memory_summary(self, memory: Optional["DialogueMemory"]) -> dict[str, Any]:
        if memory is None:
            return {}
        return {
            "is_hard_blocker": {
                name: ps.is_true_hard_blocker for name, ps in memory.participants.items()
            },
            "stated_priorities": {
                name: ps.stated_priority for name, ps in memory.participants.items()
            },
            "public_preferences": {
                name: ps.public_preference for name, ps in memory.participants.items()
            },
        }

    @property
    def paths(self) -> tuple[str, str]:
        return self.chat_file, self.eval_file
