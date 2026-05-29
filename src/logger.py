"""
logger.py
---------
DialogueLogger -- writes two files per dialogue:

  .txt          chat with persona block at the top, Outcome/Tokens footer.
  .eval.json    everything else for evaluation:
                  - dialogue metadata + outcome + tokens
                  - per-speaker / per-phase turn counts + Gini
                  - vote flips, confirmation rejections
                  - Fisher (1970) ratios  (eval-only)
                  - Stage 5: justification / reciprocity / reflexivity rates
                  - challenges + answered challenges
                  - full personas (traits + beliefs + speech signature)
                  - structured per-turn trace (acts, addressees, stance updates,
                    challenge edges)
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, TYPE_CHECKING

from config_loader import cfg
from reasoning import deliberation_metrics, evaluation_summary, fisher_ratios

if TYPE_CHECKING:
    from orchestrator import DialogueState
    from persona import Persona
    from simulator import Simulator
    from state import StructuredState


def _gini(values: list[int]) -> float:
    """Gini coefficient of participation (0 = equal, 1 = one speaker dominates)."""
    if not values or len(values) <= 1:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    numer = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    return round(numer / (n * total), 4)


def _persona_block(personas: list["Persona"]) -> str:
    """Human-readable participant introductions for the top of the chat file."""
    lines: list[str] = ["Participants"]
    lines.append("-" * 50)
    for p in personas:
        primary_tag = " (primary)" if p.is_primary else ""
        lines.append(f"{p.name}{primary_tag} -- {p.role}")
        lines.append(
            f"  traits: open={p.openness} consc={p.conscientiousness} "
            f"extra={p.extraversion} agree={p.agreeableness} "
            f"neuro={p.neuroticism} length={p.response_length}"
        )
        sig = p.speech_signature()
        lines.append(
            f"  voice: hedge={sig.hedge_propensity:.2f} direct={sig.directness:.2f} "
            f"thinkaloud={sig.thinkaloud_propensity:.2f} detail={sig.detail_orientation:.2f}"
        )
        if p.goal:
            lines.append(f"  goal: {p.goal}")
        if p.backstory:
            lines.append(f"  backstory: {p.backstory}")
        if p.beliefs:
            b = p.beliefs
            accept_others = [x for x in b.acceptable if x != b.preferred]
            accept_str = f", accepts {accept_others}" if accept_others else ""
            rejects_str = f", rejects {b.rejected}" if b.rejected else ""
            lines.append(f"  prefers Option {b.preferred}{accept_str}{rejects_str}")
            lines.append(f"  concern: {b.key_concern}")
            if b.reasons:
                lines.append(f"  reasons: {' | '.join(b.reasons)}")
            if b.reservation:
                lines.append(f"  reservation: {b.reservation}")
            if b.would_reconsider_if:
                lines.append(f"  would reconsider if: {b.would_reconsider_if}")
    lines.append("-" * 50)
    return "\n".join(lines)


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

    # ------------------------------------------------------------------

    def write_header(self, participant_names: list[str],
                     personas: list["Persona"], opening_lines: list[str]) -> None:
        header = (
            f"Dialogue ID : {self.dialogue_id}\n"
            f"Participants: {', '.join(participant_names)}\n"
            f"Topic       : {self.topic}\n"
            + "=" * 50 + "\n\n"
            + _persona_block(personas) + "\n\n"
            + "=" * 50 + "\n"
        )
        with open(self.chat_file, "w", encoding="utf-8") as f:
            f.write(header)
            for line in opening_lines:
                f.write(f"{line}\n")
            f.write("\n")

    def append_chat_line(self, line: str) -> None:
        with open(self.chat_file, "a", encoding="utf-8") as f:
            f.write(f"{line}\n\n")

    def buffer(self, line: str, selected_reason: str, state: "DialogueState",
               tokens_in: int = 0, tokens_out: int = 0,
               verification_result: Optional[dict] = None) -> None:
        if ":" not in line:
            return
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        is_moderator = speaker == "Moderator"

        record: dict = {
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
                i["code"] for i in verification_result.get("issues", [])
                if i["severity"] == "repair"
            ]
            record["repair_attempted"] = verification_result.get("repair_attempted", False)
            record["repair_succeeded"] = verification_result.get("repair_succeeded", False)

        self._turn_records.append(record)
        if not is_moderator:
            self._speaker_turn_counts[speaker] = self._speaker_turn_counts.get(speaker, 0) + 1
            self._phase_turn_counts[state.phase] = self._phase_turn_counts.get(state.phase, 0) + 1

    # ------------------------------------------------------------------

    def flush(
        self,
        outcome: str,
        sims: list["Simulator"],
        state: "DialogueState",
        structured: Optional["StructuredState"],
        setup_tokens_in: int,
        setup_tokens_out: int,
        dialogue_tokens_in: int,
        dialogue_tokens_out: int,
    ) -> None:
        total_in  = setup_tokens_in + dialogue_tokens_in
        total_out = setup_tokens_out + dialogue_tokens_out

        # --- .txt footer ------------------------------------------------
        with open(self.chat_file, "a", encoding="utf-8") as f:
            f.write(
                f"\n--- Outcome: {outcome} ---\n"
                f"--- Tokens : setup={setup_tokens_in}/{setup_tokens_out}  "
                f"dialogue={dialogue_tokens_in}/{dialogue_tokens_out}  "
                f"total={total_in}/{total_out} (in/out) ---\n"
            )

        # --- .eval.json -------------------------------------------------
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
                    k: sorted(v) for k, v in getattr(state, "explicit_accepts", {}).items()
                },
                "explicit_rejects": getattr(state, "explicit_rejects", {}),
                "confirmation_rejected_options": sorted(getattr(state, "confirmation_rejected_options", set())),
                "compromise_terms": getattr(state, "compromise_terms", {}),
                "final_candidate_option": state.candidate_option,
                "final_preferred_option": state.preferred_option,
                "outcome_reason": getattr(state, "outcome_reason", ""),
            },
            "personas": [s.persona.as_dict() for s in sims],
            "turns": self._turn_records,
        }

        # --- Verification aggregate stats ----------------------------------
        repair_attempts = sum(
            1 for t in self._turn_records if t.get("repair_attempted", False)
        )
        failed_repairs = sum(
            1 for t in self._turn_records
            if t.get("repair_attempted", False) and not t.get("repair_succeeded", True)
        )
        meta["verification"] = {
            "repair_attempts": repair_attempts,
            "failed_repairs": failed_repairs,
        }

        # --- Post-dialogue evaluation metrics ----------------------------------
        word_budgets = cfg.response_length.word_budgets
        speaker_targets = {
            s.name: word_budgets[max(0, min(4, s.persona.response_length - 1))]
            for s in sims
        }
        final_opt = state.preferred_option or state.candidate_option
        meta["evaluation"] = evaluation_summary(
            outcome=outcome,
            final_option=final_opt,
            speaker_targets=speaker_targets,
            turn_records=self._turn_records,
        )

        # Live control no longer uses StanceTable consensus as the source of
        # truth. Outcome validity must therefore be evaluated against the
        # explicit control state: final option is valid and, for success /
        # compromise_success, every participant either voted for or explicitly
        # accepted it.
        valid_letters = {"A", "B", "C", "D"}
        final_candidate = state.preferred_option or state.candidate_option
        if outcome in {"success", "compromise_success"} and final_candidate in valid_letters:
            all_accepted = True
            for sim in sims:
                name = sim.name
                voted = getattr(state, "explicit_votes", {}).get(name) == final_candidate
                accepted = final_candidate in getattr(state, "explicit_accepts", {}).get(name, set())
                if not (voted or accepted):
                    all_accepted = False
                    break
            meta["evaluation"]["outcome_valid"] = all_accepted
        elif outcome == "force_close":
            meta["evaluation"]["outcome_valid"] = final_candidate in valid_letters
        elif outcome == "failed_no_viable_compromise":
            meta["evaluation"]["outcome_valid"] = True
        else:
            meta["evaluation"]["outcome_valid"] = False

        if structured is not None:
            names = [s.name for s in sims]
            meta["fisher_ratios"] = fisher_ratios(structured)              # eval-only
            meta["deliberation"] = deliberation_metrics(structured, names) # eval-only; no live control
            meta["consensus_state_final"] = structured.consensus_state
            meta["is_hard_blocker"] = {
                name: ps.is_true_hard_blocker
                for name, ps in structured.participants.items()
            }
            meta["public_preferences_final"] = {
                name: ps.public_preference
                for name, ps in structured.participants.items()
            }
            meta["stated_priorities"] = {
                name: ps.stated_priority
                for name, ps in structured.participants.items()
            }
            meta["challenges"] = [
                {
                    "turn_id": c.challenge_turn_id,
                    "challenger": c.challenger,
                    "target": c.target,
                    "target_option": c.target_option,
                    "answered_turn_id": c.answered_turn_id,
                }
                for c in structured.discourse.challenges
            ]
            meta["structured_turns"] = [
                {
                    "id": t.turn_id,
                    "speaker": t.speaker,
                    "phase": t.phase,
                    "act": t.dialogue_act.value,
                    "mentioned_options": t.mentioned_options,
                    "is_question": t.is_question,
                    "addressees": t.addressees,
                    "reply_to": t.reply_to,
                    "answers_question_id": t.answers_question_id,
                    "challenges_turn_id": t.challenges_turn_id,
                    "answered_challenge_id": t.answered_challenge_id,
                    "stance_updates": [
                        {"option": su.option, "stance": su.stance,
                         "confidence": su.confidence, "condition": su.condition}
                        for su in t.stance_updates
                    ],
                }
                for t in structured.turns
            ]

        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    @property
    def paths(self) -> tuple[str, str]:
        return self.chat_file, self.eval_file
