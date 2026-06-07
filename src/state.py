"""Dialogue state management."""

from __future__ import annotations

from typing import Optional

from act_parser import parse_dialogue_act
from schemas import (
    ActType,
    CompromiseProposal,
    DialogueState,
    MoveIntent,
    OpenQuestion,
    OptionCoverage,
    ParticipantRuntime,
    Persona,
    Phase,
    Scenario,
    TurnRecord,
)
from utils import OptionResolver, jaccard_text, normalise_ws


def initialise_state(scenario: Scenario, personas: list[Persona]) -> DialogueState:
    return DialogueState(
        scenario=scenario,
        personas=personas,
        phase=Phase.OPENING,
        runtimes={
            persona.id: ParticipantRuntime(
                persona_id=persona.id,
                current_preference=persona.preferred_option,
                accepted_options=set(),
                rejected_options=set(),
            )
            for persona in personas
        },
        coverage={option.id: OptionCoverage() for option in scenario.options},
    )


class StateTracker:
    def __init__(self, scenario: Scenario, personas: list[Persona]) -> None:
        self.resolver = OptionResolver(scenario.options)
        self.participant_names = {p.id: p.name for p in personas}

    def apply_turn(
        self,
        state: DialogueState,
        speaker_id: str,
        speaker_name: str,
        text: str,
        phase: Phase,
        intent: Optional[MoveIntent] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        validation_issues: Optional[list[str]] = None,
    ) -> TurnRecord:
        text = normalise_ws(text)
        act = parse_dialogue_act(speaker_id, speaker_name, text, self.resolver, self.participant_names, intent)
        state.turn_index += 1
        record = TurnRecord(
            index=state.turn_index,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=text,
            phase=phase,
            act=act,
            intent=intent,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            validation_issues=validation_issues or [],
        )
        state.turns.append(record)
        if speaker_id != "moderator":
            self._update_participant_runtime(state, record)
        self._update_coverage(state, record)
        self._update_questions(state, record)
        self._update_candidate_and_progress(state, record)
        return record

    def _update_participant_runtime(self, state: DialogueState, record: TurnRecord) -> None:
        runtime = state.runtimes[record.speaker_id]
        runtime.turn_count += 1
        runtime.last_spoke_turn = record.index
        runtime.already_said.append(record.text)
        if len(runtime.already_said) > 6:
            runtime.already_said = runtime.already_said[-6:]
        act = record.act
        if act.explicit_vote:
            runtime.explicit_vote = act.explicit_vote
            runtime.current_preference = act.explicit_vote
        for option_id in act.accepts:
            runtime.accepted_options.add(option_id)
            runtime.rejected_options.discard(option_id)
        for option_id in act.rejects:
            runtime.rejected_options.add(option_id)
            runtime.accepted_options.discard(option_id)
        if act.option_refs and act.act_type in {ActType.REACT, ActType.COMPARE, ActType.PUSH_BACK, ActType.ANSWER, ActType.VOTE}:
            runtime.stated_reasons.append(record.text)
            if len(runtime.stated_reasons) > 5:
                runtime.stated_reasons = runtime.stated_reasons[-5:]

    def _update_coverage(self, state: DialogueState, record: TurnRecord) -> None:
        if record.speaker_id == "moderator":
            return
        act = record.act
        for option_id in act.option_refs:
            if option_id in state.coverage:
                state.coverage[option_id].mentions += 1
                if _looks_like_reason(record.text):
                    state.coverage[option_id].reasons += 1
        for option_id in act.accepts:
            if option_id in state.coverage:
                state.coverage[option_id].acceptances += 1
        for option_id in act.rejects:
            if option_id in state.coverage:
                state.coverage[option_id].objections += 1
        if act.proposes_option:
            state.compromise_proposals.append(
                CompromiseProposal(record.speaker_id, act.proposes_option, record.text, record.index)
            )

    def _update_questions(self, state: DialogueState, record: TurnRecord) -> None:
        if record.act.act_type == ActType.ASK and record.act.question_target_id:
            state.open_questions.append(
                OpenQuestion(record.speaker_id, record.act.question_target_id, record.text, record.index)
            )
        # Mark the oldest direct question to this speaker as answered when they speak next.
        for question in state.open_questions:
            if not question.answered and question.target_id == record.speaker_id and record.index > question.turn_index:
                question.answered = True
                break
        state.open_questions = [q for q in state.open_questions if not q.answered]

    def _update_candidate_and_progress(self, state: DialogueState, record: TurnRecord) -> None:
        before = self._progress_signature(state)
        if record.act.proposes_option:
            state.candidate_option = record.act.proposes_option
        elif record.act.explicit_vote:
            state.candidate_option = self.leading_vote_option(state) or state.candidate_option
        after = self._progress_signature(state)
        if before == after:
            state.no_progress_count += 1
        else:
            state.no_progress_count = 0

    def leading_vote_option(self, state: DialogueState) -> Optional[str]:
        counts: dict[str, int] = {}
        for runtime in state.runtimes.values():
            if runtime.explicit_vote:
                counts[runtime.explicit_vote] = counts.get(runtime.explicit_vote, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)

    def _progress_signature(self, state: DialogueState) -> tuple:
        return (
            tuple(sorted((pid, rt.explicit_vote) for pid, rt in state.runtimes.items() if rt.explicit_vote)),
            tuple(sorted((pid, tuple(sorted(rt.accepted_options)), tuple(sorted(rt.rejected_options))) for pid, rt in state.runtimes.items())),
            tuple((oid, cov.mentions, cov.reasons, cov.objections, cov.acceptances) for oid, cov in sorted(state.coverage.items())),
            state.candidate_option,
        )


def _looks_like_reason(text: str) -> bool:
    lower = text.lower()
    if any(cue in lower for cue in ["because", "since", "as ", "but", "though", "price", "time", "risk", "concern", "fits", "better", "worse"]):
        return True
    return len(text.split()) >= 9


def own_recent_similarity(state: DialogueState, speaker_id: str, text: str) -> float:
    own = [turn.text for turn in reversed(state.turns) if turn.speaker_id == speaker_id]
    if not own:
        return 0.0
    return max(jaccard_text(text, prev) for prev in own[:3])
