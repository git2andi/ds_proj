"""Pure state fixtures for deterministic controller tests.

No LLM calls: scenarios, personas, dialogue states, turns, and parsed acts are
built directly from the domain models, using the same initialisation path as
the runner (dialogue.initialise_state) so tests exercise production state.
"""

from __future__ import annotations

from typing import NamedTuple

from dialogue import initialise_state
from models import (
    ActType,
    DialogueState,
    MoveIntent,
    OptionCard,
    Persona,
    Phase,
    Scenario,
    SimulatorParameters,
    TraitProfile,
    TurnRecord,
)
from parsing import (
    OptionResolver,
    active_blocker_option,
    resolve_addressee,
    visible_commitment,
    visible_question,
)

# Acts whose display label comes from the routed intent, not visible text.
_CONTEXTUAL_ACTS = {
    ActType.OPENING, ActType.ANSWER, ActType.PROCESS,
    ActType.COMPROMISE, ActType.VOTE, ActType.CLOSING,
}


class ParsedAct(NamedTuple):
    """Test-only display/trace parse: the fields the old DialogueAct exposed,
    recomputed from the canonical deterministic helpers (production no longer
    keeps a DialogueAct — evidence is the sole semantic authority, item 7)."""

    act_type: ActType
    option_refs: list[str]
    addressee_id: str | None
    question_scope: str | None
    question_target_id: str | None


def make_scenario() -> Scenario:
    """Three-option weekend scenario with distinct, alias-safe names."""
    return Scenario(
        topic="Choose a weekend activity",
        shared_context=["Only Saturday is available.", "Budget is 60 euros per person."],
        options=[
            OptionCard(
                id="A",
                name="Museum and Cafe Day",
                short_name="Museum",
                attrs={"cost": "24 euros", "duration": "4 hours"},
                upside="low effort and easy to adjust",
                concern="may feel too quiet",
            ),
            OptionCard(
                id="B",
                name="Lake Bike Ride",
                short_name="Bike Ride",
                attrs={"cost": "12 euros", "duration": "6 hours"},
                upside="active and inexpensive",
                concern="bad fit for someone tired",
            ),
            OptionCard(
                id="C",
                name="Escape Room",
                short_name="Escape Room",
                attrs={"cost": "32 euros", "duration": "2 hours"},
                upside="interactive and memorable",
                concern="less flexible once booked",
            ),
        ],
    )


def make_persona(
    pid: str,
    name: str,
    *,
    preferred: str = "A",
    engagement: float = 0.5,
    verbosity: float = 0.5,
    directness: float = 0.5,
    stubbornness: float = 0.5,
    switch_resistance: float = 0.5,
    rejection: str | None = None,
    rejection_reason: str = "",
    age: int = 33,
) -> Persona:
    return Persona(
        id=pid,
        name=name,
        traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=SimulatorParameters(
            engagement=engagement,
            verbosity=verbosity,
            directness=directness,
            stubbornness=stubbornness,
            switch_resistance=switch_resistance,
        ),
        background=f"{name} is a test participant.",
        private_goal="wants a workable group choice",
        preferred_options=[preferred],
        age=age,
        speech_style="relaxed practical wording",
        rejection=rejection,
        rejection_reason=rejection_reason,
    )


def make_state(personas: list[Persona] | None = None, scenario: Scenario | None = None) -> DialogueState:
    scenario = scenario or make_scenario()
    personas = personas or [
        make_persona("p1", "Mira", preferred="A"),
        make_persona("p2", "Jonas", preferred="B"),
        make_persona("p3", "Lea", preferred="C"),
    ]
    return initialise_state(scenario, personas)


def make_resolver(scenario: Scenario | None = None) -> OptionResolver:
    return OptionResolver((scenario or make_scenario()).options)


def parse_text(
    state: DialogueState,
    speaker_id: str,
    text: str,
    *,
    intent: MoveIntent | None = None,
    resolver: OptionResolver | None = None,
    previous_speaker_id: str | None = None,
) -> ParsedAct:
    resolver = resolver or make_resolver(state.scenario)
    names = {p.id: p.name for p in state.personas}
    check = text.replace("’", "'").replace("‘", "'")
    option_refs = resolver.ids_in_text(check)
    question = visible_question(
        text, speaker_id=speaker_id, participant_names=names,
        previous_speaker_id=previous_speaker_id,
    )
    scope = question[0] if question else None
    target = question[1] if question else None
    addressee = resolve_addressee(text, speaker_id, names) or target
    commit = visible_commitment(
        text, resolver, sanctioned_switch=bool(intent and intent.allow_vote_change)
    )
    if commit and commit[0] in {"vote", "accept"}:
        act_type = ActType.VOTE
    elif (commit and commit[0] == "reject") or active_blocker_option(check, resolver):
        act_type = ActType.CONCERN
    elif scope:
        act_type = ActType.ASK
    elif intent is not None and intent.act in _CONTEXTUAL_ACTS:
        act_type = intent.act
    else:
        act_type = ActType.COMMENT
    return ParsedAct(act_type, option_refs, addressee, scope, target)


def append_turn(
    state: DialogueState,
    speaker_id: str,
    text: str,
    *,
    intent: MoveIntent | None = None,
    phase: Phase = Phase.DISCUSSION,
    resolver: OptionResolver | None = None,
    blocked: bool = False,
) -> TurnRecord:
    """Append one participant turn the way the runner does (record + evidence).

    This mutates only the turn list and speaker bookkeeping — semantic state
    updates (votes, coverage, threads) stay with the observer under test.
    """
    previous = next((t.speaker_id for t in reversed(state.turns) if t.speaker_id != "moderator"), None)
    resolver = resolver or make_resolver(state.scenario)
    state.turn_index += 1
    rt = state.runtimes[speaker_id]
    rt.turn_count += 1
    rt.last_spoke_turn = state.turn_index
    rt.already_said.append(text)
    from tests.evidence_adapter import derive_evidence

    record = TurnRecord(
        index=state.turn_index,
        speaker_id=speaker_id,
        speaker_name=state.name_for(speaker_id),
        text=text,
        phase=phase,
        intent=intent,
        state_mutation_blocked=blocked,
        # Every appended turn carries the typed evidence contract, mirroring
        # the runtime pipeline (deterministic test-adapter semantics).
        evidence=derive_evidence(
            text, resolver,
            speaker_id=speaker_id,
            participant_names={p.id: p.name for p in state.personas},
            intent=intent,
            previous_speaker_id=previous,
        ),
    )
    state.turns.append(record)
    return record


def vote_intent(speaker_id: str, option_id: str) -> MoveIntent:
    return MoveIntent(
        speaker_id=speaker_id,
        act=ActType.VOTE,
        reason="cast a clear visible vote",
        option_focus=[option_id],
        length_hint="short",
    )
