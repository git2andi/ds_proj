"""Pure state fixtures for deterministic controller tests.

No LLM calls: scenarios, personas, dialogue states, turns, and parsed acts are
built directly from the domain models, using the same initialisation path as
the runner (dialogue.initialise_state) so tests exercise production state.
"""

from __future__ import annotations

from dialogue import initialise_state
from models import (
    ActType,
    DialogueAct,
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
from parsing import OptionResolver, parse_dialogue_act


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
) -> DialogueAct:
    resolver = resolver or make_resolver(state.scenario)
    persona = state.persona_by_id(speaker_id)
    return parse_dialogue_act(
        speaker_id=speaker_id,
        speaker_name=persona.name,
        text=text,
        resolver=resolver,
        participant_names={p.id: p.name for p in state.personas},
        intent=intent,
        previous_speaker_id=previous_speaker_id,
    )


def append_turn(
    state: DialogueState,
    speaker_id: str,
    text: str,
    *,
    intent: MoveIntent | None = None,
    act: DialogueAct | None = None,
    phase: Phase = Phase.DISCUSSION,
    resolver: OptionResolver | None = None,
    blocked: bool = False,
) -> TurnRecord:
    """Append one participant turn the way the runner does (parse + record).

    The act is parsed from the visible text unless an explicit act is given.
    This mutates only the turn list and speaker bookkeeping — semantic state
    updates (votes, coverage, threads) stay with the observer under test.
    """
    previous = next((t.speaker_id for t in reversed(state.turns) if t.speaker_id != "moderator"), None)
    if act is None:
        act = parse_text(
            state, speaker_id, text, intent=intent, resolver=resolver, previous_speaker_id=previous
        )
    state.turn_index += 1
    rt = state.runtimes[speaker_id]
    rt.turn_count += 1
    rt.last_spoke_turn = state.turn_index
    rt.already_said.append(text)
    record = TurnRecord(
        index=state.turn_index,
        speaker_id=speaker_id,
        speaker_name=state.name_for(speaker_id),
        text=text,
        phase=phase,
        act=act,
        intent=intent,
        state_mutation_blocked=blocked,
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
