"""Shared fixtures for the test suite.

Adds src/ to sys.path so tests can import project modules directly.
Provides reusable scenario/state fixtures for validation and parsing tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable (mirrors what main.py does at startup).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
from models import (  # noqa: E402
    ActType,
    DialogueState,
    MoveIntent,
    OptionCard,
    OptionCoverage,
    Persona,
    ParticipantRuntime,
    Phase,
    Scenario,
    TraitProfile,
    TurnRecord,
    DialogueAct,
)
from parsing import OptionResolver  # noqa: E402


def _make_options() -> list[OptionCard]:
    return [
        OptionCard(id="A", name="Mountain Retreat", attrs={"cost": "$800", "days": "5"}, upside="Fresh air", tradeoff="Remote"),
        OptionCard(id="B", name="Beach Resort", attrs={"cost": "$1200", "days": "7"}, upside="Relaxing", tradeoff="Crowded"),
        OptionCard(id="C", name="City Tour", attrs={"cost": "$1000", "days": "4"}, upside="Culture", tradeoff="Tiring"),
        OptionCard(id="D", name="Road Trip", attrs={"cost": "$600", "days": "6"}, upside="Freedom", tradeoff="Long drives"),
    ]


def _make_traits(**overrides: float | int) -> TraitProfile:
    defaults = dict(
        openness=3, conscientiousness=3, extraversion=3, agreeableness=4,
        neuroticism=2, response_length=3, compromise_willingness=0.7,
        initiative=0.5, directness=0.5, detail=0.5,
    )
    defaults.update(overrides)
    return TraitProfile(**defaults)  # type: ignore[arg-type]


def _make_persona(pid: str, name: str, preferred: str = "A", **trait_kw: float | int) -> Persona:
    return Persona(
        id=pid, name=name, role="Tester", traits=_make_traits(**trait_kw),
        speech_style="casual", private_goal="find a good option",
        backstory="likes travel", main_concern="budget",
        preferred_option=preferred,
        acceptable_options=["A", "B"],
        soft_rejections=["C"],
        hard_rejections=[],
        reasons={"A": ["affordable"], "B": ["fun"]},
        reservation="none",
        reconsider_if="good argument",
        option_scores={"A": 5, "B": 4, "C": 3, "D": 2},
    )


@pytest.fixture()
def options() -> list[OptionCard]:
    return _make_options()


@pytest.fixture()
def resolver(options: list[OptionCard]) -> OptionResolver:
    return OptionResolver(options)


@pytest.fixture()
def participant_names() -> dict[str, str]:
    return {"p1": "Alice", "p2": "Bob", "p3": "Carol"}


@pytest.fixture()
def state(options: list[OptionCard]) -> DialogueState:
    scenario = Scenario(
        topic="Plan a group holiday trip",
        decision_kind="travel",
        opening_question="Where should we go?",
        options=options,
    )
    personas = [
        _make_persona("p1", "Alice", preferred="A"),
        _make_persona("p2", "Bob", preferred="B"),
        _make_persona("p3", "Carol", preferred="C"),
    ]
    st = DialogueState(scenario=scenario, personas=personas, phase=Phase.DISCUSSION)
    for p in personas:
        st.runtimes[p.id] = ParticipantRuntime(persona_id=p.id)
    for o in options:
        st.coverage[o.id] = OptionCoverage()
    return st


def make_turn(
    index: int,
    speaker_id: str,
    speaker_name: str,
    text: str,
    phase: Phase = Phase.DISCUSSION,
    act_type: ActType = ActType.REACT,
) -> TurnRecord:
    return TurnRecord(
        index=index,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        text=text,
        phase=phase,
        act=DialogueAct(speaker_id=speaker_id, text=text, act_type=act_type),
    )


def make_intent(
    speaker_id: str = "p1",
    act: ActType = ActType.REACT,
    option_focus: list[str] | None = None,
    length_hint: str = "medium",
) -> MoveIntent:
    return MoveIntent(
        speaker_id=speaker_id,
        act=act,
        reason="test",
        option_focus=option_focus or [],
        length_hint=length_hint,  # type: ignore[arg-type]
    )
