"""Simulator-level behavior controls.

This module translates broad OCEAN traits into explicit, tunable simulator
parameters and creates a small agenda for each simulated user. The agenda is not
a script; it is a set of pending communicative goals that the routing policy may
consume when the dialogue context makes them useful.
"""

from __future__ import annotations

from models import ActType, AgendaItem, AgendaStatus, Persona, SimulatorParameters, TraitProfile


def derive_simulator_parameters(traits: TraitProfile) -> SimulatorParameters:
    open01 = (traits.openness - 1) / 4
    consc01 = (traits.conscientiousness - 1) / 4
    extra01 = (traits.extraversion - 1) / 4
    agree01 = (traits.agreeableness - 1) / 4
    neuro01 = (traits.neuroticism - 1) / 4

    return SimulatorParameters(
        engagement=0.25 + 0.60 * extra01 + 0.15 * consc01,
        verbosity=0.20 + 0.55 * extra01 + 0.25 * open01,
        initiative=0.20 + 0.50 * extra01 + 0.30 * open01,
        responsiveness=0.30 + 0.45 * agree01 + 0.25 * consc01,
        stubbornness=0.10 + 0.60 * (1.0 - agree01) + 0.30 * neuro01,
        directness=0.25 + 0.35 * consc01 + 0.25 * extra01 + 0.15 * (1.0 - agree01),
        compromise_threshold=1.0 - traits.compromise_willingness,
    ).clipped()


def build_initial_agenda(persona: Persona) -> list[AgendaItem]:
    preferred = persona.preferred_option
    agenda = [
        AgendaItem(
            act=ActType.BUILD,
            option=preferred,
            reason="state one grounded reason for the initial preference",
            priority=1.0,
        ),
        AgendaItem(
            act=ActType.ASK,
            option=preferred,
            reason="ask about a practical constraint that could affect the choice",
            priority=0.7,
        ),
        AgendaItem(
            act=ActType.COMPARE,
            option=preferred,
            reason="compare the preferred option with a realistic alternative",
            priority=0.65,
        ),
    ]
    if persona.rejection:
        agenda.insert(
            1,
            AgendaItem(
                act=ActType.CHALLENGE,
                option=persona.rejection,
                reason=f"raise the hard concern about {persona.rejection}: {persona.rejection_reason}",
                priority=0.95,
            ),
        )
    return agenda


def next_agenda_item(persona: Persona) -> tuple[int, AgendaItem] | None:
    pending = [
        (idx, item)
        for idx, item in enumerate(persona.agenda)
        if item.status == AgendaStatus.PENDING
    ]
    if not pending:
        return None
    return max(pending, key=lambda pair: pair[1].priority)


def mark_agenda_done(persona: Persona, agenda_index: int | None) -> None:
    if agenda_index is None:
        return
    if 0 <= agenda_index < len(persona.agenda):
        persona.agenda[agenda_index].status = AgendaStatus.DONE
