"""Simulator-level behavior controls.

This module translates broad OCEAN traits into explicit, tunable simulator
parameters and gives each simulated user a small private communicative-goal list
(the ``agenda`` fields).

Honest status of that list: it is a WEAK HINT SYSTEM, not agenda-based user
simulation. The router consults it only in quiet moments (reactive rules and
obligations always win), and observed runs leave most items pending at the end.
Do not describe this project as an "agenda-based user simulator" in docs or
write-ups. Cross-turn continuity comes from the stance/concern state tracked on
ParticipantRuntime (commitment strength, concern threads — see observer.py), not
from agenda execution; a real goal stack remains a possible future direction and
should not be implemented opportunistically.
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


def expected_turn_share(personas: list[Persona]) -> dict[str, float]:
    """Trait-derived target participation share per sim (issue 1).

    Engagement dominates, initiative and responsiveness tilt it. The constant
    floor keeps even a fully disengaged sim at a visible minimum share. This is
    the single contract used by both the speaker router and the evaluation's
    engagement-realization metrics.
    """
    raw = {
        p.id: 0.30 + 1.00 * p.sim_params.engagement
        + 0.40 * p.sim_params.initiative
        + 0.15 * p.sim_params.responsiveness
        for p in personas
    }
    total = sum(raw.values()) or 1.0
    return {pid: value / total for pid, value in raw.items()}


def build_initial_agenda(persona: Persona) -> list[AgendaItem]:
    """A small private communicative-goal list, tuned by parameters.

    This is a weak hint system, not a script and not agenda-based simulation: the
    router consumes an item only when nothing reactive is pending, and most items
    are expected to stay pending. Final voting is owned by the controller, so
    there is no VOTE item.
    """
    preferred = persona.preferred_option
    p = persona.sim_params
    agenda = [
        AgendaItem(
            act=ActType.SUPPORT,
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
    rejected = [oid for oid, stance in (persona.option_stances or {}).items() if int(stance.rank) == 0]
    if persona.rejection and persona.rejection not in rejected:
        rejected.append(persona.rejection)
    if rejected:
        oid = rejected[0]
        reason = (persona.option_stances.get(oid).reason_against if oid in persona.option_stances else persona.rejection_reason)
        agenda.insert(
            1,
            AgendaItem(
                act=ActType.CONCERN,
                option=oid,
                reason=f"raise the hard concern about {oid}" + (f": {reason}" if reason else ""),
                priority=0.95,
            ),
        )
    # Cooperative sims carry an intent to look for common ground.
    if p.compromise_threshold <= 0.45 and not rejected:
        agenda.append(
            AgendaItem(
                act=ActType.COMPROMISE,
                option=preferred,
                reason="look for a workable compromise the group could accept",
                priority=0.55,
            )
        )
    # Stubborn sims carry an intent to push back on a rival option.
    if p.stubbornness >= 0.6:
        agenda.append(
            AgendaItem(
                act=ActType.CONCERN,
                option=None,
                reason="object to the main rival option with a concrete concern",
                priority=0.6,
            )
        )
    return agenda


def refresh_agenda(persona: Persona, active_option: str | None) -> None:
    """Mark agenda items obsolete/blocked as the sim's stance evolves.

    Keeps continuity honest: once a sim has moved off an option, its pending
    items advocating that option no longer apply, and a hard blocker never carries
    a live compromise intent toward its rejected option.
    """
    for item in persona.agenda:
        if item.status != AgendaStatus.PENDING:
            continue
        if (
            active_option
            and item.option
            and item.option != active_option
            and item.act in {ActType.SUPPORT, ActType.ASK, ActType.COMPROMISE}
            and item.option == persona.preferred_option
        ):
            item.status = AgendaStatus.OBSOLETE
        if item.act == ActType.COMPROMISE and active_option in {oid for oid, stance in (persona.option_stances or {}).items() if int(stance.rank) == 0}:
            item.status = AgendaStatus.BLOCKED


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
