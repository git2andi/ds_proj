"""Simulator-level behavior controls.

This module translates broad OCEAN traits into explicit, tunable simulator
parameters. Per-simulator communicative reasons are stored in OptionStance, while
chat-level discussion coverage is handled by DialogueState.discussion_agenda.
"""

from __future__ import annotations

from models import Persona, SimulatorParameters, TraitProfile

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
