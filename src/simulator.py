"""Simulator-level behavior controls.

This module translates hidden OCEAN traits into the four explicit, tunable
simulator parameters. Per-simulator communicative reasons are stored in
OptionStance, while chat-level discussion coverage is handled by
DialogueState.discussion_agenda.
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
        directness=0.25 + 0.35 * consc01 + 0.25 * extra01 + 0.15 * (1.0 - agree01),
        stubbornness=(
            0.45 * (1.0 - agree01)
            + 0.25 * neuro01
            + 0.20 * (1.0 - open01)
            + 0.10 * consc01
        ),
    ).clipped()


def expected_turn_share(personas: list[Persona]) -> dict[str, float]:
    """Engagement-derived target participation share per sim.

    Engagement is the only participation-share parameter; the constant floor
    keeps even a fully disengaged sim at a visible minimum share. This is the
    single contract used by both the speaker router and the evaluation's
    engagement-realization metrics.
    """
    raw = {p.id: 0.30 + p.sim_params.engagement for p in personas}
    total = sum(raw.values()) or 1.0
    return {pid: value / total for pid, value in raw.items()}
