"""
policy/personality_bias.py
---------------------------
PersonalityBias — Big Five trait values mapped to probabilistic behavioral biases.

Following McCrae & John (1992): traits are broad dimensions, not scripts.
These floats inform act_planner.plan_turn() (act sampling weights) and
turn_policy._score() (self-selection probability) but are NEVER exposed
as named phrases in prompts — that is the CLAUDE.md hard requirement.

Bias values are floats in [0.0, 1.0]; higher = stronger tendency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persona import Persona


@dataclass
class PersonalityBias:
    """Probabilistic behavioral tendencies derived from Big Five trait values."""
    talkativeness: float                 # E → propensity to speak up
    self_selection_propensity: float     # E+O+A blend → self-select likelihood
    concession_propensity: float         # A → CONCEDE / CONDITIONAL_ACCEPT
    objection_propensity: float          # low-A + N → ASSERT_OPPOSITION / CHALLENGE
    risk_salience: float                 # N → ASK_CLARIFICATION, concern tone
    detail_orientation: float            # C → ASSERT_SUPPORT with specifics
    consistency: float                   # C + low-O → holds position longer
    reframing_propensity: float          # O → PROPOSE_COMPROMISE, reframe
    clarification_propensity: float      # N + O + low-C → ASK_CLARIFICATION


def derive(persona: "Persona") -> PersonalityBias:
    """
    Map Big Five 1..5 to bias floats in [0.0, 1.0].
    Linear normalization: 1 → 0.0, 5 → 1.0.
    Trait interactions follow McCrae & John's orthogonal-factor model:
    each trait contributes independently; no caricature compounds.
    """
    def _n(v: int) -> float:
        return (max(1, min(5, int(v))) - 1) / 4.0

    e = _n(persona.extraversion)
    a = _n(persona.agreeableness)
    o = _n(persona.openness)
    c = _n(persona.conscientiousness)
    n = _n(persona.neuroticism)

    return PersonalityBias(
        talkativeness             = e,
        self_selection_propensity = e * 0.6 + o * 0.2 + a * 0.2,
        concession_propensity     = a * 0.7 + (1.0 - n) * 0.3,
        objection_propensity      = (1.0 - a) * 0.6 + n * 0.4,
        risk_salience             = n,
        detail_orientation        = c,
        consistency               = c * 0.6 + (1.0 - o) * 0.4,
        reframing_propensity      = o,
        clarification_propensity  = n * 0.5 + o * 0.3 + (1.0 - c) * 0.2,
    )
