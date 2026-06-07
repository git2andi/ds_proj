"""Compatibility layer for the old Simulator abstraction.

The active implementation uses utterance_generator.UtteranceGenerator plus
schemas.Persona.  This module intentionally contains no generation logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import Persona
from utterance_generator import UtteranceGenerator


@dataclass(slots=True)
class SimAgent:
    persona: Persona

    @property
    def id(self) -> str:
        return self.persona.id

    @property
    def name(self) -> str:
        return self.persona.name


__all__ = ["SimAgent", "UtteranceGenerator"]
