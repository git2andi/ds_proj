"""Compatibility layer.

The old policy module mixed hard-blocker sampling, speaker selection, and
repetition signals.  Routing is now implemented in router.TurnRouter and returns
MoveIntent objects instead of bare speakers.
"""

from __future__ import annotations

from router import TurnRouter
from utils import jaccard_text


def repetition_pressure(lines: list[str]) -> float:
    if len(lines) < 2:
        return 0.0
    return jaccard_text(lines[-1], lines[-2])


__all__ = ["TurnRouter", "repetition_pressure"]
