"""
config_loader.py
----------------
Loads config.yaml once at import time and exposes a typed `cfg` object.
All modules import `cfg` from here -- no magic numbers elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _Section:
    """Wraps a config dict so keys are accessible as attributes."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._raw = data
        for key, value in data.items():
            if isinstance(key, str):
                setattr(self, key, _Section(value) if isinstance(value, dict) else value)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        """Allow integer-keyed access, e.g. cfg.some_section[2]."""
        val = self._raw[key]
        return _Section(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class Config(_Section):
    # Speakers never treated as participants in turn logic.
    EXCLUDED_SPEAKERS: frozenset[str] = frozenset({"Moderator"})

    # Top-level YAML sections -- declared so Pylance can resolve cfg.<section>
    llm: _Section
    simulation: _Section
    turns: _Section
    consensus: _Section
    repetition: _Section
    personas: _Section
    participants: _Section
    output: _Section
    response_length: _Section
    turn_policy: _Section
    act_planner: _Section
    stubbornness: _Section
    phase_policy: _Section
    scoring: _Section
    moderation: _Section
    logging: _Section
    grounding: _Section
    evaluation: _Section
    prompt_budget: _Section

    def __init__(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        super().__init__(data)


# config.yaml lives in the project root, one level above src/
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
cfg = Config(_CONFIG_PATH)
