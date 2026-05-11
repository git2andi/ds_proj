"""
config_loader.py
----------------
Loads config.yaml once at import time and exposes a typed `cfg` object.
All modules import `cfg` from here — no magic numbers elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _Section:
    """Wraps a config dict so keys are accessible as attributes."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, _Section(value) if isinstance(value, dict) else value)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class Config(_Section):
    # Speakers never treated as participants in turn logic.
    EXCLUDED_SPEAKERS: frozenset[str] = frozenset({"Moderator"})

    # Top-level YAML sections — declared so Pylance can resolve cfg.<section>
    llm: _Section
    simulation: _Section
    turns: _Section
    consensus: _Section
    repetition: _Section
    personas: _Section
    participants: _Section
    output: _Section

    def __init__(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        super().__init__(data)


# config.yaml lives in the project root, one level above src/
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
cfg = Config(_CONFIG_PATH)
