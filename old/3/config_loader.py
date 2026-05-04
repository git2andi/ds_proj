"""
config_loader.py
----------------
Loads config.yaml once at import time and exposes a typed `cfg` object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _Section:
    """Wraps a config dict so keys are accessible as attributes."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, _Section(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class Config(_Section):
    # Speakers that are never treated as participants in turn selection.
    EXCLUDED_SPEAKERS: frozenset[str] = frozenset({"Moderator"})

    def __init__(self, path: Path) -> None:
        data = _load(path)
        super().__init__(data)


_CONFIG_PATH = Path(__file__).parent / "config.yaml"
cfg = Config(_CONFIG_PATH)
