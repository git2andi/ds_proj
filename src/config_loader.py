"""Configuration loader.

`config.yaml` is the only location for tunable numeric parameters.  Modules import
`cfg` from here.  The wrapper deliberately stays small: attribute access for
readability, raw mapping preservation for logging/validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


class Section:
    def __init__(self, data: Any) -> None:
        self._raw = data
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and key.isidentifier():
                    setattr(self, key, Section(value) if isinstance(value, dict) else value)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(self._raw, dict):
            value = self._raw[key]
        elif isinstance(self._raw, list):
            value = self._raw[key]
        else:
            raise TypeError(f"Config section of type {type(self._raw).__name__} is not subscriptable")
        return Section(value) if isinstance(value, dict) else value

    def __contains__(self, key: Any) -> bool:
        return isinstance(self._raw, dict) and key in self._raw

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self._raw, dict):
            return self._raw.get(key, default)
        return default

    def items(self):
        if not isinstance(self._raw, dict):
            raise TypeError("Config section is not a mapping")
        return self._raw.items()


class Config(Section):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.root = path.parent
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        super().__init__(data)
        self._validate()

    def _require(self, dotted: str) -> Any:
        value: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"Missing required config key: {dotted}")
            value = value[part]
        return value

    @staticmethod
    def _validate_range(name: str, values: Iterable[Any], low: float, high: float) -> None:
        vals = list(values)
        if len(vals) != 2:
            raise ValueError(f"{name} must contain [min, max].")
        lo, hi = float(vals[0]), float(vals[1])
        if not (low <= lo <= hi <= high):
            raise ValueError(f"{name} must satisfy {low} <= min <= max <= {high}.")

    def _validate(self) -> None:
        required = [
            "llm.provider",
            "llm.models",
            "simulation.num_participants",
            "simulation.min_participants",
            "simulation.max_participants",
            "scenario.option_labels",
            "conversation.hard_max_turns_per_participant",
            "output.log_dir",
        ]
        for key in required:
            self._require(key)

        n = int(self.simulation.num_participants)
        min_n = int(self.simulation.min_participants)
        max_n = int(self.simulation.max_participants)
        if not (min_n <= n <= max_n):
            raise ValueError("simulation.num_participants must satisfy min <= n <= max.")

        labels = list(self.scenario.option_labels)
        if len(set(labels)) != len(labels):
            raise ValueError("scenario.option_labels must be unique.")

        for name, rng in self.personas.trait_ranges.items():
            self._validate_range(f"personas.trait_ranges.{name}", rng, int(self.personas.trait_min), int(self.personas.trait_max))
        for name, rng in self.personas.hard_blocker_trait_ranges.items():
            self._validate_range(f"personas.hard_blocker_trait_ranges.{name}", rng, int(self.personas.trait_min), int(self.personas.trait_max))


cfg = Config(Path(__file__).resolve().parents[1] / "config.yaml")
