"""
Configuration loader.

The simulator treats config.yaml as the single place for tunable numbers.
Modules import ``cfg`` from here.  The wrapper intentionally stays light: it
provides attribute access while preserving the raw mapping for validation and
logging.
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

    def as_dict(self) -> Any:
        return self._raw


class Config(Section):
    EXCLUDED_SPEAKERS = frozenset({"Moderator"})

    def __init__(self, path: Path) -> None:
        self.path = path
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

    def _validate_range(self, name: str, value: Iterable[Any], low: float, high: float) -> None:
        vals = list(value)
        if len(vals) != 2:
            raise ValueError(f"{name} must contain [min, max].")
        if not (low <= float(vals[0]) <= float(vals[1]) <= high):
            raise ValueError(f"{name} must satisfy {low} <= min <= max <= {high}.")

    def _validate(self) -> None:
        for key in [
            "llm.provider", "simulation.num_participants", "simulation.min_participants",
            "simulation.max_participants", "scenario.option_count", "scenario.option_labels",
            "conversation.hard_max_total_turns", "output.log_dir",
        ]:
            self._require(key)

        n = int(self.simulation.num_participants)
        min_n = int(self.simulation.min_participants)
        max_n = int(self.simulation.max_participants)
        if min_n > max_n:
            raise ValueError("simulation.min_participants must be <= simulation.max_participants.")
        if not (min_n <= n <= max_n):
            raise ValueError(f"simulation.num_participants must be between {min_n} and {max_n}; got {n}.")

        labels = list(self.scenario.option_labels)
        if len(labels) != int(self.scenario.option_count):
            raise ValueError("scenario.option_labels length must equal scenario.option_count.")
        if len(set(labels)) != len(labels):
            raise ValueError("scenario.option_labels must be unique.")

        for trait, bounds in self.personas.trait_ranges.items():
            self._validate_range(f"personas.trait_ranges.{trait}", bounds, 1, 5)
        for name, bounds in self.personas.cooperative_defaults.items():
            self._validate_range(f"personas.cooperative_defaults.{name}", bounds, 0.0, 1.0)

        if int(self.conversation.soft_min_total_turns) > int(self.conversation.hard_max_total_turns):
            raise ValueError("conversation.soft_min_total_turns must be <= hard_max_total_turns.")
        if not (0.0 <= float(self.consensus.majority_fraction_to_test) <= 1.0):
            raise ValueError("consensus.majority_fraction_to_test must be in [0, 1].")


def _find_config() -> Path:
    here = Path(__file__).resolve().parent
    # Look next to this module (src/), then the project root (parent of src/),
    # then the current working directory.  This keeps config discovery stable
    # regardless of where the process is launched from.
    for candidate in (here / "config.yaml", here.parent / "config.yaml", Path.cwd() / "config.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find config.yaml in src/, the project root, or the current working directory.")


cfg = Config(_find_config())

# Directory holding config.yaml.  Used to anchor relative paths (e.g. log_dir)
# so artifacts land in the project root no matter the working directory.
PROJECT_ROOT = cfg.path.parent
