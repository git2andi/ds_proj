"""Configuration loader.

`config.yaml` is the only location for tunable numeric parameters.  Modules import
`cfg` from here.  The wrapper deliberately stays small: attribute access for
readability, raw mapping preservation for logging/validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


def parse_preference_shape(value: Any) -> tuple[int, ...]:
    """Parse a preference partition from ``"2-1"`` or ``[2, 1]``."""
    if isinstance(value, str):
        raw_parts = value.split("-")
    elif isinstance(value, (list, tuple)):
        raw_parts = list(value)
    else:
        raise ValueError("preference shape must be a hyphenated string or integer list")
    if not raw_parts:
        raise ValueError("preference shape must not be empty")
    try:
        parts = tuple(int(part) for part in raw_parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid preference shape: {value!r}") from exc
    if any(part <= 0 for part in parts):
        raise ValueError("preference shape parts must be positive integers")
    if tuple(sorted(parts, reverse=True)) != parts:
        raise ValueError("preference shape parts must be ordered largest to smallest")
    return parts


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
            "conversation.max_discussion_turns_per_participant",
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

        self._validate_preference_distribution(min_n, max_n, len(labels), n)

    def _validate_preference_distribution(
        self,
        min_n: int,
        max_n: int,
        option_count: int,
        active_n: int,
    ) -> None:
        distribution = self._require("personas.preference_distribution")
        if not isinstance(distribution, dict):
            raise ValueError("personas.preference_distribution must be a mapping.")
        weights_by_size = distribution.get("shape_weights")
        if not isinstance(weights_by_size, dict):
            raise ValueError("personas.preference_distribution.shape_weights must be a mapping.")

        for size in range(min_n, max_n + 1):
            raw_weights = weights_by_size.get(size, weights_by_size.get(str(size)))
            if not isinstance(raw_weights, dict) or not raw_weights:
                raise ValueError(f"Missing preference shape weights for group size {size}.")
            total = 0.0
            for raw_shape, raw_weight in raw_weights.items():
                shape = parse_preference_shape(raw_shape)
                self._validate_preference_shape(shape, size, option_count)
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Weight for preference shape {raw_shape!r} must be numeric.") from exc
                if weight < 0:
                    raise ValueError(f"Weight for preference shape {raw_shape!r} must be non-negative.")
                total += weight
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"Preference shape weights for group size {size} must sum to 1.0, got {total}.")

        forced = distribution.get("forced_shape")
        if forced is not None:
            self._validate_preference_shape(parse_preference_shape(forced), active_n, option_count)

    @staticmethod
    def _validate_preference_shape(shape: tuple[int, ...], n: int, option_count: int) -> None:
        if sum(shape) != n:
            raise ValueError(f"Preference shape {shape} must sum to participant count {n}.")
        if len(shape) > option_count:
            raise ValueError(
                f"Preference shape {shape} needs {len(shape)} distinct options, but only {option_count} exist."
            )


cfg = Config(Path(__file__).resolve().parents[1] / "config.yaml")
