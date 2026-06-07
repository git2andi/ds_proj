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
    repetition: _Section
    personas: _Section
    output: _Section
    response_length: _Section
    voice: _Section
    argument_kit: _Section
    divergence: _Section
    memory: _Section
    turn_policy: _Section
    stubbornness: _Section
    grounding: _Section
    prompt_budget: _Section
    option_generation: _Section
    structured_control: _Section
    prompt_contracts: _Section
    verification: _Section
    closure: _Section

    def __init__(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        super().__init__(data)
        self._validate()

    def _validate(self) -> None:
        n = int(self.simulation.num_participants)
        min_n = int(self.simulation.min_participants)
        max_n = int(self.simulation.max_participants)
        if min_n > max_n:
            raise ValueError("simulation.min_participants must be <= simulation.max_participants.")
        if not (min_n <= n <= max_n):
            raise ValueError(
                f"simulation.num_participants must be between {min_n} and {max_n}; got {n}."
            )
        if int(self.option_generation.option_count) != 4:
            raise ValueError(
                "option_generation.option_count must stay 4 until OptionResolver/verifier support labels beyond A-D."
            )
        if int(self.option_generation.display_attribute_limit) <= 0:
            raise ValueError("option_generation.display_attribute_limit must be positive.")
        if int(self.structured_control.state_reject_excerpt_chars) <= 0:
            raise ValueError("structured_control.state_reject_excerpt_chars must be positive.")
        if int(self.structured_control.max_candidates_to_test) <= 0:
            raise ValueError("structured_control.max_candidates_to_test must be positive.")


# config.yaml lives in the project root, one level above src/
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
cfg = Config(_CONFIG_PATH)
