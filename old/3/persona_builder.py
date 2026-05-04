"""
modules/persona_builder.py
--------------------------
Builds participant personas using a concept-first pipeline:

  1. RolePlanner assigns a topic-aligned role to each participant (one LLM call).
  2. PersonaBuilder generates a character concept + contextual focus notes (one LLM call per sim).
  3. Numeric traits are derived from the concept using group-constraint-aware sampling.
  4. Goals are generated last, with access to backstory + traits (one LLM call per sim).

Replaces template.py and role_planner.py.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import prompts as prompts
from config_loader import cfg
from modules.llm_client import get_llm_client


# ---------------------------------------------------------------------------
# Trait schema
# ---------------------------------------------------------------------------

SCALAR_TRAITS = [
    "friendliness",
    "assertiveness",
    "talkativeness",
    "initiative",
    "agreeableness",
    "flexibility",
    "patience",
    "response_length",
    "contrarian_pressure",
]

FOCUS_DIMENSIONS = ["cost", "comfort", "time", "safety", "flexibility_focus"]

# Maps qualitative levels to a sampling range (inclusive).
_LEVEL_RANGES: dict[str, tuple[int, int]] = {
    "low": (1, 2),
    "medium": (2, 4),
    "high": (4, 5),
}


# ---------------------------------------------------------------------------
# Persona dataclass (plain dict-compatible for backwards compat with generator)
# ---------------------------------------------------------------------------

@dataclass
class Persona:
    name: str
    role: str
    is_primary: bool
    goal: str
    backstory: str
    focus_notes: dict[str, str]          # topic-contextual note per focus dimension
    friendliness: int = 3
    assertiveness: int = 3
    talkativeness: int = 3
    initiative: int = 3
    agreeableness: int = 3
    flexibility: int = 3
    patience: int = 3
    response_length: int = 3
    contrarian_pressure: int = 3
    focus: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {t: getattr(self, t) for t in SCALAR_TRAITS}
        d.update({
            "name": self.name,
            "role": self.role,
            "is_primary": self.is_primary,
            "goal": self.goal,
            "backstory": self.backstory,
            "focus": self.focus,
            "focus_notes": self.focus_notes,
        })
        return d

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style access so generator.py can use persona.get() unchanged."""
        return self.as_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            object.__setattr__(self, key, value)


# ---------------------------------------------------------------------------
# Group constraint enforcement (guided mode)
# ---------------------------------------------------------------------------

def _apply_group_constraints(
    trait_sets: list[dict[str, int]],
) -> list[dict[str, int]]:
    """
    Resample individual trait values until all group-level constraints are met.
    Constraints are defined per-trait in config.personas.group_constraints.
    """
    constraints: Any = cfg.personas.get("group_constraints", None)
    if not constraints or cfg.personas.mode == "random":
        return trait_sets

    max_attempts = 50

    for trait_name in SCALAR_TRAITS:
        trait_cfg = getattr(constraints, trait_name, None)
        if trait_cfg is None:
            continue

        min_spread: Optional[int] = getattr(trait_cfg, "min_spread", None)
        no_all_below: Optional[int] = getattr(trait_cfg, "no_all_below", None)
        max_count_at_max: Optional[int] = getattr(trait_cfg, "max_count_at_max", None)

        for _ in range(max_attempts):
            values = [ts[trait_name] for ts in trait_sets]

            spread_ok = (max(values) - min(values)) >= min_spread if min_spread else True
            floor_ok = any(v > no_all_below for v in values) if no_all_below else True
            ceiling_ok = (
                sum(1 for v in values if v == 5) <= max_count_at_max
                if max_count_at_max else True
            )

            if spread_ok and floor_ok and ceiling_ok:
                break

            # Resample the sim furthest from satisfying constraints.
            idx = _worst_offender(values, min_spread, no_all_below, max_count_at_max)
            trait_sets[idx][trait_name] = random.randint(1, 5)

    return trait_sets


def _worst_offender(
    values: list[int],
    min_spread: Optional[int],
    no_all_below: Optional[int],
    max_count_at_max: Optional[int],
) -> int:
    """Return the index of the sim whose value is most responsible for constraint failure."""
    if max_count_at_max is not None and sum(1 for v in values if v == 5) > max_count_at_max:
        # Pick one of the sims with value 5.
        return next(i for i, v in enumerate(values) if v == 5)
    if no_all_below is not None and all(v <= no_all_below for v in values):
        return values.index(min(values))
    if min_spread is not None and (max(values) - min(values)) < min_spread:
        median = sorted(values)[len(values) // 2]
        return next(i for i, v in enumerate(values) if v == median)
    return 0


# ---------------------------------------------------------------------------
# Role planner (moved from role_planner.py)
# ---------------------------------------------------------------------------

class RolePlanner:
    """Assigns topic-aligned roles to all participants in one LLM call."""

    def __init__(self) -> None:
        self._llm = get_llm_client()

    def plan(self, topic: str, names: list[str]) -> dict[str, dict[str, Any]]:
        if not names:
            return {}

        fallback = {
            name: {"role": "participant", "is_primary": (i == 0)}
            for i, name in enumerate(names)
        }

        try:
            data = self._llm.generate_json(prompts.role_planning(topic, names))
            roles = data.get("roles", {})

            if not isinstance(roles, dict):
                return fallback

            cleaned: dict[str, dict[str, Any]] = {}
            for name in names:
                info = roles.get(name)
                if not isinstance(info, dict):
                    return fallback
                cleaned[name] = {
                    "role": str(info.get("role", "participant")).strip() or "participant",
                    "is_primary": bool(info.get("is_primary", False)),
                }

            # Enforce exactly one primary.
            primaries = [n for n, v in cleaned.items() if v["is_primary"]]
            if len(primaries) != 1:
                for n in cleaned:
                    cleaned[n]["is_primary"] = False
                cleaned[names[0]]["is_primary"] = True

            return cleaned

        except Exception as exc:
            print(f"!! Role planning error: {exc}")
            return fallback


# ---------------------------------------------------------------------------
# Persona builder
# ---------------------------------------------------------------------------

class PersonaBuilder:
    """
    Builds a fully populated Persona for one participant.

    Pipeline per sim:
      concept → traits (from concept + group constraints) → goal
    """

    def __init__(self, topic: str, dialogue_id: str) -> None:
        self.topic = topic
        self.dialogue_id = dialogue_id
        self._llm = get_llm_client()
        self._save_dir = os.path.join(cfg.output.config_dir, dialogue_id)
        os.makedirs(self._save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_all(
        self,
        names: list[str],
        role_plan: dict[str, dict[str, Any]],
    ) -> list[Persona]:
        """Build personas for all participants, enforcing group constraints."""
        raw_trait_sets = [
            self._random_traits() for _ in names
        ]

        if cfg.personas.mode == "guided":
            raw_trait_sets = _apply_group_constraints(raw_trait_sets)

        personas: list[Persona] = []
        for name, raw_traits in zip(names, raw_trait_sets):
            role_info = role_plan[name]
            persona = self._build_one(
                name=name,
                role=role_info["role"],
                is_primary=role_info["is_primary"],
                raw_traits=raw_traits,
            )
            self._save(persona)
            personas.append(persona)

        return personas

    def load_from_file(
        self,
        path: str,
        name: str,
        role: str,
        is_primary: bool,
    ) -> Persona:
        """Load a previously saved persona JSON and stamp new role/primary."""
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        persona = self._dict_to_persona(data, name, role, is_primary)
        self._save(persona)
        return persona

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _build_one(
        self,
        name: str,
        role: str,
        is_primary: bool,
        raw_traits: dict[str, int],
    ) -> Persona:
        backstory, focus_notes, trait_hints = self._generate_concept(name, role, is_primary)

        if cfg.personas.mode == "guided" and cfg.personas.generate_contextual_focus:
            traits = self._traits_from_hints(trait_hints, raw_traits)
        else:
            traits = raw_traits

        focus = {dim: random.randint(1, 5) for dim in FOCUS_DIMENSIONS}

        # Partial persona (no goal yet) for the goal prompt.
        partial = Persona(
            name=name,
            role=role,
            is_primary=is_primary,
            goal="",
            backstory=backstory,
            focus_notes=focus_notes,
            focus=focus,
            **traits,
        )
        partial.goal = self._generate_goal(partial)

        return partial

    def _generate_concept(
        self,
        name: str,
        role: str,
        is_primary: bool,
    ) -> tuple[str, dict[str, str], dict[str, dict[str, str]]]:
        """
        Returns (backstory, focus_notes, personality_hints).
        Falls back to empty values if the LLM call fails.
        """
        if not cfg.personas.generate_backstory:
            return "", {dim: "" for dim in FOCUS_DIMENSIONS}, {}

        try:
            data = self._llm.generate_json(
                prompts.character_concept(self.topic, name, role, is_primary)
            )
            backstory = str(data.get("backstory", "")).strip()
            raw_focus_notes: dict = data.get("focus_notes", {})
            raw_hints: dict = data.get("personality_hints", {})

            focus_notes = {
                dim: str(raw_focus_notes.get(dim, "")).strip()
                for dim in FOCUS_DIMENSIONS
            }
            return backstory, focus_notes, raw_hints

        except Exception as exc:
            print(f"!! Concept generation error for {name}: {exc}")
            return "", {dim: "" for dim in FOCUS_DIMENSIONS}, {}

    def _generate_goal(self, persona: Persona) -> str:
        traits_str = ", ".join(
            f"{t}={getattr(persona, t)}/5" for t in SCALAR_TRAITS
        )
        focus_str = ", ".join(
            f"{k}={persona.focus.get(k, 3)}/5" for k in FOCUS_DIMENSIONS
        )
        try:
            raw = self._llm.generate(
                prompts.goal_generation(
                    topic=self.topic,
                    name=persona.name,
                    role=persona.role,
                    is_primary=persona.is_primary,
                    backstory=persona.backstory,
                    traits_summary=traits_str,
                    focus_summary=focus_str,
                )
            ).strip()
            return raw or self._fallback_goal()
        except Exception as exc:
            print(f"!! Goal generation error for {persona.name}: {exc}")
            return self._fallback_goal()

    # ------------------------------------------------------------------
    # Trait helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _random_traits() -> dict[str, int]:
        return {t: random.randint(1, 5) for t in SCALAR_TRAITS}

    @staticmethod
    def _traits_from_hints(
        hints: dict[str, dict[str, str]],
        raw_traits: dict[str, int],
    ) -> dict[str, int]:
        """
        Map qualitative levels from the concept to numeric ranges,
        using raw_traits as fallback for any missing keys.
        """
        result = dict(raw_traits)
        for trait, info in hints.items():
            if trait not in SCALAR_TRAITS:
                continue
            level = str(info.get("level", "medium")).lower()
            lo, hi = _LEVEL_RANGES.get(level, (1, 5))
            result[trait] = random.randint(lo, hi)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, persona: Persona) -> None:
        path = os.path.join(self._save_dir, f"{persona.name.lower()}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(persona.as_dict(), f, indent=4)

    def _dict_to_persona(
        self,
        data: dict[str, Any],
        name: str,
        role: str,
        is_primary: bool,
    ) -> Persona:
        traits = {t: _clamp(data.get(t, 3)) for t in SCALAR_TRAITS}
        focus_raw = data.get("focus", {})
        focus = {dim: _clamp(focus_raw.get(dim, 3)) for dim in FOCUS_DIMENSIONS}
        return Persona(
            name=name,
            role=role,
            is_primary=is_primary,
            goal=str(data.get("goal", "")),
            backstory=str(data.get("backstory", "")),
            focus_notes=data.get("focus_notes", {dim: "" for dim in FOCUS_DIMENSIONS}),
            focus=focus,
            **traits,
        )

    @staticmethod
    def _fallback_goal() -> str:
        return "Support a practical outcome that fits their priorities."


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp(value: Any, lo: int = 1, hi: int = 5, default: int = 3) -> int:
    if not isinstance(value, int):
        return default
    return max(lo, min(hi, value))
