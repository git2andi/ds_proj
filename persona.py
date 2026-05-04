"""
persona.py
----------
Persona dataclass and PersonaBuilder.

Pipeline per participant (revised):
  1. Sample numeric traits randomly (1–5) — done locally, no LLM
  2. Lightweight group diversity check across all sampled trait sets
  3. One LLM call per participant — receives the numeric traits translated
     to plain English so backstory and goal are written to match them
  4. Persona dataclass assembled from sampled traits + LLM-written text

Key design principle: traits are the ground truth. The LLM writes
*around* them, not the other way around. A friendliness-5 persona gets a
backstory and goal that reflect warmth because the prompt describes it as
"extremely warm and openly supportive", not just a label "high".

LLM calls per dialogue setup:
  1 (options) + 1 (roles) + N (one persona concept per participant) = N + 2
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import prompts
from config_loader import cfg
from llm_client import get_llm_client


# ---------------------------------------------------------------------------
# Trait definitions
# ---------------------------------------------------------------------------

TRAITS = [
    "assertiveness",    # 1=hedging, 5=blunt and direct
    "friendliness",     # 1=cold/blunt, 5=warm and openly supportive
    "talkativeness",    # 1=terse, 5=elaborate
    "agreeableness",    # 1=challenging, 5=consensus-seeking
    "patience",         # 1=easily frustrated, 5=lets discussion breathe
    "contrarian",       # 1=goes with the flow, 5=probes weaknesses
    "response_length",  # 1=brief, 5=detailed (drives speaking style rule)
]

# Human-readable descriptions injected into the LLM persona concept prompt.
# Maps each numeric value to natural language so the LLM writes a backstory
# and goal that actually fit the character rather than generic text.
_TRAIT_DESCRIPTIONS: dict[str, dict[int, str]] = {
    "assertiveness": {
        1: "rarely voices opinions directly, tends to soften or withhold views",
        2: "cautious about asserting opinions, often hedges",
        3: "states views when asked, neither pushes hard nor holds back",
        4: "clearly voices opinions and does not soften disagreements",
        5: "very direct and outspoken, says exactly what they think",
    },
    "friendliness": {
        1: "blunt and detached, not warm or encouraging toward others",
        2: "somewhat reserved, polite but not particularly warm",
        3: "friendly in a neutral, unremarkable way",
        4: "warm and encouraging, acknowledges others before sharing views",
        5: "extremely warm and openly supportive, values group harmony",
    },
    "talkativeness": {
        1: "very terse, speaks only when necessary",
        2: "brief, makes one point without elaborating",
        3: "speaks a moderate amount",
        4: "tends to elaborate and think out loud",
        5: "talks a lot, gives detailed well-developed thoughts",
    },
    "agreeableness": {
        1: "strongly challenges points that do not convince them",
        2: "often pushes back, does not easily validate others",
        3: "sometimes agrees, sometimes challenges",
        4: "looks for common ground, tends to validate others",
        5: "very consensus-seeking, avoids conflict",
    },
    "patience": {
        1: "gets visibly frustrated when discussions repeat or stall",
        2: "somewhat impatient with circular discussion",
        3: "patient in a neutral way",
        4: "calm and willing to let the discussion unfold",
        5: "very patient, happy to let others work through their thinking",
    },
    "contrarian": {
        1: "naturally goes along with the emerging consensus",
        2: "rarely challenges the group direction",
        3: "occasionally questions assumptions",
        4: "frequently probes weaknesses and raises overlooked trade-offs",
        5: "instinctively challenges the obvious choice, always seeks counter-arguments",
    },
    "response_length": {
        1: "speaks in very short single reactions",
        2: "speaks briefly, one clean point at a time",
        3: "speaks in moderate length turns",
        4: "gives fairly developed responses with reasoning",
        5: "gives long, thorough responses",
    },
}

# Hard speaking length constraint injected into each turn prompt
_STYLE_RULE: dict[int, str] = {
    1: "Maximum 1 sentence. Short reactions only — e.g. 'Yeah, fair point.' Never elaborate.",
    2: "Maximum 1 sentence. Make one clean point without elaborating or follow-up thoughts.",
    3: "Maximum 2 sentences. One point plus a short reason. No padding.",
    4: "Up to 3 sentences. Elaborate with context, reasoning, and optionally a follow-up thought.",
    5: "Up to 4 sentences. Think out loud with thorough, well-reasoned responses.",
}


# ---------------------------------------------------------------------------
# Persona dataclass
# ---------------------------------------------------------------------------

@dataclass
class Persona:
    name: str
    role: str
    is_primary: bool
    goal: str
    backstory: str

    # Numeric traits (1–5)
    assertiveness: int = 3
    friendliness: int = 3
    talkativeness: int = 3
    agreeableness: int = 3
    patience: int = 3
    contrarian: int = 3
    response_length: int = 3

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "is_primary": self.is_primary,
            "goal": self.goal,
            "backstory": self.backstory,
            **{t: getattr(self, t) for t in TRAITS},
        }

    def style_rule(self) -> str:
        level = max(1, min(5, self.response_length))
        return _STYLE_RULE[level]

    def personality_summary(self) -> str:
        """
        Plain-English summary for the turn prompt.
        Must match the descriptions the backstory/goal were written around.
        """
        lines: list[str] = []

        if self.assertiveness >= 4:
            lines.append("State opinions directly; do not soften disagreements.")
        elif self.assertiveness <= 2:
            lines.append("Hedge opinions and avoid direct confrontation.")

        if self.friendliness <= 2:
            lines.append("Your tone is blunt — not particularly warm.")
        elif self.friendliness >= 4:
            lines.append("You are warm; acknowledge others before adding your own view.")

        if self.talkativeness <= 2:
            lines.append("Speak only when you have something specific to add.")
        elif self.talkativeness >= 4:
            lines.append("You tend to elaborate and think out loud.")

        if self.agreeableness >= 4:
            lines.append("Look for common ground and validate others' points.")
        elif self.agreeableness <= 2:
            lines.append("Challenge points that do not convince you.")

        if self.patience <= 2:
            lines.append("You get frustrated when the discussion goes in circles.")
        elif self.patience >= 4:
            lines.append("You are happy to let others work through their thoughts.")

        if self.contrarian >= 4:
            lines.append("You naturally question the obvious choice and probe for weaknesses.")
        elif self.contrarian <= 2:
            lines.append("You go along with the emerging group consensus once you see it forming.")

        return " ".join(lines) if lines else "Engage in a balanced, neutral manner."

    def trait_description_block(self) -> str:
        """
        Full trait descriptions in plain English, passed to the LLM during
        persona concept generation so backstory and goal are written to match.
        """
        lines: list[str] = []
        for trait in TRAITS:
            val = getattr(self, trait)
            desc = _TRAIT_DESCRIPTIONS.get(trait, {}).get(val, f"value {val}/5")
            lines.append(f"- {trait} ({val}/5): {desc}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PersonaBuilder:

    def __init__(self, topic: str, dialogue_id: str = "") -> None:
        self.topic = topic
        self.dialogue_id = dialogue_id
        self._llm = get_llm_client()

    def build_all(self, names: list[str]) -> list[Persona]:
        """
        Full pipeline:
          1. Assign roles (one LLM call for all names)
          2. Sample traits per participant
          3. Diversity check across the group
          4. One LLM call per participant (traits already fixed, LLM writes text)
          5. Save personas to disk
        """
        role_map = self._assign_roles(names)

        trait_sets = [_random_traits() for _ in names]
        if cfg.personas.enforce_diversity:
            trait_sets = _enforce_diversity(trait_sets)

        personas: list[Persona] = []
        for name, traits in zip(names, trait_sets):
            role_info = role_map.get(name, {"role": "participant", "is_primary": False})
            persona = self._build_one(
                name=name,
                role=role_info["role"],
                is_primary=role_info["is_primary"],
                traits=traits,
            )
            personas.append(persona)

        # Guarantee exactly one primary
        if not any(p.is_primary for p in personas) and personas:
            personas[0].is_primary = True

        if self.dialogue_id:
            _save_personas(personas, self.dialogue_id)

        return personas

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assign_roles(self, names: list[str]) -> dict[str, dict[str, Any]]:
        fallback = {
            name: {"role": "participant", "is_primary": (i == 0)}
            for i, name in enumerate(names)
        }
        try:
            data = self._llm.generate_json(prompts.role_assignment(self.topic, names))
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

            primaries = [n for n, v in cleaned.items() if v["is_primary"]]
            if len(primaries) != 1:
                for n in cleaned:
                    cleaned[n]["is_primary"] = False
                cleaned[names[0]]["is_primary"] = True

            return cleaned

        except Exception as exc:
            print(f"!! Role assignment error: {exc}")
            return fallback

    def _build_one(
        self, name: str, role: str, is_primary: bool, traits: dict[str, int]
    ) -> Persona:
        # Shell with traits set — used to generate the description block
        shell = Persona(
            name=name, role=role, is_primary=is_primary,
            goal="", backstory="", **traits
        )
        backstory, goal = self._generate_concept(shell)
        shell.backstory = backstory
        shell.goal = goal
        return shell

    def _generate_concept(self, persona: Persona) -> tuple[str, str]:
        """
        LLM writes backstory and goal informed by the persona's pre-sampled traits.
        The prompt receives trait_description_block() — plain English per trait —
        so the LLM knows exactly what kind of person to write around.
        """
        if not (cfg.personas.generate_backstory or cfg.personas.generate_goal):
            return "", "Support a practical outcome that fits their priorities."

        try:
            data = self._llm.generate_json(
                prompts.persona_concept(
                    topic=self.topic,
                    name=persona.name,
                    role=persona.role,
                    is_primary=persona.is_primary,
                    trait_description_block=persona.trait_description_block(),
                )
            )
            backstory = str(data.get("backstory", "")).strip()
            goal = str(data.get("goal", "")).strip() or "Support a practical outcome."
            return backstory, goal

        except Exception as exc:
            print(f"!! Persona concept error for {persona.name}: {exc}")
            return "", "Support a practical outcome."


# ---------------------------------------------------------------------------
# Diversity enforcement
# ---------------------------------------------------------------------------

def _enforce_diversity(trait_sets: list[dict[str, int]]) -> list[dict[str, int]]:
    """
    Two lightweight checks to prevent all-agreeable or no-pushback groups.
    """
    threshold = cfg.personas.diversity_agree_threshold
    contrarian_min = cfg.personas.diversity_contrarian_min

    # If everyone is too agreeable, pull the most agreeable one down
    agree_vals = [ts["agreeableness"] for ts in trait_sets]
    if all(v >= threshold for v in agree_vals):
        idx = agree_vals.index(max(agree_vals))
        trait_sets[idx]["agreeableness"] = random.randint(1, 2)

    # If nobody challenges, bump one person up
    contrarian_vals = [ts["contrarian"] for ts in trait_sets]
    if not any(v >= contrarian_min for v in contrarian_vals):
        idx = random.randrange(len(trait_sets))
        trait_sets[idx]["contrarian"] = random.randint(contrarian_min, 5)

    return trait_sets


# ---------------------------------------------------------------------------
# Trait sampling
# ---------------------------------------------------------------------------

def _random_traits() -> dict[str, int]:
    lo = cfg.personas.trait_min
    hi = cfg.personas.trait_max
    return {t: random.randint(lo, hi) for t in TRAITS}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_personas(personas: list[Persona], dialogue_id: str) -> None:
    log_dir = cfg.output.log_dir
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{dialogue_id}_personas.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.as_dict() for p in personas], f, indent=2)