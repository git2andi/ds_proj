"""
persona.py
----------
Persona dataclass, AgentBeliefs, and PersonaBuilder.

Pipeline per dialogue:
  1. Sample numeric traits randomly (1–5) — no LLM
  2. Group diversity check across all sampled trait sets
  3. One LLM call per participant — writes backstory + goal to match traits
  4. One LLM call per participant — derives belief state from persona + options
  5. Persona + AgentBeliefs assembled and saved

LLM calls per dialogue setup:
  1 (options) + 1 (roles) + N (persona concept) + N (beliefs) = 2N + 2
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

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
    "response_length",  # 1=brief, 5=detailed
]

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

# Style rules: ground even short responses in what was just said
_STYLE_RULE: dict[int, str] = {
    1: (
        "One short grounded reaction — ~8–12 words. Anchor it to what was just said. "
        'e.g. "Yeah but A\'s still cheaper." / "B — walk\'s not that bad." / "Nah, that price is rough."'
    ),
    2: (
        "One sentence — react to something specific, then your take. "
        'e.g. "Fair about the cost, but Option A\'s location still wins it for me."'
    ),
    3: (
        "React to the last point, then add one reason of your own. ~20–25 words. "
        "Conversational, not a speech."
    ),
    4: "Two casual sentences. Hook onto something just said, then develop your argument. No summaries.",
    5: (
        "Two to three sentences. Make your argument with a specific reason. "
        "Still conversational — not a formal statement."
    ),
}


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass
class AgentBeliefs:
    preferred: str          # "B" — top choice going into the discussion
    acceptable: list[str]   # ["B", "C"] — genuine compromise options
    rejected: list[str]     # ["D"] — strongly opposed
    key_concern: str        # what drives their preference
    concession: str         # concrete condition under which they'd accept a non-preferred option


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

    assertiveness: int = 3
    friendliness: int = 3
    talkativeness: int = 3
    agreeableness: int = 3
    patience: int = 3
    contrarian: int = 3
    response_length: int = 3

    beliefs: Optional[AgentBeliefs] = field(default=None, compare=False)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "is_primary": self.is_primary,
            "goal": self.goal,
            "backstory": self.backstory,
            **{t: getattr(self, t) for t in TRAITS},
        }
        if self.beliefs:
            d["beliefs"] = {
                "preferred": self.beliefs.preferred,
                "acceptable": self.beliefs.acceptable,
                "rejected": self.beliefs.rejected,
                "key_concern": self.beliefs.key_concern,
                "concession": self.beliefs.concession,
            }
        return d

    def style_rule(self) -> str:
        return _STYLE_RULE[max(1, min(5, self.response_length))]

    def personality_summary(self) -> str:
        """Plain-English personality cues injected into every turn prompt."""
        lines: list[str] = []

        # Conflicting contrarian + agreeableness → a character who probes but stays open
        if self.contrarian >= 4 and self.agreeableness >= 4:
            lines.append(
                "You question assumptions and play devil's advocate, "
                "but you're genuinely open to being convinced — you push back to understand, not to win."
            )
        else:
            if self.assertiveness >= 4:
                lines.append("State opinions directly; don't soften disagreements.")
            elif self.assertiveness <= 2:
                lines.append("Hedge opinions; avoid direct confrontation.")

            if self.contrarian >= 4:
                lines.append("You naturally question the obvious choice and probe for weaknesses.")
            elif self.contrarian <= 2:
                lines.append("You go along with the group once consensus starts forming.")

            if self.agreeableness >= 4:
                lines.append("Look for common ground and validate others when they make a fair point.")
            elif self.agreeableness <= 2:
                lines.append("Push back on points that don't convince you.")

        if self.friendliness <= 2:
            lines.append("Your tone is blunt — not particularly warm.")
        elif self.friendliness >= 4:
            lines.append("You're warm; a brief acknowledgment before your point is natural for you.")

        if self.patience <= 2:
            lines.append("You get frustrated when the discussion goes in circles.")
        elif self.patience >= 4:
            lines.append("You're patient — happy to let others work through their thinking.")

        return " ".join(lines) if lines else "Engage in a balanced, neutral way."

    def trait_description_block(self) -> str:
        """Full trait descriptions for the persona-concept LLM call."""
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
        Build personas without beliefs (options not known yet).
        Call assign_beliefs() after Orchestrator has generated the options.
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

        if not any(p.is_primary for p in personas) and personas:
            personas[0].is_primary = True

        return personas

    def assign_beliefs(self, personas: list[Persona], options: list[str]) -> None:
        """
        Generate and attach AgentBeliefs to each persona.
        Called after both personas and options are ready.
        Saves the final persona JSON (with beliefs) to disk.
        """
        for persona in personas:
            persona.beliefs = self._generate_beliefs(persona, options)

        if self.dialogue_id:
            _save_personas(personas, self.dialogue_id)

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
        shell = Persona(
            name=name, role=role, is_primary=is_primary,
            goal="", backstory="", **traits
        )
        backstory, goal = self._generate_concept(shell)
        shell.backstory = backstory
        shell.goal = goal
        return shell

    def _generate_concept(self, persona: Persona) -> tuple[str, str]:
        """LLM writes backstory and goal to match the pre-sampled traits."""
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

    def _generate_beliefs(self, persona: Persona, options: list[str]) -> AgentBeliefs:
        """
        One LLM call per participant: given their character and the options,
        produce a stable internal belief state before the conversation starts.
        """
        fallback = AgentBeliefs(
            preferred="A",
            acceptable=["A", "B"],
            rejected=[],
            key_concern="practical trade-offs",
            concession="could accept any option the group strongly prefers",
        )
        try:
            data = self._llm.generate_json(
                prompts.agent_beliefs(
                    name=persona.name,
                    role=persona.role,
                    goal=persona.goal,
                    backstory=persona.backstory,
                    personality_summary=persona.personality_summary(),
                    options_text="\n".join(f"  {o}" for o in options),
                )
            )

            preferred_raw = str(data.get("preferred", "A")).strip().upper()
            if preferred_raw not in {"A", "B", "C", "D"}:
                preferred_raw = "A"

            acceptable_raw = data.get("acceptable", [preferred_raw])
            acceptable = [
                x.strip().upper() for x in (acceptable_raw if isinstance(acceptable_raw, list) else [])
                if isinstance(x, str) and x.strip().upper() in {"A", "B", "C", "D"}
            ]
            if preferred_raw not in acceptable:
                acceptable.insert(0, preferred_raw)

            rejected_raw = data.get("rejected", [])
            rejected = [
                x.strip().upper() for x in (rejected_raw if isinstance(rejected_raw, list) else [])
                if isinstance(x, str) and x.strip().upper() in {"A", "B", "C", "D"}
                and x.strip().upper() not in acceptable
            ]

            key_concern = str(data.get("key_concern", "practical trade-offs")).strip()
            concession = str(
                data.get("concession", "could accept any option the group strongly prefers")
            ).strip()

            beliefs = AgentBeliefs(
                preferred=preferred_raw,
                acceptable=acceptable,
                rejected=rejected,
                key_concern=key_concern,
                concession=concession,
            )
            accept_others = [x for x in acceptable if x != preferred_raw]
            accept_str = f", accepts {accept_others}" if accept_others else ""
            print(f"  [{persona.name}] prefers {preferred_raw}{accept_str} | {key_concern[:45]}")
            return beliefs

        except Exception as exc:
            print(f"!! Belief generation error for {persona.name}: {exc}")
            return fallback


# ---------------------------------------------------------------------------
# Diversity enforcement
# ---------------------------------------------------------------------------

def _enforce_diversity(trait_sets: list[dict[str, int]]) -> list[dict[str, int]]:
    threshold = cfg.personas.diversity_agree_threshold
    contrarian_min = cfg.personas.diversity_contrarian_min

    agree_vals = [ts["agreeableness"] for ts in trait_sets]
    if all(v >= threshold for v in agree_vals):
        idx = agree_vals.index(max(agree_vals))
        trait_sets[idx]["agreeableness"] = random.randint(1, 2)

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
    ranges_cfg = getattr(cfg.personas, "trait_ranges", None)

    result: dict[str, int] = {}
    for t in TRAITS:
        override = getattr(ranges_cfg, t, None) if ranges_cfg else None
        if override is None:
            result[t] = random.randint(lo, hi)
        elif isinstance(override, (list, tuple)):
            result[t] = random.randint(int(override[0]), int(override[1]))
        else:
            result[t] = max(1, min(5, int(override)))
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_personas(personas: list[Persona], dialogue_id: str) -> None:
    log_dir = cfg.output.log_dir
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{dialogue_id}_personas.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.as_dict() for p in personas], f, indent=2)