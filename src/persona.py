"""
persona.py
----------
Persona dataclass, AgentBeliefs, and PersonaBuilder.

Pipeline per dialogue:
  1. Sample Big Five traits randomly (1-5) - no LLM
  2. Group diversity check across all sampled trait sets
  3. One LLM call per participant - writes backstory + goal to match traits
  4. One LLM call per participant - derives belief state from persona + options
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

PERSONALITY_TRAITS = [
    "openness",          # 1=conventional, 5=curious and imaginative
    "conscientiousness", # 1=spontaneous, 5=careful and structured
    "extraversion",      # 1=reserved, 5=outgoing and talkative
    "agreeableness",     # 1=challenging, 5=cooperative and warm
    "neuroticism",       # 1=calm, 5=emotionally reactive under stress
]

COMMUNICATION_FIELDS = [
    "response_length",   # output control, not a Big Five trait
]

TRAITS = PERSONALITY_TRAITS + COMMUNICATION_FIELDS

_TRAIT_DESCRIPTIONS: dict[str, dict[int, str]] = {
    "openness": {
        1: "prefers familiar, practical choices and dislikes overcomplicating things",
        2: "leans practical and conventional, but will consider a clearly useful new angle",
        3: "balances familiar options with some curiosity about alternatives",
        4: "curious, interested in trade-offs, and willing to explore unusual angles",
        5: "highly imaginative and novelty-seeking, enjoys reframing the problem",
    },
    "conscientiousness": {
        1: "loose, spontaneous, and comfortable deciding with incomplete details",
        2: "somewhat casual about planning and precision",
        3: "moderately organized and practical",
        4: "careful, responsible, and attentive to consequences",
        5: "very structured and planful, wants decisions to be justified clearly",
    },
    "extraversion": {
        1: "reserved, speaks briefly, and rarely jumps in unprompted",
        2: "somewhat quiet, contributes when there is a clear reason",
        3: "moderately sociable and comfortable joining the discussion",
        4: "outgoing, energetic, and likely to respond quickly",
        5: "very expressive and talkative, thinks aloud with the group",
    },
    "agreeableness": {
        1: "skeptical, competitive, and comfortable disagreeing directly",
        2: "somewhat challenging and slow to validate others",
        3: "balanced between cooperation and disagreement",
        4: "cooperative, warm, and willing to look for common ground",
        5: "highly compassionate and harmony-seeking, reluctant to create conflict",
    },
    "neuroticism": {
        1: "calm and emotionally steady even when the discussion gets stuck",
        2: "usually calm, with mild concern when plans feel shaky",
        3: "moderately sensitive to risk, conflict, or uncertainty",
        4: "worries about downsides and reacts strongly to uncertainty",
        5: "emotionally reactive, easily frustrated or anxious when things feel unresolved",
    },
}

_STYLE_RULE: dict[int, str] = {
    1: "Terse chat style. One punchy fragment or short sentence. Cut everything non-essential — no filler.",
    2: "Brief chat style. One clear point, sometimes with a short reason. Stay under two short sentences.",
    3: "Normal chat style. One or two compact sentences. Explain when needed, not by default.",
    4: "Chatty style. You can explain a bit and riff on ideas, but keep it breezy and conversational like a group chat.",
    5: "Verbose-for-chat style. Give useful detail and context, but never write an essay or a formal summary.",
}

_WORD_BUDGET: dict[int, int] = {
    1: 14,
    2: 24,
    3: 36,
    4: 48,
    5: 60,
}


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass
class AgentBeliefs:
    preferred: str
    acceptable: list[str]
    rejected: list[str]
    key_concern: str
    concession: str


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

    openness: int = 3
    conscientiousness: int = 3
    extraversion: int = 3
    agreeableness: int = 3
    neuroticism: int = 3
    response_length: int = 2

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

    def response_length_score(self) -> int:
        """Communication-style control, separate from Big Five personality."""
        return max(1, min(5, int(self.response_length)))

    def style_rule(self) -> str:
        return _STYLE_RULE[self.response_length_score()]

    def max_words(self, phase: str) -> int:
        """Hard word budget used to make generated turns feel like chat."""
        base = _WORD_BUDGET[self.response_length_score()]
        if phase == "greeting":
            return min(base, 8)
        if phase == "confirmation":
            return min(base, 10)
        if phase == "narrowing":
            return min(max(base, 18), 40)
        if phase == "emergence":
            return min(base, 42)
        return base

    def personality_summary(self) -> str:
        """Plain-English personality cues injected into every turn prompt."""
        lines: list[str] = []

        if self.openness >= 4:
            lines.append("You enjoy exploring angles others haven't raised; you might say 'wait, what about...' or flip the framing.")
        elif self.openness <= 2:
            lines.append("You prefer familiar, practical choices; you get impatient with speculation — 'let's stick to what's actually on the table'.")

        if self.conscientiousness >= 4:
            lines.append("You care about specifics and consequences; you name concrete details rather than vibes.")
        elif self.conscientiousness <= 2:
            lines.append("You decide from gut feel and get bored by excessive analysis; 'honestly just pick one' is your vibe.")

        if self.extraversion >= 4:
            lines.append("You're energetic and warm; you react quickly, use enthusiasm ('oh that's actually a good point'), and think aloud.")
        elif self.extraversion <= 2:
            lines.append("You're reserved; you only speak up when you have something specific to say, and you keep it short.")

        if self.agreeableness >= 4:
            lines.append("You're cooperative; you naturally acknowledge others before pushing back — 'I get that, but...' is your style.")
        elif self.agreeableness <= 2:
            lines.append("You're blunt and skeptical; you push back directly without softening it — 'yeah but that's not actually true' is fine for you.")

        if self.neuroticism >= 4:
            lines.append("Uncertainty makes you visibly tense; your worry comes through — 'but what if that doesn't work out?', 'I'm a bit nervous about...'")
        elif self.neuroticism <= 2:
            lines.append("You stay calm even when others are anxious or the group goes in circles.")

        return " ".join(lines) if lines else "Engage in a balanced, neutral way."

    def trait_description_block(self) -> str:
        """Full Big Five trait descriptions for the persona-concept LLM call."""
        lines: list[str] = []
        for trait in PERSONALITY_TRAITS:
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
            concession="could accept a different option if it directly addresses their main concern",
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
                data.get("concession", "could accept a different option if it addresses their main concern")
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
    extraversion_min = cfg.personas.diversity_extraversion_min

    agree_vals = [ts["agreeableness"] for ts in trait_sets]
    if all(v >= threshold for v in agree_vals):
        idx = agree_vals.index(max(agree_vals))
        trait_sets[idx]["agreeableness"] = random.randint(1, 2)

    extraversion_vals = [ts["extraversion"] for ts in trait_sets]
    if not any(v >= extraversion_min for v in extraversion_vals):
        idx = random.randrange(len(trait_sets))
        trait_sets[idx]["extraversion"] = random.randint(extraversion_min, 5)

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
