"""
persona.py
----------
Persona dataclass, AgentBeliefs, PersonaBuilder.

Pipeline per dialogue (single grouped LLM path, no per-persona fallback):
  1. One LLM call generates N names + roles tuned to the topic.
  2. Sample Big Five traits per participant from cooperative defaults.
  3. Group diversity check.
  4. One LLM call generates all backstories + goals.
  5. One LLM call generates all belief states (after options exist).

Total: 3 LLM calls for setup, regardless of N.
"""

from __future__ import annotations

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
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

TRAITS = PERSONALITY_TRAITS + ["response_length"]

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
    1: "Terse chat style. One punchy fragment or short sentence. Cut everything non-essential.",
    2: "Brief chat style. One clear point, sometimes with a short reason.",
    3: "Normal chat style. One or two compact sentences. Explain when needed.",
    4: "Chatty style. Explain a bit and riff, but keep it breezy.",
    5: "Verbose-for-chat style. Give useful detail and context, never an essay.",
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
        return max(1, min(5, int(self.response_length)))

    def style_rule(self) -> str:
        return _STYLE_RULE[self.response_length_score()]

    def max_words(self, phase: str) -> int:
        base = cfg.response_length.word_budgets[self.response_length_score() - 1]
        caps = cfg.response_length.phase_caps
        if phase == "opening":
            return min(base, caps.opening)
        if phase == "confirmation":
            return min(base, caps.confirmation)
        if phase == "narrowing":
            return min(max(base, caps.narrowing_min), caps.narrowing_max)
        if phase == "emergence":
            return min(base, caps.emergence)
        if phase == "closure":
            return min(base, caps.closure)
        return base

    def personality_summary(self) -> str:
        """Register descriptors derived from Big Five — never prescribes phrases."""
        parts: list[str] = []
        if self.openness >= 4:
            parts.append("considers angles others haven't raised; comfortable reframing the question")
        elif self.openness <= 2:
            parts.append("prefers concrete options on the table; impatient with speculation")
        if self.conscientiousness >= 4:
            parts.append("detail-oriented; names concrete specifics rather than gut impressions")
        elif self.conscientiousness <= 2:
            parts.append("relies on gut feel; comfortable deciding without exhaustive analysis")
        if self.extraversion >= 4:
            parts.append("energetic, quick to react, thinks aloud")
        elif self.extraversion <= 2:
            parts.append("reserved; speaks up only when there is something specific to add")
        if self.agreeableness >= 4:
            parts.append("acknowledges before pushing back; seeks common ground")
        elif self.agreeableness <= 2:
            parts.append("direct, skeptical, blunt; states disagreement plainly")
        if self.neuroticism >= 4:
            parts.append("sensitive to uncertainty; concern and caution show through tone")
        elif self.neuroticism <= 2:
            parts.append("calm and steady even when discussion stalls")
        if not parts:
            return "Balanced, neutral register."
        summary = "; ".join(parts)
        return summary[0].upper() + summary[1:] + "."

    def trait_description_block(self) -> str:
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

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._llm = get_llm_client()

    def generate_names_and_roles(self, n: int) -> list[dict[str, Any]]:
        """One LLM call: N names + roles tuned to the topic. Raises on failure."""
        data = self._llm.generate_json(prompts.names_and_roles(self.topic, n))
        entries = data.get("participants", [])
        if not isinstance(entries, list) or len(entries) != n:
            raise ValueError(f"Name+role generation expected {n} participants, got: {entries!r}")
        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for e in entries:
            if not isinstance(e, dict):
                raise ValueError(f"Name+role entry is not an object: {e!r}")
            name = str(e.get("name", "")).strip()
            role = str(e.get("role", "")).strip()
            if not name or not role:
                raise ValueError(f"Name+role entry missing name or role: {e!r}")
            if name in seen:
                raise ValueError(f"Name+role generation produced a duplicate name: {name!r}")
            seen.add(name)
            cleaned.append({"name": name, "role": role, "is_primary": bool(e.get("is_primary", False))})
        primaries = sum(1 for c in cleaned if c["is_primary"])
        if primaries != 1:
            for c in cleaned:
                c["is_primary"] = False
            cleaned[0]["is_primary"] = True
        return cleaned

    def build_all(self, name_role_entries: list[dict[str, Any]]) -> list[Persona]:
        """Build personas from name+role entries; assign traits and concepts."""
        trait_sets = [_random_traits() for _ in name_role_entries]
        if cfg.personas.enforce_diversity:
            trait_sets = _enforce_diversity(trait_sets)

        shells: list[Persona] = []
        for entry, traits in zip(name_role_entries, trait_sets):
            shells.append(Persona(
                name=entry["name"],
                role=entry["role"],
                is_primary=entry["is_primary"],
                goal="",
                backstory="",
                **traits,
            ))

        concepts = self._generate_concepts_group(shells)
        for persona in shells:
            if persona.name not in concepts:
                raise ValueError(f"Persona concept generation missing entry for {persona.name!r}")
            persona.backstory, persona.goal = concepts[persona.name]

        if not any(p.is_primary for p in shells) and shells:
            shells[0].is_primary = True
        return shells

    def assign_beliefs(self, personas: list[Persona], options: list[str]) -> None:
        """One LLM call for all belief states. Raises if any persona is missing."""
        group_beliefs = self._generate_beliefs_group(personas, options)
        for persona in personas:
            if persona.name not in group_beliefs:
                raise ValueError(f"Belief generation missing entry for {persona.name!r}")
            persona.beliefs = group_beliefs[persona.name]

    # ------------------------------------------------------------------

    def _generate_concepts_group(
        self, shells: list[Persona]
    ) -> dict[str, tuple[str, str]]:
        entries = [
            {
                "name": p.name,
                "role": p.role,
                "is_primary": p.is_primary,
                "trait_description_block": p.trait_description_block(),
            }
            for p in shells
        ]
        data = self._llm.generate_json(
            prompts.persona_group_generation(self.topic, entries)
        )
        raw = data.get("personas", {})
        if not isinstance(raw, dict):
            raise ValueError(f"Persona concept generation returned no 'personas' object: {data!r}")
        result: dict[str, tuple[str, str]] = {}
        for p in shells:
            entry = raw.get(p.name)
            if not isinstance(entry, dict):
                raise ValueError(f"Persona concept generation missing entry for {p.name!r}")
            backstory = str(entry.get("backstory", "")).strip()
            goal = str(entry.get("goal", "")).strip()
            if not backstory or not goal:
                raise ValueError(f"Persona concept for {p.name!r} missing backstory or goal.")
            result[p.name] = (backstory, goal)
        return result

    def _generate_beliefs_group(
        self, personas: list[Persona], options: list[str]
    ) -> dict[str, AgentBeliefs]:
        personas_text = "\n".join(
            f"{p.name} ({p.role}): goal={p.goal}  backstory={p.backstory}  "
            f"personality={p.personality_summary()}"
            for p in personas
        )
        options_text = "\n".join(f"  {o}" for o in options)
        data = self._llm.generate_json(
            prompts.agent_beliefs_group(self.topic, personas_text, options_text)
        )
        raw = data.get("beliefs", {})
        if not isinstance(raw, dict):
            raise ValueError(f"Belief generation returned no 'beliefs' object: {data!r}")
        result: dict[str, AgentBeliefs] = {}
        for persona in personas:
            entry = raw.get(persona.name)
            if not isinstance(entry, dict):
                raise ValueError(f"Belief generation missing entry for {persona.name!r}")
            result[persona.name] = self._parse_beliefs(entry, persona.name)
        return result

    def _parse_beliefs(self, data: dict, name: str) -> AgentBeliefs:
        """Parse a belief dict. Raises if required fields are missing or invalid."""
        preferred = str(data.get("preferred", "")).strip().upper()
        if preferred not in {"A", "B", "C", "D"}:
            raise ValueError(f"Belief for {name!r} has invalid 'preferred': {preferred!r}")

        acceptable_raw = data.get("acceptable")
        if not isinstance(acceptable_raw, list):
            raise ValueError(f"Belief for {name!r} has no 'acceptable' list.")
        acceptable = [
            x.strip().upper() for x in acceptable_raw
            if isinstance(x, str) and x.strip().upper() in {"A", "B", "C", "D"}
        ]
        if preferred not in acceptable:
            acceptable.insert(0, preferred)

        rejected_raw = data.get("rejected", [])
        rejected = [
            x.strip().upper() for x in (rejected_raw if isinstance(rejected_raw, list) else [])
            if isinstance(x, str) and x.strip().upper() in {"A", "B", "C", "D"}
            and x.strip().upper() not in acceptable
        ]

        key_concern = str(data.get("key_concern", "")).strip()
        concession = str(data.get("concession", "")).strip()
        if not key_concern or not concession:
            raise ValueError(f"Belief for {name!r} missing key_concern or concession.")

        beliefs = AgentBeliefs(
            preferred=preferred,
            acceptable=acceptable,
            rejected=rejected,
            key_concern=key_concern,
            concession=concession,
        )
        accept_others = [x for x in acceptable if x != preferred]
        accept_str = f", accepts {accept_others}" if accept_others else ""
        print(f"  [{name}] prefers {preferred}{accept_str} | {key_concern[:45]}")
        return beliefs


# ---------------------------------------------------------------------------
# Diversity + trait sampling
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
