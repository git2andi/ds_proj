"""
persona.py
----------
Persona dataclass, AgentBeliefs (argument kit), SpeechSignature, PersonaBuilder.

Per-dialogue setup (one grouped LLM path -- no per-persona fallback):
  1. One LLM call generates N names + roles tuned to the topic.
  2. Sample Big Five traits per participant from cooperative defaults.
  3. Group diversity check (Big Five spread).
  4. One LLM call generates all backstories + goals.
  5. One LLM call generates all belief states (Toulmin argument kit).
  6. Deterministic divergence enforcement on preferred-options.

Total: 3 LLM calls for setup, regardless of N.

Research grounding (the parts of the architecture this module owns):
  - McCrae & John (1992) -- Big Five traits, sampled with diversity constraint
    and routed into a deterministic SpeechSignature + derived conversational
    controls (this module) that scaffold distinct voices.
  - Shanahan (2023, "Role-play with LLMs") -- persona scaffolding is structural,
    not "be the character" instruction; SpeechSignature lives in the speaker card.
  - Toulmin (1958) -- AgentBeliefs carries claim + warrants (reasons) +
    reservation + would_reconsider_if so personas have material to argue with,
    not just positions to hold.
  - Liang et al. (2023) "Encouraging Divergent Thinking" -- _enforce_divergence
    spreads preferred options so the group must actually reconcile.
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
# Belief state -- Toulmin argument kit
# ---------------------------------------------------------------------------

@dataclass
class AgentBeliefs:
    """Private belief model. The 'argument kit' (reasons, reservation,
    would_reconsider_if) is what lets sims ARGUE rather than restate."""

    preferred: str
    acceptable: list[str]
    rejected: list[str]
    key_concern: str
    # Toulmin warrants -- 1-2 concrete reasons drawn from the persona's
    # goal/expertise/backstory, phrased as their knowledge/experience.
    reasons: list[str]
    # One genuine concern about a rival option -- framed to be addressable,
    # not a veto. ("I'd worry about X" not "I refuse Y".)
    reservation: str
    # The concrete thing that would move them off `preferred`. Enables genuine
    # update and keeps disagreements resolvable.
    would_reconsider_if: str


# ---------------------------------------------------------------------------
# Speech signature -- Big Five routed into distinct voice features.
# Deterministic floats fed into the speaker card and the turn prompt; never
# surfaced as named phrases or filler lists.
# ---------------------------------------------------------------------------

@dataclass
class SpeechSignature:
    hedge_propensity: float        # 0..1; "I think", "kind of", "tbh"
    directness: float              # 0..1; willingness to say "no" plainly
    thinkaloud_propensity: float   # 0..1; fragments + run-ons, less polish
    detail_orientation: float      # 0..1; cites concrete option-text fragments

    def descriptor(self) -> str:
        """Short register descriptor for the speaker card -- never prescribes phrases."""
        parts: list[str] = []
        if self.hedge_propensity >= 0.65:
            parts.append("hedges naturally (\"i think\", \"kind of\")")
        elif self.hedge_propensity <= 0.30:
            parts.append("speaks with little hedging")
        if self.directness >= 0.65:
            parts.append("blunt; will say \"no\" plainly")
        elif self.directness <= 0.30:
            parts.append("softens disagreement before stating it")
        if self.thinkaloud_propensity >= 0.65:
            parts.append("thinks aloud; sentences run a bit loose")
        elif self.thinkaloud_propensity <= 0.30:
            parts.append("tight, polished sentences")
        if self.detail_orientation >= 0.65:
            parts.append("cites concrete details from the options")
        elif self.detail_orientation <= 0.30:
            parts.append("speaks at the level of impressions, not specifics")
        return "; ".join(parts) if parts else "neutral register"


def _norm(value: int) -> float:
    return (max(1, min(5, int(value))) - 1) / 4.0


def derive_speech_signature(
    openness: int, conscientiousness: int, extraversion: int,
    agreeableness: int, neuroticism: int,
) -> SpeechSignature:
    """Map Big Five -> voice features (Shanahan 2023; Character-LLM).
    All weights come from cfg.voice; this is pure config-driven mapping."""
    w = cfg.voice
    a = _norm(agreeableness)
    o = _norm(openness)
    c = _norm(conscientiousness)
    neuro = _norm(neuroticism)
    e = _norm(extraversion)

    hedge = 0.5 + w.hedge_neuroticism_weight * (neuro - 0.5) \
                + w.hedge_agreeableness_weight * (a - 0.5)
    directness = 0.5 + w.directness_disagreeableness_weight * (0.5 - a) \
                     + w.directness_low_neuroticism_weight * (0.5 - neuro)
    thinkaloud = 0.5 + w.thinkaloud_extraversion_weight * (e - 0.5) \
                     + w.thinkaloud_low_conscientiousness_weight * (0.5 - c)
    detail = 0.5 + w.detail_conscientiousness_weight * (c - 0.5)

    # openness has a small reframing influence on hedge / directness too --
    # very open sims hedge slightly more (consider angles before committing).
    hedge = max(0.0, min(1.0, hedge + 0.05 * (o - 0.5)))

    return SpeechSignature(
        hedge_propensity=max(0.0, min(1.0, hedge)),
        directness=max(0.0, min(1.0, directness)),
        thinkaloud_propensity=max(0.0, min(1.0, thinkaloud)),
        detail_orientation=max(0.0, min(1.0, detail)),
    )


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
                "reasons": list(self.beliefs.reasons),
                "reservation": self.beliefs.reservation,
                "would_reconsider_if": self.beliefs.would_reconsider_if,
            }
        sig = self.speech_signature()
        d["speech_signature"] = {
            "hedge": round(sig.hedge_propensity, 2),
            "directness": round(sig.directness, 2),
            "thinkaloud": round(sig.thinkaloud_propensity, 2),
            "detail": round(sig.detail_orientation, 2),
        }
        d["derived_controls"] = self.derived_controls()
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
        if phase == "closure":
            return min(base, caps.closure)
        return base

    def speech_signature(self) -> SpeechSignature:
        return derive_speech_signature(
            self.openness, self.conscientiousness, self.extraversion,
            self.agreeableness, self.neuroticism,
        )

    def derived_controls(self) -> dict[str, float]:
        """Six behavioural controls derived from Big Five (each 0..1).

        Used in the speaker card instead of the verbose personality_summary +
        speech_signature descriptor so the prompt stays compact.
        """
        o = _norm(self.openness)
        c = _norm(self.conscientiousness)
        e = _norm(self.extraversion)
        a = _norm(self.agreeableness)
        n = _norm(self.neuroticism)
        return {
            "initiative":   round(min(1.0, 0.40 * e + 0.30 * (1.0 - n) + 0.30 * o), 2),
            "flexibility":  round(min(1.0, 0.50 * a + 0.30 * o + 0.20 * (1.0 - n)), 2),
            "directness":   round(min(1.0, 0.50 * (1.0 - a) + 0.30 * (1.0 - n) + 0.20 * c), 2),
            "detail_level": round(min(1.0, 0.60 * c + 0.20 * o + 0.20 * (1.0 - e)), 2),
            "warmth":       round(min(1.0, 0.60 * a + 0.20 * e + 0.20 * (1.0 - n)), 2),
            "target_response_length": round(_norm(self.response_length), 2),
        }

    def derived_controls_descriptor(self) -> str:
        """Compact one-line style descriptor for the speaker card.

        Replaces the separate Voice + Personality lines to reduce prompt bloat.
        Only expresses values that are clearly high or low (>=0.65 or <=0.30).
        """
        dc = self.derived_controls()
        parts: list[str] = []

        if dc["initiative"] >= 0.65:
            parts.append("jumps into the conversation readily")
        elif dc["initiative"] <= 0.30:
            parts.append("waits before contributing")

        if dc["flexibility"] >= 0.65:
            parts.append("genuinely open to shifting position")
        elif dc["flexibility"] <= 0.30:
            parts.append("holds position firmly")

        if dc["directness"] >= 0.65:
            parts.append("states disagreement plainly")
        elif dc["directness"] <= 0.30:
            parts.append("softens pushback")

        if dc["detail_level"] >= 0.65:
            parts.append("references concrete specifics")
        elif dc["detail_level"] <= 0.30:
            parts.append("stays at the level of impressions")

        if dc["warmth"] >= 0.65:
            parts.append("warm, friendly tone")
        elif dc["warmth"] <= 0.30:
            parts.append("cool, businesslike tone")

        return "; ".join(parts) if parts else "neutral, balanced style"

    def personality_summary(self) -> str:
        """Register descriptors derived from Big Five -- never prescribes phrases."""
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
        min_n = int(cfg.simulation.min_participants)
        max_n = int(cfg.simulation.max_participants)
        if not (min_n <= n <= max_n):
            raise ValueError(f"Participant count must be between {min_n} and {max_n}; got {n}.")
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
        """One LLM call for all belief states + deterministic divergence enforcement."""
        group_beliefs = self._generate_beliefs_group(personas, options)
        for persona in personas:
            if persona.name not in group_beliefs:
                raise ValueError(f"Belief generation missing entry for {persona.name!r}")
            persona.beliefs = group_beliefs[persona.name]

        option_letters = _option_letters_from_texts(options)
        if option_letters:
            _enforce_divergence(personas, option_letters)
            _enforce_acceptable_overlap(personas, option_letters)

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
        if not key_concern:
            raise ValueError(f"Belief for {name!r} missing key_concern.")

        reasons_raw = data.get("reasons", [])
        if not isinstance(reasons_raw, list):
            reasons_raw = []
        reasons = [str(r).strip() for r in reasons_raw if isinstance(r, str) and str(r).strip()]
        rk = cfg.argument_kit
        if len(reasons) < rk.reasons_min:
            raise ValueError(
                f"Belief for {name!r} needs at least {rk.reasons_min} reason(s); got {reasons!r}"
            )
        if len(reasons) > rk.reasons_max:
            reasons = reasons[:rk.reasons_max]

        reservation = str(data.get("reservation", "")).strip()
        if rk.reservation_required and not reservation:
            raise ValueError(f"Belief for {name!r} missing required 'reservation'.")

        would_reconsider_if = str(data.get("would_reconsider_if", "")).strip()
        if rk.reconsider_required and not would_reconsider_if:
            raise ValueError(f"Belief for {name!r} missing required 'would_reconsider_if'.")

        beliefs = AgentBeliefs(
            preferred=preferred,
            acceptable=acceptable,
            rejected=rejected,
            key_concern=key_concern,
            reasons=reasons,
            reservation=reservation,
            would_reconsider_if=would_reconsider_if,
        )
        accept_others = [x for x in acceptable if x != preferred]
        accept_str = f", accepts {accept_others}" if accept_others else ""
        print(f"  [{name}] prefers {preferred}{accept_str} | {key_concern[:45]}")
        return beliefs


# ---------------------------------------------------------------------------
# Trait diversity + sampling
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

    # Ensure response lengths are spread across the group. If the spread is
    # less than 2 points, push one sim toward a short length and one toward
    # a long length so turns feel visibly different in size.
    rl_vals = [ts["response_length"] for ts in trait_sets]
    if len(trait_sets) >= 2 and max(rl_vals) - min(rl_vals) < 2:
        idx_min = rl_vals.index(min(rl_vals))
        idx_max = rl_vals.index(max(rl_vals))
        if idx_min == idx_max:
            idx_max = (idx_min + 1) % len(trait_sets)
        rl_lo = getattr(cfg.personas.trait_ranges, "response_length", None)
        lo_floor = int(rl_lo[0]) if rl_lo else 1
        hi_ceil  = int(rl_lo[1]) if rl_lo else 5
        trait_sets[idx_min]["response_length"] = lo_floor
        trait_sets[idx_max]["response_length"] = hi_ceil

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


# ---------------------------------------------------------------------------
# Divergence enforcement (Liang 2023). The fuel for real discussion is that
# cooperative people START in different places for good reasons.
# ---------------------------------------------------------------------------

def _option_letters_from_texts(option_texts: list[str]) -> list[str]:
    import re
    letters: list[str] = []
    for opt in option_texts:
        m = re.match(r"^\s*option\s+([a-d])\b", opt, re.I)
        if m:
            letters.append(m.group(1).upper())
    return letters


def _enforce_divergence(personas: list[Persona], option_letters: list[str]) -> None:
    """If `divergence.enforce_distinct_preferred`, ensure not all `preferred`
    are identical. Pick the most-similar pair and nudge one toward an option
    still inside their `acceptable` list, preferring a not-yet-claimed letter."""
    if not cfg.divergence.enforce_distinct_preferred:
        return
    if len({p.beliefs.preferred for p in personas if p.beliefs}) > 1:
        return  # already diverse

    claimed: set[str] = set()
    # Anchor the first persona; nudge the rest off the shared preference.
    anchor = personas[0]
    if anchor.beliefs:
        claimed.add(anchor.beliefs.preferred)

    for persona in personas[1:]:
        if not persona.beliefs:
            continue
        alternatives = [
            o for o in persona.beliefs.acceptable
            if o != persona.beliefs.preferred and o not in claimed
        ] or [o for o in option_letters if o not in claimed]
        if not alternatives:
            continue
        old = persona.beliefs.preferred
        persona.beliefs.preferred = alternatives[0]
        if persona.beliefs.preferred not in persona.beliefs.acceptable:
            persona.beliefs.acceptable.insert(0, persona.beliefs.preferred)
        claimed.add(persona.beliefs.preferred)
        print(
            f"  [divergence] nudged {persona.name}: preferred {old} -> "
            f"{persona.beliefs.preferred} (keeps acceptable overlap)"
        )


def _enforce_acceptable_overlap(personas: list[Persona], option_letters: list[str]) -> None:
    """Trim each persona's `acceptable` to the configured size range, and
    guarantee at least `required_common_acceptable` options are in everyone's
    acceptable set so consensus is reachable."""
    lo = cfg.divergence.target_acceptable_min
    hi = cfg.divergence.target_acceptable_max
    required_common = cfg.divergence.required_common_acceptable

    # Step 1: trim oversize acceptable sets to `hi`, preserving preferred + variety.
    for persona in personas:
        if not persona.beliefs:
            continue
        accept = list(persona.beliefs.acceptable)
        if len(accept) > hi:
            keep = [persona.beliefs.preferred]
            for opt in accept:
                if opt not in keep and len(keep) < hi:
                    keep.append(opt)
            persona.beliefs.acceptable = keep

    # Step 2: pick a common option (the most popular acceptable letter) and
    # make sure every persona has it. This is the shared fallback.
    if required_common <= 0:
        return
    pool: dict[str, int] = {l: 0 for l in option_letters}
    for persona in personas:
        if not persona.beliefs:
            continue
        for opt in persona.beliefs.acceptable:
            pool[opt] = pool.get(opt, 0) + 1
    if not pool:
        return
    common = sorted(pool.items(), key=lambda kv: (-kv[1], kv[0]))
    common_letters = [opt for opt, _ in common][:required_common]

    for persona in personas:
        if not persona.beliefs:
            continue
        for opt in common_letters:
            if opt not in persona.beliefs.acceptable and opt not in persona.beliefs.rejected:
                persona.beliefs.acceptable.append(opt)

    # Step 3: ensure minimum acceptable size (preferred + at least one other).
    for persona in personas:
        if not persona.beliefs:
            continue
        if len(persona.beliefs.acceptable) >= lo:
            continue
        for opt in option_letters:
            if opt in persona.beliefs.acceptable or opt in persona.beliefs.rejected:
                continue
            persona.beliefs.acceptable.append(opt)
            if len(persona.beliefs.acceptable) >= lo:
                break
