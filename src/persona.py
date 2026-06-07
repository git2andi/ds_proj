"""Persona and belief-state construction.

The builder is deliberately cooperative by default.  A rare hard-blocker can be
sampled at group setup, but normal trait variation should not make most chats
collapse into non-compromise outcomes.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from schemas import OptionCard, Persona, TraitProfile
from utils import clamp


class PersonaBuilder:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        seed = cfg.simulation.get("random_seed", None)
        if seed is not None:
            random.seed(int(seed))
        self._llm = get_llm_client()

    def build(self, n: int, options: list[OptionCard]) -> list[Persona]:
        entries = self._names_and_roles(n)
        hard_blocker_id = self._sample_hard_blocker_id(entries)
        traits_by_id = {entry["id"]: self._sample_traits(entry["id"] == hard_blocker_id) for entry in entries}
        participant_summaries = [
            {
                "id": entry["id"],
                "name": entry["name"],
                "role": entry["role"],
                "traits": asdict(traits_by_id[entry["id"]]),
                "hard_blocker": entry["id"] == hard_blocker_id,
            }
            for entry in entries
        ]
        belief_data = self._beliefs(options, participant_summaries)
        personas: list[Persona] = []
        for entry in entries:
            traits = traits_by_id[entry["id"]]
            raw_belief = belief_data.get(entry["id"], {})
            personas.append(self._persona_from(entry, traits, raw_belief, options, entry["id"] == hard_blocker_id))
        self._ensure_minimum_preference_diversity(personas, options)
        self._ensure_common_compromise(personas, options)
        return personas

    def _names_and_roles(self, n: int) -> list[dict[str, str]]:
        try:
            data = self._llm.generate_json(prompts.names_and_roles(self.topic, n), profile="setup")
            rows = data.get("participants", [])
            if isinstance(rows, list) and len(rows) == n:
                cleaned = []
                used_names: set[str] = set()
                for i, row in enumerate(rows):
                    name = str(row.get("name") or "").strip()
                    if not name or name.lower() in used_names:
                        name = str(cfg.personas.fallback_names[i])
                    used_names.add(name.lower())
                    cleaned.append({"id": f"p{i+1}", "name": name, "role": str(row.get("role") or "friend").strip()})
                return cleaned
        except Exception:
            pass
        return [
            {"id": f"p{i+1}", "name": str(cfg.personas.fallback_names[i]), "role": "friend"}
            for i in range(n)
        ]

    def _sample_hard_blocker_id(self, entries: list[dict[str, str]]) -> str | None:
        if not entries:
            return None
        if random.random() >= float(cfg.personas.hard_blocker_dialogue_probability):
            return None
        max_count = int(cfg.personas.hard_blocker_max_count)
        if max_count <= 0:
            return None
        return random.choice(entries)["id"]

    def _trait_int(self, name: str) -> int:
        low, high = list(cfg.personas.trait_ranges[name])
        return random.randint(int(low), int(high))

    def _control_float(self, name: str) -> float:
        low, high = list(cfg.personas.cooperative_defaults[name])
        return random.uniform(float(low), float(high))

    def _sample_traits(self, hard_blocker: bool) -> TraitProfile:
        agree = self._trait_int("agreeableness")
        neuro = self._trait_int("neuroticism")
        compromise = self._control_float("compromise_willingness_range")
        patience = self._control_float("patience_range")
        if hard_blocker:
            agree = min(agree, 2)
            compromise = random.uniform(0.10, 0.35)
            patience = random.uniform(0.25, 0.55)
        return TraitProfile(
            openness=self._trait_int("openness"),
            conscientiousness=self._trait_int("conscientiousness"),
            extraversion=self._trait_int("extraversion"),
            agreeableness=agree,
            neuroticism=neuro,
            response_length=self._trait_int("response_length"),
            compromise_willingness=clamp(compromise, 0.0, 1.0),
            patience=clamp(patience, 0.0, 1.0),
            initiative=clamp(self._control_float("initiative_range"), 0.0, 1.0),
            conflict_directness=clamp(self._control_float("conflict_directness_range"), 0.0, 1.0),
            detail_level=clamp(self._control_float("detail_level_range"), 0.0, 1.0),
        )

    def _beliefs(self, options: list[OptionCard], participant_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        try:
            data = self._llm.generate_json(prompts.belief_generation(self.topic, options, participant_summaries), profile="setup")
            rows = data.get("beliefs", [])
            if isinstance(rows, list):
                return {str(row.get("id")): row for row in rows if isinstance(row, dict)}
        except Exception:
            pass
        # Deterministic fallback: varied preferences, broad acceptability.
        labels = [o.id for o in options]
        fallback: dict[str, dict[str, Any]] = {}
        for idx, row in enumerate(participant_summaries):
            preferred = labels[idx % len(labels)]
            acceptable = list(dict.fromkeys([preferred, labels[(idx + 1) % len(labels)]]))
            fallback[row["id"]] = {
                "private_goal": "find something that works for the group without overcomplicating it",
                "preferred_option": preferred,
                "acceptable_options": acceptable,
                "soft_rejections": [],
                "hard_rejections": [],
                "reasons": {preferred: [f"Option {preferred} fits their main priority."], acceptable[-1]: ["It could work as a compromise."]},
                "reservation": "they do not want the group to ignore the main trade-off",
                "reconsider_if": "the others show another option covers the main concern well enough",
            }
        return fallback

    def _persona_from(
        self,
        entry: dict[str, str],
        traits: TraitProfile,
        belief: dict[str, Any],
        options: list[OptionCard],
        hard_blocker: bool,
    ) -> Persona:
        labels = [o.id for o in options]
        preferred = str(belief.get("preferred_option") or labels[0]).strip().upper()
        if preferred not in labels:
            preferred = labels[0]
        acceptable = [str(x).strip().upper() for x in belief.get("acceptable_options", []) if str(x).strip().upper() in labels]
        acceptable = list(dict.fromkeys([preferred, *acceptable]))
        if not hard_blocker:
            min_accept = int(cfg.personas.non_blocker_min_acceptable)
            for label in labels:
                if len(acceptable) >= min_accept:
                    break
                if label not in acceptable:
                    acceptable.append(label)
        soft_rej = [str(x).strip().upper() for x in belief.get("soft_rejections", []) if str(x).strip().upper() in labels and str(x).strip().upper() not in acceptable]
        soft_rej = soft_rej[: int(cfg.personas.non_blocker_max_soft_rejections) if not hard_blocker else len(labels)]
        hard_rej = [str(x).strip().upper() for x in belief.get("hard_rejections", []) if str(x).strip().upper() in labels]
        if not hard_blocker:
            hard_rej = []
        reasons_raw = belief.get("reasons", {})
        reasons: dict[str, list[str]] = {}
        if isinstance(reasons_raw, dict):
            for option_id, texts in reasons_raw.items():
                oid = str(option_id).strip().upper()
                if oid in labels:
                    if isinstance(texts, list):
                        reasons[oid] = [str(t).strip() for t in texts if str(t).strip()]
                    elif str(texts).strip():
                        reasons[oid] = [str(texts).strip()]
        if preferred not in reasons or not reasons[preferred]:
            reasons[preferred] = [f"Option {preferred} best matches what they care about."]
        style = self._speech_style(traits)
        return Persona(
            id=entry["id"],
            name=entry["name"],
            role=entry["role"],
            traits=traits,
            speech_style=style,
            private_goal=str(belief.get("private_goal") or "find a workable group decision").strip(),
            preferred_option=preferred,
            acceptable_options=acceptable,
            soft_rejections=soft_rej,
            hard_rejections=hard_rej,
            reasons=reasons,
            reservation=str(belief.get("reservation") or "they worry the trade-off may be ignored").strip(),
            reconsider_if=str(belief.get("reconsider_if") or "another option clearly handles their main concern").strip(),
            is_hard_blocker=hard_blocker,
        )

    def _speech_style(self, traits: TraitProfile) -> str:
        parts: list[str] = []
        if traits.response_length <= 2:
            parts.append("brief")
        elif traits.response_length >= 4:
            parts.append("a bit chatty")
        else:
            parts.append("normal length")
        if traits.conflict_directness >= 0.65:
            parts.append("direct when disagreeing")
        elif traits.conflict_directness <= 0.35:
            parts.append("softens disagreement")
        if traits.detail_level >= 0.65:
            parts.append("uses concrete option details")
        else:
            parts.append("speaks from priorities and impressions")
        return ", ".join(parts)

    def _ensure_minimum_preference_diversity(self, personas: list[Persona], options: list[OptionCard]) -> None:
        required = int(cfg.personas.preferred_diversity_min_unique)
        if len({p.preferred_option for p in personas}) >= min(required, len(options)):
            return
        # Only adjust if no useful diversity exists, and keep reasons consistent.
        labels = [o.id for o in options]
        for idx, persona in enumerate(personas):
            new_pref = labels[idx % len(labels)]
            persona.preferred_option = new_pref
            if new_pref not in persona.acceptable_options:
                persona.acceptable_options.insert(0, new_pref)
            persona.reasons.setdefault(new_pref, [f"Option {new_pref} fits their personal priority."])
            if len({p.preferred_option for p in personas}) >= min(required, len(options)):
                break

    def _ensure_common_compromise(self, personas: list[Persona], options: list[OptionCard]) -> None:
        if not bool(cfg.personas.require_group_common_acceptable) or not personas:
            return
        labels = [o.id for o in options]
        for label in labels:
            if all(label in p.acceptable_options or p.is_hard_blocker for p in personas):
                return
        # Pick the least disruptive option: the most common currently acceptable label.
        counts = {label: 0 for label in labels}
        for p in personas:
            for label in p.acceptable_options:
                counts[label] += 1
        common = max(counts, key=counts.get)
        for p in personas:
            if not p.is_hard_blocker and common not in p.acceptable_options:
                p.acceptable_options.append(common)
                p.reasons.setdefault(common, [f"Option {common} could work as a compromise."])
