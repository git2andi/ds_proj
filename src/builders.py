"""Scenario and persona construction.

Two sequential LLM calls: the first creates the option cards, the second creates
participant belief states given those options.  Splitting keeps each call small
enough to avoid timeouts on slower endpoints.
If it cannot produce a valid world, build() raises rather than fabricating one.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict
from typing import Any

import prompts
from aliases import validated_short_alias
from config_loader import cfg
from llm_client import get_llm_client
from models import OptionCard, Persona, Scenario, TraitProfile
from utils import sample_int_range

_NAME_POOL = [
    "Amir", "Beatriz", "Callum", "Daria", "Emeka", "Faye", "Goran", "Hana",
    "Ivan", "Juno", "Kenji", "Lila", "Marco", "Nadia", "Oscar", "Priya",
    "Quinn", "Rosa", "Sven", "Tala", "Uri", "Vera", "Wyatt", "Xena",
    "Yuki", "Zara", "Anton", "Cleo", "Diego", "Elif", "Felix", "Gemma",
    "Hugo", "Isla", "Jasper", "Kira", "Leo", "Mina", "Nico", "Olga",
    "Pavel", "Rina", "Sami", "Thea", "Vince", "Wren", "Yara", "Zeke",
]


def _sample_names(n: int) -> list[str]:
    return random.sample(_NAME_POOL, min(n, len(_NAME_POOL)))


def _require(value: Any, field: str) -> str:
    """Return the stripped string value, or raise if the model omitted it.

    Setup never fabricates chat content: a missing/blank required field means the
    LLM response is unusable, so we raise (the build() retry loop then re-tries and,
    failing that, aborts the run with a clear message) instead of papering over it
    with a canned default."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"setup response missing required field: {field}")
    return text


class SetupBuilder:
    def __init__(self, topic: str) -> None:
        self.topic = topic.strip()
        seed = cfg.simulation.get("random_seed", None)
        if seed is not None:
            random.seed(int(seed))
        self._llm = get_llm_client()

    def build(self, n: int) -> tuple[Scenario, list[Persona]]:
        trait_rows = self._trait_rows(n)
        plan = self._preference_plan(n)
        pref_groups = self._preference_groups(plan)
        attempts = max(1, int(cfg.simulation.setup_generation_attempts))
        last_error = ""
        for attempt in range(attempts):
            try:
                scenario, options_json = self._generate_scenario(n)
                personas = self._generate_personas(n, trait_rows, pref_groups, options_json, scenario)
                self._validate_preference_plan(personas, plan)
                self._validate_world(scenario, personas)
                return scenario, personas
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"Scenario setup failed for topic {self.topic!r} after {attempts} attempt(s). "
            f"Last error: {last_error}. Check the LLM endpoint/provider in config.yaml."
        )

    def _generate_scenario(self, n: int) -> tuple[Scenario, list[dict]]:
        data = self._llm.generate_json(prompts.setup_scenario(self.topic, n), profile="setup")
        raw_scenario = data.get("scenario", data)
        scenario = self._parse_scenario(raw_scenario, n)
        options_json = raw_scenario.get("options", [])
        return scenario, options_json

    def _generate_personas(self, n: int, trait_rows: list[dict], pref_groups: list[list[str]],
                           options_json: list[dict], scenario: Scenario) -> list[Persona]:
        data = self._llm.generate_json(
            prompts.setup_personas(self.topic, n, trait_rows, pref_groups, options_json),
            profile="setup",
        )
        return self._parse_personas(data.get("participants", []), trait_rows, scenario)

    def _trait_rows(self, n: int) -> list[dict[str, Any]]:
        hard_id = None
        if n > 0 and random.random() < float(cfg.personas.hard_blocker_probability):
            hard_id = f"p{random.randint(1, n)}"
        names = _sample_names(n)
        rows: list[dict[str, Any]] = []
        for idx in range(n):
            pid = f"p{idx + 1}"
            stubborn = pid == hard_id
            traits = self._sample_traits(stubborn)
            rows.append({
                "id": pid,
                "name": names[idx],
                "traits": asdict(traits),
            })
        return rows

    def _preference_plan(self, n: int) -> dict[str, int]:
        """Assign each participant id to a preference "camp". Most runs keep everyone in
        their own camp (all distinct), but with coalition_probability two or more share a
        camp, producing 2v1 / 3v1 / 5v2-style splits. Always at least two camps so there
        is something to discuss."""
        labels = [str(x) for x in cfg.scenario.option_labels]
        max_camps = min(n, len(labels))
        if max_camps <= 2 or n <= 2:
            camps = max(2, max_camps) if n >= 2 else 1
        elif random.random() < float(cfg.personas.coalition_probability):
            camps = random.randint(2, max_camps - 1)   # force at least one shared pair
        else:
            camps = max_camps
        ids = [f"p{i + 1}" for i in range(n)]
        random.shuffle(ids)
        plan: dict[str, int] = {}
        for idx, pid in enumerate(ids):
            plan[pid] = idx if idx < camps else random.randint(0, camps - 1)
        return plan

    @staticmethod
    def _preference_groups(plan: dict[str, int]) -> list[list[str]]:
        camps: dict[int, list[str]] = {}
        for pid, camp in plan.items():
            camps.setdefault(camp, []).append(pid)
        return [sorted(members, key=lambda p: int(p[1:])) for _, members in sorted(camps.items())]

    def _sample_traits(self, stubborn: bool) -> TraitProfile:
        ranges = cfg.personas.hard_blocker_trait_ranges if stubborn else cfg.personas.trait_ranges
        return TraitProfile(
            openness=sample_int_range(ranges.openness),
            conscientiousness=sample_int_range(ranges.conscientiousness),
            extraversion=sample_int_range(ranges.extraversion),
            agreeableness=sample_int_range(ranges.agreeableness),
            neuroticism=sample_int_range(ranges.neuroticism),
        )

    def _parse_scenario(self, raw: Any, n: int) -> Scenario:
        if not isinstance(raw, dict):
            raise ValueError("setup.scenario must be an object")
        options_raw = raw.get("options")
        if not isinstance(options_raw, list):
            raise ValueError("scenario.options must be a list")
        labels = [str(x) for x in cfg.scenario.option_labels]
        options = [self._parse_option(item, labels[i]) for i, item in enumerate(options_raw[: len(labels)])]
        if len(options) != len(labels):
            raise ValueError("wrong number of options")
        ctx_raw = raw.get("shared_context", [])
        shared_context = [str(s).strip() for s in ctx_raw if str(s).strip()] if isinstance(ctx_raw, list) else []
        self._validate_participant_references(shared_context, n)
        return Scenario(
            topic=self.topic,
            decision_kind=_require(raw.get("decision_kind"), "scenario.decision_kind"),
            opening_question=_require(raw.get("opening_question"), "scenario.opening_question"),
            options=options,
            shared_context=shared_context,
        )

    @staticmethod
    def _validate_participant_references(shared_context: list[str], n: int) -> None:
        number_words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
        count_pattern = r"(?P<count>\d+|two|three|four|five|six|seven)"
        patterns = [
            re.compile(rf"\bgroup\s+of\s+{count_pattern}\b", re.I),
            re.compile(rf"\b{count_pattern}\s+(?:friends|students|colleagues|participants|players|group\s+members)\b", re.I),
        ]
        for fact in shared_context:
            for pattern in patterns:
                match = pattern.search(fact)
                if not match:
                    continue
                raw_count = match.group("count").lower()
                count = int(raw_count) if raw_count.isdigit() else number_words[raw_count]
                if count != n:
                    raise ValueError(f"shared_context participant count {count} does not match requested {n}")

    def _parse_option(self, raw: Any, expected_id: str) -> OptionCard:
        if not isinstance(raw, dict):
            raise ValueError("each option must be an object")
        attrs = raw.get("attrs", {})
        if not isinstance(attrs, dict):
            attrs = {}
        clean_attrs = {str(k).strip(): str(v).strip() for k, v in attrs.items() if str(k).strip() and str(v).strip()}
        attr_min = int(cfg.scenario.public_attr_min)
        attr_max = int(cfg.scenario.public_attr_max)
        clean_attrs = dict(list(clean_attrs.items())[:attr_max])
        if len(clean_attrs) < attr_min:
            raise ValueError("option has too few attributes")
        name = self._clean_name(_require(raw.get("name"), f"option {expected_id} name"))
        return OptionCard(
            id=str(raw.get("id") or expected_id).strip().upper(),
            name=name,
            short_name=validated_short_alias(name, str(raw.get("short_name") or "")),
            attrs=clean_attrs,
            upside=_require(raw.get("upside"), f"option {expected_id} upside"),
            tradeoff=_require(raw.get("tradeoff"), f"option {expected_id} tradeoff"),
            concern=_require(raw.get("concern"), f"option {expected_id} concern"),
            best_for=_require(raw.get("best_for") or raw.get("best for"), f"option {expected_id} best_for"),
        )

    def _parse_personas(self, rows: Any, trait_rows: list[dict[str, Any]], scenario: Scenario) -> list[Persona]:
        if not isinstance(rows, list):
            raise ValueError("participants must be a list")
        traits_by_id = {row["id"]: self._trait_from_row(row) for row in trait_rows}
        names_by_id = {row["id"]: row.get("name", "") for row in trait_rows}
        personas: list[Persona] = []
        for idx, row in enumerate(rows[: len(trait_rows)]):
            if not isinstance(row, dict):
                raise ValueError("participant row must be an object")
            pid = str(row.get("id") or f"p{idx + 1}")
            if pid not in traits_by_id:
                pid = f"p{idx + 1}"
            if names_by_id.get(pid):
                row["name"] = names_by_id[pid]
            personas.append(self._persona_from_row(row, traits_by_id[pid], scenario, idx, pid))
        if len(personas) != len(trait_rows):
            raise ValueError("wrong number of participants")
        return personas

    @staticmethod
    def _trait_from_row(row: dict[str, Any]) -> TraitProfile:
        raw = row["traits"]
        return TraitProfile(**raw)

    def _persona_from_row(self, row: dict[str, Any], traits: TraitProfile, scenario: Scenario, idx: int, pid: str) -> Persona:
        labels = scenario.option_ids
        # Parse preferred_options (1–2 items); fall back to old preferred_option field
        raw_prefs = row.get("preferred_options") or []
        if not isinstance(raw_prefs, list):
            raw_prefs = [raw_prefs] if raw_prefs else []
        preferred_options = [
            str(x).strip().upper() for x in raw_prefs[:2]
            if str(x).strip().upper() in labels
        ]
        if not preferred_options:
            # backward-compat with old preferred_option single field
            old = str(row.get("preferred_option") or "").strip().upper()
            if old in labels:
                preferred_options = [old]
        if not preferred_options:
            raise ValueError(f"participant {pid} has no valid preferred_options")
        # Parse optional rejection (hard blockers only, but accepted from any row and validated later)
        rej_raw = str(row.get("rejection") or "").strip().upper()
        rejection: str | None = rej_raw if rej_raw in labels and rej_raw not in preferred_options else None
        rejection_reason = str(row.get("rejection_reason") or "").strip() if rejection else ""
        return Persona(
            id=pid,
            name=_require(row.get("name"), f"participant {pid} name"),
            traits=traits,
            background=_require(
                row.get("background") or row.get("backstory"),
                f"participant {pid} background",
            ),
            private_goal=_require(row.get("private_goal"), f"participant {pid} private_goal"),
            preferred_options=preferred_options,
            rejection=rejection,
            rejection_reason=rejection_reason,
        )

    @staticmethod
    def _clean_name(raw: str) -> str:
        # Cap over-long names without cutting mid-title or appending punctuation.
        words = raw.split()
        cap = int(cfg.scenario.option_name_max_words)
        return " ".join(words[:cap]) if len(words) > cap else " ".join(words)

    def _validate_preference_plan(self, personas: list[Persona], plan: dict[str, int]) -> None:
        pref_by_id = {p.id: p.preferred_options[0] for p in personas}
        camps: dict[int, set[str]] = {}
        for pid, camp_idx in plan.items():
            if pid in pref_by_id:
                camps.setdefault(camp_idx, set()).add(pref_by_id[pid])
        camp_options: dict[int, str] = {}
        for camp_idx, prefs in sorted(camps.items()):
            if len(prefs) > 1:
                raise ValueError(
                    f"Preference camp {camp_idx} members chose conflicting options {prefs} — "
                    "retry so the LLM can honor the coalition structure."
                )
            camp_options[camp_idx] = next(iter(prefs))
        if len(set(camp_options.values())) < len(camp_options):
            raise ValueError(
                f"Different preference camps converged on the same option {camp_options} — "
                "retry to produce the required preference diversity."
            )

    def _validate_world(self, scenario: Scenario, personas: list[Persona]) -> None:
        labels = [str(x) for x in cfg.scenario.option_labels]
        if scenario.option_ids != labels:
            raise ValueError(f"option ids must be {labels}, got {scenario.option_ids}")
        names = [p.name.lower() for p in personas]
        if len(set(names)) != len(names):
            raise ValueError("participant names must be unique")
        for persona in personas:
            if not persona.preferred_options:
                raise ValueError(f"participant {persona.id} has no preferred options")
            for opt in persona.preferred_options:
                if opt not in labels:
                    raise ValueError(f"participant {persona.id} has invalid preferred option {opt}")
            if persona.rejection and persona.rejection not in labels:
                raise ValueError(f"participant {persona.id} has invalid rejection {persona.rejection}")
            if persona.rejection and persona.rejection in persona.preferred_options:
                raise ValueError(f"participant {persona.id} cannot reject a preferred option")
            if persona.traits.agreeableness == 1 and len(persona.preferred_options) > 1:
                raise ValueError(f"hard blocker {persona.id} should have exactly one preferred option")

