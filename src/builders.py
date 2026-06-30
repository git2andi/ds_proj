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
from config_loader import cfg, parse_preference_shape
from llm_client import get_llm_client
from models import OptionCard, Persona, Scenario, TraitProfile
from utils import sample_int_range

_TOPIC_COUNT_PATTERNS = [
    re.compile(r"(?P<count>\d+|two|three|four|five|six|seven)\s+(?:friends|students|colleagues|participants|players|teammates|people|of\s+us)\b", re.I),
    re.compile(r"\bgroup\s+of\s+(?P<count>\d+|two|three|four|five|six|seven)\b", re.I),
    re.compile(r"\bteam\s+of\s+(?P<count>\d+|two|three|four|five|six|seven)\b", re.I),
]
_TOPIC_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

_INCOMPLETE_NAME_ENDINGS = frozenset({"a", "an", "the", "to", "from", "and", "or", "of", "in", "at", "by", "for", "with", "on", "via", "but", "&"})

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
        self._validate_topic_participant_count(self.topic, n)
        preference_shape = self._preference_shape(n, len(cfg.scenario.option_labels))
        trait_rows = self._trait_rows(n)
        attempts = max(1, int(cfg.simulation.setup_generation_attempts))
        scenario: Scenario | None = None
        options_json: list[dict] = []
        scenario_errors: list[str] = []
        for _ in range(attempts):
            try:
                scenario, options_json = self._generate_scenario(n)
                break
            except Exception as exc:
                scenario_errors.append(f"{type(exc).__name__}: {exc}")
        if scenario is None:
            raise RuntimeError(
                f"Scenario setup failed at scenario stage for topic {self.topic!r} "
                f"after {attempts} attempt(s): {' | '.join(scenario_errors)}. "
                "Check the authorized LLM endpoint/provider in config.yaml."
            )

        required_preferences = self._preference_assignments(n, scenario.option_ids, preference_shape)
        persona_errors: list[str] = []
        for _ in range(attempts):
            try:
                personas = self._generate_personas(
                    n, trait_rows, required_preferences, options_json, scenario
                )
                self._validate_preference_assignments(personas, required_preferences)
                self._validate_world(scenario, personas)
                return scenario, personas
            except Exception as exc:
                persona_errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"Scenario setup failed at persona stage for topic {self.topic!r} "
            f"after {attempts} attempt(s): {' | '.join(persona_errors)}. "
            "The validated scenario was preserved across persona retries."
        )

    def _generate_scenario(self, n: int) -> tuple[Scenario, list[dict]]:
        data = self._llm.generate_json(prompts.setup_scenario(self.topic, n), profile="setup")
        raw_scenario = data.get("scenario", data)
        scenario = self._parse_scenario(raw_scenario, n)
        options_json = raw_scenario.get("options", [])
        return scenario, options_json

    def _generate_personas(self, n: int, trait_rows: list[dict], required_preferences: dict[str, str],
                           options_json: list[dict], scenario: Scenario) -> list[Persona]:
        data = self._llm.generate_json(
            prompts.setup_personas(self.topic, n, trait_rows, required_preferences, options_json),
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

    def _preference_shape(self, n: int, option_count: int) -> tuple[int, ...]:
        distribution = cfg.personas.preference_distribution
        forced = distribution.get("forced_shape")
        if forced is not None:
            shape = parse_preference_shape(forced)
        else:
            weights_by_size = distribution.shape_weights
            raw_weights = weights_by_size.get(n, weights_by_size.get(str(n)))
            if not isinstance(raw_weights, dict) or not raw_weights:
                raise ValueError(f"No preference shape weights configured for group size {n}")
            shape_names = list(raw_weights)
            shape = parse_preference_shape(random.choices(
                shape_names,
                weights=[float(raw_weights[name]) for name in shape_names],
                k=1,
            )[0])
        if sum(shape) != n:
            raise ValueError(f"Preference shape {shape} must sum to participant count {n}")
        if len(shape) > option_count:
            raise ValueError(
                f"Preference shape {shape} needs {len(shape)} distinct options, but only {option_count} exist"
            )
        return shape

    def _preference_assignments(
        self,
        n: int,
        option_ids: list[str],
        shape: tuple[int, ...] | None = None,
    ) -> dict[str, str]:
        """Assign a concrete required primary option to every participant before prompting."""
        shape = shape or self._preference_shape(n, len(option_ids))
        ids = [f"p{i + 1}" for i in range(n)]
        chosen_options = random.sample(option_ids, len(shape))
        random.shuffle(ids)
        assignments: dict[str, str] = {}
        cursor = 0
        for group_size, option_id in zip(shape, chosen_options):
            for pid in ids[cursor:cursor + group_size]:
                assignments[pid] = option_id
            cursor += group_size
        return assignments

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
    def _validate_topic_participant_count(topic: str, n: int) -> None:
        """Fail fast if the topic explicitly names a group size that contradicts configured n.
        Raises before any LLM call so the user gets a clear message rather than a
        contradicted world that passes structural validation."""
        for pattern in _TOPIC_COUNT_PATTERNS:
            match = pattern.search(topic)
            if not match:
                continue
            raw = match.group("count").lower()
            count = int(raw) if raw.isdigit() else _TOPIC_COUNT_WORDS.get(raw)
            if count and count != n:
                raise ValueError(
                    f"Topic mentions {count} participant(s) but simulation.num_participants={n}. "
                    f"Set num_participants={count} in config.yaml or rephrase the topic to omit the count."
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
        words = raw.split()
        cap = int(cfg.scenario.option_name_max_words)
        if len(words) > cap:
            truncated = words[:cap]
            if truncated[-1].lower() in _INCOMPLETE_NAME_ENDINGS:
                raise ValueError(
                    f"Option name truncated mid-phrase (last word {truncated[-1]!r} is a function word): {raw!r}"
                )
            return " ".join(truncated)
        return " ".join(words)

    @staticmethod
    def _validate_preference_assignments(
        personas: list[Persona], required_preferences: dict[str, str]
    ) -> None:
        personas_by_id = {persona.id: persona for persona in personas}
        if set(personas_by_id) != set(required_preferences):
            raise ValueError("persona ids do not match the required preference assignments")
        for pid, required in required_preferences.items():
            actual = personas_by_id[pid].preferred_option
            if actual != required:
                raise ValueError(
                    f"participant {pid} primary preference must be {required}, got {actual}"
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

