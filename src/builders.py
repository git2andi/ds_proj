"""Scenario and persona construction.

Two sequential LLM calls: the first creates the option cards, the second creates
participant belief states given those options.  Splitting keeps each call small
enough to avoid timeouts on slower endpoints.
If it cannot produce a valid world, build() raises rather than fabricating one.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import asdict
from typing import Any

import prompts
from aliases import validated_short_alias
from config_loader import cfg, parse_preference_shape
from llm_client import get_llm_client
from models import OptionCard, Persona, Scenario, TraitProfile
from simulator import build_initial_agenda, derive_simulator_parameters
from utils import sample_int_range

_TOPIC_COUNT_PATTERNS = [
    re.compile(r"(?P<count>\d+|two|three|four|five|six|seven)\s+(?:friends|students|colleagues|participants|players|teammates|people|of\s+us)\b", re.I),
    re.compile(r"\bgroup\s+of\s+(?P<count>\d+|two|three|four|five|six|seven)\b", re.I),
    re.compile(r"\bteam\s+of\s+(?P<count>\d+|two|three|four|five|six|seven)\b", re.I),
]
_TOPIC_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

_INCOMPLETE_NAME_ENDINGS = frozenset({"a", "an", "the", "to", "from", "and", "or", "of", "in", "at", "by", "for", "with", "on", "via", "but", "&"})

# --- I6: hard shared-context caps vs option attributes -----------------------
# Soft qualifiers make a number a guideline, not a cap.
_SOFT_CAP = re.compile(r"\b(?:around|about|roughly|approximately|moderate|flexible)\b", re.I)
_CAP_WORDS = r"(?:fixed\s+at|capped\s+at|caps?\s+at|cap\s+of|limited\s+to|max(?:imum)?\s+(?:of\s+|at\s+)?|no\s+more\s+than|at\s+most|up\s+to|under|within)"
_MONEY_CAP = re.compile(rf"\b{_CAP_WORDS}\s*[$€£]\s*([\d,]+(?:\.\d{{1,2}})?)", re.I)
_UNIT_CAP = re.compile(
    rf"\b{_CAP_WORDS}\s*([\d,]+(?:\.\d+)?)\s*(miles?|km|kilometers?|minutes?|mins?|hours?|hrs?)\b",
    re.I,
)
_PER_UNIT = re.compile(r"\bper\s+([a-z]+)\b", re.I)
_MONEY_KEYS = ("cost", "price", "budget", "fee", "rate")
# unit word -> (family, factor to the family's canonical unit).
# Canonical units: distance = km, duration = minutes. Normalizing lets an
# "under 2 hours" cap catch a "duration_minutes: 130" attribute (I15).
_UNIT_INFO = {
    "mile": ("distance", 1.609344), "miles": ("distance", 1.609344),
    "km": ("distance", 1.0), "kilometer": ("distance", 1.0), "kilometers": ("distance", 1.0),
    "minute": ("duration", 1.0), "minutes": ("duration", 1.0), "min": ("duration", 1.0), "mins": ("duration", 1.0),
    "hour": ("duration", 60.0), "hours": ("duration", 60.0), "hr": ("duration", 60.0), "hrs": ("duration", 60.0),
}
_UNIT_KEYS = {
    "distance": ("distance", "drive", "commute", "travel"),
    "duration": ("time", "duration", "travel", "wait", "setup", "commute", "length", "runtime"),
}
# A cap sentence that names an activity ("within 15 minutes walking distance")
# only binds attributes about that activity — never e.g. a wait time (I15).
_CAP_SCOPE = re.compile(r"\b(walk(?:ing)?|wait(?:ing)?|driv(?:e|ing)|commut(?:e|ing)|travel(?:ing)?|set[- ]?up)\b", re.I)
_SCOPE_TOKENS = {
    "walk": ("walk", "distance"),
    "wait": ("wait",),
    "driv": ("drive", "driving", "distance", "commute"),
    "commut": ("commute", "travel", "distance"),
    "travel": ("travel", "commute", "distance"),
    "set": ("setup", "set up", "set-up"),
}


def _to_number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _format_like(value: float, template: str) -> str:
    """Format a number in the same style as the number inside ``template``."""
    text = str(int(value)) if float(value).is_integer() else f"{value:g}"
    if "," in template and value >= 1000:
        text = f"{int(value):,}"
    return text


def shared_context_caps(shared_context: list[str]) -> list[dict]:
    """Hard numeric caps stated in shared context.

    Each cap: {"kind": "money"|"distance"|"duration", "value": float (as
    stated), "canon": float (canonical unit), "per": str|None, "source": fact}.
    Soft phrasings ("around $200") are ignored on purpose.
    """
    caps: list[dict] = []
    for fact in shared_context:
        if _SOFT_CAP.search(fact):
            continue
        per = _PER_UNIT.search(fact)
        per_word = per.group(1).lower() if per else None
        money = _MONEY_CAP.search(fact)
        if money:
            value = _to_number(money.group(1))
            caps.append({"kind": "money", "value": value, "canon": value, "per": per_word, "source": fact})
        unit = _UNIT_CAP.search(fact)
        if unit:
            family, factor = _UNIT_INFO[unit.group(2).lower()]
            value = _to_number(unit.group(1))
            scope_match = _CAP_SCOPE.search(fact)
            scope = next(
                (stem for stem in _SCOPE_TOKENS if scope_match and scope_match.group(1).lower().startswith(stem)),
                None,
            )
            caps.append({
                "kind": family, "value": value, "canon": value * factor,
                "per": per_word, "scope": scope, "source": fact,
            })
    return caps


def _attr_number(kind: str, key: str, value: str, assume_relevant: bool = False) -> tuple[float, str, float] | None:
    """(canonical value, matched substring, attr unit factor) of an attr
    relevant to a cap kind. The unit may sit in the value ("130 minutes") or in
    the attribute key itself ("duration_minutes: 130"). ``assume_relevant``
    skips the key-topic filter when the caller already scope-matched the attr."""
    key_l = key.lower().replace("_", " ")
    if kind == "money":
        if not any(k in key_l for k in _MONEY_KEYS):
            return None
        match = re.search(r"[$€£]?\s*([\d,]+(?:\.\d{1,2})?)", value)
        if not match:
            return None
        return _to_number(match.group(1)), match.group(1), 1.0
    unit_words = [u for u, (family, _) in _UNIT_INFO.items() if family == kind]
    key_unit = next((u for u in unit_words if re.search(rf"\b{u}\b", key_l)), None)
    if not assume_relevant and not key_unit and not any(k in key_l for k in _UNIT_KEYS.get(kind, ())):
        return None
    match = re.search(rf"([\d,]+(?:\.\d+)?)\s*({'|'.join(unit_words)})\b", value, re.I)
    if match:
        factor = _UNIT_INFO[match.group(2).lower()][1]
        return _to_number(match.group(1)) * factor, match.group(1), factor
    if key_unit:
        match = re.search(r"([\d,]+(?:\.\d+)?)", value)
        if match:
            factor = _UNIT_INFO[key_unit][1]
            return _to_number(match.group(1)) * factor, match.group(1), factor
    return None


def _per_basis(text: str) -> str | None:
    match = _PER_UNIT.search(text)
    return match.group(1).lower() if match else None


def enforce_shared_caps(scenario: Scenario, mutate: bool = True) -> list[str]:
    """Clamp option attributes that violate a hard shared-context cap.

    A cap and an attribute are only compared when their per-unit basis matches
    (a '$500 total' budget never clamps a 'cost per person' attribute); units
    are normalized within a family, so an "under 2 hours" cap catches a
    minutes attribute. Returns human-readable repair notes; with ``mutate``
    the offending attr values are rewritten in place (in the attr's own unit,
    floored so the result never exceeds the cap), otherwise the violations are
    only reported so the caller can retry generation instead (I15).
    """
    notes: list[str] = []
    caps = shared_context_caps(scenario.shared_context)
    for cap in caps:
        for option in scenario.options:
            for key, value in list(option.attrs.items()):
                basis = _per_basis(key) or _per_basis(value)
                if basis != cap["per"]:
                    continue
                scope = cap.get("scope")
                if scope:
                    haystack = f"{key} {value}".lower().replace("_", " ").replace("-", " ")
                    if not any(token in haystack for token in _SCOPE_TOKENS[scope]):
                        continue
                parsed = _attr_number(cap["kind"], key, value, assume_relevant=bool(scope))
                if parsed is None or parsed[0] <= cap["canon"]:
                    continue
                _, matched, factor = parsed
                clamped = math.floor((cap["canon"] / factor) * 10) / 10
                if mutate:
                    option.attrs[key] = value.replace(matched, _format_like(clamped, matched), 1)
                notes.append(
                    f"option {option.id} attr '{key}' value {matched} violates cap "
                    f"{cap['value']:g} (shared context: {cap['source']!r})"
                    + (f"; clamped to {clamped:g}" if mutate else "")
                )
    return notes

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


def repair_preferred_options(
    preferred: list[str],
    rejection: str | None,
    required: str | None,
    single_only: bool,
) -> list[str]:
    """Deterministically align a persona's preference list with its assignment.

    The controller assigns the required primary option before prompting, so a
    row that drops or reorders it is a formatting slip, not a different world —
    repair it instead of failing the whole persona batch. A rejection of the
    required option is a real contradiction and raises so the attempt retries.
    Hard blockers (``single_only``) keep exactly one preferred option.
    """
    repaired = list(preferred)
    if required:
        if rejection == required:
            raise ValueError(f"rejection {rejection} contradicts required preference {required}")
        if required in repaired:
            repaired.remove(required)
        repaired.insert(0, required)
        repaired = repaired[:2]
    if single_only:
        repaired = repaired[:1]
    return repaired


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
        for attempt in range(attempts):
            try:
                # Prefer a regenerated, genuinely valid board over clamping:
                # rewriting a violating number can fabricate a false fact about
                # a real-world named option (I15). Clamp only on the last try.
                scenario, options_json = self._generate_scenario(n, allow_clamp=(attempt == attempts - 1))
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

    def _generate_scenario(self, n: int, allow_clamp: bool = True) -> tuple[Scenario, list[dict]]:
        data = self._llm.generate_json(prompts.setup_scenario(self.topic, n), profile="setup")
        raw_scenario = data.get("scenario", data)
        scenario = self._parse_scenario(raw_scenario, n)
        # Hard numeric caps in shared context must hold for every option: an
        # invalid option must never be able to win the discussion (I6/I15).
        # Early attempts retry generation on violation; the final attempt
        # clamps deterministically.
        if not allow_clamp:
            violations = enforce_shared_caps(scenario, mutate=False)
            if violations:
                raise ValueError("options violate hard shared-context caps: " + " | ".join(violations))
        scenario.setup_notes.extend(enforce_shared_caps(scenario))
        options_json = raw_scenario.get("options", [])
        if scenario.setup_notes:
            # Keep the persona prompt consistent with the clamped option cards.
            by_id = {option.id: option for option in scenario.options}
            for row in options_json:
                oid = str(row.get("id", "")).strip().upper()
                if oid in by_id and isinstance(row.get("attrs"), dict):
                    for key in list(row["attrs"]):
                        if key.strip() in by_id[oid].attrs:
                            row["attrs"][key] = by_id[oid].attrs[key.strip()]
        return scenario, options_json

    def _generate_personas(self, n: int, trait_rows: list[dict], required_preferences: dict[str, str],
                           options_json: list[dict], scenario: Scenario) -> list[Persona]:
        data = self._llm.generate_json(
            prompts.setup_personas(self.topic, n, trait_rows, required_preferences, options_json),
            profile="setup",
        )
        return self._parse_personas(data.get("participants", []), trait_rows, scenario, required_preferences)

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

    def _parse_personas(self, rows: Any, trait_rows: list[dict[str, Any]], scenario: Scenario,
                        required_preferences: dict[str, str] | None = None) -> list[Persona]:
        if not isinstance(rows, list):
            raise ValueError("participants must be a list")
        traits_by_id = {row["id"]: self._trait_from_row(row) for row in trait_rows}
        names_by_id = {row["id"]: row.get("name", "") for row in trait_rows}
        required_preferences = required_preferences or {}
        personas: list[Persona] = []
        for idx, row in enumerate(rows[: len(trait_rows)]):
            if not isinstance(row, dict):
                raise ValueError("participant row must be an object")
            pid = str(row.get("id") or f"p{idx + 1}")
            if pid not in traits_by_id:
                pid = f"p{idx + 1}"
            if names_by_id.get(pid):
                row["name"] = names_by_id[pid]
            personas.append(self._persona_from_row(
                row, traits_by_id[pid], scenario, idx, pid, required_preferences.get(pid)
            ))
        if len(personas) != len(trait_rows):
            raise ValueError("wrong number of participants")
        return personas

    @staticmethod
    def _trait_from_row(row: dict[str, Any]) -> TraitProfile:
        raw = row["traits"]
        return TraitProfile(**raw)

    def _persona_from_row(self, row: dict[str, Any], traits: TraitProfile, scenario: Scenario, idx: int, pid: str,
                          required: str | None = None) -> Persona:
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
        # Parse optional rejection (hard blockers only, but accepted from any row and validated later)
        rej_raw = str(row.get("rejection") or "").strip().upper()
        rejection: str | None = rej_raw if rej_raw in labels and rej_raw not in preferred_options else None
        rejection_reason = str(row.get("rejection_reason") or "").strip() if rejection else ""
        preferred_options = repair_preferred_options(
            preferred_options, rejection, required, single_only=traits.agreeableness == 1
        )
        if not preferred_options:
            raise ValueError(f"participant {pid} has no valid preferred_options")
        sim_params = derive_simulator_parameters(traits)
        persona = Persona(
            id=pid,
            name=_require(row.get("name"), f"participant {pid} name"),
            traits=traits,
            sim_params=sim_params,
            background=_require(
                row.get("background") or row.get("backstory"),
                f"participant {pid} background",
            ),
            private_goal=_require(row.get("private_goal"), f"participant {pid} private_goal"),
            preferred_options=preferred_options,
            rejection=rejection,
            rejection_reason=rejection_reason,
        )
        persona.agenda = build_initial_agenda(persona)
        return persona

    @staticmethod
    def _clean_name(raw: str) -> str:
        """Normalize an option name without treating cosmetic length as a setup failure.

        The option name is not part of the factual source of truth; the full option
        card still carries attributes, upside, tradeoff, and concern. Earlier
        versions truncated at option_name_max_words and aborted when the cut ended
        on a function word. That made valid travel names such as
        "Two-stop flight with short layovers in Boston and Copenhagen" fail even
        though the scenario was usable.

        We therefore keep complete names up to a permissive cap and only shorten
        genuinely excessive names. If shortening is required, we avoid ending on
        connector words, but we do not fail the whole run for name wording.
        """
        words = str(raw).split()
        if not words:
            return ""

        configured_cap = int(cfg.scenario.option_name_max_words)
        # Cosmetic names produced by slower/local models are often 9-12 words.
        # Treat the configured cap as a display target, not as a hard validator.
        hard_cap = max(configured_cap, 12)
        if len(words) <= hard_cap:
            return " ".join(words)

        shortened = words[:hard_cap]
        while len(shortened) > 2 and shortened[-1].lower().strip(",.;:-") in _INCOMPLETE_NAME_ENDINGS:
            shortened.pop()
        return " ".join(shortened)

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

