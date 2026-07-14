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
from config_loader import PROFILE_TRAIT_NAMES, cfg, parse_preference_shape
from llm_client import get_llm_client
from models import (
    OptionCard, OptionStance, Persona, Scenario, SimulatorParameters,
    STANCE_ACCEPTABLE, STANCE_DISLIKED, STANCE_NEUTRAL, STANCE_PREFERRED, STANCE_REJECTED,
)
from utils import sample_int_range

_TOPIC_COUNT_PATTERNS = [
    re.compile(r"(?P<count>\d+|two|three|four|five|six|seven)\s+(?:friends|students|colleagues|participants|players|teammates|people|of\s+us)\b", re.I),
    re.compile(r"\bgroup\s+of\s+(?P<count>\d+|two|three|four|five|six|seven)\b", re.I),
    re.compile(r"\bteam\s+of\s+(?P<count>\d+|two|three|four|five|six|seven)\b", re.I),
]
_TOPIC_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

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


def _speech_style_for_age(age: int) -> str:
    """Return the compact, age-consistent speech-style register.

    speech_style is small register coloring only: it changes wording, never
    preferences, votes, decision behavior, turn length, or directness.
    """
    age = max(16, min(85, int(age)))
    if age <= 27:
        return "young casual wording"
    if age <= 40:
        return "relaxed practical wording"
    if age <= 58:
        return "direct workplace wording"
    return "measured traditional wording"


def _age_for_profile(profile: dict, traits: SimulatorParameters | None = None) -> int:
    """Use manual age when present, otherwise sample a stable plausible age.

    Traits get a weak influence so generated casts are not fully uniform, but age
    remains a surface-style attribute, not a decision variable.
    """
    raw = profile.get("age")
    if raw is not None and str(raw).strip():
        return max(16, min(85, int(raw)))
    # Age is lexical metadata only and is deliberately independent of the
    # behavioral traits.
    return random.randint(20, 65)


def _speech_style_for_profile(profile: dict, age: int) -> str:
    manual = str(profile.get("speech_style") or "").strip()
    return manual if manual else _speech_style_for_age(age)




_YOUNG_FAMILY_RE = re.compile(
    r"\b(?:married|spouse|husband|wife|fianc[eé]|children|kids?|toddler|baby|son|daughter|parent\s+of|mother\s+of|father\s+of)\b",
    re.I,
)
_YOUNG_ESTABLISHED_RE = re.compile(
    r"\b(?:mortgage|homeowner|owns?\s+(?:a\s+)?home|senior\s+(?:manager|lead|director)|director\s+of|executive|head\s+of|decades?\s+of)\b",
    re.I,
)
_LONG_EXPERIENCE_RE = re.compile(r"\b(\d{2})\+?\s+years?\s+(?:of\s+)?(?:experience|in\s+the\s+field|working)\b", re.I)
_OLDER_YOUTH_RE = re.compile(r"\b(?:teen(?:ager)?|high[- ]school student|first[- ]year student|apprentice living with parents)\b", re.I)


def _age_plausibility_issues(age: int, *texts: str) -> list[str]:
    """Return obvious age/backstory contradictions.

    This is intentionally conservative. It catches absurd setup artifacts such as
    a 19-year-old married parent with a mortgage or a 21-year-old senior director,
    but it does not try to model a full biography.
    """
    joined = " ".join(str(text or "") for text in texts)
    issues: list[str] = []
    if age <= 22 and _YOUNG_FAMILY_RE.search(joined):
        issues.append("very young participant has spouse/children/family role")
    if age <= 24 and _YOUNG_ESTABLISHED_RE.search(joined):
        issues.append("very young participant has established-life or senior-career marker")
    if age <= 26 and re.search(r"\bsenior\s+(?:manager|lead|director)|\bexecutive\b|\bhead\s+of\b", joined, re.I):
        issues.append("young participant has implausibly senior career")
    for match in _LONG_EXPERIENCE_RE.finditer(joined):
        years = int(match.group(1))
        if age - years < 16:
            issues.append(f"{years} years of experience is implausible for age {age}")
    if age >= 56 and _OLDER_YOUTH_RE.search(joined):
        issues.append("older participant has youth/student marker")
    return issues


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


def _sample_names(n: int, exclude: list[str] | None = None) -> list[str]:
    excluded = {name.lower() for name in (exclude or [])}
    pool = [name for name in _NAME_POOL if name.lower() not in excluded]
    return random.sample(pool, min(n, len(pool)))


def manual_environment() -> dict | None:
    """The manual environment mapping when environment.mode=manual, else None.

    Config validation (config_loader) has already checked structure, option count,
    and required fields; callers can consume the mapping directly.
    """
    environment = cfg.get("environment", None) or {}
    if str(environment.get("mode", "auto")) != "manual":
        return None
    return environment.get("manual") or {}


def manual_participant_profiles() -> list[dict]:
    """Return normalized manual profiles using direct simulator traits."""
    participants = cfg.get("participants", None) or {}
    if str(participants.get("mode", "auto")) != "manual":
        return []
    profiles: list[dict] = []
    for row in participants.get("profiles") or []:
        preferred = str(row.get("preferred_option") or "").strip().upper() or None
        rejection = str(row.get("rejection") or "").strip().upper() or None
        profiles.append({
            "name": str(row.get("name") or "").strip(),
            "description": str(row.get("description") or "").strip(),
            "private_goal": str(row.get("private_goal") or "").strip(),
            "preferred_option": preferred,
            "age": int(row["age"]) if row.get("age") is not None and str(row.get("age")).strip() else None,
            "speech_style": str(row.get("speech_style") or "").strip(),
            "hard_blocker": bool(row.get("hard_blocker", False)),
            "rejection": rejection,
            "rejection_reason": str(row.get("rejection_reason") or "").strip(),
            "traits": {key: int(value) for key, value in (row.get("traits") or {}).items()},
        })
    return profiles


def repair_preferred_options(
    preferred: list[str],
    rejection: str | None,
    required: str | None,
    single_only: bool,
) -> list[str]:
    """Deterministically align a persona's preference list with its assignment.

    The setup builder assigns the required primary option before prompting, so a
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


def _clip_reason(text: str, limit: int = 11) -> str:
    words = str(text or "").strip().split()
    if not words:
        return ""
    return " ".join(words[:limit]).rstrip(" ,.;:")


def _option_hint(option: OptionCard, positive: bool) -> str:
    if positive:
        return _clip_reason(option.upside, 10)
    return _clip_reason(option.concern, 10)


def _stance_from_option_table(row: dict[str, Any], labels: list[str], scenario: Scenario) -> dict[str, OptionStance]:
    """Parse the optional per-sim/per-option compatibility table.

    Missing rows stay neutral. The table is an initial stance guide, not a final
    script: most options should remain neutral/acceptable, with hard rejects only
    when explicitly configured or generated for the sole hard blocker.
    """
    raw = row.get("option_stances") or []
    by_id: dict[str, OptionStance] = {}
    if isinstance(raw, dict):
        iterable = [{"option": oid, **(value if isinstance(value, dict) else {"rank": value})} for oid, value in raw.items()]
    elif isinstance(raw, list):
        iterable = raw
    else:
        iterable = []
    for item in iterable:
        if not isinstance(item, dict):
            continue
        oid = str(item.get("option") or item.get("option_id") or item.get("id") or "").strip().upper()
        if oid not in labels:
            continue
        try:
            rank = int(item.get("rank", STANCE_NEUTRAL))
        except (TypeError, ValueError):
            rank = STANCE_NEUTRAL
        by_id[oid] = OptionStance(
            option_id=oid,
            rank=max(STANCE_REJECTED, min(STANCE_PREFERRED, rank)),
            reason_for=_clip_reason(item.get("reason_for", "")),
            reason_against=_clip_reason(item.get("reason_against", "")),
        ).clipped()
    for oid in labels:
        by_id.setdefault(oid, OptionStance(option_id=oid, rank=STANCE_NEUTRAL))
    return by_id


def _normalise_initial_stances(
    scenario: Scenario,
    stances: dict[str, OptionStance],
    preferred_options: list[str],
    rejection: str | None,
    rejection_reason: str,
    exclusive_blocker: bool = False,
) -> dict[str, OptionStance]:
    labels = scenario.option_ids
    normal: dict[str, OptionStance] = {}
    preferred = preferred_options[0] if preferred_options else None
    secondary = set(preferred_options[1:])
    for oid in labels:
        option = scenario.option(oid)
        source = stances.get(oid, OptionStance(option_id=oid, rank=STANCE_NEUTRAL))
        rank = source.rank
        reason_for = source.reason_for
        reason_against = source.reason_against
        if oid == preferred:
            rank = STANCE_PREFERRED
            reason_for = reason_for or _option_hint(option, True)
            reason_against = ""
        elif exclusive_blocker:
            # A sampled hard blocker hard-rejects every non-preferred
            # option with a grounded reason; the simulator must not
            # silently remain neutral or acceptable toward an alternative.
            rank = STANCE_REJECTED
            reason_against = (
                reason_against
                or _clip_reason(rejection_reason)
                or _option_hint(option, False)
                or "does not meet their one non-negotiable requirement"
            )
            reason_for = ""
        elif oid in secondary:
            rank = max(rank, STANCE_ACCEPTABLE)
            reason_for = reason_for or _option_hint(option, True)
        elif oid == rejection:
            rank = STANCE_REJECTED
            reason_against = _clip_reason(rejection_reason) or reason_against or _option_hint(option, False)
            reason_for = ""
        else:
            rank = min(max(rank, STANCE_DISLIKED), STANCE_ACCEPTABLE)
            # Avoid over-biasing neutral options: keep only one short side unless
            # the setup LLM explicitly provided it.
            if rank >= STANCE_ACCEPTABLE:
                reason_for = reason_for or _option_hint(option, True)
            elif rank <= STANCE_DISLIKED:
                reason_against = reason_against or _option_hint(option, False)
        normal[oid] = OptionStance(oid, rank, _clip_reason(reason_for), _clip_reason(reason_against)).clipped()
    return normal


class SetupBuilder:
    def __init__(
        self,
        topic: str,
        *,
        force_auto_scenario: bool = False,
        llm=None,
    ) -> None:
        self.topic = topic.strip()
        seed = cfg.simulation.get("random_seed", None)
        if seed is not None:
            random.seed(int(seed))
        # participants.mode stays independent: manual profiles combine freely
        # with an automatically generated scenario (explicit CLI topic).
        self._profiles = manual_participant_profiles()
        self._manual_env = None if force_auto_scenario else manual_environment()
        if self._manual_env:
            self.topic = str(self._manual_env["topic"]).strip()
        self._llm = llm or get_llm_client()  # one shared setup/dialogue provider
        self._hard_blocker_id: str | None = None

    def build(self, n: int) -> tuple[Scenario, list[Persona]]:
        if self._profiles and len(self._profiles) != n:
            raise ValueError(
                f"participants.profiles defines {len(self._profiles)} simulators but build() was asked for {n}"
            )
        if self._manual_env is None:
            self._validate_topic_participant_count(self.topic, n)
        # With any manually pinned preference the shape distribution is bypassed,
        # so only precompute (and fail fast on) the shape when it will be used.
        preference_shape = None if self._pinned_preferences() else self._preference_shape(n, len(cfg.scenario.option_labels))
        trait_rows = self._trait_rows(n)
        attempts = max(1, int(cfg.simulation.setup_generation_attempts))
        scenario: Scenario | None = None
        options_json: list[dict] = []
        scenario_errors: list[str] = []
        if self._manual_env is not None:
            # Manual environment: deterministic, no scenario LLM call and no
            # retry loop — any problem is a config error and should surface.
            scenario, options_json = self._manual_scenario(n)
        else:
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
        if self._profiles_complete():
            # Fully specified manual cast: no persona LLM call, no sampling noise.
            self._current_scenario = scenario
            personas = self._personas_from_profiles(trait_rows, required_preferences)
            self._validate_preference_assignments(personas, required_preferences)
            self._validate_world(scenario, personas)
            return scenario, personas
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

    def _manual_scenario(self, n: int) -> tuple[Scenario, list[dict]]:
        """Build the Scenario deterministically from environment.manual.

        The user-authored cards are the factual source of truth: numbers are never
        rewritten. If an option violates a hard cap stated in the manual shared
        context, that is a configuration contradiction and the run fails with the
        violation list. The auto path's attribute-count band and its group-size
        contradiction guards are deliberately not applied: those exist to catch
        the setup LLM disobeying the requested world, while a manual environment
        is author-owned (a shared fact like "25 colleagues will attend" describes
        the scenario, not the deciding group).
        """
        env = self._manual_env or {}
        labels = [str(x) for x in cfg.scenario.option_labels]
        options: list[OptionCard] = []
        for idx, row in enumerate(env.get("options") or []):
            name = self._clean_name(_require(row.get("name"), f"environment.manual.options[{idx}].name"))
            attrs = {
                str(k).strip(): str(v).strip()
                for k, v in (row.get("attrs") or {}).items()
                if str(k).strip() and str(v).strip()
            }
            # Manual mode has no retry loop: an invalid short_name is a config error.
            short_name = validated_short_alias(name, str(row.get("short_name") or ""))
            if not short_name:
                raise ValueError(
                    f"environment.manual.options[{idx}].short_name is missing or unusable: "
                    f"{row.get('short_name')!r} (a concise natural alias of the option name is required)"
                )
            options.append(OptionCard(
                id=labels[idx],
                name=name,
                short_name=short_name,
                attrs=attrs,
                upside=str(row.get("upside") or "").strip(),
                concern=str(row.get("concern") or "").strip(),
            ))
        self._require_unique_short_names(options)
        shared_context = [str(item).strip() for item in env.get("shared_context") or [] if str(item).strip()]
        scenario = Scenario(
            topic=self.topic,
            options=options,
            shared_context=shared_context,
        )
        violations = enforce_shared_caps(scenario, mutate=False)
        if violations:
            raise ValueError(
                "environment.manual options violate the manual shared-context caps: "
                + " | ".join(violations)
            )
        options_json = [
            {
                "id": option.id, "name": option.name, "short_name": option.short_name,
                "attrs": dict(option.attrs), "upside": option.upside, "concern": option.concern,
            }
            for option in options
        ]
        return scenario, options_json

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
            prompts.setup_personas(
                self.topic, n, trait_rows, required_preferences, options_json,
                list(scenario.shared_context),
                hard_blocker_id=self._hard_blocker_id,
            ),
            profile="setup",
        )
        return self._parse_personas(data.get("participants", []), trait_rows, scenario, required_preferences)

    def _trait_rows(self, n: int) -> list[dict[str, Any]]:
        """Sample direct simulator traits and at most one hard blocker."""
        manual_blockers = [
            f"p{idx + 1}" for idx, profile in enumerate(self._profiles)
            if profile.get("hard_blocker")
        ]
        if len(manual_blockers) > 1:
            raise ValueError("manual profiles may define at most one hard blocker")
        hard_id = manual_blockers[0] if manual_blockers else None
        if not self._profiles and n > 0 and random.random() < float(cfg.personas.hard_blocker_probability):
            hard_id = f"p{random.randint(1, n)}"
        self._hard_blocker_id = hard_id

        given_names = [p["name"] for p in self._profiles if p["name"]]
        fill_names = iter(_sample_names(n, exclude=given_names))
        rows: list[dict[str, Any]] = []
        for idx in range(n):
            pid = f"p{idx + 1}"
            profile = self._profiles[idx] if self._profiles else {}
            params = self._sample_traits(pid == hard_id, profile.get("traits") or {})
            row: dict[str, Any] = {
                "id": pid,
                "name": profile.get("name") or next(fill_names),
                "traits": asdict(params),
                "hard_blocker": pid == hard_id,
            }
            if profile.get("description"):
                row["background"] = profile["description"]
            if profile.get("private_goal"):
                row["private_goal"] = profile["private_goal"]
            if profile.get("age") is not None:
                row["age"] = profile["age"]
            rows.append(row)
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

    def _pinned_preferences(self) -> dict[str, str]:
        return {
            f"p{idx + 1}": profile["preferred_option"]
            for idx, profile in enumerate(self._profiles)
            if profile.get("preferred_option")
        }

    def _profile_for(self, pid: str) -> dict:
        if not self._profiles:
            return {}
        try:
            idx = int(pid[1:]) - 1
        except ValueError:
            return {}
        return self._profiles[idx] if 0 <= idx < len(self._profiles) else {}

    def _preference_assignments(
        self,
        n: int,
        option_ids: list[str],
        shape: tuple[int, ...] | None = None,
    ) -> dict[str, str]:
        """Assign a concrete required primary option to every participant before prompting."""
        pinned = self._pinned_preferences()
        if pinned:
            # Manual pins take precedence over the shape distribution; the
            # unpinned rest get a uniformly random option (never their own
            # rejection), documented in config.yaml.
            assignments = dict(pinned)
            for idx in range(n):
                pid = f"p{idx + 1}"
                if pid in assignments:
                    continue
                rejection = self._profile_for(pid).get("rejection")
                assignments[pid] = random.choice([o for o in option_ids if o != rejection])
            return assignments
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
        self._avoid_rejection_conflicts(assignments, option_ids)
        return assignments

    def _avoid_rejection_conflicts(self, assignments: dict[str, str], option_ids: list[str]) -> None:
        """Never require a manual profile's own rejected option as its primary preference.

        Prefer swapping two assignments (shape preserved); fall back to a random
        non-rejected option if no swap partner exists.
        """
        for pid, option in list(assignments.items()):
            rejection = self._profile_for(pid).get("rejection")
            if not rejection or rejection != option:
                continue
            for other, other_option in assignments.items():
                if other == pid or other_option == option:
                    continue
                if self._profile_for(other).get("rejection") != option:
                    assignments[pid], assignments[other] = other_option, option
                    break
            else:
                assignments[pid] = random.choice([o for o in option_ids if o != rejection])

    def _sample_traits(
        self, hard_blocker: bool, fixed: dict[str, int] | None = None
    ) -> SimulatorParameters:
        ranges = cfg.personas.trait_ranges
        values = {
            name: sample_int_range(ranges[name])
            for name in PROFILE_TRAIT_NAMES
        }
        if fixed:
            values.update({key: int(value) for key, value in fixed.items()})
        if hard_blocker:
            values["stubbornness"] = 5
        return SimulatorParameters(**values).validated(hard_blocker=hard_blocker)

    def _profiles_complete(self) -> bool:
        """True when every manual profile fully specifies the persona-level fields.

        Traits/parameters never need the LLM, so completeness only requires the
        text fields and the initial preference. A complete cast skips the persona
        LLM call entirely.
        """
        return bool(self._profiles) and all(
            profile["name"] and profile["description"] and profile["private_goal"] and profile["preferred_option"]
            for profile in self._profiles
        )

    def _personas_from_profiles(
        self, trait_rows: list[dict[str, Any]], required_preferences: dict[str, str]
    ) -> list[Persona]:
        personas: list[Persona] = []
        for row in trait_rows:
            pid = row["id"]
            profile = self._profile_for(pid)
            params = self._trait_from_row(row)
            hard_blocker = pid == self._hard_blocker_id
            preferred_options = repair_preferred_options(
                [], profile.get("rejection"), required_preferences.get(pid),
                single_only=hard_blocker,
            )
            option_stances = _normalise_initial_stances(
                self._current_scenario,
                {},
                preferred_options,
                profile.get("rejection"),
                profile.get("rejection_reason", ""),
                exclusive_blocker=hard_blocker,
            )
            age = _age_for_profile(profile, params)
            background = profile["description"]
            private_goal = profile["private_goal"]
            plausibility = _age_plausibility_issues(age, background, private_goal)
            if plausibility:
                raise ValueError(f"participant {pid} age/profile mismatch: {'; '.join(plausibility)}")
            personas.append(Persona(
                id=pid,
                name=row["name"],
                sim_params=params,
                background=background,
                private_goal=private_goal,
                preferred_options=preferred_options,
                age=age,
                speech_style=_speech_style_for_profile(profile, age),
                rejection=profile.get("rejection"),
                rejection_reason=profile.get("rejection_reason", ""),
                option_stances=option_stances,
                hard_blocker=hard_blocker,
            ))
        return personas

    def _parse_scenario(self, raw: Any, n: int) -> Scenario:
        if not isinstance(raw, dict):
            raise ValueError("setup.scenario must be an object")
        options_raw = raw.get("options")
        if not isinstance(options_raw, list):
            raise ValueError("scenario.options must be a list")
        labels = [str(x) for x in cfg.scenario.option_labels]
        parsed = [self._parse_option(item, labels[i]) for i, item in enumerate(options_raw[: len(labels)])]
        if len(parsed) != len(labels):
            raise ValueError("wrong number of options")
        options = [card for card, _proposed in parsed]
        # Alias problems never discard a substantively valid board: they get a
        # small alias-only repair call instead.
        alias_notes = self._ensure_valid_aliases(options, {card.id: prop for card, prop in parsed})
        ctx_raw = raw.get("shared_context", [])
        shared_context = [str(s).strip() for s in ctx_raw if str(s).strip()] if isinstance(ctx_raw, list) else []
        self._validate_participant_references(shared_context, n)
        scenario = Scenario(
            topic=self.topic,
            options=options,
            shared_context=shared_context,
        )
        scenario.setup_notes.extend(alias_notes)
        return scenario

    def _alias_problems(self, options: list[OptionCard], proposed: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        """(invalid, duplicate) alias diagnostics: option id -> offending alias."""
        invalid = {o.id: proposed.get(o.id, "") for o in options if not o.short_name}
        duplicates: dict[str, str] = {}
        seen: dict[str, str] = {}
        for option in options:
            if not option.short_name:
                continue
            key = option.short_name.casefold()
            if key in seen:
                duplicates[option.id] = option.short_name
            else:
                seen[key] = option.id
        return invalid, duplicates

    def _ensure_valid_aliases(self, options: list[OptionCard], proposed: dict[str, str]) -> list[str]:
        """Repair invalid/duplicate short aliases with a small alias-only LLM
        call; the option board itself is preserved. Returns setup notes.

        Repaired aliases pass the same deterministic validation as generated
        ones (words from the option name, length bounds, uniqueness); repair
        has a small explicit retry limit and a precise final error.
        """
        invalid, duplicates = self._alias_problems(options, proposed)
        if not invalid and not duplicates:
            return []
        notes = [
            *(f"invalid_alias: option {oid} short_name {alias!r} rejected" for oid, alias in sorted(invalid.items())),
            *(f"duplicate_alias: option {oid} short_name {alias!r} collides" for oid, alias in sorted(duplicates.items())),
        ]
        by_id = {o.id: o for o in options}
        rejected: dict[str, set[str]] = {oid: {alias} for oid, alias in {**invalid, **duplicates}.items()}
        for _attempt in range(2):  # explicit alias-repair retry limit
            need = sorted(set(invalid) | set(duplicates))
            used = [o.short_name for o in options if o.short_name and o.id not in need]
            option_rows = [
                {
                    "id": option.id,
                    "name": option.name,
                    "short_name": option.short_name or proposed.get(option.id, ""),
                    "attrs": dict(option.attrs),
                    "upside": option.upside,
                    "concern": option.concern,
                }
                for option in options
            ]
            prompt = prompts.alias_repair(
                topic=self.topic,
                option_rows=option_rows,
                invalid=invalid,
                duplicates=duplicates,
            )
            try:
                data = self._llm.generate_json(prompt, profile="setup")
            except Exception:
                data = {}
            if isinstance(data, dict):
                raw_aliases = data.get("short_names", data.get("aliases", data))
            else:
                raw_aliases = {}
            taken = {alias.casefold() for alias in used}
            for oid in need:
                candidate = str((raw_aliases or {}).get(oid) or "").strip()
                validated = validated_short_alias(by_id[oid].name, candidate)
                if validated and validated.casefold() not in taken:
                    by_id[oid].short_name = validated
                    taken.add(validated.casefold())
                    notes.append(f"alias_repaired: option {oid} short_name set to {validated!r}")
                elif candidate:
                    rejected.setdefault(oid, set()).add(candidate)
            invalid, duplicates = self._alias_problems(options, proposed)
            if not invalid and not duplicates:
                return notes
        remaining = {**invalid, **duplicates}
        raise ValueError(
            "alias_repair_failed: could not obtain valid unique short aliases for "
            + ", ".join(
                f"option {oid} (name {by_id[oid].name!r}; rejected: {sorted(rejected.get(oid, {alias}))})"
                for oid, alias in sorted(remaining.items())
            )
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

    def _parse_option(self, raw: Any, expected_id: str) -> tuple[OptionCard, str]:
        """Parse one option; returns (card, proposed_alias).

        Substantive option-field failures (missing name/upside/concern, too few
        attributes) still reject the scenario attempt. An unusable short_name
        leaves the card's short_name empty for the alias-only repair step — it
        is never silently derived by clipping words from the full name.
        """
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
            raise ValueError(f"option_field_failure: option {expected_id} has too few attributes")
        name = self._clean_name(_require(raw.get("name"), f"option {expected_id} name"))
        proposed = str(raw.get("short_name") or "").strip()
        card = OptionCard(
            id=str(raw.get("id") or expected_id).strip().upper(),
            name=name,
            short_name=validated_short_alias(name, proposed),
            attrs=clean_attrs,
            upside=_require(raw.get("upside"), f"option {expected_id} upside"),
            concern=_require(raw.get("concern"), f"option {expected_id} concern"),
        )
        return card, proposed

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
    def _trait_from_row(row: dict[str, Any]) -> SimulatorParameters:
        return SimulatorParameters(**row["traits"]).validated(
            hard_blocker=bool(row.get("hard_blocker", False))
        )

    def _persona_from_row(self, row: dict[str, Any], params: SimulatorParameters, scenario: Scenario, idx: int, pid: str,
                          required: str | None = None) -> Persona:
        labels = scenario.option_ids
        # Parse the generated preferred_options list (one or two items).
        raw_prefs = row.get("preferred_options") or []
        if not isinstance(raw_prefs, list):
            raw_prefs = [raw_prefs] if raw_prefs else []
        preferred_options = [
            str(x).strip().upper() for x in raw_prefs[:2]
            if str(x).strip().upper() in labels
        ]
        # Parse optional rejection (hard blockers only, but accepted from any row and validated later)
        rej_raw = str(row.get("rejection") or "").strip().upper()
        rejection: str | None = rej_raw if rej_raw in labels and rej_raw not in preferred_options else None
        rejection_reason = str(row.get("rejection_reason") or "").strip() if rejection else ""
        # Manual profile fields override whatever the LLM generated for them;
        # everything the profile leaves open keeps the generated value.
        profile = self._profile_for(pid)
        if profile.get("rejection"):
            rejection = profile["rejection"]
            rejection_reason = profile["rejection_reason"]
            preferred_options = [opt for opt in preferred_options if opt != rejection]
        preferred_options = repair_preferred_options(
            preferred_options, rejection, required, single_only=(pid == self._hard_blocker_id)
        )
        if not preferred_options:
            raise ValueError(f"participant {pid} has no valid preferred_options")
        exclusive_blocker = pid == self._hard_blocker_id
        raw_stances = _stance_from_option_table(row, labels, scenario)
        option_stances = _normalise_initial_stances(
            scenario, raw_stances, preferred_options, rejection, rejection_reason,
            exclusive_blocker=exclusive_blocker,
        )
        raw_age = row.get("age")
        profile_age = profile.get("age")
        age_source = profile_age if profile_age is not None else raw_age
        age = _age_for_profile({"age": age_source} if age_source is not None else profile, params)
        # speech_style is builder-derived from age (or a manual profile
        # override); the persona LLM never writes it.
        speech_style = _speech_style_for_profile(profile, age)
        background = profile.get("description") or _require(
            row.get("background"),
            f"participant {pid} background",
        )
        private_goal = profile.get("private_goal") or _require(
            row.get("private_goal"), f"participant {pid} private_goal"
        )
        plausibility = _age_plausibility_issues(age, background, private_goal)
        if plausibility:
            raise ValueError(f"participant {pid} age/profile mismatch: {'; '.join(plausibility)}")
        persona = Persona(
            id=pid,
            name=_require(row.get("name"), f"participant {pid} name"),
            sim_params=params,
            background=background,
            private_goal=private_goal,
            preferred_options=preferred_options,
            age=age,
            speech_style=speech_style,
            rejection=rejection,
            rejection_reason=rejection_reason,
            option_stances=option_stances,
            hard_blocker=exclusive_blocker,
        )
        return persona

    @staticmethod
    def _clean_name(raw: str) -> str:
        """Whitespace-normalize an option name. Full names are never shortened
        or otherwise mutated; concise references come from short_name."""
        return " ".join(str(raw).split())

    @staticmethod
    def _require_unique_short_names(options: list[OptionCard]) -> None:
        seen: dict[str, str] = {}
        for option in options:
            key = option.short_name.casefold()
            if key in seen:
                raise ValueError(
                    f"options {seen[key]} and {option.id} share the short_name {option.short_name!r}; "
                    "short names must be unique"
                )
            seen[key] = option.id

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
            if persona.hard_blocker and len(persona.preferred_options) > 1:
                raise ValueError(f"hard blocker {persona.id} should have exactly one preferred option")
            # Exclusive hard-blocker contract: a sampled
            # blocker has exactly one rank-5 option and hard-rejects every
            # alternative with a grounded reason; a non-blocker must never end
            # up with that exclusive pattern (at most the one manual/LLM
            # rejection). Violations raise into the existing retry path.
            stances = persona.option_stances or {}
            ranks = {oid: stances[oid].rank if oid in stances else STANCE_NEUTRAL for oid in labels}
            rejected_ids = sorted(oid for oid, rank in ranks.items() if rank == STANCE_REJECTED)
            if persona.hard_blocker:
                preferred_ids = sorted(oid for oid, rank in ranks.items() if rank == STANCE_PREFERRED)
                if len(persona.preferred_options) != 1 or preferred_ids != [persona.preferred_option]:
                    raise ValueError(
                        f"hard blocker {persona.id} must have exactly one rank-5 option "
                        f"({persona.preferred_option}), got {preferred_ids}"
                    )
                others = sorted(oid for oid in labels if oid != persona.preferred_option)
                if rejected_ids != others:
                    raise ValueError(
                        f"hard blocker {persona.id} must hard-reject every alternative; "
                        f"rejected {rejected_ids}, expected {others}"
                    )
                for oid in others:
                    if not stances[oid].reason_against:
                        raise ValueError(
                            f"hard blocker {persona.id} rejects {oid} without a grounded reason"
                        )
            elif len(rejected_ids) > (1 if persona.rejection else 0):
                raise ValueError(
                    f"participant {persona.id} was given hard rejections {rejected_ids} "
                    "without being the sampled hard blocker or having a manual rejection"
                )
            plausibility = _age_plausibility_issues(persona.age, persona.background, persona.private_goal)
            if plausibility:
                raise ValueError(
                    f"participant {persona.id} age/profile mismatch: {'; '.join(plausibility)}"
                )

