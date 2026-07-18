"""Scenario and persona construction.

Setup intentionally validates only structural properties that are reliable for
arbitrary topics. Generated facts are never rewritten locally. An invalid
automatic scenario is regenerated once with the validation feedback.
"""

from __future__ import annotations

import random
import re
from typing import Any

from aliases import (
    normalize_option_text,
    unique_generated_aliases,
    validate_unique_aliases,
    validated_alias,
)
from config_loader import DIRECT_TRAIT_NAMES, cfg, parse_preference_shape
from llm_client import get_llm_client
from models import (
    OptionCard,
    OptionStance,
    Persona,
    Scenario,
    SimulatorParameters,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
)
from prompts import setup_aliases, setup_personas, setup_scenario


_NAME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]+$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_FALLBACK_NAMES = (
    "Alex", "Maya", "Jonas", "Lea", "Nora", "Omar", "Sofia",
    "Tariq", "Mira", "Ben", "Lina", "Eli", "Zara", "Noah",
)


def _require(value: Any, field: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _validated_person_name(raw: object, *, participant_id: str) -> str:
    name = " ".join(str(raw or "").strip().split())
    if not name or len(name) > 24 or not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"participant {participant_id} requires one short first name, got {raw!r}"
        )
    return name


def normalize_shared_context(raw: Any) -> list[str]:
    if isinstance(raw, list):
        text = " ".join(str(part).strip() for part in raw if str(part).strip())
    else:
        text = " ".join(str(raw or "").strip().split())
    if not text:
        raise ValueError("shared_context is required")
    sentences = [part for part in _SENTENCE_RE.split(text) if part.strip()]
    maximum = int(cfg.scenario.get("shared_context_max_sentences", 2))
    if not 1 <= len(sentences) <= maximum:
        raise ValueError(f"shared_context must contain 1..{maximum} complete sentences")
    word_limit = int(cfg.scenario.shared_context_max_words)
    if len(text.split()) > word_limit:
        raise ValueError(f"shared_context must contain at most {word_limit} words")
    return [text]


def manual_environment() -> dict[str, Any] | None:
    section = cfg.get("environment", {}) or {}
    if str(section.get("mode", "auto")) != "manual":
        return None
    manual = section.get("manual") or {}
    return dict(manual)


def manual_participant_profiles() -> list[dict[str, Any]]:
    section = cfg.get("participants", {}) or {}
    if str(section.get("mode", "auto")) != "manual":
        return []
    return [dict(row) for row in (section.get("profiles") or [])]


def _speech_style_for_age(age: int) -> str:
    if age < 25:
        return "casual, contemporary wording"
    if age < 45:
        return "plain, conversational wording"
    if age < 65:
        return "measured, practical wording"
    return "clear, measured wording"


def style_tendencies_for(
    participant_id: str,
    speech_style: str,
    params: SimulatorParameters,
    configured: Any = None,
) -> tuple[str, ...]:
    if configured:
        values = configured if isinstance(configured, list) else [configured]
        return tuple(str(value).strip() for value in values if str(value).strip())[:3]
    tendencies = [speech_style]
    tendencies.append("uses short statements" if params.verbosity <= 2 else "adds a little explanation")
    tendencies.append("softens disagreement" if params.directness <= 2 else "states judgments plainly")
    return tuple(tendencies)


def _sample_age(rng: random.Random, profile: dict[str, Any]) -> int:
    raw = profile.get("age")
    if raw is not None:
        age = int(raw)
        if not 18 <= age <= 80:
            raise ValueError("participant age must be in [18, 80]")
        return age
    return rng.randint(20, 68)


def _clean_reason(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _default_reason(option: OptionCard, *, positive: bool) -> str:
    if positive:
        return option.upside or next(iter(option.attrs.values()), option.name)
    return option.concern or next(iter(option.attrs.values()), option.name)


def _parse_stance_table(
    row: dict[str, Any], scenario: Scenario
) -> dict[str, OptionStance]:
    raw_table = row.get("option_stances") or {}
    result: dict[str, OptionStance] = {}
    for option in scenario.options:
        raw = raw_table.get(option.id, {}) if isinstance(raw_table, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        result[option.id] = OptionStance(
            option_id=option.id,
            rank=int(raw.get("rank", STANCE_NEUTRAL)),
            reason_for=_clean_reason(raw.get("reason_for")),
            reason_against=_clean_reason(raw.get("reason_against")),
        ).clipped()
    return result


def _normalise_stances(
    scenario: Scenario,
    stances: dict[str, OptionStance],
    preferred: str,
    *,
    hard_blocker: bool,
    rejection_reason: str,
) -> dict[str, OptionStance]:
    result: dict[str, OptionStance] = {}
    for option in scenario.options:
        stance = stances.get(option.id, OptionStance(option.id)).clipped()
        if option.id == preferred:
            stance.rank = STANCE_PREFERRED
            stance.reason_for = stance.reason_for or _default_reason(option, positive=True)
            stance.reason_against = ""
        elif hard_blocker:
            stance.rank = STANCE_REJECTED
            stance.reason_against = (
                stance.reason_against or rejection_reason or _default_reason(option, positive=False)
            )
        else:
            if stance.rank == STANCE_PREFERRED:
                stance.rank = STANCE_ACCEPTABLE
            if stance.rank == STANCE_REJECTED:
                stance.rank = STANCE_DISLIKED
            if stance.rank >= STANCE_ACCEPTABLE:
                stance.reason_for = stance.reason_for or _default_reason(option, positive=True)
            if stance.rank <= STANCE_DISLIKED:
                stance.reason_against = stance.reason_against or _default_reason(option, positive=False)
        result[option.id] = stance
    return result


def _option_rows(scenario: Scenario) -> list[dict[str, Any]]:
    return [
        {
            "id": option.id,
            "name": option.name,
            "short_name": option.short_name,
            "aliases": list(option.aliases),
            "attrs": dict(option.attrs),
            "upside": option.upside,
            "concern": option.concern,
        }
        for option in scenario.options
    ]


def _fallback_person_name(participant_id: str, used_names: set[str]) -> str:
    start = max(0, int(participant_id[1:]) - 1)
    for offset in range(len(_FALLBACK_NAMES)):
        name = _FALLBACK_NAMES[(start + offset) % len(_FALLBACK_NAMES)]
        if name.casefold() not in used_names:
            return name
    raise ValueError("no unique fallback participant name is available")


def _derived_alias_candidates(option_name: str) -> list[str]:
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]+", option_name)
    if words and words[0].casefold() in {"the", "a", "an"}:
        words = words[1:]
    if len(words) < 2:
        return []
    candidates = [" ".join(words[:2])]
    if len(words) >= 3:
        candidates.append(" ".join(words[-2:]))
    return candidates[:2]


def _retarget_person_text(text: object, raw_name: object, new_name: str) -> str:
    value = " ".join(str(text or "").split())
    original = " ".join(str(raw_name or "").split())
    if not value or not original:
        return value
    candidates = [original]
    first = original.split()[0]
    if first != original:
        candidates.append(first)
    for candidate in candidates:
        value = re.sub(
            rf"(?<!\w){re.escape(candidate)}(?!\w)",
            new_name,
            value,
            flags=re.I,
        )
    return value


class SetupBuilder:
    def __init__(
        self,
        topic: str,
        *,
        force_auto_scenario: bool = False,
        llm: Any = None,
        rng: random.Random | None = None,
    ) -> None:
        self.topic = topic.strip()
        configured_seed = cfg.simulation.get("random_seed", None)
        self.rng = rng or random.Random(
            int(configured_seed)
            if configured_seed is not None
            else random.SystemRandom().randint(0, 2**31 - 1)
        )
        self._profiles = manual_participant_profiles()
        self._manual_env = None if force_auto_scenario else manual_environment()
        if self._manual_env:
            self.topic = str(self._manual_env["topic"]).strip()
        self._llm = llm or get_llm_client()
        self._hard_blocker_id: str | None = None
        self._generated_names: dict[str, str] = {}

    def build(self, n: int) -> tuple[Scenario, list[Persona]]:
        if self._profiles and len(self._profiles) != n:
            raise ValueError(
                f"participants.profiles defines {len(self._profiles)} simulators but build() requested {n}"
            )
        self._validate_topic_participant_count(self.topic, n)
        traits = self._trait_rows(n)
        shape = None if self._pinned_preferences() else self._preference_shape(n, len(cfg.scenario.option_labels))

        if self._manual_env is not None:
            scenario = self._manual_scenario()
        else:
            scenario = self._generate_scenario(n)

        self._apply_setup_names(traits, scenario)
        preferences = self._preference_assignments(n, scenario.option_ids, shape)
        if self._profiles_complete():
            personas = self._personas_from_profiles(traits, preferences, scenario)
        else:
            personas = self._generate_personas(n, traits, preferences, scenario)
        self._validate_world(scenario, personas, preferences)
        return scenario, personas

    def _manual_scenario(self) -> Scenario:
        env = self._manual_env or {}
        labels = [str(value) for value in cfg.scenario.option_labels]
        rows = env.get("options") or []
        if len(rows) != len(labels):
            raise ValueError(f"manual environment requires exactly {len(labels)} options")
        options = [self._parse_option(row, label, manual=True) for row, label in zip(rows, labels)]
        scenario = Scenario(
            topic=self.topic,
            shared_context=normalize_shared_context(env.get("shared_context")),
            options=options,
        )
        self._validate_scenario(scenario, automatic=False, aliases_ready=True)
        return scenario

    def _generate_scenario(self, n: int) -> Scenario:
        self._generated_names = {}
        attempts = max(1, int(cfg.simulation.setup_generation_attempts))
        feedback = ""
        errors: list[str] = []
        for attempt in range(attempts):
            prompt = setup_scenario(self.topic, n, validation_feedback=feedback)
            raw = self._llm.generate_json(prompt, profile="setup")
            try:
                scenario = self._parse_scenario(raw.get("scenario", raw), n)
                self._validate_scenario(scenario, automatic=True, aliases_ready=False)
                if attempt:
                    scenario.setup_notes.append("scenario_regenerated_after_validation_error")
                self._assign_generated_metadata(scenario, n)
                self._validate_scenario(scenario, automatic=True, aliases_ready=True)
                return scenario
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                errors.append(message)
                feedback = message
        raise RuntimeError(
            f"Scenario generation failed after {attempts} attempt(s): " + " | ".join(errors)
        )

    def _parse_scenario(self, raw: Any, n: int) -> Scenario:
        if not isinstance(raw, dict):
            raise ValueError("scenario must be a JSON object")
        rows = raw.get("options") or []
        labels = [str(value) for value in cfg.scenario.option_labels]
        if not isinstance(rows, list) or len(rows) != len(labels):
            raise ValueError(f"scenario requires exactly {len(labels)} option cards")
        options = [self._parse_option(row, label, manual=False) for row, label in zip(rows, labels)]
        return Scenario(
            topic=self.topic,
            shared_context=normalize_shared_context(raw.get("shared_context")),
            options=options,
        )

    def _parse_option(self, raw: Any, expected_id: str, *, manual: bool) -> OptionCard:
        if not isinstance(raw, dict):
            raise ValueError(f"option {expected_id} must be an object")
        actual_id = str(raw.get("id") or "").strip().upper()
        if actual_id != expected_id:
            raise ValueError(f"expected option id {expected_id}, got {actual_id!r}")
        name = _require(raw.get("name"), f"option {expected_id} name")
        if manual:
            short_name = validated_alias(
                name, _require(raw.get("short_name"), f"option {expected_id} short_name")
            )
            raw_aliases = raw.get("aliases") or []
            if not isinstance(raw_aliases, list):
                raise ValueError(f"option {expected_id} aliases must be a list")
            aliases = tuple(
                validated_alias(name, alias) for alias in raw_aliases if str(alias).strip()
            )
        else:
            # Automatic aliases are generated in a separate, non-destructive step.
            short_name = name
            aliases = ()
        attrs_raw = raw.get("attrs")
        if not isinstance(attrs_raw, dict):
            raise ValueError(f"option {expected_id} attrs must be an object")
        attrs = {
            " ".join(str(key).strip().split()): " ".join(str(value).strip().split())
            for key, value in attrs_raw.items()
            if str(key).strip() and str(value).strip()
        }
        if not attrs:
            raise ValueError(f"option {expected_id} requires public attributes")
        if not manual:
            minimum = int(cfg.scenario.public_attr_min)
            maximum = int(cfg.scenario.public_attr_max)
            if not minimum <= len(attrs) <= maximum:
                raise ValueError(
                    f"option {expected_id} requires {minimum}..{maximum} attributes, got {len(attrs)}"
                )
        return OptionCard(
            id=expected_id,
            name=name,
            short_name=short_name,
            aliases=aliases,
            attrs=attrs,
            upside=_require(raw.get("upside"), f"option {expected_id} upside"),
            concern=_require(raw.get("concern"), f"option {expected_id} concern"),
        )

    def _validate_scenario(
        self,
        scenario: Scenario,
        *,
        automatic: bool,
        aliases_ready: bool,
    ) -> None:
        labels = [str(value) for value in cfg.scenario.option_labels]
        if scenario.option_ids != labels:
            raise ValueError(f"option ids must be {labels}")
        names = [normalize_option_text(option.name) for option in scenario.options]
        if len(names) != len(set(names)):
            raise ValueError("full option names must be unique")
        if aliases_ready:
            validate_unique_aliases(scenario)
        if automatic:
            self._validate_participant_references(scenario.context_text)

    def _assign_generated_metadata(self, scenario: Scenario, n: int) -> None:
        """Add aliases and fixed participant names without risking the valid board."""

        rows = [{"id": option.id, "name": option.name} for option in scenario.options]
        participant_ids = [f"p{index + 1}" for index in range(n)]
        proposed: dict[str, list[str]] = {
            option.id: [] for option in scenario.options
        }
        try:
            raw = self._llm.generate_json(
                setup_aliases(rows, participant_ids), profile="setup"
            )
            alias_rows = raw.get("aliases", raw) if isinstance(raw, dict) else raw
            if not isinstance(alias_rows, list):
                raise ValueError("alias response must contain a list")
            for row in alias_rows:
                if not isinstance(row, dict):
                    continue
                option_id = str(row.get("id") or "").strip().upper()
                values = row.get("aliases") or []
                if option_id in scenario.option_ids and isinstance(values, list):
                    proposed[option_id].extend(str(value) for value in values)

            name_rows = (raw.get("participant_names") or []) if isinstance(raw, dict) else []
            if isinstance(name_rows, list):
                used: set[str] = set()
                for row in name_rows:
                    if not isinstance(row, dict):
                        continue
                    participant_id = str(row.get("id") or "").strip()
                    if participant_id not in participant_ids:
                        continue
                    try:
                        name = _validated_person_name(
                            row.get("name"), participant_id=participant_id
                        )
                    except ValueError:
                        continue
                    if name.casefold() in used:
                        continue
                    used.add(name.casefold())
                    self._generated_names[participant_id] = name
        except Exception:
            scenario.setup_notes.append("alias_generation_used_derived_fallbacks")

        for option in scenario.options:
            proposed[option.id].extend(_derived_alias_candidates(option.name))

        accepted = unique_generated_aliases(
            {option.id: option.name for option in scenario.options},
            proposed,
            max_words=int(cfg.scenario.short_alias_max_words),
        )
        generated_count = 0
        for option in scenario.options:
            aliases = accepted.get(option.id, ())
            option.aliases = aliases
            option.short_name = aliases[0] if aliases else option.name
            generated_count += len(aliases)
        if generated_count:
            scenario.setup_notes.append("generated_option_aliases")
        elif "alias_generation_used_derived_fallbacks" not in scenario.setup_notes:
            scenario.setup_notes.append("alias_generation_returned_no_usable_aliases")
        if self._generated_names:
            scenario.setup_notes.append("generated_participant_names")

    def _apply_setup_names(
        self, trait_rows: list[dict[str, Any]], scenario: Scenario
    ) -> None:
        """Pin one valid unique name per automatic participant before persona generation."""

        used = {
            str(row.get("name")).casefold()
            for row in trait_rows
            if row.get("name")
        }
        for row in trait_rows:
            if row.get("name"):
                continue
            participant_id = str(row["id"])
            proposed = self._generated_names.get(participant_id)
            try:
                name = _validated_person_name(proposed, participant_id=participant_id)
                if name.casefold() in used:
                    raise ValueError(f"duplicate generated name {name!r}")
            except ValueError:
                name = _fallback_person_name(participant_id, used)
                scenario.setup_notes.append(f"fallback_name_assigned:{participant_id}")
            row["name"] = name
            used.add(name.casefold())

    @staticmethod
    def _validate_participant_references(context: str) -> None:
        lowered = context.lower()
        if re.search(r"\b(participant|simulator|persona)\s*\d+\b", lowered):
            raise ValueError("shared_context must not refer to generated participants")

    def _generate_personas(
        self,
        n: int,
        trait_rows: list[dict[str, Any]],
        preferences: dict[str, str],
        scenario: Scenario,
    ) -> list[Persona]:
        attempts = max(1, int(cfg.simulation.setup_generation_attempts))
        errors: list[str] = []
        for _ in range(attempts):
            prompt = setup_personas(
                self.topic,
                n,
                trait_rows,
                preferences,
                _option_rows(scenario),
                scenario.context_text,
                self._hard_blocker_id,
            )
            try:
                raw = self._llm.generate_json(prompt, profile="setup")
                return self._parse_personas(raw.get("participants", raw), trait_rows, scenario, preferences)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"Persona generation failed after {attempts} attempt(s): " + " | ".join(errors)
        )

    def _trait_rows(self, n: int) -> list[dict[str, Any]]:
        blockers = [idx for idx, profile in enumerate(self._profiles) if profile.get("hard_blocker")]
        if len(blockers) > 1:
            raise ValueError("at most one hard blocker is allowed")
        if blockers:
            blocker_index = blockers[0]
        elif self.rng.random() < float(cfg.personas.hard_blocker_probability):
            blocker_index = self.rng.randrange(n)
        else:
            blocker_index = -1
        self._hard_blocker_id = f"p{blocker_index + 1}" if blocker_index >= 0 else None

        rows: list[dict[str, Any]] = []
        for index in range(n):
            pid = f"p{index + 1}"
            profile = self._profile_for(pid)
            hard = pid == self._hard_blocker_id
            traits: dict[str, int] = {}
            supplied = profile.get("traits") or {}
            for name in DIRECT_TRAIT_NAMES:
                if name in supplied:
                    value = int(supplied[name])
                else:
                    low, high = cfg.personas.trait_ranges[name]
                    value = self.rng.randint(int(low), int(high))
                if hard and name == "stubbornness":
                    value = 5
                traits[name] = value
            row: dict[str, Any] = {"id": pid, "traits": traits, "hard_blocker": hard}
            if profile.get("name"):
                row["name"] = str(profile["name"])
            rows.append(row)
        return rows

    def _preference_shape(self, n: int, option_count: int) -> tuple[int, ...]:
        forced = cfg.personas.preference_distribution.get("forced_shape", None)
        if forced is not None:
            return parse_preference_shape(forced)
        weights_by_size = cfg.personas.preference_distribution.shape_weights
        raw = weights_by_size.get(n, weights_by_size.get(str(n)))
        choices = list(raw.items())
        selected = self.rng.choices(
            [parse_preference_shape(shape) for shape, _ in choices],
            weights=[float(weight) for _, weight in choices],
            k=1,
        )[0]
        if sum(selected) != n or len(selected) > option_count:
            raise ValueError(f"invalid preference shape {selected}")
        return selected

    def _pinned_preferences(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for index, profile in enumerate(self._profiles):
            value = str(profile.get("preferred_option") or "").strip().upper()
            if value:
                result[f"p{index + 1}"] = value
        return result

    def _preference_assignments(
        self,
        n: int,
        option_ids: list[str],
        shape: tuple[int, ...] | None,
    ) -> dict[str, str]:
        pinned = self._pinned_preferences()
        assignments = dict(pinned)
        if not pinned:
            if shape is None:
                raise ValueError("preference shape is required")
            selected_options = self.rng.sample(option_ids, len(shape))
            bag: list[str] = []
            for option_id, count in zip(selected_options, shape):
                bag.extend([option_id] * count)
            self.rng.shuffle(bag)
            assignments = {f"p{idx + 1}": bag[idx] for idx in range(n)}
        else:
            for idx in range(n):
                pid = f"p{idx + 1}"
                if pid not in assignments:
                    assignments[pid] = self.rng.choice(option_ids)
        return assignments

    def _profiles_complete(self) -> bool:
        if not self._profiles:
            return False
        required = {"name", "description", "private_goal", "preferred_option"}
        return all(required <= set(profile) for profile in self._profiles)

    def _profile_for(self, participant_id: str) -> dict[str, Any]:
        index = int(participant_id[1:]) - 1
        return self._profiles[index] if 0 <= index < len(self._profiles) else {}

    def _personas_from_profiles(
        self,
        trait_rows: list[dict[str, Any]],
        preferences: dict[str, str],
        scenario: Scenario,
    ) -> list[Persona]:
        rows: list[dict[str, Any]] = []
        for trait in trait_rows:
            pid = trait["id"]
            profile = self._profile_for(pid)
            preferred = preferences[pid]
            stances = {
                option.id: {
                    "rank": STANCE_PREFERRED if option.id == preferred else STANCE_NEUTRAL,
                    "reason_for": _default_reason(option, positive=True),
                    "reason_against": _default_reason(option, positive=False),
                }
                for option in scenario.options
            }
            rows.append(
                {
                    "id": pid,
                    "name": profile["name"],
                    "background": profile["description"],
                    "private_goal": profile["private_goal"],
                    "age": profile.get("age"),
                    "preferred_options": [preferred],
                    "option_stances": stances,
                }
            )
        return self._parse_personas(rows, trait_rows, scenario, preferences)

    def _parse_personas(
        self,
        rows: Any,
        trait_rows: list[dict[str, Any]],
        scenario: Scenario,
        preferences: dict[str, str],
    ) -> list[Persona]:
        if not isinstance(rows, list) or len(rows) != len(trait_rows):
            raise ValueError("wrong number of participant cards")
        traits_by_id = {row["id"]: row for row in trait_rows}
        raw_by_id = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
        if set(raw_by_id) != set(traits_by_id):
            raise ValueError("participant ids do not match the requested ids")
        personas: list[Persona] = []
        used_names: set[str] = set()
        for pid, trait in traits_by_id.items():
            row = dict(raw_by_id[pid])
            profile = self._profile_for(pid)
            response_name = row.get("name")
            fixed_name = profile.get("name") or trait.get("name")
            used_fallback = False
            if fixed_name:
                name = _validated_person_name(fixed_name, participant_id=pid)
                if name.casefold() in used_names:
                    if profile.get("name"):
                        raise ValueError(f"duplicate configured name {name!r}")
                    name = _fallback_person_name(pid, used_names)
                    used_fallback = True
                    scenario.setup_notes.append(f"fallback_name_assigned:{pid}")
            else:
                try:
                    name = _validated_person_name(response_name, participant_id=pid)
                    if name.casefold() in used_names:
                        raise ValueError(f"duplicate generated name {name!r}")
                except ValueError:
                    name = _fallback_person_name(pid, used_names)
                    used_fallback = True
                    scenario.setup_notes.append(f"fallback_name_assigned:{pid}")
            used_names.add(name.casefold())
            hard = bool(trait.get("hard_blocker"))
            params = SimulatorParameters(**trait["traits"]).validated(hard_blocker=hard)
            preferred = preferences[pid]
            rejection_reason = _clean_reason(
                profile.get("rejection_reason") or row.get("rejection_reason")
            )
            stances = _normalise_stances(
                scenario,
                _parse_stance_table(row, scenario),
                preferred,
                hard_blocker=hard,
                rejection_reason=rejection_reason,
            )
            age = _sample_age(self.rng, {**row, **profile})
            speech_style = str(profile.get("speech_style") or _speech_style_for_age(age))
            background = profile.get("description") or row.get("background")
            private_goal = profile.get("private_goal") or row.get("private_goal")
            name_changed = bool(
                response_name
                and str(response_name).strip().casefold() != name.casefold()
            )
            if used_fallback or name_changed:
                background = _retarget_person_text(background, response_name, name)
                private_goal = _retarget_person_text(private_goal, response_name, name)
                rejection_reason = _retarget_person_text(
                    rejection_reason, response_name, name
                )
                for stance in stances.values():
                    stance.reason_for = _retarget_person_text(
                        stance.reason_for, response_name, name
                    )
                    stance.reason_against = _retarget_person_text(
                        stance.reason_against, response_name, name
                    )
            personas.append(
                Persona(
                    id=pid,
                    name=name,
                    sim_params=params,
                    background=_require(background, f"participant {pid} background"),
                    private_goal=_require(private_goal, f"participant {pid} private_goal"),
                    preferred_options=[preferred],
                    age=age,
                    speech_style=speech_style,
                    style_tendencies=style_tendencies_for(
                        pid,
                        speech_style,
                        params,
                        profile.get("style_tendencies"),
                    ),
                    rejection=None,
                    rejection_reason=rejection_reason,
                    option_stances=stances,
                    hard_blocker=hard,
                )
            )
        return personas

    @staticmethod
    def _validate_topic_participant_count(topic: str, n: int) -> None:
        if not topic.strip():
            raise ValueError("topic is required")
        minimum = int(cfg.simulation.min_participants)
        maximum = int(cfg.simulation.max_participants)
        if not minimum <= n <= maximum:
            raise ValueError(f"participant count must be in [{minimum}, {maximum}]")

    @staticmethod
    def _validate_world(
        scenario: Scenario,
        personas: list[Persona],
        preferences: dict[str, str],
    ) -> None:
        if len(personas) != len(preferences):
            raise ValueError("wrong number of personas")
        ids = {persona.id for persona in personas}
        if ids != set(preferences):
            raise ValueError("persona ids do not match preference assignments")
        if len({persona.name.casefold() for persona in personas}) != len(personas):
            raise ValueError("participant names must be unique")
        blockers = [persona for persona in personas if persona.hard_blocker]
        if len(blockers) > 1:
            raise ValueError("at most one hard blocker is allowed")
        for persona in personas:
            if persona.preferred_option != preferences[persona.id]:
                raise ValueError(f"participant {persona.id} has the wrong primary preference")
            if set(persona.option_stances) != set(scenario.option_ids):
                raise ValueError(f"participant {persona.id} has incomplete option stances")
            if persona.hard_blocker:
                for option_id, stance in persona.option_stances.items():
                    expected = STANCE_PREFERRED if option_id == persona.preferred_option else STANCE_REJECTED
                    if stance.rank != expected:
                        raise ValueError(f"hard blocker {persona.id} has invalid stance for {option_id}")
                    if option_id != persona.preferred_option and not stance.reason_against:
                        raise ValueError(f"hard blocker {persona.id} lacks a rejection reason for {option_id}")
