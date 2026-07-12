"""Configuration loader.

`config.yaml` is the only location for tunable numeric parameters.  Modules import
`cfg` from here.  The wrapper deliberately stays small: attribute access for
readability, raw mapping preservation for logging/validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

# Field names accepted in a manual participant profile (participants.profiles).
PROFILE_TRAIT_NAMES = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
PROFILE_PARAMETER_NAMES = (
    "engagement", "verbosity", "directness", "stubbornness", "switch_resistance",
)
_PROFILE_FIELDS = frozenset({
    "name", "description", "private_goal", "preferred_option",
    "age", "speech_style", "rejection", "rejection_reason", "traits", "parameters",
})

# Field names accepted in a manual environment (environment.manual).
_MANUAL_ENV_FIELDS = frozenset({"topic", "shared_context", "options"})
_MANUAL_OPTION_FIELDS = frozenset({"id", "name", "short_name", "attrs", "upside", "concern"})

# LLM roles and the providers the client layer implements.
LLM_ROLES = ("dialogue", "validator")
KNOWN_LLM_PROVIDERS = frozenset({"uni", "groq", "gemini", "gpt"})


def validate_llm_roles(llm: Any) -> None:
    """Validate the role-based LLM selection (llm.dialogue / llm.validator).

    Both roles must name a known provider with a model mapping; the same
    provider for both roles is valid. The legacy single `llm.provider` key is
    an explicit migration error rather than a silently supported alias.
    """
    if not isinstance(llm, dict):
        raise ValueError("llm must be a mapping.")
    if "provider" in llm:
        raise ValueError(
            "llm.provider was replaced by the role selectors llm.dialogue and "
            "llm.validator; set both (they may name the same provider)."
        )
    models = llm.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("llm.models must be a non-empty mapping.")
    for role in LLM_ROLES:
        provider = str(llm.get(role) or "").strip().lower()
        if not provider:
            raise ValueError(f"Missing required config key: llm.{role}")
        if provider not in KNOWN_LLM_PROVIDERS:
            raise ValueError(
                f"llm.{role} must be one of {sorted(KNOWN_LLM_PROVIDERS)}, got {provider!r}."
            )
        if provider not in models:
            raise ValueError(f"llm.{role} = {provider!r} has no entry under llm.models.")
        if provider == "uni" and not str((llm.get("endpoints") or {}).get("uni") or "").strip():
            raise ValueError(f"llm.{role} = 'uni' requires llm.endpoints.uni.")
    sampling = llm.get("sampling")
    if not isinstance(sampling, dict) or "validator" not in sampling:
        raise ValueError("llm.sampling.validator must define the cold validator profile.")


def parse_preference_shape(value: Any) -> tuple[int, ...]:
    """Parse a preference partition from ``"2-1"`` or ``[2, 1]``."""
    if isinstance(value, str):
        raw_parts = value.split("-")
    elif isinstance(value, (list, tuple)):
        raw_parts = list(value)
    else:
        raise ValueError("preference shape must be a hyphenated string or integer list")
    if not raw_parts:
        raise ValueError("preference shape must not be empty")
    try:
        parts = tuple(int(part) for part in raw_parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid preference shape: {value!r}") from exc
    if any(part <= 0 for part in parts):
        raise ValueError("preference shape parts must be positive integers")
    if tuple(sorted(parts, reverse=True)) != parts:
        raise ValueError("preference shape parts must be ordered largest to smallest")
    return parts


def apply_corpus_preset(data: dict) -> dict | None:
    """Fold the active corpus preset into runtime parameters.

    Presets are optional: with ``corpus.preset`` null the config is untouched.
    Corpus-level fields map onto existing tunables — typical discussion length
    onto the per-participant turn caps, preferred group size onto
    ``simulation.num_participants`` (clamped to the configured range). Dominance
    fields are returned for the speaker router. Returns the resolved preset
    (with its name) or ``None``.
    """
    corpus = data.get("corpus") or {}
    name = corpus.get("preset")
    if not name:
        return None
    presets = corpus.get("presets") or {}
    if name not in presets or not isinstance(presets[name], dict):
        raise ValueError(f"corpus.preset {name!r} has no mapping under corpus.presets")
    preset = presets[name]

    turns = preset.get("turns_per_participant")
    if turns is not None:
        t = float(turns)
        if t <= 0:
            raise ValueError("corpus preset turns_per_participant must be positive")
        conv = data.setdefault("conversation", {})
        conv["min_discussion_turns_per_participant"] = round(0.75 * t, 2)
        conv["target_discussion_turns_per_participant"] = t
        conv["max_discussion_turns_per_participant"] = round(1.5 * t, 2)

    size = preset.get("preferred_group_size")
    if size is not None:
        sim = data.setdefault("simulation", {})
        lo = int(sim.get("min_participants", 2))
        hi = int(sim.get("max_participants", 7))
        sim["num_participants"] = max(lo, min(hi, int(size)))

    resolved: dict = {"name": str(name)}
    if "top_speaker_share" in preset:
        share = float(preset["top_speaker_share"])
        if not 0.0 < share <= 1.0:
            raise ValueError("corpus preset top_speaker_share must be in (0, 1]")
        resolved["top_speaker_share"] = share
    if "dominance_range" in preset:
        rng = list(preset["dominance_range"])
        if len(rng) != 2:
            raise ValueError("corpus preset dominance_range must contain [min, max]")
        dom_lo, dom_hi = float(rng[0]), float(rng[1])
        if not 0.0 <= dom_lo <= dom_hi <= 1.0:
            raise ValueError("corpus preset dominance_range must satisfy 0 <= min <= max <= 1")
        resolved["dominance_range"] = (dom_lo, dom_hi)
    if "imbalance_tolerance" in preset:
        tol = float(preset["imbalance_tolerance"])
        if not 0.0 <= tol < 1.0:
            raise ValueError("corpus preset imbalance_tolerance must be in [0, 1)")
        resolved["imbalance_tolerance"] = tol
    return resolved


class Section:
    def __init__(self, data: Any) -> None:
        self._raw = data
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and key.isidentifier():
                    setattr(self, key, Section(value) if isinstance(value, dict) else value)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(self._raw, dict):
            value = self._raw[key]
        elif isinstance(self._raw, list):
            value = self._raw[key]
        else:
            raise TypeError(f"Config section of type {type(self._raw).__name__} is not subscriptable")
        return Section(value) if isinstance(value, dict) else value

    def __contains__(self, key: Any) -> bool:
        return isinstance(self._raw, dict) and key in self._raw

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self._raw, dict):
            return self._raw.get(key, default)
        return default

    def items(self):
        if not isinstance(self._raw, dict):
            raise TypeError("Config section is not a mapping")
        return self._raw.items()


class Config(Section):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.root = path.parent
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        corpus_active = apply_corpus_preset(data)
        super().__init__(data)
        self.corpus_active = corpus_active
        self._validate()

    def participant_count(self) -> int:
        """Active group size: profile count in manual participants mode, else simulation.num_participants."""
        participants = self._raw.get("participants") or {}
        if str(participants.get("mode", "auto")) == "manual":
            return len(participants.get("profiles") or [])
        return int(self.simulation.num_participants)

    def _require(self, dotted: str) -> Any:
        value: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"Missing required config key: {dotted}")
            value = value[part]
        return value

    @staticmethod
    def _validate_range(name: str, values: Iterable[Any], low: float, high: float) -> None:
        vals = list(values)
        if len(vals) != 2:
            raise ValueError(f"{name} must contain [min, max].")
        lo, hi = float(vals[0]), float(vals[1])
        if not (low <= lo <= hi <= high):
            raise ValueError(f"{name} must satisfy {low} <= min <= max <= {high}.")

    def _validate(self) -> None:
        required = [
            "simulation.num_participants",
            "simulation.min_participants",
            "simulation.max_participants",
            "scenario.option_labels",
            "conversation.max_discussion_turns_per_participant",
            "output.log_dir",
        ]
        for key in required:
            self._require(key)

        validate_llm_roles(self._raw.get("llm"))

        n = int(self.simulation.num_participants)
        min_n = int(self.simulation.min_participants)
        max_n = int(self.simulation.max_participants)
        if not (min_n <= n <= max_n):
            raise ValueError("simulation.num_participants must satisfy min <= n <= max.")

        labels = list(self.scenario.option_labels)
        if len(set(labels)) != len(labels):
            raise ValueError("scenario.option_labels must be unique.")

        for name, rng in self.personas.trait_ranges.items():
            self._validate_range(f"personas.trait_ranges.{name}", rng, int(self.personas.trait_min), int(self.personas.trait_max))
        for name, rng in self.personas.hard_blocker_trait_ranges.items():
            self._validate_range(f"personas.hard_blocker_trait_ranges.{name}", rng, int(self.personas.trait_min), int(self.personas.trait_max))

        self._validate_preference_distribution(min_n, max_n, len(labels), n)
        self._validate_participants(min_n, max_n, labels)
        self._validate_environment(labels)
        self._validate_moderator()
        self._validate_validation()

    def _validate_validation(self) -> None:
        validation = self._raw.get("validation")
        if validation is None:
            return  # absent section = critical default
        if not isinstance(validation, dict):
            raise ValueError("validation must be a mapping.")
        mode = str(validation.get("mode", "critical"))
        if mode != "critical":
            raise ValueError("validation.mode must be 'critical'.")

    def _validate_moderator(self) -> None:
        moderator = self._raw.get("moderator")
        if moderator is None:
            return  # absent section = fully-moderated default behavior
        if not isinstance(moderator, dict):
            raise ValueError("moderator must be a mapping.")
        allowed = {"enabled", "opening", "mid_discussion_nudges", "final_vote_call", "closing"}
        unknown = set(moderator) - allowed
        if unknown:
            raise ValueError(f"moderator has unknown fields: {sorted(unknown)}.")
        for key in allowed:
            if key in moderator and not isinstance(moderator[key], bool):
                raise ValueError(f"moderator.{key} must be a boolean.")

    def _validate_environment(self, option_labels: list) -> None:
        environment = self._raw.get("environment") or {}
        mode = str(environment.get("mode", "auto"))
        if mode not in ("auto", "manual"):
            raise ValueError("environment.mode must be 'auto' or 'manual'.")
        if mode == "auto":
            return
        manual = environment.get("manual")
        if not isinstance(manual, dict) or not manual:
            raise ValueError("environment.mode=manual requires a non-empty environment.manual mapping.")
        unknown = set(manual) - _MANUAL_ENV_FIELDS
        if unknown:
            raise ValueError(f"environment.manual has unknown fields: {sorted(unknown)}.")
        if not str(manual.get("topic") or "").strip():
            raise ValueError("environment.manual.topic must be a non-empty string.")
        context = manual.get("shared_context", [])
        if not isinstance(context, list) or any(not str(item).strip() for item in context):
            raise ValueError("environment.manual.shared_context must be a list of non-empty strings.")
        options = manual.get("options")
        labels = [str(x) for x in option_labels]
        if not isinstance(options, list) or len(options) != len(labels):
            raise ValueError(
                f"environment.manual.options must define exactly {len(labels)} options "
                f"(one per scenario.option_labels entry), got {len(options) if isinstance(options, list) else type(options).__name__}."
            )
        for idx, option in enumerate(options):
            where = f"environment.manual.options[{idx}]"
            if not isinstance(option, dict):
                raise ValueError(f"{where} must be a mapping.")
            unknown = set(option) - _MANUAL_OPTION_FIELDS
            if unknown:
                raise ValueError(f"{where} has unknown fields: {sorted(unknown)}.")
            option_id = str(option.get("id") or "").strip().upper()
            if option_id and option_id != labels[idx]:
                raise ValueError(f"{where}.id must be {labels[idx]!r} (options are positional), got {option_id!r}.")
            if not str(option.get("name") or "").strip():
                raise ValueError(f"{where}.name must be a non-empty string.")
            attrs = option.get("attrs")
            if not isinstance(attrs, dict) or not any(
                str(k).strip() and str(v).strip() for k, v in attrs.items()
            ):
                raise ValueError(f"{where}.attrs must be a mapping with at least one factual attribute.")

    def _validate_participants(self, min_n: int, max_n: int, option_labels: list) -> None:
        participants = self._raw.get("participants") or {}
        mode = str(participants.get("mode", "auto"))
        if mode not in ("auto", "manual"):
            raise ValueError("participants.mode must be 'auto' or 'manual'.")
        if mode == "auto":
            return
        profiles = participants.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            raise ValueError("participants.mode=manual requires a non-empty participants.profiles list.")
        if not (min_n <= len(profiles) <= max_n):
            raise ValueError(
                f"participants.profiles must contain {min_n}..{max_n} profiles, got {len(profiles)}."
            )
        labels = {str(x).upper() for x in option_labels}
        trait_lo, trait_hi = int(self.personas.trait_min), int(self.personas.trait_max)
        names: list[str] = []
        for idx, profile in enumerate(profiles):
            where = f"participants.profiles[{idx}]"
            if not isinstance(profile, dict):
                raise ValueError(f"{where} must be a mapping.")
            unknown = set(profile) - _PROFILE_FIELDS
            if unknown:
                raise ValueError(f"{where} has unknown fields: {sorted(unknown)}.")
            name = str(profile.get("name") or "").strip()
            if name:
                names.append(name.lower())
            preferred = str(profile.get("preferred_option") or "").strip().upper() or None
            if preferred is not None and preferred not in labels:
                raise ValueError(f"{where}.preferred_option must be one of {sorted(labels)}.")
            if "age" in profile and profile.get("age") is not None:
                age = int(profile["age"])
                if not (16 <= age <= 85):
                    raise ValueError(f"{where}.age must be in [16, 85].")
            if "speech_style" in profile and not str(profile.get("speech_style") or "").strip():
                raise ValueError(f"{where}.speech_style must be a non-empty string when provided.")
            rejection = str(profile.get("rejection") or "").strip().upper() or None
            if rejection is not None:
                if rejection not in labels:
                    raise ValueError(f"{where}.rejection must be one of {sorted(labels)}.")
                if rejection == preferred:
                    raise ValueError(f"{where}: rejection must differ from preferred_option.")
                if not str(profile.get("rejection_reason") or "").strip():
                    raise ValueError(f"{where}: a rejection requires a rejection_reason.")
            traits = profile.get("traits") or {}
            if not isinstance(traits, dict):
                raise ValueError(f"{where}.traits must be a mapping.")
            for key, value in traits.items():
                if key not in PROFILE_TRAIT_NAMES:
                    raise ValueError(f"{where}.traits has unknown trait {key!r}.")
                if not (trait_lo <= int(value) <= trait_hi):
                    raise ValueError(f"{where}.traits.{key} must be in [{trait_lo}, {trait_hi}].")
            # A rejection is a hard constraint, not a personality: an explicitly
            # agreeable profile may hold one (P5: "an agreeable vegan can reject
            # a steakhouse politely"). Leaving agreeableness unset keeps the
            # classic blocker persona (pinned to 1 in the builder).
            parameters = profile.get("parameters") or {}
            if not isinstance(parameters, dict):
                raise ValueError(f"{where}.parameters must be a mapping.")
            for key, value in parameters.items():
                if key not in PROFILE_PARAMETER_NAMES:
                    raise ValueError(f"{where}.parameters has unknown parameter {key!r}.")
                if not (0.0 <= float(value) <= 1.0):
                    raise ValueError(f"{where}.parameters.{key} must be in [0, 1].")
        if len(set(names)) != len(names):
            raise ValueError("participants.profiles names must be unique.")

    def _validate_preference_distribution(
        self,
        min_n: int,
        max_n: int,
        option_count: int,
        active_n: int,
    ) -> None:
        distribution = self._require("personas.preference_distribution")
        if not isinstance(distribution, dict):
            raise ValueError("personas.preference_distribution must be a mapping.")
        weights_by_size = distribution.get("shape_weights")
        if not isinstance(weights_by_size, dict):
            raise ValueError("personas.preference_distribution.shape_weights must be a mapping.")

        for size in range(min_n, max_n + 1):
            raw_weights = weights_by_size.get(size, weights_by_size.get(str(size)))
            if not isinstance(raw_weights, dict) or not raw_weights:
                raise ValueError(f"Missing preference shape weights for group size {size}.")
            total = 0.0
            for raw_shape, raw_weight in raw_weights.items():
                shape = parse_preference_shape(raw_shape)
                self._validate_preference_shape(shape, size, option_count)
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Weight for preference shape {raw_shape!r} must be numeric.") from exc
                if weight < 0:
                    raise ValueError(f"Weight for preference shape {raw_shape!r} must be non-negative.")
                total += weight
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"Preference shape weights for group size {size} must sum to 1.0, got {total}.")

        forced = distribution.get("forced_shape")
        if forced is not None:
            self._validate_preference_shape(parse_preference_shape(forced), active_n, option_count)

    @staticmethod
    def _validate_preference_shape(shape: tuple[int, ...], n: int, option_count: int) -> None:
        if sum(shape) != n:
            raise ValueError(f"Preference shape {shape} must sum to participant count {n}.")
        if len(shape) > option_count:
            raise ValueError(
                f"Preference shape {shape} needs {len(shape)} distinct options, but only {option_count} exist."
            )


cfg = Config(Path(__file__).resolve().parents[1] / "config.yaml")
