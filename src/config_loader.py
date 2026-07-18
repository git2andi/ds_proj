"""Configuration loader and validation for the simplified runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

DIRECT_TRAIT_NAMES = ("engagement", "verbosity", "directness", "stubbornness")
PROFILE_TRAIT_NAMES = DIRECT_TRAIT_NAMES
KNOWN_LLM_PROVIDERS = frozenset({"uni", "groq", "gemini", "gpt"})
_PROFILE_FIELDS = frozenset({
    "name", "description", "private_goal", "preferred_option", "age",
    "speech_style", "style_tendencies", "traits", "hard_blocker",
    "rejection", "rejection_reason",
})
_MANUAL_ENV_FIELDS = frozenset({"topic", "shared_context", "options"})
_MANUAL_OPTION_FIELDS = frozenset({"id", "name", "short_name", "aliases", "attrs", "upside", "concern"})


def parse_preference_shape(value: Any) -> tuple[int, ...]:
    raw = value.split("-") if isinstance(value, str) else list(value)
    parts = tuple(int(item) for item in raw)
    if not parts or any(item <= 0 for item in parts) or tuple(sorted(parts, reverse=True)) != parts:
        raise ValueError("preference shape parts must be positive and ordered largest to smallest")
    return parts


class Section:
    def __init__(self, data: Any) -> None:
        self._raw = data
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and key.isidentifier():
                    setattr(self, key, Section(value) if isinstance(value, dict) else value)

    def __getitem__(self, key: Any) -> Any:
        value = self._raw[key]
        return Section(value) if isinstance(value, dict) else value

    def __contains__(self, key: Any) -> bool:
        return isinstance(self._raw, dict) and key in self._raw

    def get(self, key: Any, default: Any = None) -> Any:
        return self._raw.get(key, default) if isinstance(self._raw, dict) else default

    def items(self):
        if not isinstance(self._raw, dict):
            raise TypeError("Config section is not a mapping")
        return self._raw.items()


class Config(Section):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.root = path.parent
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        super().__init__(data)
        self._validate()

    def participant_count(self) -> int:
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
    def _range(name: str, values: Iterable[Any], low: int, high: int) -> None:
        vals = list(values)
        if len(vals) != 2:
            raise ValueError(f"{name} must contain [min, max]")
        lo, hi = int(vals[0]), int(vals[1])
        if not low <= lo <= hi <= high:
            raise ValueError(f"{name} must satisfy {low} <= min <= max <= {high}")

    @staticmethod
    def _level_mapping(section: dict[str, Any], name: str, *, cast=float) -> dict[int, Any]:
        raw = section.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"{name} must map levels 1..5")
        result = {}
        for level in range(1, 6):
            value = raw.get(level, raw.get(str(level)))
            if value is None:
                raise ValueError(f"{name} is missing level {level}")
            result[level] = cast(value)
        return result

    def _validate(self) -> None:
        for key in (
            "llm.dialogue", "llm.models", "llm.sampling",
            "simulation.num_participants", "simulation.min_participants", "simulation.max_participants",
            "scenario.option_labels", "personas.trait_ranges",
            "personas.preference_distribution", "conversation.thread_turn_cap",
            "simulator.bid_probability_by_engagement",
            "simulator.movement_probability_by_stubbornness",
            "language.max_words_by_verbosity", "language.action_max_words",
            "language.directness_instructions", "output.log_dir",
        ):
            self._require(key)
        self._validate_llm()
        self._validate_group_and_traits()
        self._validate_preferences()
        self._validate_environment()
        self._validate_participants()
        self._validate_conversation()
        self._validate_behavior()

    def _validate_llm(self) -> None:
        provider = str(self.llm.dialogue).lower()
        if provider not in KNOWN_LLM_PROVIDERS:
            raise ValueError(f"llm.dialogue must be one of {sorted(KNOWN_LLM_PROVIDERS)}")
        if not str(self.llm.models.get(provider) or "").strip():
            raise ValueError(f"missing model for provider {provider}")
        if provider == "uni" and not str(self.llm.endpoints.get("uni") or "").strip():
            raise ValueError("llm.dialogue='uni' requires llm.endpoints.uni")
        for profile in ("setup", "dialogue", "repair"):
            if profile not in self.llm.sampling:
                raise ValueError(f"llm.sampling.{profile} is required")

    def _validate_group_and_traits(self) -> None:
        n = int(self.simulation.num_participants)
        low, high = int(self.simulation.min_participants), int(self.simulation.max_participants)
        if not low <= n <= high:
            raise ValueError("simulation.num_participants must satisfy min <= n <= max")
        labels = [str(value).upper() for value in self.scenario.option_labels]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("scenario.option_labels must be non-empty and unique")
        for trait in DIRECT_TRAIT_NAMES:
            self._range(
                f"personas.trait_ranges.{trait}",
                self.personas.trait_ranges[trait],
                1,
                4 if trait == "stubbornness" else 5,
            )
        probability = float(self.personas.hard_blocker_probability)
        if not 0 <= probability <= 1:
            raise ValueError("personas.hard_blocker_probability must be in [0, 1]")

    def _validate_preferences(self) -> None:
        low, high = int(self.simulation.min_participants), int(self.simulation.max_participants)
        option_count = len(self.scenario.option_labels)
        distribution = self.personas.preference_distribution
        weights_by_size = distribution.shape_weights
        for size in range(low, high + 1):
            weights = weights_by_size.get(size, weights_by_size.get(str(size)))
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"missing preference shape weights for group size {size}")
            total = 0.0
            for raw_shape, raw_weight in weights.items():
                shape = parse_preference_shape(raw_shape)
                if sum(shape) != size or len(shape) > option_count:
                    raise ValueError(f"invalid preference shape {shape} for group size {size}")
                total += float(raw_weight)
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"preference weights for group size {size} must sum to 1.0")
        forced = distribution.get("forced_shape", None)
        if forced is not None and sum(parse_preference_shape(forced)) != int(self.simulation.num_participants):
            raise ValueError("forced preference shape does not match the active group")

    def _validate_environment(self) -> None:
        env = self._raw.get("environment") or {}
        mode = str(env.get("mode", "auto"))
        if mode not in {"auto", "manual"}:
            raise ValueError("environment.mode must be auto or manual")
        if mode == "auto":
            return
        manual = env.get("manual")
        if not isinstance(manual, dict) or not manual:
            raise ValueError("manual environment is required")
        if set(manual) - _MANUAL_ENV_FIELDS:
            raise ValueError("manual environment contains unknown fields")
        options = manual.get("options")
        if not isinstance(options, list) or len(options) != len(self.scenario.option_labels):
            raise ValueError("manual environment has the wrong number of options")
        for option in options:
            if not isinstance(option, dict) or set(option) - _MANUAL_OPTION_FIELDS:
                raise ValueError("manual option contains unsupported fields")
            if not option.get("name") or not option.get("short_name") or not isinstance(option.get("attrs"), dict):
                raise ValueError("manual options require name, short_name, and attrs")

    def _validate_participants(self) -> None:
        section = self._raw.get("participants") or {}
        mode = str(section.get("mode", "auto"))
        if mode not in {"auto", "manual"}:
            raise ValueError("participants.mode must be auto or manual")
        if mode == "auto":
            return
        profiles = section.get("profiles")
        low, high = int(self.simulation.min_participants), int(self.simulation.max_participants)
        if not isinstance(profiles, list) or not low <= len(profiles) <= high:
            raise ValueError(f"manual profiles must contain {low}..{high} entries")
        blockers = 0
        names: list[str] = []
        labels = set(str(value).upper() for value in self.scenario.option_labels)
        for profile in profiles:
            if not isinstance(profile, dict) or set(profile) - _PROFILE_FIELDS:
                raise ValueError("manual profile contains unsupported fields")
            if profile.get("name"):
                names.append(str(profile["name"]).casefold())
            preferred = str(profile.get("preferred_option") or "").upper()
            if preferred and preferred not in labels:
                raise ValueError("manual preferred_option is invalid")
            hard = bool(profile.get("hard_blocker", False))
            blockers += int(hard)
            traits = profile.get("traits") or {}
            if not isinstance(traits, dict) or set(traits) - set(DIRECT_TRAIT_NAMES):
                raise ValueError("manual traits are invalid")
        if blockers > 1:
            raise ValueError("at most one manual hard blocker is allowed")
        if len(names) != len(set(names)):
            raise ValueError("manual names must be unique")

    def _validate_conversation(self) -> None:
        conv = self._raw.get("conversation") or {}
        minimum = float(conv.get("min_voluntary_turns_per_participant", 0))
        target = float(conv.get("soft_target_voluntary_turns_per_participant", 0))
        maximum = float(conv.get("hard_max_voluntary_turns_per_participant", 0))
        if not 0 <= minimum <= target <= maximum:
            raise ValueError("turn budgets must satisfy min <= target <= max")
        soft_cap = int(conv.get("soft_target_voluntary_turn_cap", 0))
        hard_cap = int(conv.get("hard_max_voluntary_turn_cap", 0))
        if not 1 <= soft_cap <= hard_cap:
            raise ValueError("absolute turn caps are invalid")
        for key in (
            "thread_turn_cap", "stagnation_no_bid_rounds", "compromise_window_max_turns",
            "recent_turns_in_prompt", "max_consecutive_turns",
        ):
            if int(conv.get(key, 0)) < 1:
                raise ValueError(f"conversation.{key} must be positive")

    def _validate_behavior(self) -> None:
        simulator = self._raw.get("simulator") or {}
        bid = self._level_mapping(simulator, "bid_probability_by_engagement", cast=float)
        movement = self._level_mapping(simulator, "movement_probability_by_stubbornness", cast=float)
        if any(not 0 <= value <= 1 for value in [*bid.values(), *movement.values()]):
            raise ValueError("behavior probabilities must be in [0, 1]")
        if any(bid[level] > bid[level + 1] for level in range(1, 5)):
            raise ValueError("bid probabilities must increase with engagement")
        if any(movement[level] < movement[level + 1] for level in range(1, 5)) or movement[5] != 0:
            raise ValueError("movement probabilities must decrease with stubbornness and end at zero")
        language = self._raw.get("language") or {}
        words = self._level_mapping(language, "max_words_by_verbosity", cast=int)
        directness = self._level_mapping(language, "directness_instructions", cast=str)
        if any(value <= 0 for value in words.values()) or any(words[level] > words[level + 1] for level in range(1, 5)):
            raise ValueError("verbosity word limits must be positive and monotonic")
        if any(not value.strip() for value in directness.values()):
            raise ValueError("directness instructions must be non-empty")
        caps = language.get("action_max_words")
        if not isinstance(caps, dict) or set(caps) != {"ask", "answer"}:
            raise ValueError("language.action_max_words must contain ask and answer")

    def level_value(self, section_name: str, mapping_name: str, level: int, *, cast=float):
        values = self._level_mapping(self._raw.get(section_name) or {}, mapping_name, cast=cast)
        try:
            return values[int(level)]
        except KeyError as exc:
            raise ValueError(f"level must be in 1..5, got {level}") from exc

    def action_word_cap(self, action_name: str) -> int:
        try:
            return int(self.language.action_max_words[action_name])
        except KeyError as exc:
            raise ValueError(f"unknown action word cap: {action_name}") from exc

    def conversation_turn_budgets(self, participant_count: int) -> tuple[int, int, int]:
        import math
        n = max(1, int(participant_count))
        conv = self._raw["conversation"]
        minimum = math.ceil(float(conv["min_voluntary_turns_per_participant"]) * n)
        target = min(
            math.ceil(float(conv["soft_target_voluntary_turns_per_participant"]) * n),
            int(conv["soft_target_voluntary_turn_cap"]),
        )
        maximum = min(
            math.ceil(float(conv["hard_max_voluntary_turns_per_participant"]) * n),
            int(conv["hard_max_voluntary_turn_cap"]),
        )
        return minimum, max(minimum, target), max(target, maximum)


cfg = Config(Path(__file__).resolve().parents[1] / "config.yaml")
