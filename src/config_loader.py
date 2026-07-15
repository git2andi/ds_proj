"""Small configuration loader for the autonomous simulator runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

DIRECT_TRAIT_NAMES = ("engagement", "verbosity", "directness", "stubbornness")
PROFILE_TRAIT_NAMES = DIRECT_TRAIT_NAMES  # public alias used by setup code
KNOWN_LLM_PROVIDERS = frozenset({"uni", "groq", "gemini", "gpt"})
_PROFILE_FIELDS = frozenset({
    "name", "description", "private_goal", "preferred_option", "age",
    "speech_style", "style_tendencies", "traits", "hard_blocker", "rejection", "rejection_reason",
})
_MANUAL_ENV_FIELDS = frozenset({"topic", "shared_context", "options"})
_MANUAL_OPTION_FIELDS = frozenset({"id", "name", "short_name", "attrs", "upside", "concern"})
_LLM_FIELDS = frozenset({
    "dialogue", "models", "endpoints", "timeouts",
    "gemini_rpm_delay_seconds", "sampling",
})


def parse_preference_shape(value: Any) -> tuple[int, ...]:
    raw = value.split("-") if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else None
    if not raw:
        raise ValueError("preference shape must be a hyphenated string or integer list")
    try:
        parts = tuple(int(part) for part in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid preference shape: {value!r}") from exc
    if any(part <= 0 for part in parts) or tuple(sorted(parts, reverse=True)) != parts:
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

    def get(self, key: str, default: Any = None) -> Any:
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

    def _validate(self) -> None:
        for key in (
            "llm.dialogue", "llm.models", "simulation.num_participants",
            "simulation.min_participants", "simulation.max_participants",
            "scenario.option_labels", "conversation.hard_max_voluntary_turns_per_participant",
            "conversation.small_group_max_participants",
            "conversation.small_group_extra_no_bid_rounds",
            "conversation.small_group_shared_acceptance_extra_turns",
            "conversation.unanimous_closure_min_voluntary_turns_per_participant",
            "conversation.large_group_min_participants",
            "conversation.large_group_optional_reaction_window_cap",
            "conversation.large_group_narrowing_issue_turn_cap",
            "conversation.max_concern_reopens",
            "simulator.bid_probability_by_engagement",
            "simulator.question_modes",
            "simulator.movement_probability_by_stubbornness",
            "language.max_words_by_verbosity",
            "language.action_max_words",
            "language.near_duplicate_similarity_threshold",
            "language.near_duplicate_recent_turns",
            "language.directness_instructions",
            "output.log_dir",
        ):
            self._require(key)
        self._validate_llm()
        n = int(self.simulation.num_participants)
        lo, hi = int(self.simulation.min_participants), int(self.simulation.max_participants)
        if not lo <= n <= hi:
            raise ValueError("simulation.num_participants must satisfy min <= n <= max")
        labels = [str(value).upper() for value in self.scenario.option_labels]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("scenario.option_labels must be non-empty and unique")
        for trait in DIRECT_TRAIT_NAMES:
            high = 4 if trait == "stubbornness" else 5
            self._range(f"personas.trait_ranges.{trait}", self.personas.trait_ranges[trait], 1, high)
        probability = float(self.personas.hard_blocker_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("personas.hard_blocker_probability must be in [0, 1]")
        self._validate_preferences(lo, hi, len(labels), n)
        self._validate_participants(lo, hi, labels)
        self._validate_environment(labels)
        self._validate_conversation()
        self._validate_behavior_mappings()
        moderator = self._raw.get("moderator") or {}
        if set(moderator) - {"enabled"}:
            raise ValueError("moderator only supports the 'enabled' flag")
        if "enabled" in moderator and not isinstance(moderator["enabled"], bool):
            raise ValueError("moderator.enabled must be a boolean")

    def _validate_llm(self) -> None:
        llm = self._raw.get("llm") or {}
        unknown = set(llm) - _LLM_FIELDS
        if unknown:
            raise ValueError(f"llm has unknown fields: {sorted(unknown)}")
        provider = str(llm.get("dialogue") or "").lower()
        if provider not in KNOWN_LLM_PROVIDERS:
            raise ValueError(f"llm.dialogue must be one of {sorted(KNOWN_LLM_PROVIDERS)}")
        models = llm.get("models") or {}
        if provider not in models:
            raise ValueError(f"llm.dialogue={provider!r} has no model mapping")
        if provider == "uni" and not str((llm.get("endpoints") or {}).get("uni") or "").strip():
            raise ValueError("llm.dialogue='uni' requires llm.endpoints.uni")
        sampling = llm.get("sampling") or {}
        for profile in ("setup", "dialogue", "repair"):
            if profile not in sampling:
                raise ValueError(f"llm.sampling.{profile} is required")

    @staticmethod
    def _level_mapping(section: dict[str, Any], name: str, *, cast=float) -> dict[int, Any]:
        raw = section.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"{name} must be a mapping with levels 1..5")
        result: dict[int, Any] = {}
        for level in range(1, 6):
            value = raw.get(level, raw.get(str(level)))
            if value is None:
                raise ValueError(f"{name} is missing level {level}")
            result[level] = cast(value)
        extra = {str(key) for key in raw} - {str(level) for level in range(1, 6)}
        if extra:
            raise ValueError(f"{name} has unsupported levels: {sorted(extra)}")
        return result

    def _validate_conversation(self) -> None:
        conv = self._raw.get("conversation") or {}
        minimum = float(conv.get("min_voluntary_turns_per_participant", 0))
        target = float(conv.get("soft_target_voluntary_turns_per_participant", 0))
        maximum = float(conv.get("hard_max_voluntary_turns_per_participant", 0))
        if not 0 <= minimum <= target <= maximum:
            raise ValueError(
                "voluntary per-participant budgets must satisfy 0 <= min <= soft target <= hard max"
            )
        soft_cap = int(conv.get("soft_target_voluntary_turn_cap", 0))
        hard_cap = int(conv.get("hard_max_voluntary_turn_cap", 0))
        if soft_cap < 1 or hard_cap < soft_cap:
            raise ValueError("absolute voluntary-turn caps must satisfy 1 <= soft cap <= hard cap")
        if int(conv.get("issue_follow_up_cap", 5)) < 1:
            raise ValueError("conversation.issue_follow_up_cap must be positive")
        if int(conv.get("direct_question_optional_follow_up_cap", 1)) < 0:
            raise ValueError("conversation.direct_question_optional_follow_up_cap must be non-negative")
        if int(conv.get("concern_external_response_cap", 2)) < 0:
            raise ValueError("conversation.concern_external_response_cap must be non-negative")
        if int(conv.get("max_concerns_per_participant", 1)) < 0:
            raise ValueError("conversation.max_concerns_per_participant must be non-negative")
        if not isinstance(conv.get("diagnostic_allow_reason_reuse", False), bool):
            raise ValueError("conversation.diagnostic_allow_reason_reuse must be a boolean")
        if int(conv.get("stagnation_no_bid_rounds", 1)) < 1:
            raise ValueError("conversation.stagnation_no_bid_rounds must be positive")
        if int(conv.get("compromise_window_max_turns", 2)) < 1:
            raise ValueError("conversation.compromise_window_max_turns must be positive")
        if int(conv.get("narrowing_reaction_turn_cap", 2)) < 0:
            raise ValueError("conversation.narrowing_reaction_turn_cap must be non-negative")
        if int(conv.get("large_group_narrowing_final_position_cap", 3)) < 1:
            raise ValueError("conversation.large_group_narrowing_final_position_cap must be positive")
        if int(conv.get("small_group_extra_no_bid_rounds", 1)) < 0:
            raise ValueError("conversation.small_group_extra_no_bid_rounds must be non-negative")
        if int(conv.get("small_group_shared_acceptance_extra_turns", 3)) < 0:
            raise ValueError("conversation.small_group_shared_acceptance_extra_turns must be non-negative")
        if float(conv.get("unanimous_closure_min_voluntary_turns_per_participant", 1.0)) < 0:
            raise ValueError(
                "conversation.unanimous_closure_min_voluntary_turns_per_participant must be non-negative"
            )
        if int(conv.get("large_group_optional_reaction_window_cap", 2)) < 0:
            raise ValueError("conversation.large_group_optional_reaction_window_cap must be non-negative")
        if int(conv.get("large_group_narrowing_issue_turn_cap", 1)) < 0:
            raise ValueError("conversation.large_group_narrowing_issue_turn_cap must be non-negative")
        if int(conv.get("max_concern_reopens", 1)) < 0:
            raise ValueError("conversation.max_concern_reopens must be non-negative")
        small_group_max = int(conv.get("small_group_max_participants", 4))
        large_group_min = int(conv.get("large_group_min_participants", 5))
        simulation_min = int(self.simulation.min_participants)
        simulation_max = int(self.simulation.max_participants)
        if not simulation_min <= small_group_max <= simulation_max:
            raise ValueError("conversation.small_group_max_participants must be within the supported group-size range")
        if not simulation_min <= large_group_min <= simulation_max:
            raise ValueError("conversation.large_group_min_participants must be within the supported group-size range")
        if small_group_max >= large_group_min:
            raise ValueError("small-group maximum must be lower than large-group minimum")
        if int(conv.get("recent_turns_in_prompt", 5)) < 1:
            raise ValueError("conversation.recent_turns_in_prompt must be positive")
        if int(conv.get("max_consecutive_turns", 2)) < 1:
            raise ValueError("conversation.max_consecutive_turns must be positive")

    def _validate_behavior_mappings(self) -> None:
        simulator = self._raw.get("simulator") or {}
        bid = self._level_mapping(simulator, "bid_probability_by_engagement", cast=float)
        movement = self._level_mapping(simulator, "movement_probability_by_stubbornness", cast=float)
        for name, values in (("bid_probability_by_engagement", bid), ("movement_probability_by_stubbornness", movement)):
            if any(not 0.0 <= value <= 1.0 for value in values.values()):
                raise ValueError(f"simulator.{name} values must be in [0, 1]")
        if any(bid[level] > bid[level + 1] for level in range(1, 5)):
            raise ValueError("bid probabilities must be non-decreasing with engagement")
        if any(movement[level] < movement[level + 1] for level in range(1, 5)):
            raise ValueError("movement probabilities must be non-increasing with stubbornness")
        if movement[5] != 0.0:
            raise ValueError("stubbornness level 5 must have zero movement probability")
        question_modes = simulator.get("question_modes")
        allowed_question_modes = {"choice_impact", "tradeoff"}
        if not isinstance(question_modes, list) or not question_modes:
            raise ValueError("simulator.question_modes must be a non-empty list")
        normalized_modes = [str(value) for value in question_modes]
        if len(normalized_modes) != len(set(normalized_modes)):
            raise ValueError("simulator.question_modes must not contain duplicates")
        unknown_modes = set(normalized_modes) - allowed_question_modes
        if unknown_modes:
            raise ValueError(f"simulator.question_modes contains unsupported values: {sorted(unknown_modes)}")
        unknown_probability = float(simulator.get("unknown_information_question_probability", 0.0))
        if not 0.0 <= unknown_probability <= 1.0:
            raise ValueError("simulator.unknown_information_question_probability must be in [0, 1]")

        language = self._raw.get("language") or {}
        words = self._level_mapping(language, "max_words_by_verbosity", cast=int)
        directness = self._level_mapping(language, "directness_instructions", cast=str)
        if any(value <= 0 for value in words.values()):
            raise ValueError("language.max_words_by_verbosity values must be positive")
        if any(words[level] > words[level + 1] for level in range(1, 5)):
            raise ValueError("word limits must be non-decreasing with verbosity")
        if any(not value.strip() for value in directness.values()):
            raise ValueError("directness instructions must be non-empty")
        action_max_words = language.get("action_max_words")
        expected_action_caps = {
            "acknowledge", "ask", "answer", "final_position", "vote", "simple_vote"
        }
        if not isinstance(action_max_words, dict):
            raise ValueError("language.action_max_words must be a mapping")
        unknown_action_caps = set(action_max_words) - expected_action_caps
        missing_action_caps = expected_action_caps - set(action_max_words)
        if unknown_action_caps or missing_action_caps:
            raise ValueError(
                "language.action_max_words must contain exactly "
                f"{sorted(expected_action_caps)}"
            )
        caps = {name: int(value) for name, value in action_max_words.items()}
        if any(value <= 0 for value in caps.values()):
            raise ValueError("language.action_max_words values must be positive")
        if caps["simple_vote"] > caps["vote"]:
            raise ValueError("language.action_max_words.simple_vote must not exceed vote")
        similarity = float(language.get("near_duplicate_similarity_threshold", 0.92))
        if not 0.0 <= similarity <= 1.0:
            raise ValueError("language.near_duplicate_similarity_threshold must be in [0, 1]")
        if int(language.get("near_duplicate_recent_turns", 3)) < 1:
            raise ValueError("language.near_duplicate_recent_turns must be positive")

    def level_value(self, section_name: str, mapping_name: str, level: int, *, cast=float):
        section = self._raw.get(section_name) or {}
        values = self._level_mapping(section, mapping_name, cast=cast)
        try:
            return values[int(level)]
        except KeyError as exc:
            raise ValueError(f"level must be in 1..5, got {level}") from exc

    def action_word_cap(self, action_name: str) -> int:
        caps = (self._raw.get("language") or {}).get("action_max_words") or {}
        try:
            return int(caps[action_name])
        except KeyError as exc:
            raise ValueError(f"unknown language action word cap: {action_name}") from exc

    def conversation_turn_budgets(self, participant_count: int) -> tuple[int, int, int]:
        n = max(1, int(participant_count))
        conv = self._raw.get("conversation") or {}
        import math
        minimum = math.ceil(float(conv["min_voluntary_turns_per_participant"]) * n)
        target = min(
            math.ceil(float(conv["soft_target_voluntary_turns_per_participant"]) * n),
            int(conv["soft_target_voluntary_turn_cap"]),
        )
        maximum = min(
            math.ceil(float(conv["hard_max_voluntary_turns_per_participant"]) * n),
            int(conv["hard_max_voluntary_turn_cap"]),
        )
        target = max(minimum, target)
        maximum = max(target, maximum)
        return minimum, target, maximum

    def _validate_environment(self, labels: list[str]) -> None:
        env = self._raw.get("environment") or {}
        mode = str(env.get("mode", "auto"))
        if mode not in {"auto", "manual"}:
            raise ValueError("environment.mode must be 'auto' or 'manual'")
        if mode == "auto":
            return
        manual = env.get("manual")
        if not isinstance(manual, dict) or not manual:
            raise ValueError("environment.mode=manual requires environment.manual")
        unknown = set(manual) - _MANUAL_ENV_FIELDS
        if unknown:
            raise ValueError(f"environment.manual has unknown fields: {sorted(unknown)}")
        if not str(manual.get("topic") or "").strip():
            raise ValueError("environment.manual.topic must be non-empty")
        options = manual.get("options")
        if not isinstance(options, list) or len(options) != len(labels):
            raise ValueError(f"environment.manual.options must contain exactly {len(labels)} options")
        for index, option in enumerate(options):
            if not isinstance(option, dict):
                raise ValueError(f"environment.manual.options[{index}] must be a mapping")
            unknown = set(option) - _MANUAL_OPTION_FIELDS
            if unknown:
                raise ValueError(f"manual option has unknown fields: {sorted(unknown)}")
            if not str(option.get("name") or "").strip() or not str(option.get("short_name") or "").strip():
                raise ValueError("manual options require name and short_name")
            if not isinstance(option.get("attrs"), dict) or not option["attrs"]:
                raise ValueError("manual options require at least one public attribute")

    def _validate_participants(self, min_n: int, max_n: int, labels: list[str]) -> None:
        participants = self._raw.get("participants") or {}
        mode = str(participants.get("mode", "auto"))
        if mode not in {"auto", "manual"}:
            raise ValueError("participants.mode must be 'auto' or 'manual'")
        if mode == "auto":
            return
        profiles = participants.get("profiles")
        if not isinstance(profiles, list) or not min_n <= len(profiles) <= max_n:
            raise ValueError(f"manual profiles must contain {min_n}..{max_n} entries")
        blockers = 0
        names: list[str] = []
        for index, profile in enumerate(profiles):
            where = f"participants.profiles[{index}]"
            if not isinstance(profile, dict):
                raise ValueError(f"{where} must be a mapping")
            unknown = set(profile) - _PROFILE_FIELDS
            if unknown:
                raise ValueError(f"{where} has unknown fields: {sorted(unknown)}")
            name = str(profile.get("name") or "").strip()
            if name:
                names.append(name.casefold())
            preferred = str(profile.get("preferred_option") or "").strip().upper()
            if preferred and preferred not in labels:
                raise ValueError(f"{where}.preferred_option must be one of {labels}")
            hard = bool(profile.get("hard_blocker", False))
            blockers += int(hard)
            rejection = str(profile.get("rejection") or "").strip().upper()
            if rejection and rejection not in labels:
                raise ValueError(f"{where}.rejection must be one of {labels}")
            if (hard or rejection) and not str(profile.get("rejection_reason") or "").strip():
                raise ValueError(f"{where} hard rejection requires rejection_reason")
            traits = profile.get("traits") or {}
            if not isinstance(traits, dict):
                raise ValueError(f"{where}.traits must be a mapping")
            for key, raw in traits.items():
                if key not in DIRECT_TRAIT_NAMES:
                    raise ValueError(f"{where}.traits has unknown trait {key!r}")
                normal_high = 5 if key != "stubbornness" else (5 if hard else 4)
                if not 1 <= int(raw) <= normal_high:
                    raise ValueError(f"{where}.traits.{key} out of range")
        if blockers > 1:
            raise ValueError("manual profiles may configure at most one hard blocker")
        if len(names) != len(set(names)):
            raise ValueError("manual profile names must be unique")

    def _validate_preferences(self, min_n: int, max_n: int, option_count: int, active_n: int) -> None:
        distribution = self._require("personas.preference_distribution")
        weights_by_size = distribution.get("shape_weights")
        if not isinstance(weights_by_size, dict):
            raise ValueError("personas.preference_distribution.shape_weights must be a mapping")
        for size in range(min_n, max_n + 1):
            weights = weights_by_size.get(size, weights_by_size.get(str(size)))
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"missing preference shape weights for group size {size}")
            total = 0.0
            for raw_shape, raw_weight in weights.items():
                shape = parse_preference_shape(raw_shape)
                if sum(shape) != size or len(shape) > option_count:
                    raise ValueError(f"invalid preference shape {shape} for group size {size}")
                weight = float(raw_weight)
                if weight < 0:
                    raise ValueError("preference weights must be non-negative")
                total += weight
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"preference weights for group size {size} must sum to 1.0")
        forced = distribution.get("forced_shape")
        if forced is not None:
            shape = parse_preference_shape(forced)
            if sum(shape) != active_n or len(shape) > option_count:
                raise ValueError("forced preference shape is incompatible with the active group")


cfg = Config(Path(__file__).resolve().parents[1] / "config.yaml")
