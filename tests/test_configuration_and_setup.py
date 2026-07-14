from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

import builders
from config_loader import Config, cfg
from models import SimulatorParameters


def _base_config() -> dict:
    return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_active_config_contains_only_one_llm_role_and_simple_moderator():
    raw = cfg._raw
    assert set(raw["llm"]) >= {"dialogue", "models", "sampling"}
    assert "validator" not in raw["llm"]
    assert set(raw["moderator"]) == {"enabled"}


def test_obsolete_runtime_controls_are_absent_from_config():
    text = Path("config.yaml").read_text(encoding="utf-8").casefold()
    for obsolete in (
        "ocean", "switch_resistance", "expected_turn_share", "validator:",
        "thread_priority", "repair_family", "fallback_family",
    ):
        assert obsolete not in text


def test_normal_direct_traits_validate_in_declared_ranges():
    params = SimulatorParameters(engagement=5, verbosity=1, directness=4, stubbornness=4)
    assert params.validated().stubbornness == 4
    with pytest.raises(ValueError):
        SimulatorParameters(3, 3, 3, 5).validated()


def test_explicit_hard_blocker_uses_reserved_stubbornness_five():
    params = SimulatorParameters(3, 3, 3, 1).validated(hard_blocker=True)
    assert params.stubbornness == 5


def test_unknown_llm_field_is_rejected(tmp_path):
    data = _base_config()
    data["llm"]["obsolete_role"] = "gpt"
    with pytest.raises(ValueError, match="unknown fields"):
        Config(_write_config(tmp_path, data))


def test_granular_moderator_flags_are_rejected(tmp_path):
    data = _base_config()
    data["moderator"]["narrowing"] = True
    with pytest.raises(ValueError, match="moderator"):
        Config(_write_config(tmp_path, data))


def test_stubbornness_generation_range_cannot_include_five(tmp_path):
    data = _base_config()
    data["personas"]["trait_ranges"]["stubbornness"] = [1, 5]
    with pytest.raises(ValueError, match="stubbornness"):
        Config(_write_config(tmp_path, data))


def test_manual_profiles_may_assign_direct_traits(tmp_path):
    data = _base_config()
    data["participants"] = {
        "mode": "manual",
        "profiles": [
            {
                "name": "Nora",
                "description": "Works on a practical project.",
                "private_goal": "needs reliable equipment",
                "preferred_option": "A",
                "age": 29,
                "speech_style": "relaxed practical wording",
                "traits": {"engagement": 5, "verbosity": 2, "directness": 4, "stubbornness": 3},
            },
            {
                "name": "Ben",
                "description": "Balances study and work.",
                "private_goal": "needs flexible access",
                "preferred_option": "B",
                "age": 38,
                "speech_style": "direct workplace wording",
                "traits": {"engagement": 2, "verbosity": 4, "directness": 5, "stubbornness": 2},
            },
        ],
    }
    data["simulation"]["min_participants"] = 2
    config = Config(_write_config(tmp_path, data))
    assert config.participant_count() == 2
    assert config.participants.profiles[0]["traits"]["engagement"] == 5


def test_manual_group_cannot_define_multiple_hard_blockers(tmp_path):
    data = _base_config()
    profile = {
        "description": "Has a strict requirement.",
        "private_goal": "requires one option",
        "preferred_option": "A",
        "age": 30,
        "speech_style": "plain wording",
        "traits": {"engagement": 3, "verbosity": 3, "directness": 3, "stubbornness": 4},
        "hard_blocker": True,
        "rejection_reason": "alternatives violate the requirement",
    }
    data["participants"] = {"mode": "manual", "profiles": [{**profile, "name": "Nora"}, {**profile, "name": "Ben"}]}
    data["simulation"]["min_participants"] = 2
    with pytest.raises(ValueError, match="hard blocker"):
        Config(_write_config(tmp_path, data))


def test_setup_builder_source_has_no_ocean_or_switch_resistance_conversion():
    source = inspect.getsource(builders).casefold()
    assert "switch_resistance" not in source
    assert "ocean" not in source
    assert "traitprofile" not in source


def test_setup_trait_sampling_uses_direct_traits_and_at_most_one_blocker(monkeypatch):
    monkeypatch.setattr(builders, "get_llm_client", lambda: object())
    monkeypatch.setattr(cfg.personas, "hard_blocker_probability", 1.0)
    builder = builders.SetupBuilder("Choose a test option")
    rows = builder._trait_rows(5)
    blockers = [row for row in rows if row["hard_blocker"]]
    assert len(blockers) == 1
    assert blockers[0]["traits"]["stubbornness"] == 5
    for row in rows:
        assert set(row["traits"]) == {"engagement", "verbosity", "directness", "stubbornness"}
        if not row["hard_blocker"]:
            assert 1 <= row["traits"]["stubbornness"] <= 4


def test_setup_trait_sampling_can_create_no_blocker(monkeypatch):
    monkeypatch.setattr(builders, "get_llm_client", lambda: object())
    monkeypatch.setattr(cfg.personas, "hard_blocker_probability", 0.0)
    builder = builders.SetupBuilder("Choose a test option")
    rows = builder._trait_rows(4)
    assert not any(row["hard_blocker"] for row in rows)
    assert builder._hard_blocker_id is None


def test_complete_manual_setup_builds_without_persona_llm(monkeypatch, tmp_path):
    data = _base_config()
    data["environment"] = {
        "type": "option_grounded_group_decision",
        "source_of_truth": "manual_option_board",
        "mode": "manual",
        "manual": {
            "topic": "Choose a study room",
            "shared_context": ["The meeting is on Saturday."],
            "options": [
                {"id": "A", "name": "Central Library", "short_name": "Library", "attrs": {"access": "public"}, "upside": "quiet", "concern": "crowded"},
                {"id": "B", "name": "Riverside Cafe", "short_name": "Cafe", "attrs": {"access": "public"}, "upside": "relaxed", "concern": "noisy"},
                {"id": "C", "name": "Engineering Lab", "short_name": "Engineering Lab", "attrs": {"access": "students"}, "upside": "equipment", "concern": "early closing"},
                {"id": "D", "name": "Online Session", "short_name": "Online", "attrs": {"access": "remote"}, "upside": "no travel", "concern": "less social"},
            ],
        },
    }
    data["participants"] = {
        "mode": "manual",
        "profiles": [
            {
                "name": "Nora", "description": "Works on a practical project.",
                "private_goal": "needs reliable equipment", "preferred_option": "C",
                "age": 29, "speech_style": "relaxed practical wording",
                "traits": {"engagement": 4, "verbosity": 3, "directness": 4, "stubbornness": 2},
            },
            {
                "name": "Ben", "description": "Balances study and work.",
                "private_goal": "needs flexible access", "preferred_option": "A",
                "age": 38, "speech_style": "direct workplace wording",
                "traits": {"engagement": 2, "verbosity": 2, "directness": 5, "stubbornness": 3},
            },
            {
                "name": "Mira", "description": "Has a strict remote-access requirement.",
                "private_goal": "must avoid travel", "preferred_option": "D",
                "age": 45, "speech_style": "measured practical wording",
                "traits": {"engagement": 3, "verbosity": 4, "directness": 3, "stubbornness": 4},
                "hard_blocker": True, "rejection_reason": "travel is not possible",
            },
        ],
    }
    manual_cfg = Config(_write_config(tmp_path, data))
    monkeypatch.setattr(builders, "cfg", manual_cfg)
    monkeypatch.setattr(builders, "get_llm_client", lambda: object())
    builder = builders.SetupBuilder("")
    scenario, personas = builder.build(3)
    assert scenario.topic == "Choose a study room"
    assert [persona.name for persona in personas] == ["Nora", "Ben", "Mira"]
    assert [persona.preferred_option for persona in personas] == ["C", "A", "D"]
    blocker = personas[2]
    assert blocker.hard_blocker
    assert blocker.sim_params.stubbornness == 5
    assert all(
        stance.rank == (5 if option_id == "D" else 1)
        for option_id, stance in blocker.option_stances.items()
    )


def test_single_option_rejection_does_not_implicitly_create_hard_blocker(monkeypatch):
    profiles = [
        {
            "name": "Nora",
            "description": "Works on a project.",
            "private_goal": "needs quiet",
            "preferred_option": "A",
            "rejection": "B",
            "rejection_reason": "background noise is unacceptable",
            "traits": {"engagement": 3, "verbosity": 3, "directness": 3, "stubbornness": 3},
        },
        {
            "name": "Ben",
            "description": "Balances work and study.",
            "private_goal": "needs access",
            "preferred_option": "C",
            "rejection": "D",
            "rejection_reason": "remote attendance is unsuitable",
            "traits": {"engagement": 3, "verbosity": 3, "directness": 3, "stubbornness": 3},
        },
    ]
    monkeypatch.setitem(cfg._raw["participants"], "mode", "manual")
    monkeypatch.setitem(cfg._raw["participants"], "profiles", profiles)
    normalized = builders.manual_participant_profiles()
    assert len(normalized) == 2
    assert not any(profile["hard_blocker"] for profile in normalized)
    assert [profile["rejection"] for profile in normalized] == ["B", "D"]


def test_setup_builder_reuses_the_runtime_llm_client():
    sentinel = object()
    builder = builders.SetupBuilder("Choose a study location", llm=sentinel)
    assert builder._llm is sentinel


def test_generated_scenario_alias_repair_uses_keyword_prompt_and_short_names_schema():
    class AliasRepairLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_json(self, prompt: str, *, profile: str = "setup") -> dict:
            self.prompts.append(prompt)
            assert profile == "setup"
            return {
                "short_names": {
                    "A": "Direct Flight",
                    "B": "One Stop",
                    "C": "Budget Route",
                    "D": "Flexible Ticket",
                }
            }

    llm = AliasRepairLLM()
    builder = builders.SetupBuilder("Book a flight to Stockholm", llm=llm)
    raw = {
        "shared_context": ["The group is travelling to Stockholm."],
        "options": [
            {
                "id": "A", "name": "Direct Stockholm Flight", "short_name": "",
                "attrs": {"stops": "none", "duration": "short", "fare": "standard"},
                "upside": "no connection", "concern": "higher fare",
            },
            {
                "id": "B", "name": "One Stop Stockholm Flight", "short_name": "",
                "attrs": {"stops": "one", "duration": "medium", "fare": "standard"},
                "upside": "more departure choices", "concern": "connection risk",
            },
            {
                "id": "C", "name": "Budget Stockholm Route", "short_name": "",
                "attrs": {"stops": "one", "duration": "long", "fare": "low"},
                "upside": "lower fare", "concern": "longer journey",
            },
            {
                "id": "D", "name": "Flexible Stockholm Ticket", "short_name": "",
                "attrs": {"stops": "varies", "duration": "varies", "fare": "flexible"},
                "upside": "changeable booking", "concern": "less predictable itinerary",
            },
        ],
    }

    scenario = builder._parse_scenario(raw, n=3)

    assert [option.short_name for option in scenario.options] == [
        "Direct Flight", "One Stop", "Budget Route", "Flexible Ticket",
    ]
    assert len(llm.prompts) == 1
    assert "Book a flight to Stockholm" in llm.prompts[0]
    assert '"short_names"' in llm.prompts[0]
    assert len([note for note in scenario.setup_notes if note.startswith("alias_repaired:")]) == 4


def test_eval_suite_uses_configured_dialogue_llm_instead_of_local_renderer():
    source = Path("eval/run_eval_suite.py").read_text(encoding="utf-8")
    assert "OfflineRenderer" not in source
    assert "get_llm_client()" in source
    assert "Using dialogue LLM" in source
