from __future__ import annotations

import copy
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import prompts  # noqa: E402
from builders import SetupBuilder  # noqa: E402
from config_loader import Config, cfg, parse_preference_shape  # noqa: E402


def _scenario_response() -> dict:
    options = []
    for option_id, name in zip("ABCD", ["Alpha Place", "Beta Place", "Gamma Place", "Delta Place"]):
        options.append({
            "id": option_id,
            "name": name,
            "short_name": name,
            "attrs": {"cost": "$10", "time": "2 hours", "capacity": "8 people"},
            "upside": f"A clear benefit for {name}.",
            "tradeoff": f"A clear trade-off for {name}.",
            "concern": f"A stable concern for {name}.",
            "best_for": f"People whose priority matches {name}.",
        })
    return {
        "scenario": {
            "decision_kind": "generic_decision",
            "opening_question": "What matters most for this choice?",
            "shared_context": ["Exactly 3 friends are deciding together."],
            "options": options,
        }
    }


def _persona_response(primary_by_id: dict[str, str]) -> dict:
    return {
        "participants": [
            {
                "id": pid,
                "name": f"Name{pid[1:]}",
                "background": f"A specific experience makes option {option_id} fit {pid}.",
                "private_goal": f"Get the benefit supplied by option {option_id}.",
                "preferred_options": [option_id],
                "rejection": None,
                "rejection_reason": "",
            }
            for pid, option_id in primary_by_id.items()
        ]
    }


class _ForcedShape:
    def __init__(self, value):
        self.value = value
        self.distribution = cfg.personas.preference_distribution
        self.previous = self.distribution._raw.get("forced_shape")

    def __enter__(self):
        self.distribution._raw["forced_shape"] = self.value

    def __exit__(self, exc_type, exc, tb):
        self.distribution._raw["forced_shape"] = self.previous


class PreferenceDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = SetupBuilder.__new__(SetupBuilder)

    def test_parse_shape_accepts_config_and_override_forms(self) -> None:
        self.assertEqual(parse_preference_shape("3-2-1"), (3, 2, 1))
        self.assertEqual(parse_preference_shape([2, 1]), (2, 1))
        for invalid in ("1-2", "2-0", [], "bad", 3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_preference_shape(invalid)

    def test_configured_shape_weights_are_complete_and_normalized(self) -> None:
        weights_by_size = cfg.personas.preference_distribution.shape_weights
        for n in range(int(cfg.simulation.min_participants), int(cfg.simulation.max_participants) + 1):
            weights = weights_by_size.get(n, weights_by_size.get(str(n)))
            self.assertIsInstance(weights, dict)
            self.assertAlmostEqual(sum(float(value) for value in weights.values()), 1.0)
            for raw_shape in weights:
                shape = parse_preference_shape(raw_shape)
                self.assertEqual(sum(shape), n)
                self.assertLessEqual(len(shape), len(cfg.scenario.option_labels))

    def test_sampling_tracks_each_group_sizes_configured_weights(self) -> None:
        random.seed(84117)
        weights_by_size = cfg.personas.preference_distribution.shape_weights
        draws = 12_000
        with _ForcedShape(None):
            for n in range(2, 8):
                expected = weights_by_size.get(n, weights_by_size.get(str(n)))
                observed = Counter(self.builder._preference_shape(n, 4) for _ in range(draws))
                for raw_shape, weight in expected.items():
                    shape = parse_preference_shape(raw_shape)
                    self.assertAlmostEqual(
                        observed[shape] / draws,
                        float(weight),
                        delta=0.02,
                        msg=f"n={n}, shape={shape}",
                    )

    def test_forced_shape_produces_exact_concrete_assignment(self) -> None:
        random.seed(22)
        with _ForcedShape([3, 2, 1]):
            assignments = self.builder._preference_assignments(6, ["A", "B", "C", "D"])
        self.assertEqual(set(assignments), {f"p{i}" for i in range(1, 7)})
        self.assertEqual(sorted(Counter(assignments.values()).values(), reverse=True), [3, 2, 1])
        self.assertTrue(set(assignments.values()) <= {"A", "B", "C", "D"})

    def test_invalid_forced_shape_fails_before_any_provider_call(self) -> None:
        data = copy.deepcopy(cfg._raw)
        data["personas"]["preference_distribution"]["forced_shape"] = [2, 2]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must sum to participant count"):
                Config(path)

        class NoCallLLM:
            calls = 0

            def generate_json(self, prompt: str, *, profile: str):
                self.calls += 1
                raise AssertionError("provider must not be called")

        builder = SetupBuilder.__new__(SetupBuilder)
        builder.topic = "Choose one"
        builder._llm = NoCallLLM()
        with _ForcedShape([2, 2]):
            with self.assertRaisesRegex(ValueError, "must sum to participant count"):
                builder.build(3)
        self.assertEqual(builder._llm.calls, 0)

    def test_persona_prompt_uses_row_local_concrete_assignments(self) -> None:
        rows = [
            {"id": "p1", "name": "Ana", "traits": {}},
            {"id": "p2", "name": "Bo", "traits": {}},
            {"id": "p3", "name": "Cy", "traits": {}},
        ]
        prompt = prompts.setup_personas(
            "Choose one",
            3,
            rows,
            {"p1": "B", "p2": "B", "p3": "D"},
            _scenario_response()["scenario"]["options"],
        )
        self.assertIn("p1 (Ana): B", prompt)
        self.assertIn("p2 (Bo): B", prompt)
        self.assertIn("p3 (Cy): D", prompt)
        self.assertIn("required primary option", prompt.lower())
        self.assertNotIn("same-camp", prompt)
        self.assertNotIn("Preference camps", prompt)

    def test_persona_retry_preserves_valid_scenario_and_assignments(self) -> None:
        required = {"p1": "A", "p2": "A", "p3": "B"}
        wrong = {"p1": "A", "p2": "A", "p3": "C"}
        trait_rows = [
            {
                "id": f"p{i}",
                "name": f"Name{i}",
                "traits": {
                    "openness": 3,
                    "conscientiousness": 3,
                    "extraversion": 3,
                    "agreeableness": 4,
                    "neuroticism": 2,
                },
            }
            for i in range(1, 4)
        ]

        class FakeLLM:
            def __init__(self):
                self.scenario_calls = 0
                self.persona_calls = 0

            def generate_json(self, prompt: str, *, profile: str):
                if prompt.startswith("Create a fictional group-decision scenario"):
                    self.scenario_calls += 1
                    return _scenario_response()
                self.persona_calls += 1
                return _persona_response(wrong if self.persona_calls == 1 else required)

        builder = SetupBuilder.__new__(SetupBuilder)
        builder.topic = "Choose one"
        builder._llm = FakeLLM()
        builder._trait_rows = lambda n: trait_rows
        builder._preference_shape = lambda n, option_count: (2, 1)
        builder._preference_assignments = lambda n, option_ids, shape: dict(required)

        scenario, personas = builder.build(3)

        self.assertEqual(scenario.option_ids, ["A", "B", "C", "D"])
        self.assertEqual({p.id: p.preferred_option for p in personas}, required)
        self.assertEqual(builder._llm.scenario_calls, 1)
        self.assertEqual(builder._llm.persona_calls, 2)


class SetupCoherenceTests(unittest.TestCase):
    def test_topic_count_mismatch_raises_before_llm(self) -> None:
        class NoCallLLM:
            calls = 0
            def generate_json(self, prompt, *, profile):
                self.calls += 1
                raise AssertionError("LLM must not be called")

        for topic in [
            "Plan a lunch for 5 colleagues",
            "Choose a city for our team of four friends",
            "Pick an activity for a group of 4 people",
        ]:
            builder = SetupBuilder.__new__(SetupBuilder)
            builder.topic = topic
            builder._llm = NoCallLLM()
            with self.assertRaises(ValueError, msg=f"Expected ValueError for topic: {topic!r}"):
                builder.build(3)
            self.assertEqual(builder._llm.calls, 0, f"LLM was called for topic: {topic!r}")

    def test_topic_without_count_does_not_raise(self) -> None:
        builder = SetupBuilder.__new__(SetupBuilder)
        builder.topic = "Plan a team lunch"
        try:
            builder._validate_topic_participant_count(builder.topic, 3)
        except ValueError:
            self.fail("Topic without explicit count should not raise")


if __name__ == "__main__":
    unittest.main()
