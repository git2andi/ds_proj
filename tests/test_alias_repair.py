"""Alias-only repair during automatic scenario generation (todo_validation item 3).

A substantively valid scenario is never discarded because a generated
short_name is invalid or duplicated: the affected aliases get one small
alias-only repair call (dialogue LLM role) with deterministic re-validation,
a small retry limit, and a precise final error.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import tests  # noqa: F401  # puts src/ on sys.path

from builders import SetupBuilder


def _scenario_payload(*, b_alias: str = "Lufthansa", c_alias: str = "Finnair") -> dict[str, Any]:
    def option(oid: str, name: str, short: str) -> dict[str, Any]:
        return {
            "id": oid,
            "name": name,
            "short_name": short,
            "attrs": {"price": f"{400 + ord(oid)} euros", "duration": "11 hours", "stops": "1 stop"},
            "upside": "reasonable overall travel time",
            "concern": "long layover on the return leg",
        }

    return {
        "scenario": {
            "shared_context": ["Budget is capped at 700 euros per person."],
            "options": [
                option("A", "SAS Direct Flight from Miami", "SAS Direct"),
                option("B", "Lufthansa Flight via London", b_alias),
                option("C", "Finnair Flight via Helsinki", c_alias),
                option("D", "Icelandair Flight via Reykjavik", "Icelandair"),
            ],
        }
    }


def _persona_payload(n: int) -> dict[str, Any]:
    return {
        "participants": [
            {
                "id": f"p{i + 1}",
                "name": f"P{i + 1}",
                "age": 30 + i,
                "background": "travels for work a few times a year and books flights often.",
                "private_goal": "wants a reasonably priced flight without a brutal layover",
                "preferred_options": ["A"],
            }
            for i in range(n)
        ]
    }


class FakeSetupLLM:
    """Serves scenario/persona/alias payloads by prompt shape; counts calls.

    ``alias_payloads`` are consumed in order across alias-repair attempts;
    the last one repeats when attempts exceed the scripted list.
    """

    def __init__(self, scenario_payload: dict[str, Any],
                 alias_payloads: list[dict[str, Any]] | None = None) -> None:
        self.scenario_payload = scenario_payload
        self.alias_payloads = list(alias_payloads or [])
        self.scenario_calls = 0
        self.alias_calls = 0
        self.alias_prompts: list[str] = []
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0

    def reset_session(self) -> None:
        self.session_tokens_in = 0
        self.session_tokens_out = 0

    def generate_json(self, prompt: str, *, profile: str = "setup") -> dict[str, Any]:
        self.last_tokens_in = max(1, len(prompt.split()))
        self.last_tokens_out = 50
        self.session_tokens_in += self.last_tokens_in
        self.session_tokens_out += self.last_tokens_out
        if "simulated users" in prompt:
            return _persona_payload(3)
        if "short alias" in prompt.lower():
            self.alias_calls += 1
            self.alias_prompts.append(prompt)
            if not self.alias_payloads:
                return {}
            index = min(self.alias_calls - 1, len(self.alias_payloads) - 1)
            return self.alias_payloads[index]
        self.scenario_calls += 1
        return self.scenario_payload


def make_builder(llm: FakeSetupLLM, topic: str = "Book a flight from Miami to Stockholm") -> SetupBuilder:
    builder = SetupBuilder.__new__(SetupBuilder)
    builder.topic = topic
    builder._profiles = []
    builder._manual_env = None
    builder._llm = llm
    builder._hard_blocker_id = None
    return builder


class AliasRepairTests(unittest.TestCase):
    def setUp(self):
        # The sampled-hard-blocker draw would make the generic persona payload
        # fail validation nondeterministically; pin the draw above threshold.
        patcher = mock.patch("builders.random.random", return_value=0.99)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_valid_aliases_build_without_repair_call(self):
        llm = FakeSetupLLM(_scenario_payload())
        scenario, personas = make_builder(llm).build(3)
        self.assertEqual(scenario.option_ids, ["A", "B", "C", "D"])
        self.assertEqual(len(personas), 3)
        self.assertEqual(llm.scenario_calls, 1)
        self.assertEqual(llm.alias_calls, 0)

    def test_invalid_alias_is_repaired_without_scenario_regeneration(self):
        llm = FakeSetupLLM(
            _scenario_payload(b_alias="London Stop"),  # "Stop" is not a word of the name
            alias_payloads=[{"aliases": {"B": "Lufthansa"}}],
        )
        scenario, _personas = make_builder(llm).build(3)
        self.assertEqual(llm.scenario_calls, 1)  # scenario kept, only aliases repaired
        self.assertEqual(llm.alias_calls, 1)
        self.assertEqual(scenario.option("B").short_name, "Lufthansa")
        self.assertTrue(any(note.startswith("invalid_alias") for note in scenario.setup_notes))
        self.assertTrue(any(note.startswith("alias_repaired") for note in scenario.setup_notes))

    def test_duplicate_alias_is_repaired(self):
        llm = FakeSetupLLM(
            _scenario_payload(b_alias="Flight", c_alias="Flight"),  # colliding aliases
            alias_payloads=[{"aliases": {"C": "Finnair"}}],
        )
        scenario, _personas = make_builder(llm).build(3)
        self.assertEqual(llm.scenario_calls, 1)
        # Only the colliding (second) option is repaired; the first keeps its alias.
        self.assertEqual(scenario.option("B").short_name, "Flight")
        self.assertEqual(scenario.option("C").short_name, "Finnair")
        aliases = [o.short_name.casefold() for o in scenario.options]
        self.assertEqual(len(set(aliases)), len(aliases))
        self.assertTrue(any(note.startswith("duplicate_alias") for note in scenario.setup_notes))

    def test_bad_repair_suggestion_is_rejected_then_retried(self):
        llm = FakeSetupLLM(
            _scenario_payload(b_alias="London Stop"),
            alias_payloads=[
                {"aliases": {"B": "Berlin Hub"}},  # invented words -> deterministically rejected
                {"aliases": {"B": "via London"}},  # words from the name -> accepted
            ],
        )
        scenario, _personas = make_builder(llm).build(3)
        self.assertEqual(llm.alias_calls, 2)
        self.assertEqual(scenario.option("B").short_name, "via London")
        self.assertEqual(llm.scenario_calls, 1)
        # The retry prompt names the previously rejected suggestions.
        self.assertIn("Berlin Hub", llm.alias_prompts[1])

    def test_alias_repair_failure_raises_precise_error(self):
        llm = FakeSetupLLM(
            _scenario_payload(b_alias="London Stop"),
            alias_payloads=[{"aliases": {"B": "Nonsense Words"}}],  # never valid
        )
        with self.assertRaises(RuntimeError) as ctx:
            make_builder(llm).build(3)
        message = str(ctx.exception)
        self.assertIn("alias_repair_failed", message)
        self.assertIn("option B", message)
        # Alias repair is bounded per scenario attempt; the scenario itself
        # still retries through the normal attempt loop.
        self.assertGreaterEqual(llm.alias_calls, 2)

    def test_repair_receives_only_affected_options_and_taken_aliases(self):
        llm = FakeSetupLLM(
            _scenario_payload(b_alias="London Stop"),
            alias_payloads=[{"aliases": {"B": "Lufthansa"}}],
        )
        make_builder(llm).build(3)
        prompt = llm.alias_prompts[0]
        self.assertIn("Lufthansa Flight via London", prompt)
        self.assertIn("London Stop", prompt)          # rejected alias shown
        self.assertIn("SAS Direct", prompt)           # taken aliases shown
        self.assertNotIn("upside", prompt)            # no option facts resent
        self.assertNotIn("shared_context", prompt)


if __name__ == "__main__":
    unittest.main()
