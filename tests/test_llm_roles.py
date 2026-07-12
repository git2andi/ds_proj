"""Item 2 (todo_validation.md): role-based LLM configuration and client lookup.

The dialogue role owns every generative call; the validator role owns
structured semantic interpretation. Both are independently configurable and
may name the same provider. No credentials are used: config validation is
tested on plain dicts and client caching through a stub class.
"""

from __future__ import annotations

import unittest
from unittest import mock

import llm_client
from config_loader import cfg, validate_llm_roles
from llm_client import LLMClient, get_llm_client


def _valid_llm_config(**overrides) -> dict:
    base = {
        "dialogue": "gpt",
        "validator": "gpt",
        "models": {"gpt": "gpt-4.1-mini", "uni": "llama3.3:latest"},
        "endpoints": {"uni": "http://example.invalid/api/generate"},
        "sampling": {
            "dialogue": {"temperature": 0.8, "top_k": 60, "top_p": 0.94},
            "validator": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
        },
    }
    base.update(overrides)
    return base


class RoleConfigValidation(unittest.TestCase):
    def test_same_provider_for_both_roles_is_valid(self) -> None:
        validate_llm_roles(_valid_llm_config())

    def test_cross_provider_configuration_is_valid(self) -> None:
        validate_llm_roles(_valid_llm_config(validator="uni"))

    def test_legacy_single_provider_key_is_a_migration_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "llm.provider was replaced"):
            validate_llm_roles(_valid_llm_config(provider="gpt"))

    def test_missing_role_is_rejected(self) -> None:
        config = _valid_llm_config()
        del config["validator"]
        with self.assertRaisesRegex(ValueError, "llm.validator"):
            validate_llm_roles(config)

    def test_unknown_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "llm.dialogue must be one of"):
            validate_llm_roles(_valid_llm_config(dialogue="claude"))

    def test_provider_without_model_is_rejected(self) -> None:
        config = _valid_llm_config(validator="groq")
        with self.assertRaisesRegex(ValueError, "no entry under llm.models"):
            validate_llm_roles(config)

    def test_uni_role_requires_endpoint(self) -> None:
        config = _valid_llm_config(validator="uni", endpoints={})
        with self.assertRaisesRegex(ValueError, "llm.endpoints.uni"):
            validate_llm_roles(config)

    def test_missing_validator_sampling_profile_is_rejected(self) -> None:
        config = _valid_llm_config()
        del config["sampling"]["validator"]
        with self.assertRaisesRegex(ValueError, "llm.sampling.validator"):
            validate_llm_roles(config)

    def test_repo_config_defines_both_roles(self) -> None:
        validate_llm_roles(dict(cfg.llm.items()))


class RoleAwareClientLookup(unittest.TestCase):
    def tearDown(self) -> None:
        llm_client._CLIENTS.clear()

    def test_clients_are_cached_per_role(self) -> None:
        created: list[str] = []

        class FakeClient:
            def __init__(self, role: str) -> None:
                self.role = role
                created.append(role)

        with mock.patch.object(llm_client, "LLMClient", FakeClient):
            llm_client._CLIENTS.clear()
            dialogue = get_llm_client("dialogue")
            validator = get_llm_client("validator")
            self.assertIs(get_llm_client("dialogue"), dialogue)
            self.assertIsNot(dialogue, validator)
            self.assertEqual(created, ["dialogue", "validator"])
            self.assertEqual(dialogue.role, "dialogue")
            self.assertEqual(validator.role, "validator")

    def test_unconfigured_role_fails_before_building_a_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "No provider configured for LLM role"):
            LLMClient("no_such_role")

    def test_validator_sampling_profile_is_cold(self) -> None:
        client = LLMClient.__new__(LLMClient)
        sampling = client._sampling("validator")
        self.assertEqual(sampling["temperature"], 0.0)
        self.assertEqual(sampling["top_k"], 1)
        self.assertEqual(sampling["top_p"], 1.0)

    def test_dialogue_sampling_profile_stays_warm(self) -> None:
        client = LLMClient.__new__(LLMClient)
        self.assertGreater(client._sampling("dialogue")["temperature"], 0.5)


if __name__ == "__main__":
    unittest.main()
