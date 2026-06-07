"""Scenario and option generation."""

from __future__ import annotations

from typing import Any

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from schemas import OptionCard, Scenario


class ScenarioBuilder:
    def __init__(self) -> None:
        self._llm = get_llm_client()

    def build(self, topic: str) -> Scenario:
        last_error = ""
        attempts = int(cfg.scenario.max_generation_attempts)
        for _ in range(max(1, attempts)):
            try:
                data = self._llm.generate_json(prompts.option_generation(topic), profile="setup")
                scenario = self._parse(topic, data)
                self._validate(scenario)
                return scenario
            except Exception as exc:  # generation errors should retry with a fresh sample
                last_error = str(exc)
        raise ValueError(f"Could not generate a valid scenario after {attempts} attempts: {last_error}")

    def _parse(self, topic: str, data: dict[str, Any]) -> Scenario:
        options_raw = data.get("options")
        if not isinstance(options_raw, list):
            raise ValueError("Scenario JSON must contain a list field 'options'.")
        options = [self._parse_option(raw, expected_id=str(label)) for raw, label in zip(options_raw, cfg.scenario.option_labels)]
        return Scenario(
            topic=topic,
            decision_kind=str(data.get("decision_kind") or "generic_decision").strip(),
            opening_question=str(data.get("opening_question") or "What matters most to everyone here?").strip(),
            options=options,
        )

    def _parse_option(self, raw: Any, expected_id: str) -> OptionCard:
        if isinstance(raw, dict):
            attrs = raw.get("attrs", {})
            if not isinstance(attrs, dict):
                attrs = {}
            return OptionCard(
                id=str(raw.get("id") or expected_id).strip().upper(),
                name=str(raw.get("name") or f"Option {expected_id}").strip(),
                attrs={str(k): str(v) for k, v in attrs.items()},
                upside=str(raw.get("upside") or "").strip(),
                tradeoff=str(raw.get("tradeoff") or "").strip(),
                concern=str(raw.get("concern") or "").strip(),
                fit=str(raw.get("fit") or "").strip(),
                risk=str(raw.get("risk") or "").strip(),
                best_for=str(raw.get("best_for") or raw.get("best for") or "").strip(),
            )
        if isinstance(raw, str):
            # Backward-compatible parser for older prompt outputs.
            text = raw.strip()
            label = expected_id
            if text.lower().startswith("option"):
                parts = text.split("-", 1)
                if parts:
                    label = parts[0].replace("Option", "").strip().upper() or expected_id
                    text = parts[1].strip() if len(parts) > 1 else text
            name = text.split(":", 1)[0].strip() or f"Option {expected_id}"
            return OptionCard(id=label, name=name, upside=text)
        raise ValueError(f"Invalid option object: {raw!r}")

    def _validate(self, scenario: Scenario) -> None:
        labels = [str(x) for x in cfg.scenario.option_labels]
        if [o.id for o in scenario.options] != labels:
            raise ValueError(f"Options must have ids {labels}, got {[o.id for o in scenario.options]}")
        if len({o.name.lower() for o in scenario.options}) != len(scenario.options):
            raise ValueError("Option names must be distinct.")
        forbidden = [str(x).lower() for x in cfg.scenario.forbidden_live_fact_terms]
        for option in scenario.options:
            text = option.source_text().lower()
            blocked = [term for term in forbidden if term and term in text]
            if blocked:
                raise ValueError(f"Option {option.id} contains forbidden live-fact terms: {blocked}")
            required = [option.name, option.upside, option.tradeoff, option.concern, option.fit, option.risk, option.best_for]
            if any(not str(field).strip() for field in required):
                raise ValueError(f"Option {option.id} is missing one or more required fields.")
