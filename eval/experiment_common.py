"""Shared in-process helpers for the run-producing scripts in ``eval2/``.

Configuration overrides are applied in memory only. ``config.yaml`` is never
modified, so an interrupted experiment cannot leave the project in a patched
state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from evaluation_metrics import EVAL_DIR, ROOT

# Transcripts contain Unicode characters that can fail on a Windows console
# using cp1252. Reconfigure when the stream supports it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config_loader import Section, cfg  # noqa: E402
from builders import SetupBuilder  # noqa: E402
from dialogue import DialogueRunner  # noqa: E402
from eval import flat_metrics_for  # noqa: E402
from llm_client import get_llm_client  # noqa: E402
from models import Persona, Scenario  # noqa: E402

SCENARIOS_PATH = EVAL_DIR / "scenarios.txt"


@dataclass(frozen=True)
class ScenarioCase:
    index: int
    participants: int
    topic: str


def read_scenarios(path: Path = SCENARIOS_PATH) -> list[ScenarioCase]:
    """Parse ``participant_count | topic`` lines with strict validation."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    cases: list[ScenarioCase] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"{path}:{line_number}: expected 'participant_count | topic'")
        count_text, topic = (part.strip() for part in line.split("|", 1))
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid participant count {count_text!r}") from exc
        minimum = int(cfg.simulation.min_participants)
        maximum = int(cfg.simulation.max_participants)
        if not minimum <= count <= maximum:
            raise ValueError(f"{path}:{line_number}: participant count must be {minimum}..{maximum}")
        if not topic:
            raise ValueError(f"{path}:{line_number}: topic must not be empty")
        cases.append(ScenarioCase(len(cases) + 1, count, topic))
    if not cases:
        raise ValueError(f"{path} contains no scenarios")
    duplicates = [topic for topic, count in Counter(case.topic for case in cases).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate topics in {path}: {duplicates}")
    return cases


def set_config_value(section_name: str, key: str, value: Any) -> Any:
    """Override one config value in memory and return the previous raw value."""
    section_raw = cfg._raw[section_name]
    if key not in section_raw:
        raise KeyError(f"unknown config key: {section_name}.{key}")
    previous = section_raw[key]
    section_raw[key] = value
    section = getattr(cfg, section_name)
    setattr(section, key, Section(value) if isinstance(value, dict) else value)
    return previous


@contextmanager
def config_overrides(overrides: dict[tuple[str, str], Any]) -> Iterator[None]:
    """Apply in-memory overrides and always restore the previous values."""
    previous: dict[tuple[str, str], Any] = {}
    try:
        for (section_name, key), value in overrides.items():
            previous[(section_name, key)] = set_config_value(section_name, key, value)
        yield
    finally:
        for (section_name, key), value in reversed(list(previous.items())):
            set_config_value(section_name, key, value)


def setup_fingerprint(scenario: Scenario, personas: list[Persona]) -> str:
    """Return a stable identifier proving that paired runs used one setup."""
    payload = {
        "scenario": asdict(scenario),
        "personas": [asdict(persona) for persona in personas],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def prepare_dialogue_setup(
    topic: str,
    *,
    participants: int,
    seed: int,
    llm: Any = None,
) -> tuple[Scenario, list[Persona], str]:
    """Generate one scenario/persona setup for a paired experiment replicate.

    Config variants in the sweep affect dialogue behavior only. Reusing this
    setup prevents stochastic LLM setup differences from being mistaken for a
    configuration effect and also avoids repeating the two setup calls for
    every candidate.
    """
    client = llm or get_llm_client()
    with config_overrides({("simulation", "num_participants"): int(participants)}):
        client.reset_session()
        builder = SetupBuilder(
            topic,
            force_auto_scenario=True,
            llm=client,
            rng=random.Random(seed),
        )
        scenario, personas = builder.build(participants)
    return scenario, personas, setup_fingerprint(scenario, personas)


def run_dialogue(
    topic: str,
    *,
    participants: int | None = None,
    seed: int | None = None,
    llm: Any = None,
    log_dir: str | None = None,
    scenario: Scenario | None = None,
    personas: list[Persona] | None = None,
) -> dict[str, Any]:
    """Run one automatic-scenario dialogue and return one flat result row.

    Setup or protocol failures become rows with ``outcome='error'`` so batch
    scripts continue and preserve partial progress.
    """
    overrides: dict[tuple[str, str], Any] = {}
    if participants is not None:
        overrides[("simulation", "num_participants")] = int(participants)
    if log_dir is not None:
        overrides[("output", "log_dir")] = log_dir
    row: dict[str, Any] = {
        "topic": topic,
        "participants": participants if participants is not None else cfg.participant_count(),
        "seed": seed if seed is not None else "",
    }
    try:
        if (scenario is None) != (personas is None):
            raise ValueError("scenario and personas must be supplied together")
        with config_overrides(overrides):
            runner = DialogueRunner(
                topic,
                force_auto_scenario=scenario is None,
                scenario=copy.deepcopy(scenario),
                personas=copy.deepcopy(personas),
                llm=llm,
                seed=seed,
            )
            result = runner.run()
        row.update(flat_metrics_for(result.state, result.outcome))
        row["log_dir"] = result.log_paths["dir"]
        row["error"] = ""
    except Exception as exc:  # keep batches alive across individual failures
        row.update({"outcome": "error", "log_dir": "", "error": f"{type(exc).__name__}: {exc}"})
    return row
