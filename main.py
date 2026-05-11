"""
main.py
-------
Entry point for the dialogue simulator.

Two runtime modes:
  1. Interactive — prompted topic, single dialogue.
  2. Batch       — scenario file as first argument, all topics run sequentially.

Optional flags (both modes):
  --personas PATH   Path to a JSON file specifying persona overrides.

Scenario file format (scenarios.txt):
  One topic per line. Lines starting with # are ignored.

  Example:
    Book a flight to Stockholm
    Where should we hold the team offsite?

Usage:
  python main.py                              # interactive, random personas
  python main.py scenarios.txt               # batch, random personas
  python main.py --personas my_personas.json # interactive, fixed personas
  python main.py scenarios.txt --personas my_personas.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Optional

# Make all src/ modules importable without package prefix.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config_loader import cfg
from llm_client import get_llm_client
from orchestrator import Orchestrator
from persona import Persona, PersonaBuilder, TRAITS
from simulator import Simulator


# ---------------------------------------------------------------------------
# Participant count
# ---------------------------------------------------------------------------

def _num_participants(n_override: Optional[int] = None) -> int:
    if n_override is not None:
        return n_override
    if cfg.simulation.num_participants_random:
        return random.randint(
            cfg.simulation.num_participants_min,
            cfg.simulation.num_participants_max,
        )
    return cfg.simulation.num_participants


# ---------------------------------------------------------------------------
# Scenario parsing
# ---------------------------------------------------------------------------

def _parse_scenario(line: str) -> str:
    """Return the topic, stripping any legacy '| mode' suffix."""
    if "|" in line:
        return line.rsplit("|", 1)[0].strip()
    return line.strip()


# ---------------------------------------------------------------------------
# Persona override loading
# ---------------------------------------------------------------------------

def _load_persona_overrides(path: str) -> list[dict]:
    """
    Load a list of persona override dicts from a JSON file.

    Each entry can specify any subset of:
      name, role, is_primary, goal, backstory,
      assertiveness, friendliness, talkativeness, agreeableness,
      patience, contrarian, response_length

    Unspecified fields get random values (traits) or empty strings (text).
    The number of entries determines the number of participants.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Persona override file must be a non-empty JSON array: {path}")
    return raw


def _personas_from_overrides(
    overrides: list[dict],
    topic: str,
    dialogue_id: str,
) -> list[Persona]:
    """
    Build personas from override dicts.
    Traits not specified are sampled randomly.
    Text fields (backstory, goal) not specified trigger an LLM call
    only if cfg.personas.generate_backstory / generate_goal are true.
    """
    name_pool = list(_DEFAULT_NAMES)
    random.shuffle(name_pool)
    pool_idx = 0

    names = []
    for ov in overrides:
        raw_name = ov.get("name") or ""
        if not str(raw_name).strip():
            name = name_pool[pool_idx % len(name_pool)]
            pool_idx += 1
        else:
            name = str(raw_name).strip()
        names.append(name)

    builder = PersonaBuilder(topic=topic, dialogue_id=dialogue_id)
    role_map = builder._assign_roles(names)

    personas: list[Persona] = []
    for name, ov in zip(names, overrides):
        role_info = role_map.get(name, {"role": "participant", "is_primary": False})
        role = ov.get("role", role_info["role"])
        is_primary = ov.get("is_primary", role_info["is_primary"])

        lo, hi = cfg.personas.trait_min, cfg.personas.trait_max

        def _resolve(val) -> int:
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return random.randint(int(val[0]), int(val[1]))
            return max(1, min(5, int(val)))

        traits = {
            t: _resolve(ov[t]) if t in ov else random.randint(lo, hi)
            for t in TRAITS
        }

        shell = Persona(
            name=name, role=role, is_primary=is_primary,
            goal="", backstory="", **traits
        )

        if "backstory" in ov and "goal" in ov:
            shell.backstory = str(ov["backstory"])
            shell.goal = str(ov["goal"])
        else:
            backstory, goal = builder._generate_concept(shell)
            shell.backstory = ov.get("backstory", backstory)
            shell.goal = ov.get("goal", goal)

        personas.append(shell)

    if not any(p.is_primary for p in personas) and personas:
        personas[0].is_primary = True

    return personas


# ---------------------------------------------------------------------------
# Single dialogue
# ---------------------------------------------------------------------------

def run_dialogue(
    topic: str,
    persona_overrides: Optional[list[dict]] = None,
) -> None:
    if persona_overrides is None:
        cfg_part = getattr(cfg, "participants", None)
        if cfg_part and getattr(cfg_part, "from_config", False):
            entries = list(getattr(cfg_part, "entries", []) or [])
            if entries:
                persona_overrides = entries

    moderator_style: str = cfg.simulation.moderator_style

    n_override = len(persona_overrides) if persona_overrides else None
    n = _num_participants(n_override)

    print(f"\n{'='*60}")
    print(f"Topic    : {topic}")
    print(f"Sims: {n}  |  Moderator: {moderator_style}")
    if persona_overrides:
        print(f"Personas : override from file ({len(persona_overrides)} entries)")
    print(f"{'='*60}")

    orch = Orchestrator(topic, moderator_style=moderator_style)
    builder = PersonaBuilder(topic=topic, dialogue_id=orch.dialogue_id)

    if persona_overrides:
        personas = _personas_from_overrides(
            persona_overrides, topic=topic, dialogue_id=orch.dialogue_id
        )
    else:
        names = _default_names(n)
        personas = builder.build_all(names)

    # Assign beliefs after both options (from Orchestrator) and personas are ready.
    # This is the per-agent preference model that grounds all position discipline.
    print("\nGenerating belief states...")
    builder.assign_beliefs(personas, orch.options)

    _llm = get_llm_client()
    setup_tokens_in = _llm.session_tokens_in
    setup_tokens_out = _llm.session_tokens_out
    _llm.reset_session()

    print("\nParticipants:")
    for persona in personas:
        primary_tag = " [PRIMARY]" if persona.is_primary else ""
        trait_str = (
            f"friendly={persona.friendliness} agree={persona.agreeableness} "
            f"contrarian={persona.contrarian} length={persona.response_length}"
        )
        print(f"  {persona.name}{primary_tag} | role: {persona.role} | {trait_str}")
        print(f"    goal: {persona.goal}")
        if persona.beliefs:
            b = persona.beliefs
            accept_str = f", accepts {[x for x in b.acceptable if x != b.preferred]}" if len(b.acceptable) > 1 else ""
            print(f"    beliefs: prefers {b.preferred}{accept_str} | {b.key_concern}")
    print()

    for persona in personas:
        sim = Simulator(persona=persona, topic=topic, options=orch.options)
        orch.add_sim(sim)

    orch.run_simulation(setup_tokens_in=setup_tokens_in, setup_tokens_out=setup_tokens_out)


# ---------------------------------------------------------------------------
# Name pool
# ---------------------------------------------------------------------------

_DEFAULT_NAMES = [
    "Alex", "Jordan", "Morgan", "Taylor", "Casey",
    "Riley", "Drew", "Quinn", "Avery", "Blake",
]


def _default_names(n: int) -> list[str]:
    pool = list(_DEFAULT_NAMES)
    random.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    return pool + [f"Participant{i}" for i in range(1, n - len(pool) + 1)]


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(path: str, persona_overrides: Optional[list[dict]] = None) -> None:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    scenarios = [
        _parse_scenario(line)
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    if not scenarios:
        print(f"No scenarios found in {path}.")
        return

    print(f"Batch mode: {len(scenarios)} dialogue(s) from '{path}'")
    for i, topic in enumerate(scenarios, start=1):
        print(f"\n[{i}/{len(scenarios)}]")
        try:
            run_dialogue(topic, persona_overrides=persona_overrides)
        except Exception as exc:
            print(f"!! Dialogue failed for '{topic}': {exc}")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(persona_overrides: Optional[list[dict]] = None) -> None:
    raw = input("Enter the dialogue topic: ").strip()
    if not raw:
        print("Topic cannot be empty.")
        return
    topic = _parse_scenario(raw)
    run_dialogue(topic, persona_overrides=persona_overrides)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[Optional[str], Optional[list[dict]]]:
    """Returns (scenario_file_or_None, persona_overrides_or_None)."""
    args = sys.argv[1:]
    scenario_file: Optional[str] = None
    persona_path: Optional[str] = None

    i = 0
    while i < len(args):
        if args[i] == "--personas" and i + 1 < len(args):
            persona_path = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            scenario_file = args[i]
            i += 1
        else:
            i += 1

    overrides: Optional[list[dict]] = None
    if persona_path:
        try:
            overrides = _load_persona_overrides(persona_path)
            print(f"Loaded {len(overrides)} persona override(s) from '{persona_path}'")
        except Exception as exc:
            print(f"!! Could not load persona overrides: {exc}")
            sys.exit(1)

    return scenario_file, overrides


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenario_file, persona_overrides = _parse_args()
    if scenario_file:
        run_batch(scenario_file, persona_overrides=persona_overrides)
    else:
        run_interactive(persona_overrides=persona_overrides)
