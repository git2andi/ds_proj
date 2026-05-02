"""
main.py
-------
Entry point for the dialogue simulator.

Usage:
    python main.py

All tuneable parameters (turns, participants, moderator style, etc.)
are set in config.yaml. CLI prompts only ask for what cannot be
predetermined: the topic and participant names.
"""

from __future__ import annotations

import os

from config_loader import cfg
from modules.generator import MultiUserSimulator
from modules.orchestrator import Orchestrator
from modules.persona_builder import PersonaBuilder, RolePlanner


def _parse_participant_input(raw: str) -> tuple[str, str | None]:
    """
    Returns (display_name, source_json_path_or_None).
    If the user types a path to an existing .json file, load from it.
    Otherwise treat the input as a plain name and build fresh.
    """
    stripped = raw.strip()
    if stripped.endswith(".json") and os.path.isfile(stripped):
        name = os.path.splitext(os.path.basename(stripped))[0].capitalize()
        return name, stripped
    return stripped, None


def _ask(prompt: str, default: str = "") -> str:
    raw = input(prompt).strip()
    return raw if raw else default


def run() -> None:
    topic = _ask("Enter the dialogue topic: ")
    if not topic:
        print("Topic cannot be empty.")
        return

    num_participants_default = str(cfg.simulation.num_participants)
    num_raw = _ask(f"Number of participants [{num_participants_default}]: ", num_participants_default)
    try:
        num_participants = int(num_raw)
    except ValueError:
        print("Invalid number — using default.")
        num_participants = cfg.simulation.num_participants

    print(
        "\nTip: enter a name to create a fresh participant, "
        "or a path to an existing .json file to reuse one.\n"
    )
    raw_inputs: list[tuple[str, str | None]] = []
    for i in range(num_participants):
        raw = _ask(f"  Participant {i + 1}: ")
        raw_inputs.append(_parse_participant_input(raw))

    names = [name for name, _ in raw_inputs]

    style_default = cfg.simulation.moderator_style
    moderator_style = _ask(
        f"Moderator style [active/minimal/passive, default={style_default}]: ",
        default=style_default,
    ).lower()
    if moderator_style not in {"active", "minimal", "passive"}:
        moderator_style = style_default

    # ------------------------------------------------------------------
    # Build orchestrator (generates options + opening question).
    # ------------------------------------------------------------------
    orch = Orchestrator(topic, moderator_style=moderator_style)

    # ------------------------------------------------------------------
    # Build personas: roles → concepts → traits → goals.
    # ------------------------------------------------------------------
    role_planner = RolePlanner()
    role_plan = role_planner.plan(topic, names)

    builder = PersonaBuilder(topic=topic, dialogue_id=orch.dialogue_id)

    # Separate fresh builds from file loads.
    file_inputs = {name: path for name, path in raw_inputs if path is not None}

    # Build all fresh personas together so group constraints apply.
    fresh_names = [name for name, path in raw_inputs if path is None]
    fresh_role_plan = {n: role_plan[n] for n in fresh_names}
    fresh_personas = builder.build_all(fresh_names, fresh_role_plan) if fresh_names else []
    fresh_by_name = {p.name: p for p in fresh_personas}

    print("\nParticipants:")
    for name, source_path in raw_inputs:
        role_info = role_plan[name]

        if source_path is not None:
            persona = builder.load_from_file(
                path=source_path,
                name=name,
                role=role_info["role"],
                is_primary=role_info["is_primary"],
            )
        else:
            persona = fresh_by_name[name]

        sim = MultiUserSimulator(persona=persona, topic=topic, options=orch.options)
        orch.add_sim(sim)

        primary_tag = " [PRIMARY]" if persona.get("is_primary") else ""
        source_tag = f" (from {source_path})" if source_path else ""
        print(
            f"  {name}{primary_tag}{source_tag} | "
            f"role: {persona.get('role')} | goal: {persona.goal}"
        )

    print()
    orch.run_simulation()


if __name__ == "__main__":
    run()
