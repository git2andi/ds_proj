import os

from configs.template import PersonaManager
from modules.generator import MultiUserSimulator
from modules.orchestrator import Orchestrator
from modules.role_planner import RolePlanner

MODERATOR_STYLES = ("active", "minimal", "passive")


def _parse_participant_input(raw: str) -> tuple[str, str | None]:
    """
    Returns (display_name, source_path_or_None).

    If the user typed a path to an existing .json file, use it as the source.
    Otherwise treat the input as a plain name and create a fresh persona.
    """
    stripped = raw.strip()
    if stripped.endswith(".json") and os.path.isfile(stripped):
        name = os.path.splitext(os.path.basename(stripped))[0].capitalize()
        return name, stripped
    return stripped, None


def _ask_moderator_style() -> str:
    print("\n  Moderator style:")
    print("    active  — moderator narrows and confirms when needed (default)")
    print("    minimal — moderator only steps in if the group is genuinely stuck")
    print("    passive — moderator stays silent after the opening; group self-organises")
    raw = input("  Choose style [active/minimal/passive, default=active]: ").strip().lower()
    return raw if raw in MODERATOR_STYLES else "active"


def run_project() -> None:
    setting = input("Enter the dialogue setting: ").strip()
    num_sims = int(input("Enter number of participants: ").strip())

    print(
        "Tip: enter a name to create a fresh participant, or a path to an existing .json file to reuse one.\n"
    )

    raw_inputs: list[tuple[str, str | None]] = []
    for i in range(num_sims):
        raw = input(f"  Participant {i + 1}: ").strip()
        raw_inputs.append(_parse_participant_input(raw))

    names = [name for name, _ in raw_inputs]

    moderator_style = _ask_moderator_style()

    orch = Orchestrator(setting, moderator_style=moderator_style)
    dialogue_id = orch.dialogue_id

    pm = PersonaManager(dialogue_id)
    role_planner = RolePlanner()

    generated_roles = role_planner.plan_roles(setting, names)
    role_plan = pm.assign_roles(names, generated_roles=generated_roles)

    for name, source_path in raw_inputs:
        role_info = role_plan[name]

        if source_path is not None:
            print(f"  Loading {name} from {source_path}")
            persona = pm.load_from_path(source_path, name)
        else:
            persona = pm.create_fresh(name)

        persona = pm.apply_role(
            persona,
            role=role_info["role"],
            is_primary=role_info["is_primary"],
        )

        sim = MultiUserSimulator(
            persona=persona,
            setting=setting,
            options=orch.options,
            persona_manager=pm,
        )

        orch.add_sim(sim)

        primary_tag = " [PRIMARY]" if persona.get("is_primary") else ""
        source_tag = f" (from {source_path})" if source_path else ""
        print(
            f"   {name}{primary_tag}{source_tag} | "
            f"role: {persona.get('role')} | goal: {sim.goal}"
        )

    orch.run_simulation(max_turns=30)


if __name__ == "__main__":
    run_project()
