from configs.template import PersonaManager
from modules.generator import MultiUserSimulator
from modules.orchestrator import Orchestrator
from modules.role_planner import RolePlanner


def run_project():
    setting = input("Enter the dialogue setting: ").strip()
    num_sims = int(input("Enter number of participants: ").strip())

    names = []
    for i in range(num_sims):
        name = input(f"  Name for participant {i + 1}: ").strip()
        names.append(name)

    pm = PersonaManager()
    role_planner = RolePlanner()

    # Generate options once here so sims and orchestrator share the same list.
    orch = Orchestrator(setting)

    generated_roles = role_planner.plan_roles(setting, names)
    role_plan = pm.assign_roles(names, generated_roles=generated_roles)

    for name in names:
        role_info = role_plan[name]

        persona = pm.get_or_create_persona(
            name=name,
            role=role_info["role"],
            is_primary=role_info["is_primary"],
        )

        sim = MultiUserSimulator(
            persona=persona,
            setting=setting,
            options=orch.options,       # pass the already-generated options
            persona_manager=pm,
        )

        orch.add_sim(sim)

        primary_tag = " [PRIMARY]" if persona.get("is_primary") else ""
        print(f"   {name}{primary_tag} | role: {persona.get('role')} | goal: {sim.goal}")

    orch.run_simulation(max_turns=15)


if __name__ == "__main__":
    run_project()
