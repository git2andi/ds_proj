import time
from configs.template import PersonaManager
from modules.generator import MultiUserSimulator
from modules.orchestrator import Orchestrator


def run_project():
    setting = input("Enter the dialogue setting: ").strip()
    num_sims = int(input("Enter number of simulators: ").strip())

    pm = PersonaManager()
    orch = Orchestrator(setting)

    for i in range(num_sims):
        name = input(f"Enter name for Simulator {i+1}: ").strip()

        persona_base = pm.get_or_create_persona(name)
        sim = MultiUserSimulator(persona_base, setting, persona_manager=pm)

        orch.add_sim(sim)
        print(f"   {name} ready with goal: {sim.goal}")

        # small pause during setup to avoid bursting API requests
        time.sleep(12)

    orch.run_simulation(max_turns=6)


if __name__ == "__main__":
    run_project()