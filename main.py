# main.py
import time
from configs.template import PersonaManager
from modules.generator import MultiUserSimulator
from modules.orchestrator import Orchestrator

def run_project():
    setting = input("Enter the dialogue setting: ")
    pm = PersonaManager()
    orch = Orchestrator(setting)

    for i in range(2):
        name = input(f"Enter name for Simulator {i+1}: ")
        persona_base = pm.get_or_create_persona(name)
        sim = MultiUserSimulator(persona_base, setting)
        orch.add_sim(sim)
        print(f"   {name} ready with goal: {sim.goal}")
        time.sleep(1) # Small pause to prevent API spam during setup

    orch.run_simulation(max_turns=10)

if __name__ == "__main__":
    run_project()