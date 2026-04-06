# modules/orchestrator.py
import os, datetime

class Orchestrator:
    def __init__(self, topic):
        self.topic = topic
        # Clear start for the agents to react to
        self.history = [f"Travel Agent: Hi! Let's plan your trip to Malmö. What's your budget looking like?"]
        self.sims = []
        
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/conv_{timestamp}.txt"

    def add_sim(self, sim):
        self.sims.append(sim)

    def run_simulation(self, max_turns=10):
        names = [sim.name for sim in self.sims]
        header = f"Participants: {', '.join(names)}\nTopic: {self.topic}\n" + ("="*40) + "\n"
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(header + f"{self.history[0]}\n\n")

        print(f"\n--- Simulation Started ---\n-> {self.history[0]}\n")
        
        for turn in range(max_turns):
            active_round = False
            for sim in self.sims:
                # We provide the history to each sim
                thought, text = sim.generate_turn(self.history)
                
                if text and "[SILENCE]" not in text.upper():
                    entry = f"{sim.name}: {text}"
                    self.history.append(entry)
                    print(f"[{sim.name} THOUGHT]: {thought}\n-> {entry}\n")
                    
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"THOUGHT: {thought}\n{entry}\n\n")
                    active_round = True
                else:
                    print(f"[{sim.name}]: (Silent)")

            if not active_round and turn > 0:
                print("--- Discussion concluded. ---")
                break