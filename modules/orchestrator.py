import os
import datetime


class Orchestrator:
    def __init__(self, topic):
        self.topic = topic
        self.sims = []

        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/conv_{timestamp}.txt"

        self.history = self._build_initial_history(topic)

    def add_sim(self, sim):
        self.sims.append(sim)

    def _build_initial_history(self, topic):
        topic_lower = topic.lower()

        # Add concrete options for decision-oriented scenarios
        if "flight" in topic_lower or "book a flight" in topic_lower or "trip" in topic_lower:
            return [
                f"Moderator: Let's discuss: {self.topic}",
                "Moderator: Available options:",
                "Moderator: Option A - €120, 6h total, 1 stop, standard comfort.",
                "Moderator: Option B - €185, 3h total, direct, standard comfort.",
                "Moderator: Option C - €160, 4h total, 1 stop, extra comfort.",
                "Moderator: Option D - €110, 8h total, 2 stops, lowest price.",
                "Moderator: What matters most to each of you, and which option seems like the best compromise?"
            ]

        # Generic fallback
        return [
            f"Moderator: Let's discuss: {self.topic}",
            "Moderator: What matters most to each of you, and what would be a practical compromise?"
        ]

    def select_speakers(self):
        last_speaker = None
        if len(self.history) > 1 and ":" in self.history[-1]:
            last_speaker = self.history[-1].split(":", 1)[0].strip()

        scored = []
        for sim in self.sims:
            score = sim.should_speak(self.history, last_speaker=last_speaker)
            scored.append((score, sim))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return []

        best_score = scored[0][0]
        selected = []

        # Relative threshold from the best candidate
        for score, sim in scored:
            if score >= max(0.35, best_score - 0.12):
                selected.append(sim)

        # Diversity rule: include one not-yet-spoken participant if close enough
        spoken_names = {
            line.split(":", 1)[0].strip()
            for line in self.history
            if ":" in line
        }

        for score, sim in scored:
            if sim.name not in spoken_names and score >= best_score - 0.20:
                if sim not in selected:
                    selected.append(sim)
                break

        # Always guarantee at least one speaker
        if not selected:
            selected = [scored[0][1]]

        # Soft API-safety cap: dynamic selection, but not unlimited
        max_dynamic = 2 if len(self.sims) <= 3 else 3
        return selected[:max_dynamic]

    def run_simulation(self, max_turns=10):
        names = [sim.name for sim in self.sims]
        header = f"Participants: {', '.join(names)}\nTopic: {self.topic}\n" + ("=" * 40) + "\n"

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(header)
            for line in self.history:
                f.write(f"{line}\n")
            f.write("\n")

        print(f"\n--- Simulation Started ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        for turn in range(max_turns):
            selected = self.select_speakers()
            active_round = False

            for sim in selected:
                text = sim.generate_turn(self.history)

                if text and "[SILENCE]" not in text.upper():
                    entry = f"{sim.name}: {text}"
                    self.history.append(entry)
                    print(f"-> {entry}\n")

                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"{entry}\n\n")

                    active_round = True

            # If nobody contributed, stop
            if not active_round:
                print("--- Discussion concluded. ---")
                break

            # Optional moderator nudge if discussion seems stuck
            if turn >= 2:
                last_msgs = [
                    line for line in self.history[-3:]
                    if ":" in line and not line.startswith("Moderator:")
                ]
                if len(last_msgs) == 3:
                    self.history.append(
                        "Moderator: Can we narrow this down to one preferred option and one backup option?"
                    )
                    print("-> Moderator: Can we narrow this down to one preferred option and one backup option?\n")

                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write("Moderator: Can we narrow this down to one preferred option and one backup option?\n\n")