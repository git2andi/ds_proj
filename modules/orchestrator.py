import os
import re
import datetime
from collections import Counter

from modules.llm_client import get_llm_client


class Orchestrator:
    def __init__(self, topic):
        self.topic = topic
        self.sims = []
        self.llm = get_llm_client()
        self.options = self._generate_options(topic)
        self.has_asked_narrowing = False

        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/conv_{timestamp}.txt"

        self.history = self._build_initial_history()

    def add_sim(self, sim):
        self.sims.append(sim)

    def _generate_options(self, topic):
        prompt = f"""
You are a moderator preparing a group decision discussion.

Topic:
{topic}

Task:
Generate exactly 4 concrete decision options for this topic.

Requirements:
- Return valid JSON only.
- Use this exact schema:
{{
  "options": [
    "Option A - ...",
    "Option B - ...",
    "Option C - ...",
    "Option D - ..."
  ]
}}
- Each option must be directly relevant to the topic.
- All 4 options must use the same attribute structure.
- For each option, include exactly:
  1. a name or label,
  2. a price or cost level,
  3. a travel time / distance / convenience detail,
  4. a comfort / quality / atmosphere detail,
  5. one drawback.
- Keep each option to one concise line.
- The 4 options must be directly comparable.
- Do not include markdown.
- Do not include explanations outside the JSON.
"""

        fallback = [
            "Option A - Lower cost, simpler choice.",
            "Option B - Faster or more convenient, but more expensive.",
            "Option C - Better comfort or quality, with moderate trade-offs.",
            "Option D - Flexible alternative with mixed advantages and disadvantages.",
        ]

        try:
            data = self.llm.generate_json(prompt)
            options = data.get("options", [])

            if not isinstance(options, list) or len(options) != 4:
                return fallback

            cleaned = []
            for i, option in enumerate(options):
                if not isinstance(option, str) or not option.strip():
                    return fallback

                label = chr(ord("A") + i)
                text = option.strip()

                if not text.lower().startswith(f"option {label.lower()}"):
                    text = f"Option {label} - {text}"

                cleaned.append(text)

            return cleaned

        except Exception as e:
            print(f"!! Option generation error: {e}")
            return fallback

    def _build_initial_history(self):
        history = [f"Moderator: Let's discuss: {self.topic}"]

        if self.options:
            history.append("Moderator: Available options:")
            for option in self.options:
                history.append(f"Moderator: {option}")
            history.append(
                "Moderator: What matters most to each of you, and which option seems like the best compromise?"
            )
        else:
            history.append(
                "Moderator: What matters most to each of you, and what would be a practical compromise?"
            )

        return history

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

        for score, sim in scored:
            if score >= max(0.35, best_score - 0.12):
                selected.append(sim)

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

        if not selected:
            selected = [scored[0][1]]

        max_dynamic = 2 if len(self.sims) <= 3 else 3
        return selected[:max_dynamic]

    def _extract_option_mentions(self, text):
        return re.findall(r"\boption\s+([A-D])\b", text.lower())

    def _detect_consensus(self):
        recent_messages = []
        for line in reversed(self.history):
            if ":" not in line or line.startswith("Moderator:"):
                continue
            recent_messages.append(line)
            if len(recent_messages) >= max(6, len(self.sims) * 3):
                break

        preferred_votes = []
        backup_votes = []

        for line in recent_messages:
            _, msg = line.split(":", 1)
            msg_lower = msg.lower()
            mentions = [m.upper() for m in self._extract_option_mentions(msg)]

            if not mentions:
                continue

            # "Option B, with A as backup." -> preferred=B, backup=A
            if "backup" in msg_lower or "second choice" in msg_lower:
                preferred_votes.append(mentions[0])
                if len(mentions) >= 2:
                    backup_votes.append(mentions[1])
                continue

            # default: first mentioned option is preferred
            preferred_votes.append(mentions[0])

            # second mention can count as a soft backup candidate
            if len(mentions) >= 2:
                backup_votes.append(mentions[1])

        if len(preferred_votes) < len(self.sims):
            return None

        preferred_counts = Counter(preferred_votes)
        preferred_option, preferred_count = preferred_counts.most_common(1)[0]

        if preferred_count < max(2, len(self.sims) - 1):
            return None

        backup_option = None
        if backup_votes:
            backup_counts = Counter([b for b in backup_votes if b != preferred_option])
            if backup_counts:
                backup_option, _ = backup_counts.most_common(1)[0]

        return preferred_option, backup_option

    def run_simulation(self, max_turns=10):
        names = [sim.name for sim in self.sims]
        header = f"Participants: {', '.join(names)}\nTopic: {self.topic}\n" + ("=" * 40) + "\n"

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(header)
            for line in self.history:
                f.write(f"{line}\n")
            f.write("\n")

        print("\n--- Simulation Started ---")
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

            consensus = self._detect_consensus()
            if consensus:
                preferred_option, backup_option = consensus

                if backup_option:
                    closing = (
                        f"Moderator: It sounds like you agree on Option {preferred_option} "
                        f"as the preferred choice and Option {backup_option} as the backup. "
                        f"Discussion concluded."
                    )
                else:
                    closing = (
                        f"Moderator: It sounds like you agree on Option {preferred_option} "
                        f"as the preferred choice. Discussion concluded."
                    )

                self.history.append(closing)
                print(f"-> {closing}\n")

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{closing}\n\n")

                break

            if not active_round:
                print("--- Discussion concluded. ---")
                break

            if turn >= 2 and not self.has_asked_narrowing:
                last_msgs = [
                    line for line in self.history[-3:]
                    if ":" in line and not line.startswith("Moderator:")
                ]
                if len(last_msgs) == 3:
                    nudge = "Moderator: Can we narrow this down to one preferred option and one backup option?"
                    self.history.append(nudge)
                    print(f"-> {nudge}\n")

                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"{nudge}\n\n")

                    self.has_asked_narrowing = True