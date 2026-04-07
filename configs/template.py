import os
import json
import random


class PersonaManager:
    def __init__(self):
        self.config_dir = "configs"
        os.makedirs(self.config_dir, exist_ok=True)

        self.trait_pool = {
            "style": ["polite", "friendly", "grumpy", "enthusiastic", "serious"],
            "participation": ["active", "reserved", "reactive", "opinionated"],
            "focus": ["saving money", "safety first", "quickest route", "comfort", "fewest layovers"],
            "length": ["very short", "short", "medium"],
            "speech_mode": ["chatty", "natural", "direct", "casual"],
            "agreeableness": [0.2, 0.4, 0.6, 0.8],
            "flexibility": [0.2, 0.4, 0.6, 0.8],
            "initiative": [0.2, 0.4, 0.6, 0.8]
        }

    def get_or_create_persona(self, name):
        file_path = os.path.join(self.config_dir, f"{name.lower()}.json")

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        traits = {
            "name": name,
            "style": random.choice(self.trait_pool["style"]),
            "participation": random.choice(self.trait_pool["participation"]),
            "focus": random.choice(self.trait_pool["focus"]),
            "length": random.choice(self.trait_pool["length"]),
            "agreeableness": random.choice(self.trait_pool["agreeableness"]),
            "flexibility": random.choice(self.trait_pool["flexibility"]),
            "initiative": random.choice(self.trait_pool["initiative"]),
            "speech_mode": random.choice(self.trait_pool["speech_mode"]),
            "goal": None
        }

        traits["behavior"] = (
            f"You are {traits['name']}. "
            f"Your style is {traits['style']}. "
            f"In groups, you are {traits['participation']}. "
            f"Your main priority is {traits['focus']}. "
            f"Keep responses {traits['length']}. "
            f"Your speaking style is {traits['speech_mode']}. "
            f"You can compromise depending on your flexibility ({traits['flexibility']}) "
            f"and agreeableness ({traits['agreeableness']})."
        )

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(traits, f, indent=4)

        return traits

    def save_persona(self, persona):
        file_path = os.path.join(self.config_dir, f"{persona['name'].lower()}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(persona, f, indent=4)