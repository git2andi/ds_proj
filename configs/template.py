# configs/template.py
import os, json, random

class PersonaManager:
    def __init__(self):
        self.config_dir = "configs"
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.trait_pool = {
            "style": ["polite", "friendly", "rude", "grumpy", "enthusiastic", "serious"],
            # Changed 'silent' to 'reserved' to allow for actual dialogue
            "participation": ["active", "reserved", "reactive", "opinionated"],
            "focus": ["saving money", "YOLO - luxury travel", "quickest route", "time-restricted", "safety first"],
            "length": ["short", "medium", "long"]
        }

    def get_or_create_persona(self, name):
        file_path = os.path.join(self.config_dir, f"{name.lower()}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        traits = {
            "name": name,
            "style": random.choice(self.trait_pool["style"]),
            "participation": random.choice(self.trait_pool["participation"]),
            "focus": random.choice(self.trait_pool["focus"]),
            "length": random.choice(self.trait_pool["length"])
        }
        
        traits["behavior"] = (
            f"You are {traits['name']}. Your style is {traits['style']}. "
            f"In groups, you are {traits['participation']}. Your priority is {traits['focus']}. "
            f"Keep responses {traits['length']}. Speak when the topic relates to your priority."
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(traits, f, indent=4)
        return traits