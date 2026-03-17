import json
import random
from pathlib import Path

from src.ds_proj.profiles import get_default_profiles
from src.ds_proj.turn_selector import choose_next_speaker
from src.ds_proj.policy import choose_dialogue_act
from src.ds_proj.generator import generate_utterance


def get_profile_by_name(profiles, name):
    for p in profiles:
        if p["name"] == name:
            return p
    raise ValueError(f"Unknown speaker: {name}")


def run_simulation(num_turns=20, topic="which cards to check", seed=42):
    rng = random.Random(seed)
    profiles = get_default_profiles()
    history = []

    for turn_id in range(num_turns):
        speaker = choose_next_speaker(profiles, [h["speaker"] for h in history], rng=rng)
        profile = get_profile_by_name(profiles, speaker)

        act = choose_dialogue_act(profile, history, rng=rng)
        text = generate_utterance(profile, act, history, topic=topic)

        turn = {
            "turn_id": turn_id,
            "speaker": speaker,
            "role": profile["role"],
            "act": act,
            "text": text,
        }
        history.append(turn)

    return history


if __name__ == "__main__":
    dialogue = run_simulation(num_turns=25, topic="which cards to check", seed=42)

    for turn in dialogue:
        print(f"[{turn['turn_id']:02d}] {turn['speaker']} ({turn['role']}, {turn['act']}): {turn['text']}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "simulated_dialogue_001.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dialogue, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")