from src.ds_proj.profiles import get_default_profiles
from src.ds_proj.generator import generate_utterance

profiles = get_default_profiles()
profile = profiles[0]

history = [
    {"speaker": "Speaker_B", "act": "propose"},
]

for act in ["propose", "agree", "disagree", "ask"]:
    text = generate_utterance(profile, act, history, topic="which cards to check")
    print(f"{act}: {text}")