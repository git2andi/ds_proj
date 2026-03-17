from collections import Counter
import random

from src.ds_proj.profiles import get_default_profiles
from src.ds_proj.turn_selector import choose_next_speaker

profiles = get_default_profiles()
rng = random.Random(42)

history = []
for _ in range(30):
    speaker = choose_next_speaker(profiles, history, rng=rng)
    history.append(speaker)

counts = Counter(history)

print("Generated speaker sequence:")
print(history)
print()

print("Turn counts:")
for name, count in counts.items():
    print(f"{name}: {count}")

print()
print("Turn fractions:")
total = len(history)
for name, count in counts.items():
    print(f"{name}: {count / total:.3f}")