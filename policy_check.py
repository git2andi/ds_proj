from collections import Counter
import random

from src.ds_proj.profiles import get_default_profiles
from src.ds_proj.policy import choose_dialogue_act

profiles = get_default_profiles()
rng = random.Random(42)

# test each profile separately
for profile in profiles:
    history = [{"speaker": "Other", "act": "propose"}]

    acts = []
    for _ in range(100):
        act = choose_dialogue_act(profile, history, rng=rng)
        acts.append(act)

    counts = Counter(acts)

    print(f"Profile: {profile['name']} ({profile['role']})")
    for act, count in sorted(counts.items()):
        print(f"  {act}: {count}")
    print()