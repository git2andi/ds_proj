import random


def choose_next_speaker(profiles, history, rng=None):
    """
    Select the next speaker based on:
    - target turn share
    - slight penalty if the same speaker spoke last turn

    profiles: list of dicts
    history: list of speaker names, e.g. ["Speaker_A", "Speaker_B", ...]
    """
    if rng is None:
        rng = random.Random()

    weights = []
    last_speaker = history[-1] if history else None

    for p in profiles:
        weight = p["target_turn_share"]

        # discourage immediate self-repeat a bit
        if p["name"] == last_speaker:
            weight *= 0.3

        weights.append(weight)

    chosen = rng.choices(profiles, weights=weights, k=1)[0]
    return chosen["name"]