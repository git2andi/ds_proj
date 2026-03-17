import random


def choose_dialogue_act(profile, history, rng=None):
    """
    Choose a dialogue act for the selected speaker.

    profile: dict for one speaker
    history: list of dicts, each at least containing:
             {"speaker": ..., "act": ...}
    """
    if rng is None:
        rng = random.Random()

    # First turn or empty history: someone has to start
    if not history:
        return "propose"

    acts = ["propose", "agree", "disagree", "ask"]

    # Base weights
    weights = {
        "propose": 0.20,
        "agree": 0.35,
        "disagree": 0.20,
        "ask": 0.25,
    }

    # More initiative -> more proposing
    weights["propose"] += 0.30 * profile["initiative"]

    # More agreeableness -> more agreement, less disagreement
    weights["agree"] += 0.30 * profile["agreeableness"]
    weights["disagree"] += 0.30 * (1.0 - profile["agreeableness"])

    # Normalize into ordered list
    ordered_weights = [weights[a] for a in acts]

    return rng.choices(acts, weights=ordered_weights, k=1)[0]