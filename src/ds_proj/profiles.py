def get_default_profiles():
    """
    First empirical baseline profiles inspired by DeliData corpus stats.

    Target setup:
    - 4 participants
    - one relatively dominant speaker
    - two medium participants
    - one quieter participant
    """

    return [
        {
            "name": "Speaker_A",
            "role": "dominant",
            "target_turn_share": 0.40,
            "talkativeness": 0.90,
            "initiative": 0.80,
            "agreeableness": 0.50,
        },
        {
            "name": "Speaker_B",
            "role": "regular",
            "target_turn_share": 0.25,
            "talkativeness": 0.65,
            "initiative": 0.55,
            "agreeableness": 0.60,
        },
        {
            "name": "Speaker_C",
            "role": "regular",
            "target_turn_share": 0.20,
            "talkativeness": 0.55,
            "initiative": 0.45,
            "agreeableness": 0.55,
        },
        {
            "name": "Speaker_D",
            "role": "quiet",
            "target_turn_share": 0.15,
            "talkativeness": 0.35,
            "initiative": 0.25,
            "agreeableness": 0.65,
        },
    ]