from config_loader import cfg


def test_compact_conversation_configuration():
    keys = set(cfg.conversation._raw)
    assert keys == {
        "min_voluntary_turns_per_participant",
        "soft_target_voluntary_turns_per_participant",
        "hard_max_voluntary_turns_per_participant",
        "soft_target_voluntary_turn_cap",
        "hard_max_voluntary_turn_cap",
        "thread_turn_cap",
        "stagnation_no_bid_rounds",
        "compromise_window_max_turns",
        "recent_turns_in_prompt",
        "max_consecutive_turns",
    }


def test_turn_budgets_are_ordered_and_capped():
    minimum, target, maximum = cfg.conversation_turn_budgets(7)
    assert 0 < minimum <= target <= maximum <= cfg.conversation.hard_max_voluntary_turn_cap


def test_behavior_mappings_have_five_levels():
    for mapping in (
        cfg.simulator.bid_probability_by_engagement,
        cfg.simulator.movement_probability_by_stubbornness,
        cfg.language.max_words_by_verbosity,
        cfg.language.directness_instructions,
    ):
        assert {str(key) for key in mapping._raw} == {"1", "2", "3", "4", "5"}
