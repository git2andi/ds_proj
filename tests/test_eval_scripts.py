from summarize_runs import hedge_rate_per_100_words, spearman, trait_correlations


def test_hedge_rate_counts_lexical_proxy():
    assert hedge_rate_per_100_words("Maybe I think this could work") > 0
    assert hedge_rate_per_100_words("This works.") == 0


def test_spearman_detects_monotonic_relation():
    assert spearman([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0


def test_trait_correlations_expose_only_report_metrics():
    rows = [
        {"engagement": 1, "normalized_voluntary_share": 0.1, "verbosity": 1, "avg_words": 5, "stubbornness": 1, "showed_flexibility": True, "directness": 1, "inverse_hedge_rate": -4},
        {"engagement": 3, "normalized_voluntary_share": 0.3, "verbosity": 3, "avg_words": 12, "stubbornness": 2, "showed_flexibility": True, "directness": 3, "inverse_hedge_rate": -2},
        {"engagement": 5, "normalized_voluntary_share": 0.6, "verbosity": 5, "avg_words": 20, "stubbornness": 4, "showed_flexibility": False, "directness": 5, "inverse_hedge_rate": 0},
    ]
    assert set(trait_correlations(rows)) == {
        "engagement_vs_voluntary_share",
        "verbosity_vs_avg_words",
        "stubbornness_vs_flexibility",
        "directness_vs_inverse_hedge_rate",
    }
