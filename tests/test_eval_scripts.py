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


def test_participant_trait_rows_use_equal_share_baseline_and_exclude_votes(tmp_path):
    from summarize_runs import participant_trait_rows

    payload = {
        "personas": [
            {"id": "p1", "name": "A", "sim_params": {"engagement": 5, "verbosity": 5, "directness": 5, "stubbornness": 1}},
            {"id": "p2", "name": "B", "sim_params": {"engagement": 1, "verbosity": 1, "directness": 1, "stubbornness": 4}},
        ],
        "runtime": {"p1": {}, "p2": {}},
        "turns": [
            {"speaker_id": "p1", "text": "A longer generated discussion contribution", "voluntary": True, "moderator": False, "action": {"act": "support"}},
            {"speaker_id": "p1", "text": "Another generated contribution", "voluntary": True, "moderator": False, "action": {"act": "react"}},
            {"speaker_id": "p1", "text": "My final vote is A", "voluntary": False, "moderator": False, "action": {"act": "vote"}},
        ],
    }
    rows = {row["participant_id"]: row for row in participant_trait_rows(payload, tmp_path)}
    assert rows["p1"]["normalized_voluntary_share"] == 2.0
    assert rows["p2"]["normalized_voluntary_share"] == 0.0
    assert rows["p1"]["avg_words"] == 4.0


def test_invalid_option_votes_do_not_pass_protocol_metrics(tmp_path):
    from summarize_runs import extract_run_metrics

    payload = {
        "scenario": {"topic": "x", "options": [{"id": "A"}, {"id": "B"}]},
        "personas": [{"id": "p1"}, {"id": "p2"}],
        "turns": [],
        "votes": {"p1": "Z", "p2": "Z"},
        "outcome": {"status": "majority", "final_option": "Z"},
        "metrics": {},
        "protocol_errors": [],
    }
    row = extract_run_metrics(payload, tmp_path)
    assert row["valid_final_votes"] == 0
    assert row["protocol_pass"] is False


def test_judge_prompt_treats_moderator_as_interaction_not_persona():
    from judge_transcripts import judge_prompt

    prompt = judge_prompt("referee", "evaluate", "context")
    assert "moderator is not a simulated participant" in prompt.lower()
    assert "never include the moderator in persona_consistency" in prompt.lower()


def test_judge_aggregate_averages_each_dimension_once():
    from judge_transcripts import aggregate_assessments

    assessments = [
        {"judge": "a", "verdict": "ok", "scores": {"naturalness": 1, "coherence": 2, "groundedness": 3, "persona_consistency": 4, "deliberation_quality": 5}},
    ]
    row = aggregate_assessments(assessments)
    assert row["overall"] == 3.0


def test_scenario_worker_captures_console_without_polluting_batch_row(monkeypatch):
    import run_scenarios

    case = run_scenarios.ScenarioCase(4, 3, "Choose a test topic")

    def fake_run_dialogue(topic, **kwargs):
        print(f"dialogue for {topic}")
        return {
            "topic": topic,
            "participants": kwargs["participants"],
            "seed": kwargs["seed"],
            "outcome": "majority",
            "log_dir": "run-dir",
            "error": "",
        }

    monkeypatch.setattr(run_scenarios, "run_dialogue", fake_run_dialogue)
    row, console = run_scenarios.run_case_worker(
        case,
        seed=504,
        output_root="logs",
    )

    assert row["case_index"] == 4
    assert row["seed"] == 504
    assert "console_output" not in row
    assert console.strip() == "dialogue for Choose a test topic"


def test_scenario_runner_defaults_to_two_workers(monkeypatch):
    import run_scenarios

    monkeypatch.setattr("sys.argv", ["run_scenarios.py"])
    assert run_scenarios.parse_args().workers == 2
