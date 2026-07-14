from __future__ import annotations

import json

from logger import DialogueLogger, metrics_for
from tests.fixtures import ActionRendererLLM, make_runner


def test_structured_log_exposes_candidate_and_selected_action(monkeypatch, tmp_path):
    from config_loader import cfg

    monkeypatch.setattr(cfg.output, "log_dir", str(tmp_path.relative_to(cfg.root)) if tmp_path.is_relative_to(cfg.root) else str(tmp_path))
    runner = make_runner(llm=ActionRendererLLM())
    action = runner._simulators["p1"].opening_action(runner.state)
    runner._realize_and_commit(action, mandatory=True, voluntary=False)
    outcome = __import__("models").RunOutcome("majority", "A", {"p1": "A"}, "test")
    paths = DialogueLogger("logging test").write_run(runner.state, outcome, seed=1)
    payload = json.loads(__import__("pathlib").Path(paths["json"]).read_text(encoding="utf-8"))
    participant = next(turn for turn in payload["turns"] if turn["speaker_id"] == "p1")
    assert participant["candidate_action"] == participant["selected_action"] == participant["action"]


def test_metrics_report_voluntary_word_lengths_and_direct_traits():
    runner = make_runner()
    action = runner._simulators["p1"].propose(runner.state, liveness_forced=True)
    runner._realize_and_commit(action, mandatory=False, voluntary=True)
    outcome = __import__("models").RunOutcome("majority", "A", {}, "test")
    metrics = metrics_for(runner.state, outcome)
    assert "average_voluntary_words_by_id" in metrics["turns"]
    assert metrics["traits"]["p1"]["engagement"] == runner.state.persona("p1").sim_params.engagement


def test_verbosity_metric_uses_comparable_voluntary_actions_only():
    from models import ActionType, RunOutcome, UserAction

    runner = make_runner()
    runner._commit_action(
        UserAction("p1", True, 0.5, ActionType.ACKNOWLEDGE, ("A",), reason="ack"),
        "Yes.", mandatory=False, voluntary=True, liveness_forced=False, repair_count=0,
    )
    runner._commit_action(
        UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet"),
        "Option A is quiet and predictable for focused work.",
        mandatory=False, voluntary=True, liveness_forced=False, repair_count=0,
    )
    metrics = metrics_for(runner.state, RunOutcome("majority", "A", {}, "test"))
    assert metrics["turns"]["comparable_voluntary_turns_by_id"]["p1"] == 1
    assert metrics["turns"]["average_comparable_voluntary_words_by_id"]["p1"] == 9.0


def test_trait_metrics_report_switch_opportunities_and_hard_blocker_violations():
    from models import RunOutcome

    runner = make_runner()
    runtime = runner.state.runtimes["p1"]
    runtime.switch_opportunities = 4
    runtime.visible_switches = 1
    metrics = metrics_for(runner.state, RunOutcome("majority", "A", {}, "test"))
    assert metrics["traits"]["p1"]["switch_opportunities"] == 4
    assert metrics["traits"]["p1"]["switch_rate_per_opportunity"] == 0.25
    assert metrics["traits"]["p1"]["hard_blocker_nonpreferred_acceptances"] == 0
    assert metrics["traits"]["p1"]["hard_blocker_nonpreferred_votes"] == 0


def test_word_budget_diagnostics_log_intended_and_realized_counts():
    from models import RunOutcome

    runner = make_runner()
    action = runner._simulators["p1"].propose(runner.state, liveness_forced=True)
    runner._realize_and_commit(action, mandatory=False, voluntary=True)
    metrics = metrics_for(runner.state, RunOutcome("majority", "A", {}, "test"))
    diagnostic = metrics["realization"]["word_budget_by_id"]["p1"][0]
    assert diagnostic["intended_min"] > 0
    assert diagnostic["intended_max"] >= diagnostic["intended_min"]
    assert diagnostic["realized"] > 0


def test_transcript_keeps_full_diagnostics_in_run_json_only(monkeypatch, tmp_path):
    from config_loader import cfg
    from models import RunOutcome
    from pathlib import Path

    log_dir = str(tmp_path.relative_to(cfg.root)) if tmp_path.is_relative_to(cfg.root) else str(tmp_path)
    monkeypatch.setattr(cfg.output, "log_dir", log_dir)
    runner = make_runner(llm=ActionRendererLLM())
    action = runner._simulators["p1"].propose(runner.state, liveness_forced=True)
    runner._realize_and_commit(action, mandatory=False, voluntary=True)
    paths = DialogueLogger("compact transcript").write_run(
        runner.state, RunOutcome("majority", "A", {}, "test"), seed=1
    )
    transcript = Path(paths["transcript"]).read_text(encoding="utf-8")
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert '"word_budget_by_id"' not in transcript
    assert '"switch_decisions"' not in transcript
    assert "word_budget_by_id" in payload["metrics"]["realization"]
    assert "switch_decisions" in payload["metrics"]["stances"]
