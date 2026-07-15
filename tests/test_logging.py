from __future__ import annotations

import json
from pathlib import Path

from config_loader import cfg
from dialogue import DialogueRunner
from logger import DialogueLogger, metrics_for
from tests.fixtures import ActionRendererLLM, make_personas, make_scenario


def test_transcript_contains_compact_metrics_not_large_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.output, "log_dir", str(tmp_path))
    logger = DialogueLogger("compact")
    result = DialogueRunner(
        "", scenario=make_scenario(), personas=make_personas(("A", "B", "C")),
        llm=ActionRendererLLM(), logger=logger, seed=5,
    ).run()
    text = Path(result.log_paths["transcript"]).read_text(encoding="utf-8")
    assert "## Public option board" in text
    assert "## Run summary" in text
    assert "### Participant summary" in text
    assert "```json" not in text
    assert len(text.splitlines()) < len(result.state.turns) + 80


def test_run_json_omits_deep_debug_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.output, "log_dir", str(tmp_path))
    monkeypatch.setattr(cfg.output, "debug_metrics", False)
    result = DialogueRunner(
        "", scenario=make_scenario(), personas=make_personas(("A", "B", "C")),
        llm=ActionRendererLLM(), logger=DialogueLogger("json"), seed=6,
    ).run()
    payload = json.loads(Path(result.log_paths["json"]).read_text(encoding="utf-8"))
    assert "generation_attempts" not in payload
    assert "validation_failures" not in payload
    assert "metrics" in payload


def test_metrics_are_comparison_oriented_and_compact():
    result = DialogueRunner(
        "", scenario=make_scenario(), personas=make_personas(("A", "B", "C")),
        llm=ActionRendererLLM(), logger=type("L", (), {"write_prompt": lambda *_: "", "write_run": lambda *_args, **_kwargs: {"dir":"","transcript":"","json":"","metrics_csv":""}})(), seed=7,
    ).run()
    metrics = metrics_for(result.state, result.outcome)
    assert set(metrics) == {"turns", "generation", "questions", "issues", "stances", "compromise", "coverage", "votes", "tokens", "participants", "outcome"}
    assert set(metrics["participants"]) == {"p1", "p2", "p3"}


def test_metrics_include_compact_repair_causes_and_revote_skip():
    from models import RunOutcome
    from tests.fixtures import make_state

    state = make_state(("A", "B", "C"))
    state.validation_failures["clear_vote_is_ambiguous_or_missing"] = 2
    state.revote_skipped_no_movement = True
    outcome = RunOutcome("unresolved", None, {}, "no movement")
    metrics = metrics_for(state, outcome)
    assert metrics["generation"]["repair_causes"] == {
        "clear_vote_is_ambiguous_or_missing": 2
    }
    assert metrics["votes"]["revote_skipped"] is True


def test_failed_attempts_are_preserved_compactly_without_full_debug(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.output, "log_dir", str(tmp_path))
    monkeypatch.setattr(cfg.output, "debug_metrics", False)

    class VoteFailingLLM(ActionRendererLLM):
        def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
            if "State only one short" in prompt or (profile == "repair" and "vote" in prompt.casefold()):
                text = "Library or Cafe."
                self.prompts.append(prompt)
                self.profiles.append(profile)
                self.calls += 1
                self.last_tokens_in = max(1, len(prompt.split()))
                self.last_tokens_out = len(text.split())
                self.session_tokens_in += self.last_tokens_in
                self.session_tokens_out += self.last_tokens_out
                self.session_calls += 1
                return text
            return super().generate(prompt, profile=profile)

    result = DialogueRunner(
        "", scenario=make_scenario(), personas=make_personas(("A", "A")),
        llm=VoteFailingLLM(), logger=DialogueLogger("failed-attempts"), seed=43,
    ).run()
    payload = json.loads(Path(result.log_paths["json"]).read_text(encoding="utf-8"))
    assert "generation_attempts" not in payload
    assert payload["failed_generation_attempts"]
    assert any(row["final_status"] in {"dropped", "fallback"} for row in payload["failed_generation_attempts"])


def test_transcript_renders_shared_context_as_paragraph(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.output, "log_dir", str(tmp_path))
    logger = DialogueLogger("context-paragraph")
    result = DialogueRunner(
        "", scenario=make_scenario(), personas=make_personas(("A", "A")),
        llm=ActionRendererLLM(), logger=logger, seed=44,
    ).run()
    transcript = Path(result.log_paths["transcript"]).read_text(encoding="utf-8")
    assert "## Scenario context" in transcript
    assert "The group meets on Saturday. The budget is capped at 20 euros per person." in transcript
    assert "- Shared:" not in transcript
    payload = json.loads(Path(result.log_paths["json"]).read_text(encoding="utf-8"))
    assert isinstance(payload["scenario"]["shared_context"], str)
    assert payload["scenario"]["shared_context"].startswith("The group meets on Saturday")
