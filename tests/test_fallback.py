"""I1: still-blocking LLM text must be replaced by a deterministic safe fallback
before it reaches the transcript. No LLM calls."""

from __future__ import annotations

from dialogue import DialogueRunner, ValidationReport, initialise_state
from models import (
    ActType,
    MoveIntent,
    OptionCard,
    Persona,
    Scenario,
    SimulatorParameters,
    TraitProfile,
)
from parsing import OptionResolver, visible_commitment


def _params() -> SimulatorParameters:
    return SimulatorParameters(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)


def _persona(pid: str, pref: str, rejection: str | None = None) -> Persona:
    return Persona(
        id=pid,
        name=pid.upper(),
        traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=_params(),
        background="b",
        private_goal="g",
        preferred_options=[pref],
        rejection=rejection,
        rejection_reason="grounded reason" if rejection else "",
    )


def _world(rejection: str | None = None):
    options = [
        OptionCard(id="A", name="Sunny Side Cafe", attrs={"cost": "$20"}),
        OptionCard(id="B", name="Green Garden Bistro", attrs={"cost": "$25"}),
        OptionCard(id="C", name="Retro Diner", attrs={"cost": "$15"}),
        OptionCard(id="D", name="Riverside Patio", attrs={"cost": "$22"}),
    ]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    personas = [_persona("p1", "B", rejection), _persona("p2", "A"), _persona("p3", "C")]
    state = initialise_state(scenario, personas)
    return state, personas[0], OptionResolver(options)


def test_blocker_fallback_never_accepts_rejected_option():
    state, persona, resolver = _world(rejection="A")
    intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", option_focus=["A", "B"], allow_vote_change=True)
    report = ValidationReport(["HARD_BLOCKER_ACCEPTED_REJECTED_OPTION"], True)
    text = DialogueRunner._safe_fallback_text(state, persona, intent, report)
    commitment = visible_commitment(text, resolver)
    assert commitment is not None, f"fallback must parse as a commitment: {text}"
    stance, option_id = commitment
    assert stance == "vote"
    assert option_id != "A"
    assert option_id == "B"  # the blocker's own current preference


def test_unclear_vote_fallback_commits_clearly():
    state, persona, resolver = _world()
    intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", option_focus=["B"])
    report = ValidationReport(["UNCLEAR_VISIBLE_COMMITMENT"], True)
    text = DialogueRunner._safe_fallback_text(state, persona, intent, report)
    commitment = visible_commitment(text, resolver)
    assert commitment == ("vote", "B")


def test_fallback_rotates_commitment_family():
    state, persona, resolver = _world()
    intent = MoveIntent(
        speaker_id="p1",
        act=ActType.VOTE,
        reason="r",
        option_focus=["B"],
        avoid_phrases=["gets my vote"],
    )
    report = ValidationReport(["UNCLEAR_VISIBLE_COMMITMENT"], True)
    text = DialogueRunner._safe_fallback_text(state, persona, intent, report)
    assert "gets my vote" not in text.lower()
    assert visible_commitment(text, resolver) == ("vote", "B")


def test_coverage_fallback_mentions_required_option():
    state, persona, resolver = _world()
    intent = MoveIntent(speaker_id="p1", act=ActType.COMPARE, reason="not yet been socially processed", option_focus=["D"])
    report = ValidationReport(["MISSING_REQUIRED_OPTION_FOCUS"], True)
    text = DialogueRunner._safe_fallback_text(state, persona, intent, report)
    assert "D" in resolver.ids_in_text(text)
    assert visible_commitment(text, resolver) is None  # a coverage nudge is not a vote


def test_discussion_fallback_has_no_commitment():
    state, persona, resolver = _world()
    intent = MoveIntent(speaker_id="p1", act=ActType.BUILD, reason="r")
    report = ValidationReport(["INVALID_OPTION_REFERENCE"], True)
    text = DialogueRunner._safe_fallback_text(state, persona, intent, report)
    assert visible_commitment(text, resolver) is None


class _FakeLLM:
    """Always returns the same bad line, both for generation and repair."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.last_tokens_in = 0
        self.last_tokens_out = 0

    def generate(self, prompt: str, profile: str = "dialogue") -> str:
        return self._text


class _FakeLogger:
    def write_prompt(self, prompt: str, kind: str) -> str:
        return ""


def test_generate_and_append_replaces_blocking_text():
    """End-to-end (no LLM): a hard blocker whose generated vote accepts the
    rejected option must end up with a printed line that does NOT accept it,
    and with state moved to a valid alternative."""
    state, persona, resolver = _world(rejection="A")
    runner = DialogueRunner.__new__(DialogueRunner)
    runner._llm = _FakeLLM("Count me in for Sunny Side Cafe.")
    runner._resolver = resolver
    runner.logger = _FakeLogger()

    intent = MoveIntent(
        speaker_id="p1",
        act=ActType.VOTE,
        reason="r",
        option_focus=["A", "B"],
        allow_vote_change=True,
    )
    record = runner._generate_and_append(state, intent)

    assert record.used_fallback is True
    assert state.fallback_turn_count == 1
    assert state.invalid_printed_turn_count == 0
    committed = visible_commitment(record.text, resolver)
    assert committed is not None and committed[1] != "A"
    assert state.runtimes["p1"].explicit_vote == "B"
    assert record.state_mutation_blocked is False
