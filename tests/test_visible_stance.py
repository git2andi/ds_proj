"""I4: public stance and latent lean move only on visible parsed evidence.
No LLM calls."""

from __future__ import annotations

from dialogue import DialogueRunner, initialise_state
from models import (
    ActType,
    DialogueAct,
    MoveIntent,
    OptionCard,
    Persona,
    Phase,
    Scenario,
    SimulatorParameters,
    TraitProfile,
    TurnRecord,
)
from parsing import OptionResolver


def _params(compromise_threshold: float = 0.5) -> SimulatorParameters:
    return SimulatorParameters(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, compromise_threshold)


def _persona(pid: str, pref: str, rejection: str | None = None, ct: float = 0.5) -> Persona:
    return Persona(
        id=pid,
        name=pid.upper(),
        traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=_params(ct),
        background="b",
        private_goal="g",
        preferred_options=[pref],
        rejection=rejection,
    )


def _world(prefs=("B", "A", "C"), rejection: str | None = None):
    options = [
        OptionCard(id=x, name=n)
        for x, n in [("A", "Sunny Side Cafe"), ("B", "Green Garden Bistro"), ("C", "Retro Diner"), ("D", "Riverside Patio")]
    ]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    personas = [_persona(f"p{i+1}", pref, rejection if i == 0 else None) for i, pref in enumerate(prefs)]
    state = initialise_state(scenario, personas)
    return state, personas, OptionResolver(options)


def _runner(resolver) -> DialogueRunner:
    runner = DialogueRunner.__new__(DialogueRunner)
    runner._resolver = resolver
    return runner


def _record(state, speaker_id: str, act: DialogueAct, intent: MoveIntent | None = None) -> TurnRecord:
    state.turn_index += 1
    rec = TurnRecord(
        index=state.turn_index,
        speaker_id=speaker_id,
        speaker_name=speaker_id.upper(),
        text=act.text,
        phase=Phase.DISCUSSION,
        act=act,
        intent=intent,
    )
    state.turns.append(rec)
    return rec


def test_routing_intent_alone_does_not_move_lean():
    state, personas, resolver = _world()
    runner = _runner(resolver)
    intent = MoveIntent(speaker_id="p1", act=ActType.AGREE, reason="r", option_focus=["A"])
    act = DialogueAct(speaker_id="p1", text="Fair point about the light.", act_type=ActType.AGREE)
    runner._apply_semantics(state, _record(state, "p1", act, intent))
    assert state.runtimes["p1"].current_preference == "B"  # unchanged


def test_visible_compromise_offer_may_move_lean():
    state, personas, resolver = _world()
    personas[0].sim_params.compromise_threshold = 0.0  # dice always pass
    runner = _runner(resolver)
    act = DialogueAct(
        speaker_id="p1",
        text="Could we all live with Sunny Side Cafe?",
        act_type=ActType.PROPOSE_COMPROMISE,
        option_refs=["A"],
        offers_compromise="A",
    )
    runner._apply_semantics(state, _record(state, "p1", act))
    assert state.runtimes["p1"].current_preference == "A"


def test_vote_for_actively_blocked_option_is_not_applied():
    state, personas, resolver = _world()
    rt = state.runtimes["p1"]
    rt.hard_rejections["A"] = "dealbreaker: no vegetarian options"
    runner = _runner(resolver)
    act = DialogueAct(
        speaker_id="p1",
        text="My pick is Sunny Side Cafe.",
        act_type=ActType.VOTE,
        option_refs=["A"],
        explicit_vote="A",
    )
    runner._apply_semantics(state, _record(state, "p1", act))
    assert rt.explicit_vote is None


def test_resolution_plus_acceptance_clears_blocker_and_votes():
    state, personas, resolver = _world()
    rt = state.runtimes["p1"]
    rt.hard_rejections["A"] = "dealbreaker"
    runner = _runner(resolver)
    act = DialogueAct(
        speaker_id="p1",
        text="That fixes my concern; I can live with Sunny Side Cafe.",
        act_type=ActType.ACCEPT,
        option_refs=["A"],
        explicit_vote="A",
        accepts=["A"],
        resolves_blocker="A",
    )
    runner._apply_semantics(state, _record(state, "p1", act))
    assert "A" not in rt.hard_rejections
    assert rt.explicit_vote == "A"


def test_validation_flags_blocked_option_acceptance():
    state, personas, resolver = _world()
    state.runtimes["p1"].hard_rejections["A"] = "dealbreaker"
    runner = _runner(resolver)
    intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", option_focus=["A"])
    act = DialogueAct(speaker_id="p1", text="My pick is Sunny Side Cafe.", act_type=ActType.VOTE, option_refs=["A"], explicit_vote="A")
    report = runner._validate_turn_text(act.text, state, personas[0], intent, act)
    assert "BLOCKED_OPTION_ACCEPTED" in report.issues and report.block_state_mutation


def test_validation_allows_same_line_resolution():
    state, personas, resolver = _world()
    state.runtimes["p1"].hard_rejections["A"] = "dealbreaker"
    runner = _runner(resolver)
    intent = MoveIntent(speaker_id="p1", act=ActType.ACCEPT, reason="r", option_focus=["A"])
    act = DialogueAct(
        speaker_id="p1", text="That fixes my concern; I can live with Sunny Side Cafe.",
        act_type=ActType.ACCEPT, option_refs=["A"], explicit_vote="A", accepts=["A"], resolves_blocker="A",
    )
    report = runner._validate_turn_text(act.text, state, personas[0], intent, act)
    assert "BLOCKED_OPTION_ACCEPTED" not in report.issues


def test_off_target_switch_is_blocked():
    state, personas, resolver = _world(prefs=("B", "A", "C"))
    runner = _runner(resolver)
    # Sanctioned move: accept winner D or restate own pick B — voting C is off-target.
    intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", option_focus=["D", "B"], allow_vote_change=True)
    act = DialogueAct(speaker_id="p1", text="I'd go with Retro Diner.", act_type=ActType.VOTE, option_refs=["C"], explicit_vote="C")
    report = runner._validate_turn_text(act.text, state, personas[0], intent, act)
    assert "OFF_TARGET_SWITCH" in report.issues and report.block_state_mutation


def test_switch_back_to_initial_preference_is_allowed():
    state, personas, resolver = _world(prefs=("B", "A", "C"))
    state.runtimes["p1"].explicit_vote = "C"  # earlier vote drifted
    runner = _runner(resolver)
    intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", option_focus=["D"], allow_vote_change=True)
    act = DialogueAct(speaker_id="p1", text="I'd go with Green Garden Bistro.", act_type=ActType.VOTE, option_refs=["B"], explicit_vote="B")
    report = runner._validate_turn_text(act.text, state, personas[0], intent, act)
    assert "OFF_TARGET_SWITCH" not in report.issues


def test_compromise_requires_visible_support():
    state, personas, resolver = _world(prefs=("B", "A", "C"))
    personas[0].sim_params.compromise_threshold = 0.0  # maximally willing
    runner = _runner(resolver)
    # Nobody has visibly voted/accepted/proposed A: no compromise pressure.
    assert runner._should_compromise_to_candidate(state, personas[0], "A") is False
    # One visible vote for A from someone else: compromise becomes possible.
    state.runtimes["p2"].explicit_vote = "A"
    assert any(runner._should_compromise_to_candidate(state, personas[0], "A") for _ in range(50))


def test_can_shift_to_respects_runtime_blockers():
    state, personas, resolver = _world()
    runner = _runner(resolver)
    assert runner._can_shift_to(state, personas[0], "A") is True
    state.runtimes["p1"].hard_rejections["A"] = "dealbreaker"
    assert runner._can_shift_to(state, personas[0], "A") is False


def test_switch_event_recorded_with_reason():
    state, personas, resolver = _world()
    runner = _runner(resolver)
    act = DialogueAct(
        speaker_id="p1",
        text="I'd switch to Sunny Side Cafe because it solves the timing issue.",
        act_type=ActType.VOTE,
        option_refs=["A"],
        explicit_vote="A",
    )
    runner._apply_semantics(state, _record(state, "p1", act))
    events = state.runtimes["p1"].switch_events
    assert events and events[0]["from"] == "B" and events[0]["to"] == "A" and events[0]["has_reason"] is True
