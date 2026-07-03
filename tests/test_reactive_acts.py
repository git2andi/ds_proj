"""I9: adjacency-pair-driven act selection. No LLM calls."""

from __future__ import annotations

import random

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


def _persona(pid: str, pref: str) -> Persona:
    return Persona(
        id=pid,
        name=pid.upper(),
        traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=SimulatorParameters(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        background="b",
        private_goal="g",
        preferred_options=[pref],
    )


def _world(prefs=("A", "B", "C")):
    options = [OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    personas = [_persona(f"p{i+1}", p) for i, p in enumerate(prefs)]
    state = initialise_state(scenario, personas)
    return state, personas


def _turn(state, pid: str, text: str, intent: MoveIntent | None = None, **act_kwargs) -> TurnRecord:
    state.turn_index += 1
    act = DialogueAct(speaker_id=pid, text=text, act_type=(intent.act if intent else ActType.BUILD), **act_kwargs)
    rec = TurnRecord(index=state.turn_index, speaker_id=pid, speaker_name=pid.upper(),
                     text=text, phase=Phase.DISCUSSION, act=act, intent=intent)
    state.turns.append(rec)
    return rec


def _runner() -> DialogueRunner:
    runner = DialogueRunner.__new__(DialogueRunner)
    runner._resolver = OptionResolver([OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]])
    return runner


def test_challenge_routes_defense_to_advocate():
    random.seed(1)
    state, personas = _world()
    intent = MoveIntent(speaker_id="p3", act=ActType.CHALLENGE, reason="r", option_focus=["A"])
    _turn(state, "p3", "Option A Name seems risky, that worries me.", intent,
          option_refs=["A"], soft_rejects={"A": "risk"})
    runner = _runner()
    seen = [runner._reactive_intent(state) for _ in range(50)]
    hits = [i for i in seen if i is not None and i.act == ActType.BUILD]
    assert hits, "defense should fire regularly"
    assert all(i.speaker_id == "p1" and i.option_focus == ["A"] for i in hits)  # p1 backs A


def test_answer_gets_follow_up():
    random.seed(2)
    state, personas = _world()
    intent = MoveIntent(speaker_id="p2", act=ActType.ANSWER, reason="r", option_focus=["B"])
    answer = _turn(state, "p2", "It costs about the same.", intent, option_refs=["B"])
    runner = _runner()
    hits = [i for i in (runner._reactive_intent(state) for _ in range(80)) if i is not None]
    follow = [i for i in hits if i.respond_to_turn == answer.index]
    assert follow, "follow-ups to answers should fire"
    assert all(i.act in {ActType.AGREE, ActType.CHALLENGE, ActType.ASK} for i in follow)
    assert all(i.speaker_id != "p2" for i in follow)


def test_blocker_on_leading_option_is_probed_once():
    random.seed(3)
    state, personas = _world()
    _turn(state, "p2", "plain line.")
    state.runtimes["p1"].explicit_vote = "B"
    state.runtimes["p2"].accepted_options.add("B")
    state.runtimes["p3"].hard_rejections["B"] = "dealbreaker"
    runner = _runner()
    intent = runner._reactive_intent(state)
    assert intent is not None and intent.act == ActType.ASK
    assert intent.addressee_id == "p3" and intent.option_focus == ["B"]
    assert "B" in state.blocker_probes
    # Second probe on the same option never fires.
    followups = [runner._reactive_intent(state) for _ in range(30)]
    assert all(i is None or i.addressee_id != "p3" or i.act != ActType.ASK or i.option_focus != ["B"] for i in followups)


def test_visible_split_triggers_comparison():
    random.seed(4)
    state, personas = _world()
    _turn(state, "p1", "plain line.")
    state.runtimes["p1"].accepted_options.add("A")
    state.runtimes["p2"].accepted_options.add("B")
    runner = _runner()
    hits = [i for i in (runner._reactive_intent(state) for _ in range(60)) if i is not None]
    compares = [i for i in hits if i.act in {ActType.COMPARE, ActType.PROPOSE_COMPROMISE}]
    assert compares
    assert all(set(i.option_focus) == {"A", "B"} for i in compares)


def test_no_reactive_intent_on_plain_flow():
    random.seed(5)
    state, personas = _world()
    _turn(state, "p1", "plain statement without hooks.")
    runner = _runner()
    assert all(runner._reactive_intent(state) is None for _ in range(20))


# --- I20: bounded stagnation rescue ---


def _stagnating_state(question_in_window: bool = False):
    """n=2 world where both sides keep restating with no movement signal."""
    state, personas = _world(prefs=("A", "B"))
    lines = [
        ("p1", "Option A Name nails what the garden really stands for."),
        ("p2", "Names tied to local history last longer, so Option B Name."),
        ("p1", "Option A Name drives home the real local impact."),
        ("p2", "Focus on lasting neighborhood ties, Option B Name has that."),
        ("p1", "The message of Option A Name is what sticks with people."),
        ("p2", "Option B Name is the name people will still use in ten years." + (" Right?" if question_in_window else "")),
    ]
    for pid, text in lines:
        _turn(state, pid, text, option_refs=["A"] if pid == "p1" else ["B"])
    return state


def test_stagnating_thread_gets_one_rescue_beat():
    random.seed(6)
    state = _stagnating_state()
    runner = _runner()
    intent = runner._reactive_intent(state)
    assert intent is not None
    assert intent.act in {ActType.PROPOSE_COMPROMISE, ActType.ASK}
    assert set(intent.option_focus) == {"A", "B"}
    assert "circling" in intent.reason
    assert state.stagnation_break_done
    # once per run
    assert all(
        i is None or "circling" not in i.reason
        for i in (runner._reactive_intent(state) for _ in range(30))
    )


def test_question_in_window_counts_as_movement():
    random.seed(7)
    state = _stagnating_state(question_in_window=True)
    runner = _runner()
    hits = [runner._reactive_intent(state) for _ in range(20)]
    assert all(i is None or "circling" not in i.reason for i in hits)
    assert not state.stagnation_break_done


def test_challenge_reason_never_targets_own_pick():
    state, personas = _world()
    runner = _runner()
    reason = runner._reason_for_act(state, personas[0], ActType.CHALLENGE, ["B", "A"], None)
    assert "not your own pick" in reason and "Option B Name" in reason
    reason_self = runner._reason_for_act(state, personas[0], ActType.CHALLENGE, ["A"], None)
    assert "still holds up" in reason_self
