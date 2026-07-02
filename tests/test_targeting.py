"""I8: target selection scores threads (open questions, objections, minority
voices) instead of always taking the latest turn. No LLM calls."""

from __future__ import annotations

import random
from collections import Counter

from dialogue import DialogueRunner, initialise_state
from models import (
    ActType,
    DialogueAct,
    OpenQuestion,
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


def _state(prefs=("A", "B", "C", "D")):
    options = [OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    personas = [_persona(f"p{i+1}", p) for i, p in enumerate(prefs)]
    return initialise_state(scenario, personas), personas


def _turn(state, pid: str, text: str, **act_kwargs) -> TurnRecord:
    state.turn_index += 1
    act = DialogueAct(speaker_id=pid, text=text, act_type=ActType.BUILD, **act_kwargs)
    rec = TurnRecord(index=state.turn_index, speaker_id=pid, speaker_name=pid.upper(),
                     text=text, phase=Phase.DISCUSSION, act=act)
    state.turns.append(rec)
    return rec


def _runner() -> DialogueRunner:
    runner = DialogueRunner.__new__(DialogueRunner)
    runner._resolver = OptionResolver([OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]])
    return runner


def test_open_question_outranks_latest_turn():
    random.seed(7)
    state, personas = _state()
    q = _turn(state, "p2", "How long does Option B take to set up?", option_refs=["B"])
    state.open_questions.append(OpenQuestion(turn_id=q.index, asked_by="p2", target_id="p3", text=q.text, option_focus=["B"]))
    _turn(state, "p3", "Option C is cheaper.", option_refs=["C"])
    _turn(state, "p4", "I like the vibe.")
    runner = _runner()
    picks = Counter()
    for _ in range(300):
        runner._last_target_speaker = None
        chosen = runner._choose_target_turn(state, personas[0], ActType.CHALLENGE)
        picks[chosen.index] += 1
    assert picks[q.index] > picks.most_common()[-1][1]
    assert picks[q.index] >= max(count for idx, count in picks.items() if idx != q.index)


def test_non_latest_turns_are_reachable():
    random.seed(11)
    state, personas = _state()
    for i, pid in enumerate(["p2", "p3", "p4", "p2"]):
        _turn(state, pid, f"Plain statement {i}.")
    runner = _runner()
    picks = Counter()
    for _ in range(200):
        runner._last_target_speaker = None
        picks[runner._choose_target_turn(state, personas[0], ActType.BUILD).index] += 1
    assert len(picks) >= 2  # recency is a tilt, not a lock


def test_answer_act_targets_the_pending_question():
    state, personas = _state()
    q = _turn(state, "p2", "What does Option A cost?", option_refs=["A"])
    state.open_questions.append(OpenQuestion(turn_id=q.index, asked_by="p2", target_id="p1", text=q.text, option_focus=["A"]))
    _turn(state, "p3", "Later remark.")
    runner = _runner()
    chosen = runner._choose_target_turn(state, personas[0], ActType.ANSWER)
    assert chosen.index == q.index


def test_objection_turn_scores_above_plain_statement():
    random.seed(3)
    state, personas = _state()
    objection = _turn(state, "p2", "The cost worries me on Option B.", option_refs=["B"], soft_rejects={"B": "cost"})
    _turn(state, "p3", "Plain remark.")
    runner = _runner()
    picks = Counter()
    for _ in range(300):
        runner._last_target_speaker = None
        picks[runner._choose_target_turn(state, personas[0], ActType.BUILD).index] += 1
    assert picks[objection.index] > picks[objection.index + 1]
