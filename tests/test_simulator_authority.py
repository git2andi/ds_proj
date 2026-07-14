from __future__ import annotations

import inspect
import random

import prompts
from dialogue import DialogueRunner
from models import ActionType, Phase, QuestionIntent, StanceUpdateKind, TurnRecord, UserAction
from simulator import FloorManager, UserSimulator
from tests.fixtures import ActionRendererLLM, make_persona, make_runner, make_state


def test_opening_action_owns_act_option_and_reason():
    state = make_state(("B", "A", "C"))
    simulator = UserSimulator(state.personas[0], random.Random(1))
    action = simulator.opening_action(state)
    assert action.act is ActionType.OPENING
    assert action.option_focus == ("B",)
    assert action.reason
    assert action.speaker_id == "p1"


def test_simulator_candidate_contains_complete_action():
    state = make_state()
    simulator = UserSimulator(state.personas[0], random.Random(2))
    action = simulator.propose(state, liveness_forced=True)
    assert action.wants_to_speak
    assert action.speaker_id == "p1"
    assert isinstance(action.act, ActionType)
    assert action.option_focus
    assert action.reason


def test_floor_returns_the_exact_bid_object():
    state = make_state()
    bid = UserAction("p1", True, 0.9, ActionType.SUPPORT, ("A",), reason="grounded reason")
    selected = FloorManager(random.Random(1)).select(state, [bid])
    assert selected is not None
    assert selected.action is bid


def test_floor_does_not_rewrite_action_fields():
    state = make_state()
    bid = UserAction(
        "p2", True, 0.8, ActionType.COMPARE, ("B", "A"),
        addressee_id="p1", reason="compare the public trade-off",
    )
    before = bid.copy()
    selected = FloorManager(random.Random(3)).select(state, [bid]).action
    assert selected == before


def test_dialogue_selection_passes_selected_action_unchanged(monkeypatch):
    runner = make_runner()
    action = UserAction("p1", True, 1.0, ActionType.SUPPORT, ("A",), reason="reason")
    captured = []

    def fake_commit(selected, **_kwargs):
        captured.append(selected)
        return object()

    monkeypatch.setattr(runner, "_realize_and_commit", fake_commit)
    assert runner._select_and_realize([action], phase=runner.state.phase)
    assert captured == [action]
    assert captured[0] is action


def test_controller_has_no_participant_action_constructor_for_concessions():
    source = inspect.getsource(DialogueRunner)
    assert "UserAction(" not in source
    assert "switch_probability" not in source
    assert "controller-selected" not in source.casefold()


def test_open_floor_bidding_makes_zero_llm_calls():
    state = make_state()
    llm = ActionRendererLLM()
    simulator = UserSimulator(state.personas[0], random.Random(4))
    for _ in range(50):
        simulator.propose(state)
    assert llm.calls == 0


def test_one_accepted_turn_normally_makes_one_realization_call():
    llm = ActionRendererLLM()
    runner = make_runner(llm=llm)
    action = runner._simulators["p1"].opening_action(runner.state)
    assert runner._realize_and_commit(action, mandatory=True, voluntary=False) is not None
    assert llm.calls == 1


def test_repair_adds_at_most_one_call():
    llm = ActionRendererLLM(scripted=["", "Hi everyone. I prefer Option A because it fits my priority."])
    runner = make_runner(llm=llm)
    action = runner._simulators["p1"].opening_action(runner.state)
    assert runner._realize_and_commit(action, mandatory=True, voluntary=False) is not None
    assert llm.calls == 2
    assert runner.state.stats.repair_calls == 1


def test_action_prompt_treats_structure_as_authoritative():
    state = make_state()
    action = UserAction("p1", True, 0.7, ActionType.SUPPORT, ("A",), reason="quiet and predictable")
    prompt = prompts.realization_prompt(state, state.personas[0], action)
    assert "AUTHORITATIVE ACTION" in prompt
    assert "Do not choose a different act or option" in prompt


def test_support_reason_is_not_repeated_without_a_challenge():
    state = make_state()
    simulator = UserSimulator(state.personas[0], random.Random(7))
    runtime = state.runtimes["p1"]
    first = simulator._support_action(state, runtime)
    assert first is not None
    runtime.stated_reason_keys.add(simulator._reason_key(first.act, first.reason_source, first.reason))
    second = simulator._support_action(state, runtime)
    assert second is None or simulator._reason_key(
        second.act, second.reason_source, second.reason
    ) != simulator._reason_key(first.act, first.reason_source, first.reason)


def test_same_question_key_is_not_reopened_but_other_objectives_remain_possible():
    state = make_state()
    simulator = UserSimulator(state.personas[0], random.Random(8))
    runtime = state.runtimes["p1"]
    action = simulator._ask_action(state, runtime)
    assert action is not None and action.question_key
    runtime.asked_question_keys.add(action.question_key)
    next_action = simulator._ask_action(state, runtime)
    assert next_action is None or next_action.question_key != action.question_key


def test_question_can_clarify_a_recent_visible_claim():
    state = make_state()
    state.turns.append(TurnRecord(
        index=1,
        phase=Phase.DISCUSSION,
        speaker_id="p2",
        speaker_name="P2",
        text="Option B fits the equipment requirement.",
        action=UserAction(
            "p2", True, 0.6, ActionType.SUPPORT, ("B",),
            reason="the equipment requirement",
        ),
        voluntary=True,
    ))
    simulator = UserSimulator(state.personas[0], random.Random(18))
    action = simulator._ask_action(state, state.runtimes["p1"])
    assert action is not None
    assert action.question_intent is QuestionIntent.CLARIFICATION
    assert action.addressee_id == "p2"
    assert action.option_focus == ("B",)


def test_support_uses_unused_public_reason_before_repeating():
    state = make_state()
    simulator = UserSimulator(state.personas[0], random.Random(12))
    runtime = state.runtimes["p1"]
    first = simulator._support_action(state, runtime)
    assert first is not None
    runtime.stated_reason_keys.add(simulator._reason_key(first.act, first.reason_source, first.reason))
    second = simulator._support_action(state, runtime)
    assert second is not None
    assert simulator._reason_key(second.act, second.reason_source, second.reason) != simulator._reason_key(
        first.act, first.reason_source, first.reason
    )


def test_support_stops_after_all_relevant_reasons_are_used():
    state = make_state()
    simulator = UserSimulator(state.personas[0], random.Random(13))
    runtime = state.runtimes["p1"]
    candidates = simulator._positive_reason_candidates(state, runtime.preferred_option)
    assert len(candidates) >= 2
    for reason, source in candidates:
        runtime.stated_reason_keys.add(simulator._reason_key(ActionType.SUPPORT, source, reason))
    assert simulator._support_action(state, runtime) is None


def test_question_targets_publicly_relevant_participant_and_has_specific_intent():
    state = make_state()
    state.runtimes["p2"].public_preference = "B"
    state.public_supporters["B"].add("p2")
    simulator = UserSimulator(state.personas[0], random.Random(14))
    action = simulator._ask_action(state, state.runtimes["p1"])
    assert action is not None
    assert action.addressee_id == "p2"
    assert action.question_intent is not None
    assert "workable or problematic" not in action.reason.casefold()


def test_duplicate_question_key_is_suppressed():
    state = make_state()
    state.runtimes["p2"].public_preference = "B"
    state.public_supporters["B"].add("p2")
    simulator = UserSimulator(state.personas[0], random.Random(15))
    runtime = state.runtimes["p1"]
    first = simulator._ask_action(state, runtime)
    assert first is not None and first.question_key
    runtime.asked_question_keys.add(first.question_key)
    second = simulator._ask_action(state, runtime)
    assert second is None or second.question_key != first.question_key
