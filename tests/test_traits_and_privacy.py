from __future__ import annotations

import random

import pytest

import prompts
from models import ActionType, SimulatorParameters, TurnRecord, UserAction
from simulator import UserSimulator, switch_probability
from tests.fixtures import make_persona, make_scenario, make_state


def _bid_count(engagement: int, trials: int = 1500) -> int:
    state = make_state()
    persona = make_persona("p1", "Nora", engagement=engagement)
    state.personas[0] = persona
    state.runtimes["p1"].preferred_option = "A"
    simulator = UserSimulator(persona, random.Random(1234))
    return sum(simulator.propose(state).wants_to_speak for _ in range(trials))


def test_higher_engagement_produces_higher_bid_rate():
    assert _bid_count(5) > _bid_count(3) > _bid_count(1)


def test_verbosity_maps_to_larger_realization_budget():
    low = prompts.word_budget(ActionType.SUPPORT, 1)
    high = prompts.word_budget(ActionType.SUPPORT, 5)
    assert high[0] >= low[0]
    assert high[1] > low[1]


def test_short_natural_acts_remain_short_for_high_verbosity():
    low = prompts.word_budget(ActionType.ACKNOWLEDGE, 1)
    high = prompts.word_budget(ActionType.ACKNOWLEDGE, 5)
    assert low[0] == high[0] == 2


def test_directness_changes_prompt_instruction_only():
    state = make_state()
    low = make_persona("p1", "Nora", directness=1)
    high = make_persona("p1", "Nora", directness=5)
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    state.personas[0] = low
    low_prompt = prompts.realization_prompt(state, low, action)
    state.personas[0] = high
    high_prompt = prompts.realization_prompt(state, high, action)
    assert "tentative" in low_prompt
    assert "very clear and direct" in high_prompt


def test_directness_does_not_change_policy_action():
    state_low = make_state()
    state_high = make_state()
    low = make_persona("p1", "Nora", directness=1)
    high = make_persona("p1", "Nora", directness=5)
    state_low.personas[0] = low
    state_high.personas[0] = high
    left = UserSimulator(low, random.Random(91)).propose(state_low, liveness_forced=True)
    right = UserSimulator(high, random.Random(91)).propose(state_high, liveness_forced=True)
    assert left.act == right.act
    assert left.option_focus == right.option_focus
    assert left.stance_update == right.stance_update


def test_verbosity_does_not_change_bid_or_action_policy():
    state_low = make_state()
    state_high = make_state()
    low = make_persona("p1", "Nora", verbosity=1)
    high = make_persona("p1", "Nora", verbosity=5)
    state_low.personas[0] = low
    state_high.personas[0] = high
    left = UserSimulator(low, random.Random(7)).propose(state_low)
    right = UserSimulator(high, random.Random(7)).propose(state_high)
    assert (left.wants_to_speak, left.act, left.option_focus) == (
        right.wants_to_speak, right.act, right.option_focus
    )


def test_age_and_speech_style_do_not_change_policy():
    state_a = make_state()
    state_b = make_state()
    young = make_persona("p1", "Nora", age=22, speech_style="young casual wording")
    older = make_persona("p1", "Nora", age=65, speech_style="measured traditional wording")
    state_a.personas[0] = young
    state_b.personas[0] = older
    left = UserSimulator(young, random.Random(17)).propose(state_a, liveness_forced=True)
    right = UserSimulator(older, random.Random(17)).propose(state_b, liveness_forced=True)
    assert (left.act, left.option_focus, left.stance_update) == (
        right.act, right.option_focus, right.stance_update
    )


def test_higher_stubbornness_lowers_switch_probability():
    probabilities = [switch_probability(level, 0.9) for level in (1, 2, 3, 4)]
    assert probabilities == sorted(probabilities, reverse=True)
    assert len(set(probabilities)) == 4


def test_normal_stubbornness_cannot_be_five():
    with pytest.raises(ValueError):
        SimulatorParameters(3, 3, 3, 5).validated(hard_blocker=False)


def test_hard_blocker_uses_five_and_never_switches():
    blocker = make_persona("p1", "Nora", hard_blocker=True)
    assert blocker.sim_params.stubbornness == 5
    assert switch_probability(5, 1.0, hard_blocker=True) == 0.0


def test_public_snapshot_excludes_private_persona_fields():
    state = make_state()
    snapshot = state.public_snapshot()
    rendered = repr(snapshot)
    assert "private_goal" not in rendered
    assert "background" not in rendered
    assert "stubbornness" not in rendered
    assert "hard_blocker" not in rendered


def test_prompt_exposes_only_current_speaker_private_card():
    state = make_state()
    state.personas[0].private_goal = "NORA_SECRET"
    state.personas[1].private_goal = "BEN_SECRET"
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    prompt = prompts.realization_prompt(state, state.personas[0], action)
    assert "NORA_SECRET" in prompt
    assert "BEN_SECRET" not in prompt
    assert state.personas[1].background not in prompt


def test_visible_transcript_context_keeps_speaker_labels():
    state = make_state()
    state.turns.append(TurnRecord(1, state.phase, "p2", "Ben", "The Cafe works for me."))
    action = UserAction("p1", True, 0.5, ActionType.ACKNOWLEDGE, addressee_id="p2", reason="acknowledge")
    prompt = prompts.realization_prompt(state, state.personas[0], action)
    assert "Ben: The Cafe works for me." in prompt


def test_personal_context_becomes_public_only_when_spoken():
    state = make_state()
    private = state.personas[0].private_goal
    assert private not in repr(state.public_snapshot())
    state.turns.append(TurnRecord(1, state.phase, "p1", "Nora", f"I need this because I {private}."))
    action = UserAction("p2", True, 0.4, ActionType.ACKNOWLEDGE, addressee_id="p1", reason="acknowledge")
    prompt = prompts.realization_prompt(state, state.personas[1], action)
    assert private in prompt


def test_normal_discussion_word_budgets_have_clear_verbosity_separation():
    assert prompts.word_budget(ActionType.SUPPORT, 1) == (4, 11)
    assert prompts.word_budget(ActionType.SUPPORT, 5) == (22, 44)


def test_realization_prompt_marks_private_context_as_optional_and_nonrepetitive():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    prompt = prompts.realization_prompt(state, state.personas[0], action)
    assert "Personal background and private goal are optional" in prompt
    assert "do not repeat" in prompt.casefold()


def test_realization_prompt_includes_recent_own_language_separately():
    state = make_state()
    state.turns.extend([
        TurnRecord(1, state.phase, "p1", "Nora", "I value the quiet setting."),
        TurnRecord(2, state.phase, "p2", "Ben", "The cafe stays open later."),
        TurnRecord(3, state.phase, "p1", "Nora", "The free cost also helps."),
    ])
    action = UserAction("p1", True, 0.5, ActionType.ACKNOWLEDGE, ("A",), reason="acknowledge")
    prompt = prompts.realization_prompt(state, state.personas[0], action)
    assert "AVOID REPEATING YOUR OWN WORDING OR POINT" in prompt
    assert "I value the quiet setting." in prompt
    assert "The free cost also helps." in prompt
    assert "The cafe stays open later." not in prompt.split("AVOID REPEATING YOUR OWN WORDING OR POINT", 1)[1]


def test_issue_effect_prompt_requires_visible_structured_effect():
    from models import ActiveIssue, IssueEffect, IssueKind, IssueStatus

    state = make_state()
    state.active_issue = ActiveIssue(
        id="i001", kind=IssueKind.CONCERN, option_focus=("B",), opened_by="p1",
        addressed_to=None, summary="noise", status=IssueStatus.OPEN,
        opened_at_turn=1, last_relevant_turn=1,
    )
    action = UserAction(
        "p1", True, 0.8, ActionType.ACKNOWLEDGE, ("B",),
        issue_id="i001", issue_effect=IssueEffect.PARTIAL,
        reason="the response helped but not fully",
    )
    prompt = prompts.realization_prompt(state, state.personas[0], action)
    assert "helped but did not fully solve" in prompt
