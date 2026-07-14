from __future__ import annotations

from pathlib import Path

from models import (
    ActionType,
    IssueEffect,
    StanceUpdate,
    StanceUpdateKind,
    TurnRecord,
    UserAction,
)
from tests.fixtures import make_persona, make_scenario, make_state
from validation import validate_action, validate_realization


def test_invalid_option_in_structured_action_is_blocked():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("Z",), reason="invalid")
    assert any("invalid option" in error for error in validate_action(state, state.personas[0], action))


def test_invalid_reason_source_is_blocked():
    from models import ReasonSource
    state = make_state()
    action = UserAction(
        "p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="wrong",
        reason_source=ReasonSource("A", "cost", "99 euros"),
    )
    assert any("reason source" in error for error in validate_action(state, state.personas[0], action))


def test_invented_price_is_blocked():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    result = validate_realization("I support Option A because it costs 15 euros.", state, state.personas[0], action)
    assert not result.ok
    assert any("invented concrete value" in error for error in result.errors)


def test_invented_time_is_blocked():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    result = validate_realization("I support Option A because it stays open until 23:00.", state, state.personas[0], action)
    assert not result.ok


def test_invented_distance_is_blocked():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    result = validate_realization("I support Option A because it is only 12 km away.", state, state.personas[0], action)
    assert not result.ok


def test_invented_named_feature_is_blocked():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    result = validate_realization("I support Option A because it has a gym.", state, state.personas[0], action)
    assert not result.ok
    assert any("invented option feature" in error for error in result.errors)


def test_subjective_preference_is_allowed():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.CONCERN, ("B",), reason="too expensive for me")
    result = validate_realization("Option B seems too expensive for me.", state, state.personas[0], action)
    assert result.ok


def test_grounded_numeric_paraphrase_is_allowed():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("B",), reason="cost")
    result = validate_realization("I support Option B because 8 euros is manageable for me.", state, state.personas[0], action)
    assert result.ok


def test_unrelated_direct_answer_is_blocked():
    state = make_state()
    action = UserAction("p2", True, 1.0, ActionType.ANSWER, ("B",), reason="answer the noise question")
    result = validate_realization(
        "My favorite color is blue.", state, state.personas[1], action,
        target_question="What makes Option B workable despite the noise?",
    )
    assert not result.ok
    assert "direct answer is unrelated" in result.errors


def test_short_yes_answer_is_allowed():
    state = make_state()
    action = UserAction("p2", True, 1.0, ActionType.ANSWER, ("B",), reason="answer")
    result = validate_realization("Yes, that works for me.", state, state.personas[1], action, target_question="Can you accept Option B?")
    assert result.ok


def test_ambiguous_vote_is_blocked():
    state = make_state()
    action = UserAction("p1", True, 1.0, ActionType.VOTE, ("A",), reason="vote", vote_option="A")
    result = validate_realization("I vote for Option A or Option B.", state, state.personas[0], action)
    assert not result.ok
    assert any("ambiguous" in error for error in result.errors)


def test_vote_contradicting_action_is_blocked():
    state = make_state()
    action = UserAction("p1", True, 1.0, ActionType.VOTE, ("A",), reason="vote", vote_option="A")
    result = validate_realization("I vote for Option B.", state, state.personas[0], action)
    assert not result.ok


def test_switch_vote_may_mention_old_preference_without_becoming_ambiguous():
    state = make_state()
    state.runtimes["p1"].public_preference = "A"
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("B",), reason="changed balance", vote_option="B",
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "B", previous_option_id="A"),
    )
    text = "I preferred Option A, but the discussion changed my mind, so I vote for Option B."
    assert validate_realization(text, state, state.personas[0], action).ok


def test_formal_vote_switch_accepts_previously_preferring_bridge():
    state = make_state()
    runtime = state.runtimes["p1"]
    runtime.preferred_option = "B"
    runtime.public_preference = "B"
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("A",), reason="changed balance", vote_option="A",
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "A", previous_option_id="B"),
    )
    text = "Previously preferring Option B, I now vote for Option A."
    assert validate_realization(text, state, state.personas[0], action).ok


def test_formal_vote_switch_accepts_switching_my_vote_from_old_to_new():
    state = make_state()
    runtime = state.runtimes["p1"]
    runtime.preferred_option = "B"
    runtime.public_preference = "B"
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("C",), reason="changed balance", vote_option="C",
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "C", previous_option_id="B"),
    )
    text = "I’m switching my vote from Option B to Option C because the discussion changed the balance."
    result = validate_realization(text, state, state.personas[0], action)
    assert result.ok, result.errors


def test_formal_vote_switch_accepts_changing_preferred_choice_from_old_to_new():
    state = make_state()
    runtime = state.runtimes["p1"]
    runtime.preferred_option = "B"
    runtime.public_preference = "B"
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("C",), reason="changed balance", vote_option="C",
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "C", previous_option_id="B"),
    )
    text = (
        "I am changing my preferred choice from Option B to Option C. "
        "Therefore, I formally vote for Option C."
    )
    result = validate_realization(text, state, state.personas[0], action)
    assert result.ok, result.errors


def test_formal_vote_switch_accepts_changed_mind_when_old_preference_is_public():
    state = make_state()
    runtime = state.runtimes["p1"]
    runtime.preferred_option = "B"
    runtime.public_preference = "B"
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("A",), reason="changed balance", vote_option="A",
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "A", previous_option_id="B"),
    )
    text = "I changed my mind; my vote is for Option A."
    assert validate_realization(text, state, state.personas[0], action).ok


def test_formal_vote_switch_rejects_conflicting_explicit_vote_target():
    state = make_state()
    action = UserAction("p1", True, 1.0, ActionType.VOTE, ("A",), reason="vote", vote_option="A")
    result = validate_realization(
        "I am fine with Option A, but Option B gets my vote.",
        state,
        state.personas[0],
        action,
    )
    assert not result.ok
    assert "formal vote is ambiguous or contradicts the structured vote" in result.errors


def test_ordinary_act_wording_mismatch_is_not_rejected():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    result = validate_realization("Option A seems workable for me.", state, state.personas[0], action)
    assert result.ok


def test_nonexistent_option_is_blocked():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.COMMENT, reason="comment")
    result = validate_realization("Option Z sounds best.", state, state.personas[0], action)
    assert not result.ok


def test_near_verbatim_repetition_is_blocked():
    state = make_state()
    state.turns.append(TurnRecord(1, state.phase, "p1", "Nora", "I support Option A because it fits my priorities."))
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="same")
    result = validate_realization("I support Option A because it fits my priorities.", state, state.personas[0], action)
    assert not result.ok
    assert "near-verbatim repetition" in result.errors


def test_visible_switch_language_is_required():
    state = make_state()
    action = UserAction(
        "p1", True, 0.8, ActionType.COMPROMISE, ("A", "B"), reason="switch",
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "B", previous_option_id="A"),
    )
    result = validate_realization("Option A and Option B both have merits.", state, state.personas[0], action)
    assert not result.ok
    assert "preferred-option switch is not visible" in result.errors


def test_visible_acceptance_language_is_required():
    state = make_state()
    action = UserAction(
        "p1", True, 0.8, ActionType.COMPROMISE, ("B",), reason="accept",
        stance_update=StanceUpdate(StanceUpdateKind.MAKE_ACCEPTABLE, "B"),
    )
    bad = validate_realization("Option B is interesting.", state, state.personas[0], action)
    good = validate_realization("Option B now seems workable and acceptable to me.", state, state.personas[0], action)
    assert not bad.ok
    assert good.ok


def test_hard_blocker_action_cannot_switch_or_vote_elsewhere():
    scenario = make_scenario()
    blocker = make_persona("p1", "Nora", preferred="A", hard_blocker=True)
    from dialogue import initialise_state
    state = initialise_state(scenario, [blocker, make_persona("p2", "Ben", preferred="B")])
    action = UserAction("p1", True, 1.0, ActionType.VOTE, ("B",), reason="bad", vote_option="B")
    errors = validate_action(state, blocker, action)
    assert any("hard blocker" in error for error in errors)


def test_hard_blocker_text_contradiction_is_blocked():
    scenario = make_scenario()
    blocker = make_persona("p1", "Nora", preferred="A", hard_blocker=True)
    from dialogue import initialise_state
    state = initialise_state(scenario, [blocker, make_persona("p2", "Ben", preferred="B")])
    action = UserAction("p1", True, 0.5, ActionType.CONCERN, ("B",), reason="reject alternative")
    result = validate_realization("I can accept and support Option B.", state, blocker, action)
    assert not result.ok
    assert "hard-blocker contradiction" in result.errors


def test_validator_llm_configuration_and_client_are_gone():
    root = Path(__file__).resolve().parents[1]
    config = (root / "config.yaml").read_text(encoding="utf-8")
    client = (root / "src" / "llm_client.py").read_text(encoding="utf-8")
    assert "validator:" not in config
    assert "get_llm_client(role" not in client
    assert "validator" not in client.casefold()


def test_article_a_is_not_misread_as_option_a():
    state = make_state()
    persona = state.personas[2]
    action = UserAction(
        persona.id, True, 1.0, ActionType.VOTE, ("C",),
        reason="a non-negotiable requirement", vote_option="C",
    )
    result = validate_realization(
        "I vote for Option C because it remains a non-negotiable requirement.",
        state,
        persona,
        action,
    )
    assert result.ok, result.errors


def test_public_number_from_another_option_is_not_reassigned():
    state = make_state()
    action = UserAction("p1", True, 0.5, ActionType.SUPPORT, ("A",), reason="quiet")
    result = validate_realization(
        "I support Option A because it costs 8 euros.",
        state,
        state.personas[0],
        action,
    )
    assert not result.ok
    assert any("contradicts the public card" in error for error in result.errors)


def test_public_feature_from_another_option_is_not_reassigned():
    state = make_state()
    action = UserAction("p2", True, 0.5, ActionType.SUPPORT, ("B",), reason="equipment")
    result = validate_realization(
        "I support Option B because it has specialist workstations.",
        state,
        state.personas[1],
        action,
    )
    assert not result.ok
    assert any("invented option feature" in error for error in result.errors)


def test_authorized_persona_age_may_be_revealed():
    state = make_state()
    persona = state.personas[0]
    action = UserAction("p1", True, 0.4, ActionType.COMMENT, reason="brief personal context")
    result = validate_realization(
        f"I am {persona.age}, so that routine feels familiar to me.",
        state,
        persona,
        action,
    )
    assert result.ok, result.errors


def test_non_owner_cannot_resolve_or_maintain_another_participants_concern():
    state = make_state()
    from models import ActiveIssue, IssueEffect, IssueKind, IssueStatus

    state.active_issue = ActiveIssue(
        id="i001",
        kind=IssueKind.CONCERN,
        option_focus=("B",),
        opened_by="p1",
        addressed_to=None,
        summary="background noise",
        status=IssueStatus.OPEN,
        opened_at_turn=1,
        last_relevant_turn=1,
    )
    action = UserAction(
        "p2", True, 0.8, ActionType.ACKNOWLEDGE, ("B",),
        issue_id="i001", issue_effect=IssueEffect.RESOLVE,
    )
    assert any("concern owner" in error for error in validate_action(state, state.persona("p2"), action))


def test_response_obligation_requires_the_addressee_to_submit_an_answer_action():
    state = make_state()
    state.response_obligation = "p2"
    action = UserAction("p2", True, 1.0, ActionType.SUPPORT, ("B",), reason="support")
    assert any("requires an answer" in error for error in validate_action(state, state.persona("p2"), action))


def test_ordinary_phrase_option_details_is_not_treated_as_an_option_id():
    state = make_state()
    persona = state.persona("p1")
    action = UserAction("p1", True, 0.7, ActionType.SUPPORT, ("A",), reason="quiet")
    text = "I support Option A. The public option details support that view."
    assert validate_realization(text, state, persona, action).ok


def test_vote_language_outside_formal_vote_is_rejected():
    state = make_state()
    action = UserAction("p1", True, 0.7, ActionType.OPENING, ("A",), reason="quiet")
    result = validate_realization(
        "I vote for Option A because it is quiet.", state, state.persona("p1"), action
    )
    assert not result.ok
    assert "formal vote language is not allowed outside voting" in result.errors


def test_natural_acceptance_phrases_are_visible():
    state = make_state()
    action = UserAction(
        "p1", True, 0.8, ActionType.COMPROMISE, ("B",), reason="accept",
        stance_update=StanceUpdate(StanceUpdateKind.MAKE_ACCEPTABLE, "B"),
    )
    for text in (
        "I'm okay with Option B too.",
        "I'm willing to accept Option B.",
        "Option B works for me as well.",
        "I could go with Option B.",
    ):
        result = validate_realization(text, state, state.persona("p1"), action)
        assert result.ok, (text, result.errors)


def test_natural_old_to_new_switch_is_visible():
    state = make_state()
    action = UserAction(
        "p1", True, 0.8, ActionType.COMPROMISE, ("A", "B"), reason="switch",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED, "B", previous_option_id="A"
        ),
    )
    texts = (
        "I was initially leaning toward Option A, but now I think Option B is better for us.",
        "I could lean toward Option B now, even though I initially preferred A.",
    )
    for text in texts:
        result = validate_realization(text, state, state.persona("p1"), action)
        assert result.ok, (text, result.errors)


def test_plural_option_list_counts_each_required_option():
    state = make_state()
    action = UserAction("p1", True, 0.7, ActionType.COMPARE, ("A", "C"), reason="compare")
    result = validate_realization(
        "Between Options C and A, the Library fits me better.",
        state,
        state.persona("p1"),
        action,
    )
    assert result.ok, result.errors


def test_formal_revote_may_repeat_previous_vote_wording():
    state = make_state()
    state.turns.append(TurnRecord(
        1, state.phase, "p1", "Nora", "I vote for Option A because it remains my best fit.",
        action=UserAction("p1", True, 1.0, ActionType.VOTE, ("A",), vote_option="A"),
    ))
    action = UserAction("p1", True, 1.0, ActionType.VOTE, ("A",), reason="same vote", vote_option="A")
    result = validate_realization(
        "I vote for Option A because it remains my best fit.",
        state,
        state.persona("p1"),
        action,
    )
    assert result.ok, result.errors


def test_switch_vote_explicit_from_to_bridge_is_accepted():
    state = make_state()
    state.runtimes["p1"].public_preference = "A"
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("B",), reason="changed balance", vote_option="B",
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "B", previous_option_id="A"),
    )
    result = validate_realization(
        "I vote for Option B, switching my preference from Option A to Option B.",
        state,
        state.persona("p1"),
        action,
    )
    assert result.ok, result.errors


def test_comparative_claim_is_checked_for_the_option_it_describes():
    from dialogue import DialogueRunner
    from eval.run_eval_suite import scenario_for
    from tests.fixtures import ActionRendererLLM, NullLogger, make_persona
    import random

    scenario = scenario_for("flight")
    persona = make_persona("p1", "Nora", "A")
    runner = DialogueRunner(
        "", scenario=scenario, personas=[persona], llm=ActionRendererLLM(),
        logger=NullLogger(), rng=random.Random(1), seed=1,
    )
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("B",), reason="lower price", vote_option="B",
        stance_update=StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "B", previous_option_id="A"),
    )
    bad = validate_realization(
        "I vote for Option B after switching from Option A, which had a lower price.",
        runner.state,
        persona,
        action,
    )
    assert not bad.ok
    assert "concrete comparison contradicts public values" in bad.errors


def _flight_validation_state():
    import random
    from dialogue import DialogueRunner
    from eval.run_eval_suite import EvalCase, personas_for, scenario_for
    from tests.fixtures import ActionRendererLLM, NullLogger

    scenario = scenario_for("flight")
    case = EvalCase("validation", "", ("B", "C", "A"), 109, scenario_key="flight")
    personas = personas_for(case, scenario)
    runner = DialogueRunner(
        "",
        scenario=scenario,
        personas=personas,
        llm=ActionRendererLLM(),
        logger=NullLogger(),
        rng=random.Random(1),
        seed=1,
    )
    return runner.state


def test_exact_live_pairwise_price_comparisons_are_allowed():
    from models import ReasonSource

    state = _flight_validation_state()
    action_bc = UserAction(
        "p1", True, 0.7, ActionType.COMPARE, ("B", "C"),
        reason="lower price", reason_source=ReasonSource("B", "upside", "lower price"),
    )
    action_ba = UserAction(
        "p1", True, 0.7, ActionType.COMPARE, ("B", "A"),
        reason="lower price", reason_source=ReasonSource("B", "upside", "lower price"),
    )
    assert validate_realization(
        "Option B has a lower price than Option C.", state, state.persona("p1"), action_bc
    ).ok
    assert validate_realization(
        "Option B has a lower price than Option A.", state, state.persona("p1"), action_ba
    ).ok


def test_hyphenated_public_duration_and_card_superlatives_are_allowed():
    from models import ReasonSource

    state = _flight_validation_state()
    budget = UserAction(
        "p1", True, 0.7, ActionType.SUPPORT, ("D",),
        reason="lowest price", reason_source=ReasonSource("D", "upside", "lowest price"),
    )
    direct = UserAction(
        "p3", True, 0.7, ActionType.SUPPORT, ("A",),
        reason="shortest travel time", reason_source=ReasonSource("A", "upside", "shortest travel time"),
    )
    assert validate_realization(
        "Option D has a 16-hour travel time with two stops.",
        state,
        state.persona("p1"),
        budget,
    ).ok
    assert validate_realization(
        "Option A has the shortest travel time despite its higher price.",
        state,
        state.persona("p3"),
        direct,
    ).ok


def test_pairwise_comparison_is_not_misread_as_global_superlative():
    from models import ReasonSource

    state = _flight_validation_state()
    action = UserAction(
        "p1", True, 0.7, ActionType.COMPARE, ("B", "C"),
        reason="lower price", reason_source=ReasonSource("B", "upside", "lower price"),
    )
    result = validate_realization(
        "Option B has the lowest price.", state, state.persona("p1"), action
    )
    assert not result.ok
    assert "concrete comparison contradicts public values" in result.errors


def test_discussion_switch_only_needs_new_option_and_change_language():
    state = make_state()
    action = UserAction(
        "p1", True, 0.8, ActionType.COMPROMISE, ("A", "B"), reason="switch",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED, "B", previous_option_id="A"
        ),
    )
    texts = (
        "Actually, I’m starting to lean a bit more toward Option B.",
        "I’m now leaning more toward Option B.",
        "I’ve changed my mind; Option B works better for me.",
        "Actually, I now prefer Option B.",
        "Option B has become my preferred choice.",
    )
    for text in texts:
        result = validate_realization(text, state, state.persona("p1"), action)
        assert result.ok, (text, result.errors)


def test_exact_live_vote_switch_bridge_is_allowed():
    state = make_state()
    state.runtimes["p1"].public_preference = "C"
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("A",), reason="switch", vote_option="A",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED, "A", previous_option_id="C"
        ),
    )
    result = validate_realization(
        "I am changing my preference from Option C to Option A. I vote for Option A.",
        state,
        state.persona("p1"),
        action,
    )
    assert result.ok, result.errors


def test_vote_switch_without_old_to_new_bridge_remains_blocked():
    state = make_state()
    state.runtimes["p1"].public_preference = "C"
    action = UserAction(
        "p1", True, 1.0, ActionType.VOTE, ("A",), reason="switch", vote_option="A",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED, "A", previous_option_id="C"
        ),
    )
    result = validate_realization(
        "I have decided to vote for Option A.", state, state.persona("p1"), action
    )
    assert not result.ok
    assert "vote switch lacks a visible bridge" in result.errors


def test_exact_latest_live_switch_phrases_are_visible():
    state = make_state(("B", "A", "A"))
    action = UserAction(
        "p1", True, 0.8, ActionType.COMPROMISE, ("B", "A"), reason="switch",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED, "A", previous_option_id="B"
        ),
    )
    texts = (
        "I'm willing to switch my preference to Option A.",
        "I'm ready to move from Option B to Option A.",
        "I'm changing my preference from Option B to Option A.",
        "I'm now leaning toward Option A.",
    )
    for text in texts:
        result = validate_realization(text, state, state.persona("p1"), action)
        assert result.ok, (text, result.errors)


def test_exact_latest_live_acceptance_phrases_are_visible():
    state = make_state()
    action = UserAction(
        "p1", True, 0.8, ActionType.COMPROMISE, ("B",), reason="accept",
        stance_update=StanceUpdate(StanceUpdateKind.MAKE_ACCEPTABLE, "B"),
    )
    texts = (
        "I can see why Option B's relaxed atmosphere could work well for us too.",
        "I'm happy to go with Option B.",
        "Option B would work for me.",
    )
    for text in texts:
        result = validate_realization(text, state, state.persona("p1"), action)
        assert result.ok, (text, result.errors)


def test_exact_latest_live_grounded_comparisons_are_allowed():
    from models import ReasonSource

    state = _flight_validation_state()
    compare = UserAction(
        "p1", True, 0.7, ActionType.COMPARE, ("B", "C"),
        reason="lower price", reason_source=ReasonSource("B", "price", "520 dollars"),
    )
    result = validate_realization(
        "Option B has the lower price of 520 dollars than Option C.",
        state, state.persona("p1"), compare,
    )
    assert result.ok, result.errors

    study = make_state()
    earlier = UserAction(
        "p1", True, 0.7, ActionType.CONCERN, ("C",),
        reason="earlier closing time",
        reason_source=ReasonSource("C", "concern", "earlier closing time"),
    )
    result = validate_realization(
        "Option C has an earlier closing time.", study, study.persona("p1"), earlier,
    )
    assert result.ok, result.errors


def test_pairwise_comparison_blocks_only_the_claimed_pair():
    from models import ReasonSource

    state = _flight_validation_state()
    action = UserAction(
        "p1", True, 0.7, ActionType.COMPARE, ("B", "D"),
        reason="compare prices", reason_source=ReasonSource("B", "price", "520 dollars"),
    )
    result = validate_realization(
        "Option B has a lower price than Option D.", state, state.persona("p1"), action
    )
    assert not result.ok
    assert "concrete comparison contradicts public values" in result.errors


def test_non_comparative_discourse_words_do_not_trigger_grounding():
    state = make_state()
    action = UserAction(
        "p1", True, 0.7, ActionType.SUPPORT, ("B",), reason="relaxed atmosphere"
    )
    result = validate_realization(
        "I still lean toward Option B despite potential crowding.",
        state, state.persona("p1"), action,
    )
    assert result.ok, result.errors


def _state_with_active_concern():
    from models import ActiveIssue, IssueKind, IssueStatus, ReasonSource

    state = make_state()
    state.active_issue = ActiveIssue(
        id="i001", kind=IssueKind.CONCERN, option_focus=("B",), opened_by="p1",
        addressed_to=None, summary="background noise", status=IssueStatus.OPEN,
        opened_at_turn=1, last_relevant_turn=1,
        reason_source=ReasonSource("B", "noise", "moderate"), issue_key=("B", "noise"),
    )
    return state


def test_concern_maintain_effect_must_remain_visible():
    state = _state_with_active_concern()
    action = UserAction(
        "p1", True, 0.7, ActionType.CONCERN, ("B",), reason="noise",
        issue_id="i001", issue_effect=IssueEffect.MAINTAIN,
    )
    assert validate_realization(
        "The noise concern still rules out Option B for me.", state, state.persona("p1"), action
    ).ok
    invalid = validate_realization(
        "Option B has a relaxed atmosphere.", state, state.persona("p1"), action
    )
    assert "maintained concern is not visible" in invalid.errors


def test_concern_partial_effect_must_show_help_without_full_resolution():
    state = _state_with_active_concern()
    action = UserAction(
        "p1", True, 0.7, ActionType.ACKNOWLEDGE, ("B",), reason="partly helped",
        issue_id="i001", issue_effect=IssueEffect.PARTIAL,
    )
    result = validate_realization(
        "That helps somewhat, but the noise is not fully resolved for Option B.",
        state, state.persona("p1"), action,
    )
    assert result.ok, result.errors


def test_concern_resolve_effect_must_be_visible():
    state = _state_with_active_concern()
    action = UserAction(
        "p1", True, 0.7, ActionType.COMPROMISE, ("B",), reason="addressed",
        issue_id="i001", issue_effect=IssueEffect.RESOLVE,
    )
    assert validate_realization(
        "That addresses the noise concern enough; Option B is workable for me.",
        state, state.persona("p1"), action,
    ).ok
