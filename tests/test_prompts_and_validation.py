from __future__ import annotations

from models import (
    ActionType,
    BidPriority,
    IssueEffect,
    ReasonSource,
    StanceUpdate,
    StanceUpdateKind,
    TurnRecord,
    UserAction,
)
import prompts
from validation import validate_realization
from tests.fixtures import make_state


def test_prompt_is_compact_and_excludes_irrelevant_traits_and_options():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.SUPPORT,
        option_focus=("A",), reason="quiet and predictable",
        reason_source=ReasonSource("A", "upside", "quiet and predictable"),
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "Directness: 3/5" in prompt
    assert "engagement" not in prompt.casefold()
    assert "stubbornness" not in prompt.casefold()
    assert "Riverside Cafe" not in prompt
    assert "Engineering Lab" not in prompt
    assert len(prompt.split()) < 260


def test_prompt_uses_configured_word_limit():
    state = make_state(("A", "B", "C"))
    state.persona("p1").sim_params.verbosity = 1
    action = UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="quiet")
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "Maximum 10 words" in prompt


def test_prompt_includes_relevant_personal_context_only_when_selected():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.ANSWER,
        option_focus=("A",), reason="it works after my shift",
        personal_context="works an evening shift",
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "works an evening shift" in prompt
    action.personal_context = None
    prompt2 = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "works an evening shift" not in prompt2


def test_prompt_adds_exact_old_target_question():
    from models import ActiveIssue, IssueKind, IssueStatus
    state = make_state(("A", "B", "C"))
    state.turns = [TurnRecord(i, state.phase, "p1", "Nora", f"old {i}") for i in range(8)]
    state.active_issue = ActiveIssue(
        "i1", IssueKind.QUESTION, ("A",), "p2", "p1",
        "whether the closing time works", IssueStatus.OPEN, 0, 0,
        source_text="Ben: Would the 20:00 closing time work for you?",
    )
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.ANSWER,
        option_focus=("A",), addressee_id="p2", reason="yes", issue_id="i1",
        issue_effect=IssueEffect.RESPOND,
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "Exact question being answered" in prompt
    assert "Would the 20:00 closing time work" in prompt


def test_personal_schedule_number_is_not_rejected_as_option_fact():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.ANSWER,
        option_focus=("A",), reason="I can arrive after 7", personal_context="works an evening shift",
    )
    errors = validate_realization(state, state.persona("p1"), action, "I can usually get there after 7.")
    assert not any("unsupported concrete value" in error for error in errors)


def test_unsupported_objective_number_is_rejected():
    state = make_state(("A", "B", "C"))
    action = UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="cheap")
    errors = validate_realization(state, state.persona("p1"), action, "The library costs 200 euros.")
    assert any("unsupported concrete value" in error for error in errors)


def test_common_word_budget_does_not_resolve_as_option_alias():
    from models import OptionCard, Scenario
    from aliases import resolve_option_mentions
    scenario = Scenario("x", [
        OptionCard("A", "Budget Flight", short_name="Budget"),
        OptionCard("B", "Direct Flight", short_name="Direct"),
    ])
    assert "A" not in resolve_option_mentions("That keeps us within budget.", scenario)
    assert "A" in resolve_option_mentions("I prefer the Budget Flight.", scenario)


def test_natural_vote_phrase_is_accepted():
    state = make_state(("A", "B", "C"))
    action = UserAction("p1", True, BidPriority.REQUIRED, ActionType.VOTE, ("B",), vote_option="B")
    assert validate_realization(state, state.persona("p1"), action, "I'll go with Cafe.") == []


def test_ambiguous_vote_is_rejected():
    state = make_state(("A", "B", "C"))
    action = UserAction("p1", True, BidPriority.REQUIRED, ActionType.VOTE, ("B",), vote_option="B")
    errors = validate_realization(state, state.persona("p1"), action, "I could vote for Cafe or Lab.")
    assert any("ambiguous" in error for error in errors)


def test_visible_switch_is_required():
    state = make_state(("A", "B", "C"))
    update = StanceUpdate(StanceUpdateKind.SWITCH_PREFERRED, "B", "A")
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.COMPROMISE,
        ("B",), reason="better fit", stance_update=update,
    )
    errors = validate_realization(state, state.persona("p1"), action, "Cafe seems reasonable.")
    assert any("stance change" in error for error in errors)
    assert not validate_realization(state, state.persona("p1"), action, "I changed my mind and now prefer Cafe.")


def test_concern_maintenance_accepts_natural_wording():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.ISSUE_OWNER_REACTION, ActionType.CONCERN,
        ("B",), reason="noise", issue_id="i1", issue_effect=IssueEffect.MAINTAIN,
    )
    errors = validate_realization(state, state.persona("p1"), action, "That doesn't solve the noise issue for me.")
    assert not any("continued concern" in error for error in errors)


def test_near_duplicate_is_rejected_but_revote_can_repeat_vote():
    state = make_state(("A", "B", "C"))
    state.turns.append(TurnRecord(0, state.phase, "p1", "Nora", "Library works best for my project."))
    support = UserAction("p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="fit")
    assert any("near-verbatim" in error for error in validate_realization(state, state.persona("p1"), support, "Library works best for my project."))
    vote = UserAction("p1", True, BidPriority.REQUIRED, ActionType.VOTE, ("A",), vote_option="A")
    assert not validate_realization(state, state.persona("p1"), vote, "I vote for Library.")


def test_direct_answer_can_be_relevant_through_its_structured_reason():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.ANSWER,
        option_focus=("A",), reason="The closing time is too early after work",
    )
    errors = validate_realization(
        state, state.persona("p1"), action,
        "The closing time is too early after work.",
    )
    assert not any("unrelated" in error for error in errors)


def test_near_duplicate_threshold_is_configured():
    from config_loader import cfg
    assert 0.0 <= float(cfg.language.near_duplicate_similarity_threshold) <= 1.0


def test_clear_changed_vote_is_valid_without_fixed_bridge_phrase():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.VOTE,
        ("B",), vote_option="B",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED,
            "B",
            previous_option_id="A",
        ),
    )
    errors = validate_realization(
        state,
        state.persona("p1"),
        action,
        "I vote for Cafe.",
    )
    assert not errors


def test_incomplete_comparison_is_not_a_hard_validation_error():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.COMPARE,
        ("A", "B"), reason="quiet versus atmosphere",
    )
    errors = validate_realization(
        state,
        state.persona("p1"),
        action,
        "Library is quieter for me.",
    )
    assert not errors


def test_natural_vote_forms_are_valid():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.VOTE,
        ("A",), vote_option="A",
    )
    for text in (
        "Library gets my vote.",
        "I’m going with Library.",
        "I’ll stick with Library.",
        "For me, it is Library.",
    ):
        assert not validate_realization(state, state.persona("p1"), action, text)


def test_unchanged_vote_prompt_does_not_repeat_reason():
    from prompts import realization_prompt

    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.VOTE,
        ("A",), reason="quiet and predictable", vote_option="A",
    )
    prompt = realization_prompt(state, state.persona("p1"), action)
    assert "do not repeat your reason" in prompt.casefold()
    assert "Maximum 8 words" in prompt
    assert "quiet and predictable" not in prompt


def test_lowercase_option_as_common_noun_is_not_unknown_label():
    state = make_state()
    persona = state.personas[0]
    action = UserAction(
        speaker_id=persona.id,
        wants_to_speak=True,
        priority=BidPriority.ISSUE_RESPONSE,
        act=ActionType.COMMENT,
        option_focus=("A",),
        reason="the option can still work for me",
    )
    errors = validate_realization(
        state, persona, action, "The drawback matters, but the option can still work for me."
    )
    assert not any("unknown option" in error for error in errors)


def test_contextual_vote_without_vote_verb_is_accepted():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.VOTE,
        ("A",), vote_option="A",
    )
    assert not validate_realization(state, state.persona("p1"), action, "Library for me.")


def test_contextual_vote_with_competing_option_is_rejected():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.VOTE,
        ("A",), vote_option="A",
    )
    errors = validate_realization(
        state,
        state.persona("p1"),
        action,
        "Library or Cafe would both be fine.",
    )
    assert any("ambiguous" in error for error in errors)


def test_semantic_answer_prompt_uses_tradeoff_fields():
    from models import ResponseMode

    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.ANSWER,
        option_focus=("A",), addressee_id="p2",
        reason="can become crowded",
        response_mode=ResponseMode.ACCEPT_TRADEOFF,
        decisive_reason="quiet and predictable",
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "recognize the concern and still prefer" in prompt
    assert "quiet and predictable" in prompt
    assert "the drawback matters, but" not in prompt.casefold()
    assert "deal-breaker" not in prompt.casefold()


def test_movement_prompt_requires_concrete_grounded_reason():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.COMPROMISE,
        ("B",), reason="relaxed atmosphere",
        decisive_reason="relaxed atmosphere",
        stance_update=StanceUpdate(
            StanceUpdateKind.MAKE_ACCEPTABLE,
            "B",
            previous_option_id="A",
            movement_reason="relaxed atmosphere",
            movement_basis="common_ground",
        ),
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "concrete reason: relaxed atmosphere" in prompt
    assert "vague fairness reason" in prompt


def test_vague_movement_without_grounded_reason_is_rejected():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.COMPROMISE,
        ("B",), reason="relaxed atmosphere",
        stance_update=StanceUpdate(
            StanceUpdateKind.MAKE_ACCEPTABLE,
            "B",
            previous_option_id="A",
            movement_reason="relaxed atmosphere",
            movement_basis="common_ground",
        ),
    )
    errors = validate_realization(
        state,
        state.persona("p1"),
        action,
        "Cafe seems reasonable enough for me.",
    )
    assert any("grounded movement reason" in error for error in errors)


def test_grounded_movement_reason_is_accepted():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.COMPROMISE,
        ("B",), reason="relaxed atmosphere",
        stance_update=StanceUpdate(
            StanceUpdateKind.MAKE_ACCEPTABLE,
            "B",
            previous_option_id="A",
            movement_reason="relaxed atmosphere",
            movement_basis="common_ground",
        ),
    )
    assert not validate_realization(
        state,
        state.persona("p1"),
        action,
        "I can accept Cafe because the relaxed atmosphere works for me.",
    )


def test_vote_after_explained_acceptance_stays_short():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.VOTE,
        ("B",), vote_option="B",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED,
            "B",
            previous_option_id="A",
            movement_reason="relaxed atmosphere",
            movement_basis="previous_acceptance",
            reason_already_public=True,
        ),
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "reason was already explained publicly" in prompt
    assert "do not invent a new reason" in prompt


def test_partial_concern_reaction_keeps_original_concern_as_hesitation():
    from models import ResponseMode

    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.ISSUE_OWNER_REACTION, ActionType.COMMENT,
        ("A",), reason="can become crowded", decisive_reason="quiet and predictable",
        issue_id="issue-1", issue_effect=IssueEffect.PARTIAL,
        response_mode=ResponseMode.MAINTAIN_CONCERN,
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "Acknowledge this response" in prompt
    assert "quiet and predictable" in prompt
    assert "original concern still remains: can become crowded" in prompt


def test_first_acceptance_prompt_preserves_previous_priority():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.COMPROMISE,
        ("B",), reason="relaxed atmosphere",
        decisive_reason="relaxed atmosphere",
        stance_update=StanceUpdate(
            StanceUpdateKind.MAKE_ACCEPTABLE,
            "B",
            previous_option_id="A",
            movement_reason="relaxed atmosphere",
            movement_basis="common_ground",
        ),
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "previously preferred Library" in prompt
    assert "fits my main priority" in prompt
    assert "preserve that priority" in prompt


def test_realization_prompt_guards_qualitative_grounding():
    state = make_state(("A", "B", "C"))
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.SUPPORT,
        ("A",), reason="quiet and predictable",
    )
    prompt = prompts.realization_prompt(state, state.persona("p1"), action)
    assert "option subtypes" in prompt
    assert "facilities" in prompt
    assert "stronger/weaker versions" in prompt


def test_moderator_vote_request_is_neutral_and_combines_unanimity():
    state = make_state(("A", "A", "A"))
    neutral = prompts.moderator_vote_request(revote=False)
    assert neutral == "Let’s take the final vote. Please name the one option you’re choosing."
    assert "Now give" not in neutral

    unanimous = prompts.moderator_vote_request(
        revote=False,
        scenario=state.scenario,
        unanimous_option="A",
    )
    assert "already has everyone’s support" in unanimous
    assert "confirm it with a final vote" in unanimous
