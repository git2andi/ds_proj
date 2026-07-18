from models import ActionType, BidPriority, Phase, TurnRecord, UserAction
from prompts import moderator_compromise_prompt, realization_prompt
from tests.fixtures import make_state


def test_ordinary_discussion_prompt_does_not_force_option_restatement():
    state = make_state()
    prior = UserAction(
        "p2", True, BidPriority.NORMAL, ActionType.SUPPORT, ("B",), reason="relaxed"
    )
    state.turns.append(
        TurnRecord(0, Phase.DISCUSSION, "p2", "Ben", "Cafe feels relaxed.", action=prior)
    )
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="quiet"
    )
    prompt = realization_prompt(state, action)
    assert "Do not repeat an option name only for formality" in prompt


def test_opening_still_requests_an_explicit_option_reference():
    state = make_state()
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.OPENING, ("A",), reason="quiet"
    )
    prompt = realization_prompt(state, action)
    assert "Use one allowed option reference" in prompt


def test_reaction_prompt_discourages_name_first_openings():
    state = make_state()
    prior = UserAction(
        "p2", True, BidPriority.NORMAL, ActionType.SUPPORT, ("B",), reason="relaxed"
    )
    state.turns.append(
        TurnRecord(0, Phase.DISCUSSION, "p2", "Ben", "Cafe feels relaxed.", action=prior)
    )
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.REACT, ("B",),
        addressee_id="p2", reason="noise"
    )
    prompt = realization_prompt(state, action)
    assert "without automatically starting with their name" in prompt
    assert "Do not routinely begin with the option name" in prompt


def test_comparison_prompt_labels_each_fact_without_fixed_template():
    from models import ReasonSource

    state = make_state()
    action = UserAction(
        "p1",
        True,
        BidPriority.NORMAL,
        ActionType.COMPARE,
        ("A", "B"),
        comparison_sources=(
            ReasonSource("A", "closing time", "20:00"),
            ReasonSource("B", "closing time", "22:00"),
        ),
    )
    prompt = realization_prompt(state, action)
    assert "Fact for Library only: closing time: 20:00" in prompt
    assert "Fact for Cafe only: closing time: 22:00" in prompt
    assert "use one fixed contrast template" in prompt.lower()
    assert "same public attribute" in prompt


def test_opening_prompt_avoids_meta_opening_language():
    state = make_state()
    action = UserAction(
        "p1", True, BidPriority.REQUIRED, ActionType.OPENING, ("A",), reason="quiet"
    )
    prompt = realization_prompt(state, action)
    assert "Join the opening" not in prompt
    assert "narrate the act of speaking" in prompt


def test_prompt_encourages_varied_continuation_without_but_or_helps_template():
    state = make_state()
    prior = UserAction(
        "p2", True, BidPriority.NORMAL, ActionType.SUPPORT, ("B",), reason="relaxed"
    )
    state.turns.append(
        TurnRecord(0, Phase.DISCUSSION, "p2", "Ben", "The cafe feels relaxed.", action=prior)
    )
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.REACT, ("B",), reason="noise"
    )
    prompt = realization_prompt(state, action)
    assert "short reaction or continuation" in prompt
    assert "“But” is fine occasionally" in prompt
    assert "helps/limits/makes" in prompt
    assert "Do not routinely begin" in prompt


def test_support_instruction_does_not_lead_with_option_name():
    state = make_state()
    action = UserAction(
        "p1", True, BidPriority.NORMAL, ActionType.SUPPORT, ("A",), reason="quiet"
    )
    prompt = realization_prompt(state, action)
    assert "Continue the exchange with why the focused choice suits you" in prompt
    assert "Explain naturally why Library" not in prompt


def test_moderator_compromise_prompt_handles_singular_and_plural():
    state = make_state()
    singular = moderator_compromise_prompt(state.scenario, ("A",), variant=2)
    plural = moderator_compromise_prompt(state.scenario, ("A", "B"), variant=0)
    assert "Library is the leading option" in singular
    assert "Library and Cafe are the main options" in plural
    assert "Would either" in plural
