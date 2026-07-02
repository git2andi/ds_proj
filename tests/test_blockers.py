"""I3: parser vocabulary for active blockers, resolutions, conditional support,
compromise offers, and switch reasons. No LLM calls."""

from __future__ import annotations

from models import ActType, MoveIntent, OptionCard
from parsing import (
    OptionResolver,
    active_blocker_option,
    blocker_resolution_option,
    commitment_has_reason,
    compromise_offer_option,
    conditional_support_option,
    parse_dialogue_act,
    visible_commitment,
)


def _resolver() -> OptionResolver:
    return OptionResolver([
        OptionCard(id="A", name="Sunny Side Cafe", short_name="Sunny Side"),
        OptionCard(id="B", name="Green Garden Bistro", short_name="Green Garden"),
        OptionCard(id="C", name="Retro Diner", short_name="Retro Diner"),
        OptionCard(id="D", name="Riverside Patio", short_name="Riverside"),
    ])


def _parse(text: str, intent: MoveIntent | None = None):
    return parse_dialogue_act(
        speaker_id="p1",
        speaker_name="Isla",
        text=text,
        resolver=_resolver(),
        participant_names={"p1": "Isla", "p2": "Zeke"},
        intent=intent,
    )


# --- active blockers ---

def test_dealbreaker_is_active_blocker():
    text = "Sunny Side is a dealbreaker for me because of the vegetarian gap."
    assert active_blocker_option(text, _resolver()) == "A"


def test_negated_dealbreaker_is_not_a_blocker():
    text = "Honestly the noise at Retro Diner is not a dealbreaker for me."
    assert active_blocker_option(text, _resolver()) is None


def test_doesnt_work_for_me_is_blocker():
    text = "Riverside just doesn't work for me with the weather risk."
    assert active_blocker_option(text, _resolver()) == "D"


def test_blocker_lands_in_hard_rejects():
    act = _parse("Sunny Side is a dealbreaker for me, plain and simple.")
    assert "A" in act.hard_rejects


def test_blocker_without_option_is_ignored():
    act = _parse("Slow service is a dealbreaker for me.")
    assert not act.hard_rejects


# --- resolutions ---

def test_resolution_with_acceptance():
    text = "That fixes my concern; I can live with Sunny Side."
    act = _parse(text)
    assert act.resolves_blocker == "A"
    assert visible_commitment(text, _resolver()) == ("accept", "A")


def test_conditional_resolution_does_not_resolve():
    text = "I can live with Sunny Side only if they add vegan options."
    assert blocker_resolution_option(text, _resolver()) is None
    assert visible_commitment(text, _resolver()) is None


def test_raise_and_resolve_in_one_line_does_not_resolve():
    act = _parse("Sunny Side is a dealbreaker for me — I can't live with that menu.")
    assert act.resolves_blocker is None


# --- conditional support ---

def test_conditional_support_detected():
    act = _parse("I can support Green Garden, but only if we keep dessert elsewhere.")
    assert act.explicit_vote is None
    assert act.conditional_support == "B"


def test_plain_vote_is_not_conditional_support():
    act = _parse("I vote for Green Garden.", MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r"))
    assert act.explicit_vote == "B"
    assert act.conditional_support is None


# --- compromise offers ---

def test_compromise_offer_question_detected():
    act = _parse("Could we all live with Retro Diner, given the budget?")
    assert act.offers_compromise == "C"


def test_meet_in_the_middle_detected():
    act = _parse("Maybe we meet in the middle on Riverside and book early.")
    assert act.offers_compromise == "D"


# --- switch phrasing + reasons ---

def test_switch_phrasing_is_a_vote():
    text = "I'd switch to Green Garden because it solves the dietary issue."
    assert visible_commitment(text, _resolver()) == ("vote", "B")


def test_commitment_reason_detected():
    assert commitment_has_reason("I'd switch to Green Garden because it solves the dietary issue.")
    assert commitment_has_reason("Green Garden gets my vote for its inclusive menu.")
    assert not commitment_has_reason("I vote for Green Garden.")


def test_compare_mention_still_creates_no_support():
    act = _parse("Green Garden costs more than Retro Diner but covers more diets.")
    assert act.explicit_vote is None and not act.accepts
