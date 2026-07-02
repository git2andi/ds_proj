"""I2: on sanctioned switch turns (allow_vote_change) a bridge clause after the
commitment must not void it; everywhere else the conservative rules stand."""

from __future__ import annotations

from models import ActType, MoveIntent, OptionCard
from parsing import OptionResolver, parse_dialogue_act, visible_commitment


def _resolver() -> OptionResolver:
    return OptionResolver([
        OptionCard(id="A", name="Retro Neon Workout Crew", short_name="Retro Neon"),
        OptionCard(id="B", name="Classic Movie Monsters", short_name="Movie Monsters"),
        OptionCard(id="C", name="Board Game Pieces", short_name="Board Game"),
        OptionCard(id="D", name="Space Explorers Squad", short_name="Space Explorers"),
    ])


# The exact failure from logs/archive/20260702_151019_800548: Gemma's compliant
# bridge-clause switch was parsed as nothing and her vote silently stayed B.
GEMMA = "My pick is the Retro Neon Workout Crew—I'm good with the outfits as long as we keep things cool and comfortable."


def test_bridge_clause_counts_on_sanctioned_switch():
    assert visible_commitment(GEMMA, _resolver(), sanctioned_switch=True) == ("vote", "A")


def test_bridge_clause_still_blocked_without_sanction():
    assert visible_commitment(GEMMA, _resolver()) is None


def test_sanctioned_soft_commit_with_concession_counts_as_accept():
    text = "Honestly, Space Explorers works for me even though the Board Game was my first pick."
    stance, option = visible_commitment(text, _resolver(), sanctioned_switch=True)
    assert stance == "accept" and option == "D"


def test_sanctioned_prerequisite_still_blocks():
    text = "I'd go with Retro Neon only if we swap the tight outfits."
    assert visible_commitment(text, _resolver(), sanctioned_switch=True) is None


def test_sanctioned_unless_still_blocks():
    text = "Count me in for Retro Neon unless the budget grows."
    assert visible_commitment(text, _resolver(), sanctioned_switch=True) is None


def test_sanctioned_question_still_blocks():
    text = "I'd go with Retro Neon, but are we sure about the outfits?"
    assert visible_commitment(text, _resolver(), sanctioned_switch=True) is None


def test_parse_dialogue_act_wires_allow_vote_change():
    intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", allow_vote_change=True)
    act = parse_dialogue_act(
        speaker_id="p1",
        speaker_name="Gemma",
        text=GEMMA,
        resolver=_resolver(),
        participant_names={"p1": "Gemma", "p2": "Oscar"},
        intent=intent,
    )
    assert act.explicit_vote == "A"


def test_parse_dialogue_act_stays_strict_without_allow_vote_change():
    intent = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", allow_vote_change=False)
    act = parse_dialogue_act(
        speaker_id="p1",
        speaker_name="Gemma",
        text=GEMMA,
        resolver=_resolver(),
        participant_names={"p1": "Gemma", "p2": "Oscar"},
        intent=intent,
    )
    assert act.explicit_vote is None
