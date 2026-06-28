"""Essential tests for src/parsing.py — trailer extraction, commitment gating, option resolution."""

from __future__ import annotations

import pytest
from conftest import make_intent
from models import ActType, OptionCard
from parsing import (
    OptionResolver,
    TurnMove,
    parse_dialogue_act,
    parse_trailer,
    _HEDGED_ACCEPT,
    _resolve_move,
)


# ── parse_trailer ────────────────────────────────────────────────────────

class TestParseTrailer:
    IDS = ["A", "B", "C", "D"]

    def test_bracketed_trailer(self):
        text, move = parse_trailer("Sounds good. [act=accept; opt=A; stance=accept]", self.IDS)
        assert text == "Sounds good." and move.present and move.act == ActType.ACCEPT and move.option == "A"

    def test_bare_trailer(self):
        text, move = parse_trailer("I like this. act=vote; opt=B; stance=vote", self.IDS)
        assert "act=" not in text and move.present and move.stance == "vote"

    def test_missing_trailer(self):
        text, move = parse_trailer("Just a comment.", self.IDS)
        assert text == "Just a comment." and not move.present

    def test_invalid_option_ignored(self):
        _, move = parse_trailer("Ok. [act=accept; opt=Z; stance=accept]", self.IDS)
        assert move.option is None


# ── Commitment gating ───────────────────────────────────────────────────

class TestCommitmentGating:
    def test_discussion_accept_clamped(self):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="A", stance="accept")
        stance, _, _ = _resolve_move(move, make_intent(act=ActType.SUPPORT), ["A"], None)
        assert stance == "neutral"

    def test_routed_vote_honoured(self):
        move = TurnMove(present=True, act=ActType.VOTE, option="A", stance="vote")
        stance, _, act = _resolve_move(move, make_intent(act=ActType.VOTE), ["A"], None)
        assert stance == "vote" and act == ActType.VOTE

    def test_opening_clamped(self):
        move = TurnMove(present=True, act=ActType.VOTE, option="A", stance="vote")
        stance, _, act = _resolve_move(move, make_intent(act=ActType.OPENING), ["A"], None)
        assert stance == "neutral" and act == ActType.OPENING

    def test_hedged_accept_clamped(self):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="A", stance="accept")
        stance, _, _ = _resolve_move(move, make_intent(act=ActType.ACCEPT), ["A"], None, hedged=True)
        assert stance == "neutral"

    def test_question_not_credited_as_accept(self):
        move = TurnMove(present=False)
        stance, _, _ = _resolve_move(move, make_intent(act=ActType.ACCEPT), ["A"], "p2")
        assert stance == "neutral"

    def test_question_not_credited_as_vote(self):
        move = TurnMove(present=False)
        stance, _, _ = _resolve_move(move, make_intent(act=ActType.VOTE), ["B"], "p1")
        assert stance == "neutral"

    def test_no_trailer_vote_invisible_option_not_ghosted(self):
        # No trailer, option_refs empty, routing focuses on A → focus_opt must be None (KF03)
        move = TurnMove(present=False)
        _, focus, _ = _resolve_move(move, make_intent(act=ActType.VOTE, option_focus=["A"]), [], None)
        assert focus is None

    def test_no_trailer_accept_invisible_option_not_ghosted(self):
        # Same rule for ACCEPT intent
        move = TurnMove(present=False)
        _, focus, _ = _resolve_move(move, make_intent(act=ActType.ACCEPT, option_focus=["B"]), [], None)
        assert focus is None

    def test_no_trailer_compare_still_uses_intent_focus(self):
        # Non-binding acts may still use intent.option_focus when no trailer
        move = TurnMove(present=False)
        _, focus, _ = _resolve_move(move, make_intent(act=ActType.COMPARE, option_focus=["C"]), [], None)
        assert focus == "C"

    def test_trailer_present_vote_uses_intent_focus_when_no_option_refs(self):
        # Trailer present (commitment explicit) → intent.option_focus allowed as option fallback
        move = TurnMove(present=True, stance="vote")
        _, focus, _ = _resolve_move(move, make_intent(act=ActType.VOTE, option_focus=["A"]), [], None)
        assert focus == "A"


# ── OptionResolver ──────────────────────────────────────────────────────

class TestOptionResolver:
    def test_name_match(self, resolver: OptionResolver):
        assert "A" in resolver.ids_in_text("Mountain Retreat is great.")

    def test_invalid_refs(self, resolver: OptionResolver):
        assert resolver.invalid_option_refs("Option Z is wild.") == ["Z"]


# ── parse_dialogue_act ──────────────────────────────────────────────────

class TestDialogueAct:
    def test_vote_records_explicit(self, resolver, participant_names):
        move = TurnMove(present=True, act=ActType.VOTE, option="B", stance="vote")
        act = parse_dialogue_act(speaker_id="p1", speaker_name="Alice",
                                  text="Beach Resort gets my vote.", resolver=resolver,
                                  participant_names=participant_names, move=move,
                                  intent=make_intent(act=ActType.VOTE))
        assert act.explicit_vote == "B"

    def test_hedged_accept_stays_neutral(self, resolver, participant_names):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="A", stance="accept")
        act = parse_dialogue_act(speaker_id="p1", speaker_name="Alice",
                                  text="Mountain Retreat might work if the weather holds.",
                                  resolver=resolver, participant_names=participant_names,
                                  move=move, intent=make_intent(act=ActType.ACCEPT))
        assert act.accepts == []
