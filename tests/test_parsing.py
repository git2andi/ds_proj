"""Unit tests for src/parsing.py — trailer extraction, commitment gating, hedge detection."""

from __future__ import annotations

import pytest
from conftest import make_intent
from models import ActType, MoveIntent, OptionCard
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
    OPTION_IDS = ["A", "B", "C", "D"]

    def test_bracketed_trailer(self):
        raw = "Sounds good to me. [act=accept; opt=A; stance=accept]"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert text == "Sounds good to me."
        assert move.present is True
        assert move.act == ActType.ACCEPT
        assert move.option == "A"
        assert move.stance == "accept"

    def test_bare_trailer_no_brackets(self):
        raw = "I like this one. act=vote; opt=B; stance=vote"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert "act=" not in text
        assert move.present is True
        assert move.act == ActType.VOTE
        assert move.option == "B"
        assert move.stance == "vote"

    def test_missing_trailer(self):
        raw = "Just a regular comment about the trip."
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert text == "Just a regular comment about the trip."
        assert move.present is False
        assert move.act is None
        assert move.option is None
        assert move.stance == "neutral"

    def test_partial_trailer_act_only(self):
        raw = "Nice. [act=react]"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert text == "Nice."
        assert move.present is True
        assert move.act == ActType.REACT
        assert move.option is None
        assert move.stance == "neutral"

    def test_trailer_with_extra_brackets(self):
        raw = "Good one. [act=support; opt=[C]; stance=neutral]"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert move.present is True
        assert move.option == "C"

    def test_unknown_act_value(self):
        raw = "Hmm. [act=foobar; opt=A; stance=neutral]"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert move.present is True
        assert move.act is None  # unknown act -> None

    def test_unknown_stance_defaults_neutral(self):
        raw = "Ok. [act=react; stance=confused]"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert move.stance == "neutral"

    def test_invalid_option_letter_ignored(self):
        raw = "I agree. [act=accept; opt=Z; stance=accept]"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert move.option is None

    def test_dangling_option_letter_stripped(self):
        raw = "That's a solid pick for our group C"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert not text.rstrip().endswith("C")
        assert move.present is False

    def test_message_preserved_before_trailer(self):
        raw = "I'd rather go with the mountain option. [act=support; opt=A; stance=neutral]"
        text, move = parse_trailer(raw, self.OPTION_IDS)
        assert "mountain option" in text
        assert "act=" not in text


# ── _resolve_move (commitment gating) ────────────────────────────────────

class TestResolveMove:
    def test_discussion_accept_clamped_to_neutral(self):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="A", stance="accept")
        intent = make_intent(act=ActType.SUPPORT)  # discussion, not a decision turn
        stance, opt, act_type = _resolve_move(move, intent, ["A"], None)
        assert stance == "neutral"
        assert act_type != ActType.ACCEPT

    def test_discussion_vote_clamped_to_neutral(self):
        move = TurnMove(present=True, act=ActType.VOTE, option="B", stance="vote")
        intent = make_intent(act=ActType.COMPARE)
        stance, opt, act_type = _resolve_move(move, intent, ["B"], None)
        assert stance == "neutral"

    def test_routed_vote_honoured(self):
        move = TurnMove(present=True, act=ActType.VOTE, option="A", stance="vote")
        intent = make_intent(act=ActType.VOTE)
        stance, opt, act_type = _resolve_move(move, intent, ["A"], None)
        assert stance == "vote"
        assert act_type == ActType.VOTE

    def test_routed_accept_honoured(self):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="C", stance="accept")
        intent = make_intent(act=ActType.ACCEPT)
        stance, opt, act_type = _resolve_move(move, intent, ["C"], None)
        assert stance == "accept"
        assert act_type == ActType.ACCEPT

    def test_routed_reject_honoured(self):
        move = TurnMove(present=True, act=ActType.REJECT, option="D", stance="reject")
        intent = make_intent(act=ActType.REJECT)
        stance, opt, act_type = _resolve_move(move, intent, ["D"], None)
        assert stance == "reject"
        assert act_type == ActType.REJECT

    def test_opening_clamped_to_neutral(self):
        move = TurnMove(present=True, act=ActType.VOTE, option="A", stance="vote")
        intent = make_intent(act=ActType.OPENING)
        stance, opt, act_type = _resolve_move(move, intent, ["A"], None)
        assert stance == "neutral"
        assert act_type == ActType.OPENING

    def test_no_trailer_falls_back_to_intent(self):
        move = TurnMove(present=False)
        intent = make_intent(act=ActType.VOTE)
        stance, opt, act_type = _resolve_move(move, intent, ["B"], None)
        assert stance == "vote"
        assert act_type == ActType.VOTE


# ── Hedged accept detection ──────────────────────────────────────────────

class TestHedgedAccept:
    @pytest.mark.parametrize("text", [
        "Mountain Retreat might work for us.",
        "Maybe we can go with Beach Resort.",
        "That's fine if the weather is good.",
        "Works if we leave early enough.",
        "I guess that's okay.",
        "I suppose Beach Resort could work.",
        "Perhaps Mountain Retreat is the way to go.",
        "Good if we keep costs under control.",
        "Only if we get a discount.",
        "As long as everyone is comfortable.",
        "Provided that we book in advance.",
    ])
    def test_hedged_phrases_detected(self, text: str):
        assert _HEDGED_ACCEPT.search(text), f"Should detect hedge in: {text!r}"

    @pytest.mark.parametrize("text", [
        "Mountain Retreat is perfect.",
        "I'm fully on board with Beach Resort.",
        "Let's do it.",
        "Absolutely, City Tour gets my vote.",
        "Road Trip works for me.",
    ])
    def test_firm_accepts_not_flagged(self, text: str):
        assert not _HEDGED_ACCEPT.search(text), f"Should NOT detect hedge in: {text!r}"

    def test_hedged_accept_clamped_in_resolve_move(self):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="A", stance="accept")
        intent = make_intent(act=ActType.ACCEPT)
        stance, opt, act_type = _resolve_move(move, intent, ["A"], None, hedged=True)
        assert stance == "neutral"

    def test_firm_accept_not_clamped(self):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="A", stance="accept")
        intent = make_intent(act=ActType.ACCEPT)
        stance, opt, act_type = _resolve_move(move, intent, ["A"], None, hedged=False)
        assert stance == "accept"


# ── OptionResolver ───────────────────────────────────────────────────────

class TestOptionResolver:
    def test_ids_in_text_by_name(self, resolver: OptionResolver):
        assert "A" in resolver.ids_in_text("Mountain Retreat is great.")

    def test_ids_in_text_by_explicit_ref(self, resolver: OptionResolver):
        assert "B" in resolver.ids_in_text("Option B looks good.")

    def test_ids_in_text_no_match(self, resolver: OptionResolver):
        assert resolver.ids_in_text("Nothing relevant here.") == []

    def test_invalid_option_refs(self, resolver: OptionResolver):
        assert resolver.invalid_option_refs("Option Z is wild.") == ["Z"]
        assert resolver.invalid_option_refs("Option A and Option B.") == []

    def test_alias_collision_excluded(self):
        # If two options share a word, that word shouldn't resolve to either
        opts = [
            OptionCard(id="A", name="Bowling Night"),
            OptionCard(id="B", name="Game Night"),
        ]
        r = OptionResolver(opts)
        # "Night" is shared, so it shouldn't uniquely resolve
        # but "Bowling" and "Game" should each resolve uniquely
        assert "A" in r.ids_in_text("Bowling sounds fun.")
        assert "B" in r.ids_in_text("Game sounds fun.")

    def test_option_text(self, resolver: OptionResolver):
        text = resolver.option_text("A")
        assert "Mountain Retreat" in text
        assert "$800" in text


# ── parse_dialogue_act ───────────────────────────────────────────────────

class TestParseDialogueAct:
    def test_basic_act(self, resolver: OptionResolver, participant_names: dict[str, str]):
        move = TurnMove(present=True, act=ActType.SUPPORT, option="A", stance="neutral")
        intent = make_intent(act=ActType.SUPPORT, option_focus=["A"])
        act = parse_dialogue_act(
            speaker_id="p1", speaker_name="Alice",
            text="Mountain Retreat is solid.",
            resolver=resolver, participant_names=participant_names,
            move=move, intent=intent,
        )
        assert "A" in act.option_refs
        assert act.act_type == ActType.SUPPORT

    def test_addressee_detection(self, resolver: OptionResolver, participant_names: dict[str, str]):
        act = parse_dialogue_act(
            speaker_id="p1", speaker_name="Alice",
            text="Bob, what do you think about Mountain Retreat?",
            resolver=resolver, participant_names=participant_names,
        )
        assert act.addressee_id == "p2"

    def test_vote_act_records_explicit_vote(self, resolver: OptionResolver, participant_names: dict[str, str]):
        move = TurnMove(present=True, act=ActType.VOTE, option="B", stance="vote")
        intent = make_intent(act=ActType.VOTE)
        act = parse_dialogue_act(
            speaker_id="p1", speaker_name="Alice",
            text="Beach Resort gets my vote.",
            resolver=resolver, participant_names=participant_names,
            move=move, intent=intent,
        )
        assert act.explicit_vote == "B"

    def test_accept_records_accepts(self, resolver: OptionResolver, participant_names: dict[str, str]):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="A", stance="accept")
        intent = make_intent(act=ActType.ACCEPT)
        act = parse_dialogue_act(
            speaker_id="p1", speaker_name="Alice",
            text="Mountain Retreat works for me.",
            resolver=resolver, participant_names=participant_names,
            move=move, intent=intent,
        )
        assert "A" in act.accepts

    def test_hedged_accept_stays_neutral(self, resolver: OptionResolver, participant_names: dict[str, str]):
        move = TurnMove(present=True, act=ActType.ACCEPT, option="A", stance="accept")
        intent = make_intent(act=ActType.ACCEPT)
        act = parse_dialogue_act(
            speaker_id="p1", speaker_name="Alice",
            text="Mountain Retreat might work if the weather holds.",
            resolver=resolver, participant_names=participant_names,
            move=move, intent=intent,
        )
        assert act.accepts == []
        assert act.explicit_vote is None
