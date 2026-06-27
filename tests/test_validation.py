"""Essential tests for src/validation.py — core guardrails and repair triggers."""

from __future__ import annotations

import pytest
from conftest import make_intent, make_turn
from models import ActType, Phase
from parsing import OptionResolver, TurnMove
from validation import (
    MessageValidator,
    classify_claim_slots,
    classify_discourse_frames,
    covered_slots_hint,
    fix_collective_voice,
    recent_frame_hint,
)


@pytest.fixture()
def validator(resolver: OptionResolver, participant_names: dict[str, str]) -> MessageValidator:
    return MessageValidator(resolver, participant_names)


def _validate(validator, text, state, intent=None, move=None):
    if intent is None:
        intent = make_intent(act=ActType.REACT)
    if move is None:
        move = TurnMove()
    return validator.validate(text, state, intent, move)


# ── Structure ───────────────────────────────────────────────────────────

class TestStructure:
    def test_empty_is_fatal(self, validator, state):
        r = _validate(validator, "", state)
        assert not r.ok and "EMPTY" in r.codes()

    def test_speaker_prefix_triggers_repair(self, validator, state):
        r = _validate(validator, "Alice: I think Mountain Retreat sounds nice.", state)
        assert "SPEAKER_PREFIX" in r.codes()

    def test_invalid_option_ref_is_fatal(self, validator, state):
        r = _validate(validator, "Option Z looks interesting.", state)
        assert "INVALID_OPTION_REFERENCE" in r.codes()

    def test_invented_number_detected(self, validator, state):
        r = _validate(validator, "Mountain Retreat costs $3500 per person.", state)
        assert "INVENTED_OPTION_ATTRIBUTE" in r.codes()

    def test_incomplete_turn_detected(self, validator, state):
        r = _validate(validator, "compared to Beach Resort's", state)
        assert "INCOMPLETE_TURN" in r.codes()


# ── Repetition / echo ──────────────────────────────────────────────────

class TestRepetition:
    def test_duplicate_turn_detected(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "Mountain Retreat really appeals to me because of the fresh air and the remote location."))
        r = _validate(validator, "Mountain Retreat really appeals to me because of the fresh air and the remote location.", state, intent=make_intent(speaker_id="p1"))
        assert "DUPLICATE_TURN" in r.codes()

    def test_self_repetition_detected(self, validator, state):
        state.runtimes["p1"].already_said.append("The cost factor makes Mountain Retreat very attractive for our group budget.")
        r = _validate(validator, "The cost factor makes Mountain Retreat really attractive for our group budget.", state, intent=make_intent(speaker_id="p1"))
        assert "SELF_REPETITION" in r.codes()

    def test_self_repetition_fires_on_accept_intent(self, validator, state):
        state.runtimes["p1"].already_said.append("Taco Loco offers a unique twist.")
        r = _validate(validator, "Taco Loco offers a unique twist.", state,
                      intent=make_intent(speaker_id="p1", act=ActType.ACCEPT))
        assert "SELF_REPETITION" in r.codes()

    def test_question_echo_is_repair(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "Do they have space at the venue for our group?"))
        r = _validate(validator, "Do they have space at the venue for our group of ten?", state, intent=make_intent(speaker_id="p1"))
        assert "QUESTION_ECHO" in r.codes()
        assert any(i.severity == "repair" for i in r.issues if i.code == "QUESTION_ECHO")


# ── Opener variety ─────────────────────────────────────────────────────

class TestOpenerVariety:
    def test_i_streak_detected(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "I think Beach Resort is the best."))
        state.turns.append(make_turn(1, "p3", "Carol", "I'm leaning toward City Tour."))
        r = _validate(validator, "I really want Mountain Retreat.", state, intent=make_intent(act=ActType.REACT))
        assert "REPETITIVE_OPENER" in r.codes()

    def test_vote_exempt_from_opener_check(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "I think Beach Resort is the best."))
        state.turns.append(make_turn(1, "p3", "Carol", "I'm leaning toward City Tour."))
        r = _validate(validator, "I vote for Mountain Retreat.", state, intent=make_intent(act=ActType.VOTE))
        assert "REPETITIVE_OPENER" not in r.codes()


# ── Decision clarity ───────────────────────────────────────────────────

class TestDecisionClarity:
    def test_vote_without_vote_stance(self, validator, state):
        r = _validate(validator, "Mountain Retreat is nice.", state,
                       intent=make_intent(act=ActType.VOTE),
                       move=TurnMove(present=True, stance="neutral"))
        assert "UNCLEAR_VOTE" in r.codes()

    def test_question_in_confirmation(self, validator, state):
        r = _validate(validator, "Mountain Retreat works, but what about parking?", state,
                       intent=make_intent(act=ActType.ACCEPT),
                       move=TurnMove(present=True, stance="accept", option="A"))
        assert "QUESTION_IN_CONFIRMATION" in r.codes()


# ── Question chain ─────────────────────────────────────────────────────

class TestQuestionChain:
    def test_chain_detected(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "What about the cost?"))
        state.turns.append(make_turn(1, "p3", "Carol", "Yeah, is $800 per person?"))
        r = _validate(validator, "Does Mountain Retreat include meals?", state, intent=make_intent(act=ActType.REACT))
        assert "QUESTION_CHAIN" in r.codes()

    def test_answer_exempt_from_chain(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "What about the cost?"))
        state.turns.append(make_turn(1, "p3", "Carol", "How far is it?"))
        r = _validate(validator, "I think it's around $800 per person?", state, intent=make_intent(act=ActType.ANSWER))
        assert "QUESTION_CHAIN" not in r.codes()


# ── Robotic phrasing ───────────────────────────────────────────────────

class TestRoboticPhrasing:
    @pytest.mark.parametrize("text", [
        "The cost outweighs the benefits here.",
        "Beach Resort seems like the best fit for everyone.",
        "Ride edges ahead for me because of the playtime.",
        "Do they offer a group discount at Mountain Retreat?",
        "The views are stunning, why that matters is we have seniors.",
        "That's a great question, I hadn't thought of that.",
        "That is a good question actually.",
    ])
    def test_robotic_detected(self, validator, state, text):
        assert "ROBOTIC_TEMPLATE" in _validate(validator, text, state).codes()

    def test_considering_opener_is_repair(self, validator, state):
        r = _validate(validator, "Considering the budget, Mountain Retreat seems expensive.", state)
        assert "ROBOTIC_TEMPLATE" in r.codes()
        assert any(i.severity == "repair" for i in r.issues if i.code == "ROBOTIC_TEMPLATE")

    def test_considering_mid_sentence_not_flagged(self, validator, state):
        r = _validate(validator, "I'm not sure, considering the weather might be an issue.", state)
        assert not any(i.code == "ROBOTIC_TEMPLATE" and i.severity == "repair" for i in r.issues)

    def test_normal_text_passes(self, validator, state):
        assert "ROBOTIC_TEMPLATE" not in _validate(validator, "Five days in the mountains sounds perfect.", state).codes()

    def test_possessive_opener(self, validator, state):
        assert "POSSESSIVE_SUBJECT" in _validate(validator, "Mountain Retreat's fresh air is a big draw.", state).codes()


# ── Collective voice ───────────────────────────────────────────────────

class TestCollectiveVoice:
    def test_fix_we_to_i(self):
        assert fix_collective_voice("We consider this option best.") == "I consider this option best."

    def test_fix_non_collective_untouched(self):
        assert fix_collective_voice("We should split the cost evenly.") == "We should split the cost evenly."


# ── Other checks ───────────────────────────────────────────────────────

class TestOtherChecks:
    def test_self_narration_detected(self, validator, state):
        assert "SELF_NARRATION" in _validate(validator, "I should consider the wine list at Beach Resort.", state).codes()

    def test_unwanted_question_in_vote(self, validator, state):
        r = _validate(validator, "Should we go with Mountain Retreat?", state, intent=make_intent(act=ActType.VOTE))
        assert "UNWANTED_QUESTION" in r.codes()

    def test_unwanted_question_in_answer(self, validator, state):
        r = _validate(validator, "Do they have a kids' menu at Mountain Retreat?", state,
                      intent=make_intent(act=ActType.ANSWER))
        assert "UNWANTED_QUESTION" in r.codes()
        assert any(i.severity == "repair" for i in r.issues if i.code == "UNWANTED_QUESTION")


# ── Severity boundary ──────────────────────────────────────────────────

class TestSeverity:
    def test_style_issues_do_not_block(self, validator, state):
        r = _validate(validator, "I should consider the budget carefully.", state)
        assert r.ok  # warn-level, not repair

    def test_structural_issue_blocks(self, validator, state):
        r = _validate(validator, "Alice: Mountain Retreat sounds nice.", state)
        assert not r.ok


# ── Discourse frames & claim slots ─────────────────────────────────────

class TestDiscourseAndSlots:
    def test_echo_pivot_detected(self):
        assert "echo_pivot" in classify_discourse_frames("Variety is great, but I prefer something simpler.")

    def test_recent_overuse_triggers_hint(self):
        hint = recent_frame_hint(["agreement_preface", "agreement_preface"], [])
        assert "Avoid" in hint

    def test_cost_slot_detected(self):
        assert "cost" in classify_claim_slots("The price is $50 per person.")

    def test_multiple_slots(self):
        slots = classify_claim_slots("The price is low and the atmosphere is cozy.")
        assert "cost" in slots and "comfort" in slots

    def test_covered_slots_hint_fires_at_three(self):
        hint = covered_slots_hint("A", {"cost", "comfort", "quality"})
        assert "cost" in hint or "comfort" in hint or "quality" in hint
        assert "new angle" in hint

    def test_covered_slots_hint_silent_below_threshold(self):
        assert covered_slots_hint("A", {"cost", "comfort"}) == ""

    def test_covered_slots_hint_all_covered(self):
        all_slots = {"cost", "time", "comfort", "flexibility", "quality", "effort", "distance", "group_fit", "novelty", "risk"}
        hint = covered_slots_hint("A", all_slots)
        assert "new detail" in hint
