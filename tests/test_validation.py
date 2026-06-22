"""Unit tests for src/validation.py — deterministic guardrails."""

from __future__ import annotations

import pytest
from conftest import make_intent, make_turn
from models import ActType, Phase
from parsing import OptionResolver, TurnMove
from validation import (
    MessageValidator,
    _longest_run,
    fix_collective_voice,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def validator(resolver: OptionResolver, participant_names: dict[str, str]) -> MessageValidator:
    return MessageValidator(resolver, participant_names)


def _validate(validator, text, state, intent=None, move=None):
    if intent is None:
        intent = make_intent(act=ActType.REACT)
    if move is None:
        move = TurnMove()
    return validator.validate(text, state, intent, move)


# ── EMPTY ────────────────────────────────────────────────────────────────

class TestEmpty:
    def test_empty_string(self, validator, state):
        r = _validate(validator, "", state)
        assert not r.ok
        assert "EMPTY" in r.codes()

    def test_whitespace_only(self, validator, state):
        r = _validate(validator, "   \n  ", state)
        assert not r.ok
        assert "EMPTY" in r.codes()


# ── SPEAKER_PREFIX ───────────────────────────────────────────────────────

class TestSpeakerPrefix:
    def test_prefix_detected(self, validator, state):
        r = _validate(validator, "Alice: I think Mountain Retreat sounds nice.", state)
        assert "SPEAKER_PREFIX" in r.codes()

    def test_no_prefix(self, validator, state):
        r = _validate(validator, "Mountain Retreat sounds like a solid option.", state)
        assert "SPEAKER_PREFIX" not in r.codes()


# ── OPTION REFS ──────────────────────────────────────────────────────────

class TestOptionRefs:
    def test_valid_option_ref(self, validator, state):
        r = _validate(validator, "Option A is my top pick.", state)
        assert "INVALID_OPTION_REFERENCE" not in r.codes()

    def test_invalid_option_ref(self, validator, state):
        r = _validate(validator, "Option Z looks interesting.", state)
        assert "INVALID_OPTION_REFERENCE" in r.codes()


# ── GROUNDING NUMBERS ────────────────────────────────────────────────────

class TestGroundingNumbers:
    def test_grounded_number_passes(self, validator, state):
        # $800 exists in Mountain Retreat's attrs
        r = _validate(validator, "Mountain Retreat at $800 is a steal.", state)
        assert "INVENTED_OPTION_ATTRIBUTE" not in r.codes()

    def test_invented_number_fails(self, validator, state):
        r = _validate(validator, "Mountain Retreat costs $3500 per person.", state)
        assert "INVENTED_OPTION_ATTRIBUTE" in r.codes()

    def test_no_numbers_passes(self, validator, state):
        r = _validate(validator, "I like the sound of Beach Resort.", state)
        assert "INVENTED_OPTION_ATTRIBUTE" not in r.codes()
        assert "UNGROUNDED_NUMERIC_FACT" not in r.codes()


# ── REPETITION / ECHO GUARD ─────────────────────────────────────────────

class TestRepetition:
    def test_duplicate_turn_detected(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "Mountain Retreat really appeals to me because of the fresh air and the remote location."))
        r = _validate(
            validator,
            "Mountain Retreat really appeals to me because of the fresh air and the remote location.",
            state,
            intent=make_intent(speaker_id="p1"),
        )
        assert "DUPLICATE_TURN" in r.codes()

    def test_self_repetition_detected(self, validator, state):
        state.runtimes["p1"].already_said.append(
            "The cost factor makes Mountain Retreat very attractive for our group budget."
        )
        r = _validate(
            validator,
            "The cost factor makes Mountain Retreat really attractive for our group budget.",
            state,
            intent=make_intent(speaker_id="p1"),
        )
        assert "SELF_REPETITION" in r.codes()

    def test_echoed_phrase_detected(self, validator, state):
        # A 6+ word shared run (with option names masked) should trigger ECHOED_PHRASE
        state.turns.append(make_turn(
            0, "p2", "Bob",
            "I think the fresh air and remote setting really makes it stand out from the rest.",
        ))
        r = _validate(
            validator,
            "The fresh air and remote setting really makes it a winner for sure.",
            state,
            intent=make_intent(speaker_id="p1"),
        )
        assert "ECHOED_PHRASE" in r.codes()

    def test_naming_same_option_not_echoed(self, validator, state):
        # Two turns naming the same option shouldn't trigger echo if phrasing differs
        state.turns.append(make_turn(0, "p2", "Bob", "Mountain Retreat has great value."))
        r = _validate(
            validator,
            "I'd go with Mountain Retreat for the scenery.",
            state,
            intent=make_intent(speaker_id="p1"),
        )
        assert "ECHOED_PHRASE" not in r.codes()


# ── LONGEST RUN (echo guard helper) ─────────────────────────────────────

class TestLongestRun:
    def test_identical(self):
        assert _longest_run(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_partial_overlap(self):
        assert _longest_run(["a", "b", "c", "d"], ["x", "b", "c", "y"]) == 2

    def test_no_overlap(self):
        assert _longest_run(["a", "b"], ["c", "d"]) == 0

    def test_empty(self):
        assert _longest_run([], ["a", "b"]) == 0

    def test_mask_never_matches(self):
        assert _longest_run(["\x00", "b"], ["\x00", "b"]) == 1  # mask breaks the run


# ── OPENER VARIETY ───────────────────────────────────────────────────────

class TestOpenerVariety:
    def test_streak_triggers_repair(self, validator, state):
        # Add turns that all start with "I" to exceed max_consecutive_first_person_openers
        state.turns.append(make_turn(0, "p2", "Bob", "I think Beach Resort is the best."))
        state.turns.append(make_turn(1, "p3", "Carol", "I'm leaning toward City Tour."))
        r = _validate(
            validator,
            "I really want Mountain Retreat.",
            state,
            intent=make_intent(speaker_id="p1", act=ActType.REACT),
        )
        assert "REPETITIVE_OPENER" in r.codes()

    def test_varied_opener_passes(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "I think Beach Resort is the best."))
        state.turns.append(make_turn(1, "p3", "Carol", "That $1200 price tag is steep though."))
        r = _validate(
            validator,
            "I really want Mountain Retreat.",
            state,
            intent=make_intent(speaker_id="p1", act=ActType.REACT),
        )
        assert "REPETITIVE_OPENER" not in r.codes()

    def test_vote_exempt(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "I think Beach Resort is the best."))
        state.turns.append(make_turn(1, "p3", "Carol", "I'm leaning toward City Tour."))
        r = _validate(
            validator,
            "I vote for Mountain Retreat.",
            state,
            intent=make_intent(speaker_id="p1", act=ActType.VOTE),
        )
        assert "REPETITIVE_OPENER" not in r.codes()


# ── DECISION CLARITY ─────────────────────────────────────────────────────

class TestDecisionClarity:
    def test_vote_without_vote_stance(self, validator, state):
        intent = make_intent(act=ActType.VOTE)
        move = TurnMove(present=True, stance="neutral")
        r = _validate(validator, "Mountain Retreat is nice.", state, intent=intent, move=move)
        assert "UNCLEAR_VOTE" in r.codes()

    def test_vote_with_vote_stance(self, validator, state):
        intent = make_intent(act=ActType.VOTE)
        move = TurnMove(present=True, stance="vote", option="A")
        r = _validate(validator, "Mountain Retreat gets my vote.", state, intent=intent, move=move)
        assert "UNCLEAR_VOTE" not in r.codes()

    def test_question_in_confirmation(self, validator, state):
        intent = make_intent(act=ActType.ACCEPT)
        move = TurnMove(present=True, stance="accept", option="A")
        r = _validate(validator, "Mountain Retreat works, but what about parking?", state, intent=intent, move=move)
        assert "QUESTION_IN_CONFIRMATION" in r.codes()


# ── QUESTION CHAIN ───────────────────────────────────────────────────────

class TestQuestionChain:
    def test_chain_detected(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "What about the cost?"))
        state.turns.append(make_turn(1, "p3", "Carol", "Yeah, is $800 per person?"))
        r = _validate(
            validator,
            "Does Mountain Retreat include meals?",
            state,
            intent=make_intent(act=ActType.REACT),
        )
        assert "QUESTION_CHAIN" in r.codes()

    def test_no_chain(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "Mountain Retreat has great reviews."))
        state.turns.append(make_turn(1, "p3", "Carol", "What about the cost?"))
        r = _validate(
            validator,
            "Does it include meals?",
            state,
            intent=make_intent(act=ActType.REACT),
        )
        assert "QUESTION_CHAIN" not in r.codes()

    def test_answer_exempt(self, validator, state):
        state.turns.append(make_turn(0, "p2", "Bob", "What about the cost?"))
        state.turns.append(make_turn(1, "p3", "Carol", "How far is it?"))
        r = _validate(
            validator,
            "I think it's around $800 per person?",
            state,
            intent=make_intent(act=ActType.ANSWER),
        )
        assert "QUESTION_CHAIN" not in r.codes()


# ── COMPLETENESS ─────────────────────────────────────────────────────────

class TestCompleteness:
    def test_dangling_possessive(self, validator, state):
        r = _validate(validator, "compared to Beach Resort's", state)
        assert "INCOMPLETE_TURN" in r.codes()

    def test_dangling_preposition(self, validator, state):
        r = _validate(validator, "that reminds me of", state)
        assert "INCOMPLETE_TURN" in r.codes()

    def test_complete_sentence(self, validator, state):
        r = _validate(validator, "That sounds like a plan.", state)
        assert "INCOMPLETE_TURN" not in r.codes()

    def test_casual_no_period(self, validator, state):
        r = _validate(validator, "yeah totally agree", state)
        assert "INCOMPLETE_TURN" not in r.codes()


# ── UNWANTED QUESTION ────────────────────────────────────────────────────

class TestUnwantedQuestion:
    def test_question_in_vote(self, validator, state):
        intent = make_intent(act=ActType.VOTE)
        r = _validate(validator, "Should we go with Mountain Retreat?", state, intent=intent)
        assert "UNWANTED_QUESTION" in r.codes()

    def test_question_in_opening(self, validator, state):
        intent = make_intent(act=ActType.OPENING)
        r = _validate(validator, "What does everyone think about Mountain Retreat?", state, intent=intent)
        assert "UNWANTED_QUESTION" in r.codes()

    def test_question_in_discussion_ok(self, validator, state):
        intent = make_intent(act=ActType.ASK)
        r = _validate(validator, "What's the wifi like at Mountain Retreat?", state, intent=intent)
        assert "UNWANTED_QUESTION" not in r.codes()


# ── ROBOTIC PHRASING ─────────────────────────────────────────────────────

class TestRoboticPhrasing:
    def test_outweigh_detected(self, validator, state):
        r = _validate(validator, "The cost outweighs the benefits here.", state)
        assert "ROBOTIC_TEMPLATE" in r.codes()

    def test_makes_me_think(self, validator, state):
        r = _validate(validator, "That makes me think we should reconsider.", state)
        assert "ROBOTIC_TEMPLATE" in r.codes()

    def test_point_is_valid(self, validator, state):
        r = _validate(validator, "Your point about cost is valid, but the location is great.", state)
        assert "ROBOTIC_TEMPLATE" in r.codes()

    def test_considering_opener(self, validator, state):
        r = _validate(validator, "Considering our budget, Mountain Retreat wins.", state)
        assert "ROBOTIC_TEMPLATE" in r.codes()

    def test_seems_like_best_fit(self, validator, state):
        r = _validate(validator, "Beach Resort seems like the best fit for everyone.", state)
        assert "ROBOTIC_TEMPLATE" in r.codes()

    def test_given_the_discussion(self, validator, state):
        r = _validate(validator, "Given the discussion, let's go with A.", state)
        assert "ROBOTIC_TEMPLATE" in r.codes()

    def test_since_mentioned(self, validator, state):
        r = _validate(validator, "Since Bob mentioned the budget, I think A works.", state)
        assert "ROBOTIC_TEMPLATE" in r.codes()

    def test_normal_text_passes(self, validator, state):
        r = _validate(validator, "Five days in the mountains sounds perfect.", state)
        assert "ROBOTIC_TEMPLATE" not in r.codes()

    def test_possessive_opener(self, validator, state):
        r = _validate(validator, "Mountain Retreat's fresh air is a big draw.", state)
        assert "POSSESSIVE_SUBJECT" in r.codes()

    def test_possessive_mid_sentence_ok(self, validator, state):
        # The regex allows up to 2 leading words, so "I like <Option>'s" still matches.
        # A genuinely mid-sentence use needs more than 2 words before the option name.
        r = _validate(validator, "But honestly I really like Mountain Retreat's vibe.", state)
        assert "POSSESSIVE_SUBJECT" not in r.codes()


# ── COLLECTIVE VOICE ─────────────────────────────────────────────────────

class TestCollectiveVoice:
    def test_we_consider(self, validator, state):
        r = _validate(validator, "We consider Mountain Retreat the best.", state)
        assert "COLLECTIVE_VOICE" in r.codes()

    def test_we_prioritize(self, validator, state):
        r = _validate(validator, "We prioritize comfort over cost.", state)
        assert "COLLECTIVE_VOICE" in r.codes()

    def test_were_drawn(self, validator, state):
        r = _validate(validator, "We're drawn to the mountain option.", state)
        assert "COLLECTIVE_VOICE" in r.codes()

    def test_we_should_not_flagged(self, validator, state):
        # Collective action framing is fine — only cognition verbs are flagged
        r = _validate(validator, "We should book early to get a discount.", state)
        assert "COLLECTIVE_VOICE" not in r.codes()

    def test_we_need_not_flagged(self, validator, state):
        r = _validate(validator, "We need to decide by Friday.", state)
        assert "COLLECTIVE_VOICE" not in r.codes()

    def test_we_benefit(self, validator, state):
        r = _validate(validator, "We benefit from Green Scene's extra healthy options.", state)
        assert "COLLECTIVE_VOICE" in r.codes()

    def test_we_deserve(self, validator, state):
        r = _validate(validator, "We deserve a place with better food.", state)
        assert "COLLECTIVE_VOICE" in r.codes()


# ── fix_collective_voice ─────────────────────────────────────────────────

class TestFixCollectiveVoice:
    def test_we_to_i(self):
        assert fix_collective_voice("We consider this option best.") == "I consider this option best."

    def test_were_to_im(self):
        result = fix_collective_voice("We're drawn to Mountain Retreat.")
        assert result.startswith("I'm drawn")

    def test_we_value(self):
        result = fix_collective_voice("We value comfort over cost.")
        assert result == "I value comfort over cost."

    def test_non_collective_untouched(self):
        text = "We should split the cost evenly."
        assert fix_collective_voice(text) == text

    def test_our_group_untouched(self):
        text = "Our group prioritizes relaxation."
        assert fix_collective_voice(text) == text


# ── CARD READING ─────────────────────────────────────────────────────────

class TestCardReading:
    def test_verbatim_upside_detected(self, validator, state):
        # Mountain Retreat's upside is "Fresh air" (only 2 words, too short to trigger).
        # Beach Resort's upside is "Relaxing" (1 word). Use a longer card field.
        # Road Trip's tradeoff is "Long drives" (2 words).
        # Let's use the full upside/tradeoff of a card that has 4+ words.
        # Beach Resort: upside="Relaxing", tradeoff="Crowded" — too short.
        # We need to test with the actual option cards from conftest. Let's check:
        # OptionCard(id="A", name="Mountain Retreat", upside="Fresh air", tradeoff="Remote")
        # These are short. We need a test with longer card text.
        # Let's create a custom validator with richer options for this test.
        from models import OptionCard
        from parsing import OptionResolver
        rich_options = [
            OptionCard(id="A", name="Bistro Bliss", attrs={"cost": "$25"},
                       upside="Cozy atmosphere and excellent service",
                       tradeoff="Limited parking options"),
            OptionCard(id="B", name="Tasty Bites", attrs={"cost": "$20"},
                       upside="Quick service and generous portions",
                       tradeoff="Less intimate setting"),
        ]
        rich_resolver = OptionResolver(rich_options)
        rich_validator = MessageValidator(rich_resolver, {"p1": "Alice", "p2": "Bob"})
        r = _validate(rich_validator, "Bistro Bliss has cozy atmosphere and excellent service.", state)
        assert "CARD_READING" in r.codes()

    def test_paraphrased_passes(self, validator, state):
        from models import OptionCard
        from parsing import OptionResolver
        rich_options = [
            OptionCard(id="A", name="Bistro Bliss", attrs={"cost": "$25"},
                       upside="Cozy atmosphere and excellent service",
                       tradeoff="Limited parking options"),
        ]
        rich_resolver = OptionResolver(rich_options)
        rich_validator = MessageValidator(rich_resolver, {"p1": "Alice"})
        r = _validate(rich_validator, "The vibe at Bistro Bliss is really nice and the staff are great.", state)
        assert "CARD_READING" not in r.codes()

    def test_attr_values_exempt(self, validator, state):
        # Referencing factual attr values like "$800" is fine
        r = _validate(validator, "Mountain Retreat at $800 is a steal.", state)
        assert "CARD_READING" not in r.codes()


# ── SELF NARRATION ───────────────────────────────────────────────────────

class TestSelfNarration:
    def test_should_consider(self, validator, state):
        r = _validate(validator, "I should consider the wine list at Beach Resort.", state)
        assert "SELF_NARRATION" in r.codes()

    def test_should_prioritize(self, validator, state):
        r = _validate(validator, "I should prioritize comfort over cost.", state)
        assert "SELF_NARRATION" in r.codes()

    def test_need_to_think_about(self, validator, state):
        r = _validate(validator, "I need to think about what matters most here.", state)
        assert "SELF_NARRATION" in r.codes()

    def test_normal_statement_ok(self, validator, state):
        r = _validate(validator, "Honestly the price is what bugs me.", state)
        assert "SELF_NARRATION" not in r.codes()

    def test_should_in_suggestion_ok(self, validator, state):
        # "We should go with X" is a suggestion, not self-narration
        r = _validate(validator, "We should just pick the cheapest one.", state)
        assert "SELF_NARRATION" not in r.codes()
