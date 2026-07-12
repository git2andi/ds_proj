"""Participant-prompt realization contract (todo_new item 3).

The prompt is organized as Voice / Move / Context / Output and must not repeat
information: no full option board when a focus exists, no full stance map for a
single-stance move, no re-quoted target that is already in the recent chat.
"""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

import prompts
from models import ActType, MoveIntent, Phase, TurnRecord, ValidationIssue

from tests.fixtures import append_turn, make_persona, make_state
from tests.stubs import make_runner


def _prompt(state, intent, *, recent=None, focus=None, addressee=None):
    persona = state.persona_by_id(intent.speaker_id)
    return prompts.sim_utterance(
        persona=persona,
        state=state,
        intent=intent,
        recent_lines=recent or [],
        focus_options=focus or [],
        addressee_name=addressee,
        max_words=15,
        min_words=6,
    )


class PromptContract(unittest.TestCase):
    def test_focus_limits_the_option_cards(self):
        state = make_state()
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="support it", option_focus=["B"]
        )
        text = _prompt(state, intent, focus=[state.scenario.option("B")])
        facts = text.split("Allowed facts:")[1].split("Recent chat:")[0]
        self.assertIn("Lake Bike Ride", facts)
        self.assertNotIn("Escape Room", facts)
        self.assertNotIn("Museum and Cafe Day", facts)

    def test_stance_line_is_restricted_to_move_relevant_options(self):
        state = make_state()  # p1 prefers A; B and C exist
        state.runtimes["p1"].set_rank("C", 2, reason_against="too rigid")
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="support it", option_focus=["B"]
        )
        text = _prompt(state, intent, focus=[state.scenario.option("B")])
        stance_line = next(l for l in text.splitlines() if l.startswith("Your stance:"))
        self.assertIn("Museum", stance_line)          # current pick, always shown
        self.assertNotIn("Escape Room", stance_line)  # not part of this move

    def test_initial_preference_appears_only_when_it_differs_from_current(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support it")
        self.assertNotIn("initial preference", _prompt(state, intent))
        state.runtimes["p1"].promote_to_preferred("B")
        state.runtimes["p1"].option_ranks["A"] = 3
        self.assertIn("initial preference Museum", _prompt(state, intent))

    def test_target_inside_recent_window_is_not_requoted_in_full(self):
        state = make_state()
        long_text = (
            "The Bike Ride is cheap and active and honestly the lake route is the "
            "nicest part of the whole plan for a Saturday"
        )
        record = append_turn(state, "p2", long_text)
        recent = [f"{t.speaker_name}: {t.text}" for t in state.turns[-4:]]
        intent = MoveIntent(
            speaker_id="p1",
            act=ActType.SUPPORT,
            reason="respond to it",
            respond_to_turn=record.index,
        )
        text = _prompt(state, intent, recent=recent)
        self.assertIn("Respond to Jonas's last message in the recent chat.", text)
        # The long target text appears exactly once (in the recent block).
        self.assertEqual(text.count("nicest part of the whole plan"), 1)

    def test_target_outside_recent_window_gets_its_own_compact_quote(self):
        state = make_state()
        old = append_turn(state, "p2", "The Bike Ride keeps the cost low for everyone here.")
        for i in range(5):
            append_turn(state, "p3" if i % 2 else "p1", f"Filler message number {i}.")
        recent = [f"{t.speaker_name}: {t.text}" for t in state.turns[-4:]]
        intent = MoveIntent(
            speaker_id="p1",
            act=ActType.SUPPORT,
            reason="respond to it",
            respond_to_turn=old.index,
        )
        text = _prompt(state, intent, recent=recent)
        self.assertIn("Respond to this earlier point — Jonas:", text)

    def test_opening_board_turn_is_not_repeated_as_recent_chat(self):
        state = make_state()
        board_text = prompts.moderator_opening(state.scenario)
        state.turn_index += 1
        state.turns.append(
            TurnRecord(
                index=state.turn_index,
                speaker_id="moderator",
                speaker_name="Moderator",
                text=board_text,
                phase=Phase.OPENING,
            )
        )
        append_turn(state, "p2", "The Bike Ride sounds like the easy pick to me.")
        window = prompts.recent_turn_window(state, 4)
        self.assertEqual([t.speaker_id for t in window], ["p2"])
        intent = MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="light beat")
        recent = [f"{t.speaker_name}: {t.text}" for t in window]
        text = _prompt(state, intent, recent=recent)
        self.assertNotIn("Today we're deciding", text)

    def test_prompt_sections_are_present_in_order(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="light beat")
        text = _prompt(state, intent)
        positions = [text.index(marker) for marker in ("Voice:", "Move:", "Allowed facts:", "Recent chat:", "Output:")]
        self.assertEqual(positions, sorted(positions))

    def test_generation_requests_the_utterance_envelope(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="light beat")
        self.assertIn("<utterance></utterance>", _prompt(state, intent))

    def test_repair_requests_the_utterance_envelope(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="light beat")
        text = prompts.repair_utterance(
            original_text="Something broken.",
            issues=[ValidationIssue(code="MALFORMED_UTTERANCE")],
            persona=state.persona_by_id("p1"),
            state=state,
            recent_lines=[],
            intent=intent,
            max_words=15,
        )
        self.assertIn("<utterance></utterance>", text)


class ActRequirements(unittest.TestCase):
    """Every act carries one concise semantic requirement (todo_new item 4)."""

    _GIST = {
        ActType.OPENING: "current favorite and one grounded reason",
        ActType.SUPPORT: "acknowledgment alone is not support",
        ActType.CONCERN: "an actual objection",
        ActType.ASK: "exactly one genuine, answerable question",
        ActType.ANSWER: "Answer the question first, directly",
        ActType.COMPARE: "at least one real difference or trade-off",
        ActType.COMMENT: "light comment, not an argument",
        ActType.COMPROMISE: "exactly ONE of the existing options",
        ActType.PROCESS: "one concrete procedural suggestion",
        ActType.VOTE: "commitment unambiguous",
    }

    def test_each_act_prompt_states_its_semantic_requirement(self):
        state = make_state()
        for act, gist in self._GIST.items():
            intent = MoveIntent(speaker_id="p1", act=act, reason="do the move", option_focus=["A"])
            text = _prompt(state, intent)
            self.assertIn(gist, text, f"missing requirement for {act.value}")

    def test_vote_turn_still_carries_the_commitment_instruction(self):
        state = make_state()
        intent = MoveIntent(
            speaker_id="p1",
            act=ActType.VOTE,
            reason="cast a clear visible vote",
            option_focus=["A"],
            required_vote="A",
        )
        text = _prompt(state, intent)
        self.assertIn("commit clearly to Museum", text)


class OptionReferenceAndAddressing(unittest.TestCase):
    """No universal placement bans; addressing is optional (item 7)."""

    def test_no_universal_ban_on_option_name_openings(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support it")
        text = _prompt(state, intent)
        self.assertNotIn("do not start with an option name", text)
        self.assertIn("anywhere in the message", text)

    def test_adaptive_suppression_still_produces_the_style_note(self):
        state = make_state()
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="support it",
            suppress_option_opening=True,
        )
        text = _prompt(state, intent)
        self.assertIn("lead with your point this time", text)

    def test_vote_prompt_does_not_force_a_sentence_position(self):
        state = make_state()
        intent = MoveIntent(
            speaker_id="p1", act=ActType.VOTE, reason="cast a clear visible vote",
            option_focus=["A"], required_vote="A",
        )
        text = _prompt(state, intent)
        self.assertNotIn("preferably at the start", text)
        self.assertNotIn("Start with the final vote", text)
        self.assertIn("with the option name right next to it", text)  # parser-visible clarity stays

    def test_addressee_is_an_option_not_a_requirement(self):
        state = make_state()
        intent = MoveIntent(
            speaker_id="p1", act=ActType.SUPPORT, reason="support it", addressee_id="p2"
        )
        text = _prompt(state, intent, addressee="Jonas")
        self.assertIn("Address Jonas if it sounds natural.", text)


class TraitRealization(unittest.TestCase):
    """Trait cues stay compact, glossed, and in their own lanes (item 6)."""

    def _voice_line(self, **params) -> str:
        state = make_state(personas=[
            make_persona("p1", "Mira", **params),
            make_persona("p2", "Jonas", preferred="B"),
        ])
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support it")
        text = _prompt(state, intent)
        return next(l for l in text.splitlines() if l.startswith("Voice:"))

    def test_directness_and_stubbornness_are_glossed_and_scaled(self):
        blunt = self._voice_line(directness=0.95, stubbornness=0.9)
        soft = self._voice_line(directness=0.05, stubbornness=0.1)
        self.assertIn("Directness 5/5", blunt)
        self.assertIn("Stubbornness 5/5", blunt)
        self.assertIn("Directness 1/5", soft)
        self.assertIn("Stubbornness 1/5", soft)
        self.assertIn("blunt plain wording", blunt)
        self.assertIn("concedes slowly", blunt)

    def test_speech_style_is_a_compact_cue_in_the_voice_line(self):
        line = self._voice_line()
        self.assertIn("relaxed practical wording", line)

    def test_engagement_and_switch_resistance_stay_out_of_utterance_prompts(self):
        state = make_state(personas=[
            make_persona("p1", "Mira", engagement=0.95, switch_resistance=0.95),
            make_persona("p2", "Jonas", preferred="B"),
        ])
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support it")
        text = _prompt(state, intent).lower()
        self.assertNotIn("engagement", text)
        self.assertNotIn("switch resistance", text)
        self.assertNotIn("switch_resistance", text)

    def test_verbosity_reaches_the_prompt_only_as_a_numeric_word_range(self):
        state = make_state()
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support it")
        text = _prompt(state, intent)
        self.assertIn("6-15 words", text)  # the passed min/max range
        self.assertNotIn("verbosity", text.lower())


class ModeratorVoteRepairPrompts(unittest.TestCase):
    """Item 9: moderator verbalizes, switch bridges are required, repair keeps intent."""

    def test_switch_vote_prompt_requires_a_visible_bridge(self):
        state = make_state()
        intent = MoveIntent(
            speaker_id="p1", act=ActType.VOTE, reason="final decision: switch",
            option_focus=["B", "A"], required_vote="B", old_preference="A",
            allow_vote_change=True, allowed_reason="active and inexpensive",
        )
        text = _prompt(state, intent)
        self.assertIn("MUST make the change of mind visible", text)
        self.assertIn("active and inexpensive", text)
        self.assertNotIn("You may briefly mention the earlier pick", text)

    def test_moderator_nudge_verbalizes_a_decided_intervention(self):
        state = make_state()
        text = prompts.moderator_nudge_prompt(
            state, "the discussion is circling", "Museum",
            target_name="Jonas", requested_action="ask Jonas for his strongest concern",
        )
        self.assertIn("already been decided", text)
        self.assertIn("Do: ask Jonas for his strongest concern", text)
        self.assertIn("Address: Jonas", text)
        self.assertNotIn("MUCA", text)
        self.assertNotIn("Visible state:", text)

    def test_repair_prompt_preserves_move_voice_and_soft_length(self):
        state = make_state()
        intent = MoveIntent(
            speaker_id="p1", act=ActType.ANSWER, reason="answer it",
            option_focus=["A"], route_source="answer_required",
        )
        text = prompts.repair_utterance(
            original_text="Interesting point about the weekend.",
            issues=[ValidationIssue(code="ANSWER_DOES_NOT_ADDRESS_QUESTION")],
            persona=state.persona_by_id("p1"),
            state=state,
            recent_lines=["Jonas: How does the Museum handle a rainy day?"],
            intent=intent,
            max_words=14,
        )
        self.assertIn("keep its intended move", text)
        self.assertIn("Move: answer.", text)
        self.assertIn("relaxed practical wording", text)
        self.assertIn("around 14 words", text)
        self.assertNotIn("under 14 words", text)
        self.assertIn("answer that question first, directly", text)


class PromptSizeCleanups(unittest.TestCase):
    """Move-irrelevant persona context and duplicate stance lines are dropped (item 8)."""

    def test_vote_answer_process_prompts_omit_background_and_goal(self):
        state = make_state()
        for act in (ActType.VOTE, ActType.ANSWER, ActType.PROCESS):
            intent = MoveIntent(speaker_id="p1", act=act, reason="do it", option_focus=["A"])
            text = _prompt(state, intent)
            self.assertNotIn("Background:", text, act.value)

    def test_content_moves_keep_background_and_goal(self):
        state = make_state()
        for act in (ActType.OPENING, ActType.SUPPORT, ActType.CONCERN, ActType.COMPARE):
            intent = MoveIntent(speaker_id="p1", act=act, reason="do it", option_focus=["A"])
            text = _prompt(state, intent)
            self.assertIn("Background:", text, act.value)

    def test_bare_current_pick_is_not_restated_in_the_stance_list(self):
        state = make_state()  # p1 prefers A with no stored reason
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support", option_focus=["A"])
        text = _prompt(state, intent)
        stance_line = next(l for l in text.splitlines() if l.startswith("Your stance:"))
        self.assertIn("current pick Museum", stance_line)
        self.assertNotIn("Museum=preferred", stance_line)

    def test_current_pick_with_a_reason_still_appears(self):
        state = make_state()
        state.runtimes["p1"].reasons_for["A"] = "calm and cheap"
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support", option_focus=["A"])
        text = _prompt(state, intent)
        self.assertIn("Museum=preferred (calm and cheap)", text)


class ClauseFragments(unittest.TestCase):
    def test_card_text_becomes_a_clause_fragment(self):
        from utils import clause_fragment

        self.assertEqual(
            clause_fragment("Offers a wide variety of dishes.", "Green Garden Bistro"),
            "offers a wide variety of dishes",
        )
        self.assertEqual(
            clause_fragment("Nespresso pods included.", "Nespresso Vertuo Next"),
            "Nespresso pods included",
        )

    def test_hedged_or_long_stored_reasons_are_not_embedded_in_decision_lines(self):
        from utils import usable_reason_fragment

        self.assertEqual(
            usable_reason_fragment("Low overhead and targeted support.", "Fund"),
            "low overhead and targeted support",
        )
        # A whole earlier utterance with a hedge would void the vote parse.
        self.assertEqual(
            usable_reason_fragment("I could live with it, but only if the cost concern is settled somehow."),
            "",
        )
        self.assertEqual(usable_reason_fragment("is it really the best fit for us?"), "")
        self.assertEqual(
            usable_reason_fragment(
                "The scale's precision really improves meal quality and the battery type "
                "is standard so replacements stay easy to find around here."
            ),
            "",
        )


# Merged from test_style_flags.py (item 8): lexical variation flags are part of
# the prompt-realization surface (they shape the generated line's wording).
def _lexical_flags(intent: MoveIntent) -> list[str]:
    flags = []
    if intent.avoid_pattern:
        flags.append("avoid_pattern")
    for name in ("vary_opening", "suppress_option_opening", "suppress_name_prefix",
                 "suppress_i_opening", "suppress_we_opening"):
        if getattr(intent, name):
            flags.append(name)
    return flags


class StyleFlagTests(unittest.TestCase):
    def setUp(self):
        random.seed(3)
        self.state = make_state()
        self.runner = make_runner(self.state)

    def _intent(self, act=ActType.SUPPORT, **kwargs) -> MoveIntent:
        return MoveIntent(speaker_id="p1", act=act, reason="say it", **kwargs)

    def test_fresh_conversation_sets_no_variation_flags(self):
        for _ in range(50):
            intent = self._intent()
            self.runner._apply_style_flags(self.state, intent)
            self.assertEqual(_lexical_flags(intent), [])
            self.assertFalse(intent.suppress_name_prefix)

    def test_multiple_tripwires_yield_exactly_one_note(self):
        for speaker in ("p1", "p2", "p3", "p1"):
            append_turn(self.state, speaker,
                        "I worry the Escape Room is pricey, but the vibe is great.")
        intent = self._intent()
        self.runner._apply_style_flags(self.state, intent)
        self.assertEqual(_lexical_flags(intent), ["avoid_pattern"])

    def test_repeated_opening_word_gets_the_vary_opening_note(self):
        for speaker, text in (
            ("p1", "Honestly the Museum day is fine."),
            ("p2", "Honestly the cost matters more."),
            ("p3", "Honestly we should just decide."),
        ):
            append_turn(self.state, speaker, text)
        intent = self._intent(act=ActType.COMMENT)
        self.runner._apply_style_flags(self.state, intent)
        self.assertEqual(_lexical_flags(intent), ["vary_opening"])

    def test_functional_naming_is_never_suppressed(self):
        for speaker, text in (
            ("p1", "Lea, the Museum keeps it easy."),
            ("p2", "Mira, the cost matters."),
            ("p3", "Jonas, what do you prefer here."),
            ("p1", "Lea, that works."),
        ):
            append_turn(self.state, speaker, text)
        intent = self._intent(act=ActType.ASK, addressee_id="p2")
        self.runner._apply_style_flags(self.state, intent)
        self.assertFalse(intent.suppress_name_prefix)

    def test_tail_question_flag_is_independent_flow_control(self):
        for speaker, text in (
            ("p1", "Is the Museum too quiet for us?"),
            ("p2", "Would the Bike Ride tire anyone out?"),
        ):
            append_turn(self.state, speaker, text)
        intent = self._intent(act=ActType.COMMENT)
        self.runner._apply_style_flags(self.state, intent)
        self.assertTrue(intent.suppress_tail_question)


if __name__ == "__main__":
    unittest.main()
