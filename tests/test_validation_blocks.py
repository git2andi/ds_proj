from __future__ import annotations

import unittest

import tests  # noqa: F401
from models import ActType, MoveIntent
from models import OptionCard, Scenario
from tests.evidence_adapter import derive_evidence
from tests.fixtures import make_state
from tests.stubs import make_runner


class CriticalValidationTests(unittest.TestCase):
    def setUp(self):
        self.state = make_state()
        self.runner = make_runner(self.state)

    def assess(self, pid: str, text: str, intent: MoveIntent):
        evidence = derive_evidence(
            text, self.runner._resolver, speaker_id=pid,
            participant_names={p.id: p.name for p in self.state.personas}, intent=intent,
        )
        return self.runner._assess_candidate(
            text=text, state=self.state, persona=self.state.persona_by_id(pid),
            intent=intent, evidence=evidence,
        )

    def codes(self, pid: str, text: str, intent: MoveIntent) -> set[str]:
        return {issue.code for issue in self.assess(pid, text, intent).issues}

    def test_empty_and_malformed_output_block(self):
        intent = MoveIntent("p1", ActType.SUPPORT, "support")
        self.assertIn("EMPTY_UTTERANCE", self.codes("p1", "", intent))
        self.assertIn("MALFORMED_UTTERANCE", self.codes("p1", "Just to be clear.", intent))

    def test_invalid_option_and_missing_required_focus_block(self):
        intent = MoveIntent("p1", ActType.SUPPORT, "support")
        self.assertIn("INVALID_OPTION_REFERENCE", self.codes("p1", "Option D looks best.", intent))
        coverage = MoveIntent("p1", ActType.COMPARE, "compare", route_source="coverage", option_focus=["C"])
        self.assertIn("MISSING_REQUIRED_OPTION_FOCUS", self.codes("p1", "The Museum is simple.", coverage))

    def test_required_question_and_comparison_focus(self):
        ask = MoveIntent("p1", ActType.ASK, "ask")
        self.assertIn("QUESTION_REQUIRED", self.codes("p1", "The Museum is calm.", ask))
        compare = MoveIntent("p1", ActType.COMPARE, "compare", option_focus=["A", "B"])
        self.assertIn("MISSING_REQUIRED_OPTION_FOCUS", self.codes("p1", "The Museum is calm.", compare))

    def test_vote_must_be_clear_and_targeted(self):
        vote = MoveIntent("p1", ActType.VOTE, "vote", option_focus=["B"], required_vote="B")
        self.assertIn("UNCLEAR_VISIBLE_COMMITMENT", self.codes("p1", "I'm torn.", vote))
        self.assertIn("REQUIRED_VOTE_MISMATCH", self.codes("p1", "I vote for the Museum.", vote))

    def test_blocked_acceptance_and_same_turn_resolution(self):
        self.state.runtimes["p1"].mark_rejected("C", reason_against="unavailable")
        vote = MoveIntent("p1", ActType.VOTE, "vote", option_focus=["C"], allow_vote_change=True)
        self.assertIn("BLOCKED_OPTION_ACCEPTED", self.codes("p1", "I vote for the Escape Room.", vote))
        self.assertNotIn(
            "BLOCKED_OPTION_ACCEPTED",
            self.codes("p1", "That fixes my concern; I can live with the Escape Room.", vote),
        )

    def test_hybrid_compromise_blocks(self):
        intent = MoveIntent("p1", ActType.COMPROMISE, "compromise")
        self.assertIn(
            "HYBRID_COMPROMISE",
            self.codes("p1", "Let's combine the Museum and the Bike Ride.", intent),
        )



    def test_public_board_exposes_every_fact_available_to_generation(self):
        option = self.state.scenario.option("A")
        option.attrs["booking_window"] = "2h 15m"
        option.upside = "all details remain public and adjustable"
        line = option.public_line()
        card = option.prompt_card()
        self.assertIn("booking window: 2h 15m", line)
        self.assertIn("booking window=2h 15m", card)
        self.assertIn(option.upside, line)

    def test_unlisted_exact_numeric_detail_blocks_but_listed_value_passes(self):
        intent = MoveIntent("p1", ActType.SUPPORT, "support", option_focus=["A"])
        self.assertIn(
            "UNLISTED_NUMERIC_DETAIL",
            self.codes("p1", "The Museum has a 2h15m visit window.", intent),
        )
        self.assertNotIn(
            "UNLISTED_NUMERIC_DETAIL",
            self.codes("p1", "The Museum lasts 4 hours.", intent),
        )

    def test_natural_exact_card_values_are_allowed_and_unlisted_variants_block(self):
        option = self.state.scenario.option("A")
        option.attrs["cost"] = "500€"
        option.attrs["duration"] = "2h"
        intent = MoveIntent("p1", ActType.SUPPORT, "support", option_focus=["A"])
        safe = self.codes(
            "p1", "The Museum costs 500€ and takes 2h, so it still fits the plan.", intent
        )
        self.assertNotIn("UNLISTED_NUMERIC_DETAIL", safe)
        self.assertNotIn("ATTRIBUTE_CONTRADICTION", safe)
        unsafe = self.codes(
            "p1", "The Museum costs 650€ and takes 3h.", intent
        )
        self.assertTrue(
            {"UNLISTED_NUMERIC_DETAIL", "ATTRIBUTE_CONTRADICTION"} & unsafe
        )

    def test_listed_values_pass_with_singular_units_hyphens_and_public_focus(self):
        scenario = Scenario(
            topic="Choose an activity",
            shared_context=["Saturday evening should remain free."],
            options=[
                OptionCard(
                    id="A", name="Museum and Cafe Day", short_name="Museum",
                    attrs={"cost": "24 euros", "duration": "4 hours"},
                    upside="low effort", concern="may feel quiet",
                ),
                OptionCard(
                    id="B", name="Lake Bike Ride", short_name="Bike Ride",
                    attrs={"cost": "12 euros", "duration": "6 hours"},
                    upside="active", concern="bad fit when tired",
                ),
                OptionCard(
                    id="C", name="Home Cooking Night", short_name="Cooking",
                    attrs={"cost": "18 euros", "duration": "5 hours"},
                    upside="flexible", concern="may feel ordinary",
                ),
            ],
        )
        state = make_state(scenario=scenario)
        runner = make_runner(state)
        intent = MoveIntent("p1", ActType.CONCERN, "respond", option_focus=["B"])
        evidence = derive_evidence(
            "True, 6 hours is long, but the 12 euro cost helps.",
            runner._resolver, speaker_id="p1",
            participant_names={p.id: p.name for p in state.personas}, intent=intent,
        )
        assessment = runner._assess_candidate(
            text="True, 6 hours is long, but the 12 euro cost helps.",
            state=state, persona=state.persona_by_id("p1"), intent=intent,
            evidence=evidence,
        )
        self.assertFalse(any(issue.blocking for issue in assessment.issues))
        self.assertEqual(runner._resolver.ids_in_text("Saturday night should stay free."), [])

    def test_categorical_values_bind_to_their_own_attribute(self):
        scenario = Scenario(
            topic="Choose a presentation format",
            shared_context=[],
            options=[
                OptionCard(
                    id="A", name="Live Coding Walkthrough", short_name="Live Coding",
                    attrs={"prep_time": "low", "risk": "high"},
                    upside="authentic", concern="may fail live",
                ),
                OptionCard(
                    id="B", name="Recorded Screencast", short_name="Screencast",
                    attrs={"prep_time": "medium", "risk": "low"},
                    upside="safe", concern="less lively",
                ),
                OptionCard(
                    id="C", name="Slide Deck", short_name="Slides",
                    attrs={"prep_time": "low", "risk": "low"},
                    upside="quick", concern="less convincing",
                ),
            ],
        )
        state = make_state(scenario=scenario)
        runner = make_runner(state)
        issues = runner._deterministic_fact_issues(
            "The low risk and medium prep time with Screencast mean fewer surprises.",
            state,
        )
        self.assertNotIn("CROSS_OPTION_VALUE", {code for code, _option, _why in issues})
        bad = runner._deterministic_fact_issues(
            "Screencast has low prep time and low risk.", state
        )
        self.assertIn("CROSS_OPTION_VALUE", {code for code, _option, _why in bad})

    def test_travel_time_does_not_bind_to_duration_attribute(self):
        scenario = Scenario(
            topic="Choose a retreat",
            shared_context=[],
            options=[
                OptionCard(
                    id="A", name="Nature Hike", short_name="Nature Hike",
                    attrs={"duration_hours": "5", "travel_time_minutes": "45"},
                    upside="outdoors", concern="physically demanding",
                ),
                OptionCard(
                    id="B", name="Cooking Class", short_name="Cooking Class",
                    attrs={"duration_hours": "4", "travel_time_minutes": "20"},
                    upside="shared meal", concern="higher cost",
                ),
                OptionCard(
                    id="C", name="Park Picnic", short_name="Park Picnic",
                    attrs={"duration_hours": "6", "travel_time_minutes": "15"},
                    upside="relaxed", concern="weather dependent",
                ),
            ],
        )
        state = make_state(scenario=scenario)
        runner = make_runner(state)
        for text in (
            "The high activity level and 45-minute travel time make Nature Hike tough.",
            "The high activity level and 5-hour duration make Nature Hike tough.",
        ):
            with self.subTest(text=text):
                issues = runner._deterministic_fact_issues(text, state)
                self.assertNotIn(
                    "ATTRIBUTE_CONTRADICTION",
                    {code for code, _option, _why in issues},
                )
        bad = runner._deterministic_fact_issues(
            "Nature Hike has a 50-minute travel time.", state
        )
        self.assertIn("ATTRIBUTE_CONTRADICTION", {code for code, _option, _why in bad})

    def test_soft_realization_mismatches_do_not_block(self):
        support = MoveIntent("p1", ActType.SUPPORT, "support", option_focus=["A"])
        assessment = self.assess("p1", "The Museum has been discussed already.", support)
        self.assertFalse(any(issue.blocking for issue in assessment.issues))
        self.assertNotIn("SUPPORT_NOT_REALIZED", {i.code for i in assessment.issues})

    def test_unlisted_asserted_feature_and_location_are_blocked(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum has airport lounges.", self.state
        )
        self.assertIn("UNLISTED_FEATURE_DETAIL", {code for code, _option, _why in issues})

        issues = self.runner._deterministic_fact_issues(
            "The Museum is downtown.", self.state
        )
        self.assertIn("UNLISTED_FEATURE_DETAIL", {code for code, _option, _why in issues})





    def test_unlisted_group_seating_claim_is_blocked(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum has easy group seating.", self.state
        )
        self.assertIn("UNLISTED_FEATURE_DETAIL", {code for code, _option, _why in issues})

    def test_listed_group_seating_claim_is_allowed(self):
        self.state.scenario.option("A").attrs["seating"] = "easy group seating"
        issues = self.runner._deterministic_fact_issues(
            "The Museum has easy group seating.", self.state
        )
        self.assertNotIn("UNLISTED_FEATURE_DETAIL", {code for code, _option, _why in issues})

    def test_total_duration_does_not_bind_to_separate_layover_attribute(self):
        option = self.state.scenario.option("A")
        option.attrs["duration"] = "15 hours"
        option.attrs["layover_durations"] = "1h 20m + 1h 10m"
        for text in (
            "The Museum is 15 hours plus two layovers.",
            "The Museum's 15 hour total duration is the longest.",
        ):
            with self.subTest(text=text):
                issues = self.runner._deterministic_fact_issues(text, self.state)
                self.assertNotIn(
                    "ATTRIBUTE_CONTRADICTION",
                    {code for code, _option, _why in issues},
                )

    def test_textual_time_contradiction_against_listed_duration_is_blocked(self):
        option = self.state.scenario.option("A")
        option.attrs["layover_duration"] = "1h 40m"
        bad = self.runner._deterministic_fact_issues(
            "The Museum's layover is under an hour.", self.state
        )
        self.assertIn(
            "ATTRIBUTE_CONTRADICTION", {code for code, _option, _why in bad}
        )
        good = self.runner._deterministic_fact_issues(
            "The Museum's layover is over an hour.", self.state
        )
        self.assertNotIn(
            "ATTRIBUTE_CONTRADICTION", {code for code, _option, _why in good}
        )

    def test_explicit_attribute_value_contradiction_is_blocked(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum costs 60 euros.", self.state
        )
        self.assertIn(
            ("ATTRIBUTE_CONTRADICTION", "A"),
            {(code, option) for code, option, _why in issues},
        )

    def test_correct_attribute_value_with_other_context_number_passes(self):
        issues = self.runner._deterministic_fact_issues(
            "With our 60 euro budget, the Museum costs 24 euros.", self.state
        )
        self.assertNotIn(
            "ATTRIBUTE_CONTRADICTION", {code for code, _option, _why in issues}
        )

    def test_multi_option_comparison_checks_each_unambiguous_local_clause(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum takes 2h15m while the Bike Ride lasts 6 hours.", self.state
        )
        numeric_targets = {
            option for code, option, _why in issues if code == "UNLISTED_NUMERIC_DETAIL"
        }
        self.assertEqual(numeric_targets, {"A"})

    def test_multi_option_comparison_still_catches_transferred_card_value(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum costs 12 euros while the Bike Ride costs 12 euros.", self.state
        )
        transferred = {
            option for code, option, _why in issues if code == "CROSS_OPTION_VALUE"
        }
        self.assertIn("A", transferred)
        self.assertNotIn("B", transferred)

    def test_ambiguous_shared_multi_option_predicate_is_not_guessed(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum and Bike Ride both have airport lounges.", self.state
        )
        self.assertEqual(issues, [])



    def test_listed_feature_with_subjective_adjective_is_not_blocked(self):
        option = self.state.scenario.option("A")
        option.attrs["layover_duration"] = "2h 15m"
        option.upside = "a well-timed short layover"
        for text in (
            "The Museum has a quick layover.",
            "The Museum's layover is solid for a break.",
            "The Museum's connection tightness is still a question.",
        ):
            with self.subTest(text=text):
                issues = self.runner._deterministic_fact_issues(text, self.state)
                self.assertNotIn(
                    "UNLISTED_FEATURE_DETAIL",
                    {code for code, _option, _why in issues},
                )

    def test_reproducible_multi_option_arithmetic_is_not_blocked(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum costs over 10 euros more than the Bike Ride.", self.state
        )
        self.assertNotIn(
            "UNLISTED_NUMERIC_DETAIL", {code for code, _option, _why in issues}
        )

    def test_explicit_unlisted_capability_and_fit_claims_are_blocked(self):
        for text in (
            "The Museum can send real-time alerts.",
            "The Museum fits on a small counter.",
            "The Museum fits the small counter.",
            "The Museum's quick brew might help in the morning.",
            "The Museum's schedule is guaranteed reliable.",
        ):
            with self.subTest(text=text):
                issues = self.runner._deterministic_fact_issues(text, self.state)
                self.assertIn(
                    "UNLISTED_FEATURE_DETAIL",
                    {code for code, _option, _why in issues},
                )



    def test_listed_compactness_can_ground_counter_fit(self):
        option = self.state.scenario.option("A")
        option.upside = "compact footprint and easy setup"
        self.state.scenario.shared_context.append("The counter is small.")
        issues = self.runner._deterministic_fact_issues(
            "The Museum fits the small counter.", self.state
        )
        self.assertNotIn(
            "UNLISTED_FEATURE_DETAIL",
            {code for code, _option, _why in issues},
        )

    def test_possessive_unknown_scope_is_respected(self):
        direct_unknown = self.runner._deterministic_fact_issues(
            "We don't know about the Museum's Wi-Fi reliability here.", self.state
        )
        self.assertNotIn(
            "UNLISTED_FEATURE_DETAIL",
            {code for code, _option, _why in direct_unknown},
        )
        contrast_assertion = self.runner._deterministic_fact_issues(
            "We don't know exact timing here, but the Museum's quick brew might help.",
            self.state,
        )
        self.assertIn(
            "UNLISTED_FEATURE_DETAIL",
            {code for code, _option, _why in contrast_assertion},
        )

    def test_listed_possessive_feature_is_not_blocked(self):
        option = self.state.scenario.option("A")
        option.attrs["brew_time"] = "quick brew"
        option.attrs["live_updates"] = "real-time alerts"
        for text in (
            "The Museum's quick brew might help in the morning.",
            "The Museum's real-time updates would be useful.",
        ):
            with self.subTest(text=text):
                issues = self.runner._deterministic_fact_issues(text, self.state)
                self.assertNotIn(
                    "UNLISTED_FEATURE_DETAIL",
                    {code for code, _option, _why in issues},
                )

    def test_listed_delay_concern_is_not_blocked_but_external_delay_magnitude_is(self):
        option = self.state.scenario.option("A")
        option.concern = "possible layover delays"
        safe = self.runner._deterministic_fact_issues(
            "The Museum could face layover delays.", self.state
        )
        self.assertNotIn(
            "UNLISTED_FEATURE_DETAIL", {code for code, _option, _why in safe}
        )
        unsafe = self.runner._deterministic_fact_issues(
            "The Museum delays could add 3 hours.", self.state
        )
        self.assertIn(
            "UNLISTED_NUMERIC_DETAIL", {code for code, _option, _why in unsafe}
        )

    def test_generic_workability_language_is_not_a_feature_claim(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum can work for our group.", self.state
        )
        self.assertNotIn(
            "UNLISTED_FEATURE_DETAIL", {code for code, _option, _why in issues}
        )

    def test_opinion_and_reasonable_implication_are_not_feature_claims(self):
        issues = self.runner._deterministic_fact_issues(
            "The Museum leaves more room in the budget and feels easier.", self.state
        )
        self.assertNotIn("UNLISTED_FEATURE_DETAIL", {code for code, _option, _why in issues})
        issues = self.runner._deterministic_fact_issues(
            "The Museum has a better overall fit for our group.", self.state
        )
        self.assertNotIn("UNLISTED_FEATURE_DETAIL", {code for code, _option, _why in issues})


if __name__ == "__main__":
    unittest.main()
