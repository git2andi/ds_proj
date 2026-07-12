"""Item 7 (todo_validation.md): validator-LLM semantic interpretation.

A stubbed validator returns structured JSON; these tests pin that (a) natural
wording from the item-1 fixture corpus maps to correct typed evidence with NO
endpoint-specific regex growth, (b) deterministic verification rejects
invented spans/options, unverified context resolutions, and critical
commitments that fail the deterministic parser, and (c) validator failures
fail closed with at most one retry.
"""

from __future__ import annotations

import unittest

from interpreter import InterpretationResult, TurnInterpreter, derive_primary_act
from models import ActType, MoveIntent, VisibleEvidence
from tests import semantic_fixtures as sf
from tests.fixtures import make_resolver, make_scenario

NAMES = {"p1": "Mira", "p2": "Jonas", "p3": "Lea"}


class FakeValidatorLLM:
    """Returns scripted JSON payloads (or raises scripted exceptions)."""

    def __init__(self, payloads=None):
        self.payloads = list(payloads or [])
        self.calls = 0
        self.prompts: list[str] = []
        self.last_tokens_in = 0
        self.last_tokens_out = 0

    def generate_json(self, prompt, *, profile="validator"):
        self.calls += 1
        self.prompts.append(prompt)
        self.last_tokens_in = max(1, len(prompt.split()))
        self.last_tokens_out = 5
        item = self.payloads.pop(0) if self.payloads else {}
        if isinstance(item, Exception):
            raise item
        return item


def make_interpreter(payloads=None, mode: str = "selective") -> tuple[TurnInterpreter, FakeValidatorLLM]:
    llm = FakeValidatorLLM(payloads)
    scenario = make_scenario()
    return TurnInterpreter(llm, make_resolver(scenario), scenario, NAMES, mode=mode), llm


class NaturalWordingMapsToEvidence(unittest.TestCase):
    def test_indirect_support_maps_without_regexes(self) -> None:
        fixture = sf.by_id("support_indirect")
        interp, _ = make_interpreter([{
            "supports": [{"option": "A", "strength": "weak",
                          "span": "feels like the easiest day for everyone"}],
            "claims": [{"span": "feels like the easiest day for everyone",
                        "kind": "opinion", "option": "A"}],
            "primary_act": "support",
        }])
        result = interp.interpret(text=fixture.text, speaker_id="p1")
        self.assertEqual(
            [(s.option_id, s.strength) for s in result.evidence.supports],
            [("A", sf.SUPPORT_WEAK)],
        )
        self.assertEqual(result.verification_issues, [])
        self.assertEqual(result.evidence.primary_act, ActType.SUPPORT)

    def test_natural_concern_maps(self) -> None:
        fixture = sf.by_id("concern_natural_hesitant")
        interp, _ = make_interpreter([{
            "concerns": [{"option": "C", "severity": "ordinary",
                          "span": "I'm hesitant about the Escape Room cost"}],
            "primary_act": "concern",
        }])
        result = interp.interpret(text=fixture.text, speaker_id="p1")
        self.assertEqual(
            [(c.option_id, c.severity) for c in result.evidence.concerns],
            [("C", sf.CONCERN_ORDINARY)],
        )
        self.assertEqual(result.verification_issues, [])

    def test_menuless_vote_maps_and_passes_critical_checks(self) -> None:
        fixture = sf.by_id("vote_menuless_backing")  # "I'm backing B."
        interp, _ = make_interpreter([{
            "commitments": [{"kind": "vote", "option": "B", "span": "I'm backing B."}],
            "primary_act": "vote",
        }])
        result = interp.interpret(text=fixture.text, speaker_id="p1")
        self.assertEqual(
            [(c.kind, c.option_id) for c in result.evidence.commitments], [("vote", "B")]
        )
        self.assertEqual(result.verification_issues, [])

    def test_multi_function_switch_turn_keeps_every_evidence_type(self) -> None:
        fixture = sf.by_id("switch_with_reason_multi")
        interp, _ = make_interpreter([{
            "concerns": [{"option": "A", "severity": "ordinary",
                          "span": "I still dislike the Museum's price"}],
            "commitments": [{"kind": "vote", "option": "B",
                             "span": "I'm switching to the Bike Ride"}],
            "switches": [{"source": "A", "target": "B",
                          "reason_span": "because it's cheaper and more flexible",
                          "span": "I'm switching to the Bike Ride"}],
            "questions": [{"scope": "group", "kind": "proposal", "addressee": None,
                           "options": ["B"], "span": "Would that work for everyone?"}],
            "claims": [{"span": "it's cheaper", "kind": "arithmetic", "option": "B",
                        "sources": ["B.cost", "A.cost"]}],
            "primary_act": "vote",
            "intended_move": {"realized": True, "explanation": "committed with reason"},
        }])
        result = interp.interpret(
            text=fixture.text, speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="switch",
                              option_focus=["B"], allow_vote_change=True),
        )
        evidence = result.evidence
        self.assertEqual(result.verification_issues, [])
        self.assertEqual([c.option_id for c in evidence.concerns], ["A"])
        self.assertEqual([c.option_id for c in evidence.commitments], ["B"])
        self.assertEqual(evidence.switches[0].source, "A")
        self.assertIsNotNone(evidence.switches[0].reason_span)
        # The trailing check-in question is detected deterministically.
        self.assertEqual(evidence.questions[0].scope, "group")

    def test_pronoun_concern_resolves_via_public_context(self) -> None:
        fixture = sf.by_id("pronoun_clear")
        interp, _ = make_interpreter([{
            "concerns": [{"option": "C", "severity": "ordinary",
                          "span": "the most expensive one"}],
            "claims": [{"span": "the most expensive one", "kind": "arithmetic",
                        "option": "C", "sources": ["C.cost", "A.cost", "B.cost"]}],
            "primary_act": "concern",
        }])
        result = interp.interpret(
            text=fixture.text, speaker_id="p1", context_candidates=("C",)
        )
        self.assertEqual(result.verification_issues, [])
        self.assertEqual([c.option_id for c in result.evidence.concerns], ["C"])
        context_mentions = [m for m in result.evidence.mentions if m.resolution == "context"]
        self.assertEqual([m.option_id for m in context_mentions], ["C"])


class VerificationRejectsBadProposals(unittest.TestCase):
    def test_invented_span_is_dropped(self) -> None:
        interp, _ = make_interpreter([{
            "supports": [{"option": "A", "strength": "weak", "span": "totally invented words"}],
        }])
        result = interp.interpret(text="The Museum sounds fine.", speaker_id="p1")
        self.assertEqual(result.evidence.supports, [])
        self.assertIn("SPAN_NOT_IN_UTTERANCE:support", result.verification_issues)

    def test_invalid_option_is_dropped(self) -> None:
        interp, _ = make_interpreter([{
            "concerns": [{"option": "Z", "severity": "ordinary", "span": "sounds fine"}],
        }])
        result = interp.interpret(text="The Museum sounds fine.", speaker_id="p1")
        self.assertEqual(result.evidence.concerns, [])
        self.assertIn("INVALID_OPTION:concern:Z", result.verification_issues)

    def test_unverified_context_resolution_is_dropped(self) -> None:
        interp, _ = make_interpreter([{
            "concerns": [{"option": "C", "severity": "ordinary", "span": "It is pricey"}],
        }])
        result = interp.interpret(
            text="It is pricey.", speaker_id="p1", context_candidates=("A", "B")
        )
        self.assertEqual(result.evidence.concerns, [])
        self.assertTrue(any(
            issue.startswith("UNVERIFIED_CONTEXT_RESOLUTION") for issue in result.verification_issues
        ))

    def test_conditional_commitment_fails_the_critical_parser(self) -> None:
        text = "I'd go with the Escape Room only if we can move the booking."
        interp, _ = make_interpreter([{
            "commitments": [{"kind": "vote", "option": "C", "span": text}],
        }])
        result = interp.interpret(text=text, speaker_id="p1")
        self.assertEqual(result.evidence.commitments, [])
        self.assertIn("COMMITMENT_REJECTED:CONDITIONAL_COMMITMENT", result.verification_issues)

    def test_rejected_option_commitment_is_blocked_without_resolution(self) -> None:
        text = "The Escape Room works for me."
        interp, _ = make_interpreter([{
            "commitments": [{"kind": "accept", "option": "C", "span": text}],
        }])
        result = interp.interpret(text=text, speaker_id="p1", rejected_options=("C",))
        self.assertEqual(result.evidence.commitments, [])
        self.assertIn("COMMITMENT_REJECTED:BLOCKED_OPTION_ACCEPTED", result.verification_issues)

    def test_vote_for_unnamed_option_is_rejected(self) -> None:
        # The option binding itself is unverifiable: B is neither named nor a
        # unique public context candidate, so the proposal is dropped at the
        # binding layer (fail-closed) before the critical parser even runs.
        interp, _ = make_interpreter([{
            "commitments": [{"kind": "vote", "option": "B", "span": "count me in"}],
        }])
        result = interp.interpret(text="Alright, count me in.", speaker_id="p1")
        self.assertEqual(result.evidence.commitments, [])
        self.assertIn(
            "UNVERIFIED_CONTEXT_RESOLUTION:commitment:B", result.verification_issues
        )

    def test_vote_via_context_still_requires_explicit_naming(self) -> None:
        # Even with an unambiguous public candidate, kind "vote" must name its
        # option; only soft acceptance may bind through context.
        interp, _ = make_interpreter([{
            "commitments": [{"kind": "vote", "option": "B", "span": "That works for me."}],
        }])
        result = interp.interpret(
            text="That works for me.", speaker_id="p1", context_candidates=("B",)
        )
        self.assertEqual(result.evidence.commitments, [])
        self.assertIn(
            "COMMITMENT_REJECTED:COMMITMENT_TARGET_NOT_NAMED", result.verification_issues
        )

    def test_acceptance_via_unambiguous_context_is_allowed(self) -> None:
        interp, _ = make_interpreter([{
            "commitments": [{"kind": "accept", "option": "B", "span": "That works for me."}],
        }])
        result = interp.interpret(
            text="That works for me.", speaker_id="p1", context_candidates=("B",)
        )
        self.assertEqual(
            [(c.kind, c.option_id) for c in result.evidence.commitments], [("accept", "B")]
        )

    def test_same_line_raise_and_resolve_is_stripped(self) -> None:
        text = "The Escape Room is a dealbreaker, but that fixes my concern."
        interp, _ = make_interpreter([{
            "blockers": [
                {"option": "C", "action": "raised", "span": "a dealbreaker"},
                {"option": "C", "action": "resolved", "span": "that fixes my concern"},
            ],
        }])
        result = interp.interpret(text=text, speaker_id="p1")
        self.assertEqual(result.evidence.blockers, [])
        self.assertIn("BLOCKER_RAISED_AND_RESOLVED:C", result.verification_issues)

    def test_switch_without_commitment_is_dropped(self) -> None:
        text = "Maybe the Bike Ride someday."
        interp, _ = make_interpreter([{
            "switches": [{"source": "A", "target": "B", "span": "Maybe the Bike Ride someday."}],
        }])
        result = interp.interpret(text=text, speaker_id="p1")
        self.assertEqual(result.evidence.switches, [])
        self.assertIn("SWITCH_WITHOUT_COMMITMENT:B", result.verification_issues)

    def test_invalid_primary_act_is_rederived(self) -> None:
        interp, _ = make_interpreter([{
            "supports": [{"option": "A", "strength": "firm", "span": "I really like the Museum"}],
            "primary_act": "celebration",
        }])
        result = interp.interpret(text="I really like the Museum here.", speaker_id="p1")
        self.assertEqual(result.evidence.primary_act, ActType.SUPPORT)


class FailureBehavior(unittest.TestCase):
    def test_operational_failure_fails_closed_after_one_retry(self) -> None:
        interp, llm = make_interpreter([ValueError("bad json"), ValueError("still bad")])
        result = interp.interpret(text="The Museum sounds fine to me.", speaker_id="p1")
        self.assertTrue(result.operational_failure)
        self.assertIsNone(result.evidence)
        self.assertEqual(llm.calls, 2)

    def test_single_retry_recovers(self) -> None:
        interp, llm = make_interpreter([
            ValueError("bad json"),
            {"supports": [{"option": "A", "strength": "weak", "span": "sounds fine"}],
             "primary_act": "support"},
        ])
        result = interp.interpret(text="The Museum sounds fine to me.", speaker_id="p1")
        self.assertFalse(result.operational_failure)
        self.assertEqual(len(result.evidence.supports), 1)
        self.assertEqual(llm.calls, 2)


class DeterministicFastPath(unittest.TestCase):
    def test_minimal_fact_free_vote_skips_the_validator(self) -> None:
        interp, llm = make_interpreter([])
        result = interp.interpret(
            text="I vote for the Bike Ride.", speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="vote",
                              option_focus=["B"]),
        )
        self.assertTrue(result.fast_path)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            [(c.kind, c.option_id) for c in result.evidence.commitments], [("vote", "B")]
        )
        self.assertEqual(result.evidence.primary_act, ActType.VOTE)

    def test_line_with_numbers_goes_to_the_validator(self) -> None:
        interp, llm = make_interpreter([{
            "commitments": [{"kind": "vote", "option": "B",
                             "span": "I vote for the Bike Ride"}],
            "claims": [{"span": "12 euros", "kind": "listed_fact", "option": "B",
                        "sources": ["B.cost"]}],
            "primary_act": "vote",
        }])
        result = interp.interpret(
            text="I vote for the Bike Ride at 12 euros.", speaker_id="p1"
        )
        self.assertFalse(result.fast_path)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(len(result.evidence.commitments), 1)

    def test_rejected_option_never_takes_the_fast_path(self) -> None:
        interp, llm = make_interpreter([{}])
        result = interp.interpret(
            text="I vote for the Escape Room.", speaker_id="p1", rejected_options=("C",)
        )
        self.assertFalse(result.fast_path)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result.evidence.commitments, [])

    def test_sanctioned_switch_with_bridge_takes_the_fast_path(self) -> None:
        interp, llm = make_interpreter([])
        result = interp.interpret(
            text="I vote for the Bike Ride; the Museum was my pick, but I can go with the group here.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="switch",
                              option_focus=["B"], allow_vote_change=True,
                              old_preference="A", required_vote="B"),
        )
        self.assertTrue(result.fast_path)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            [(c.kind, c.option_id) for c in result.evidence.commitments], [("vote", "B")]
        )
        self.assertEqual(result.evidence.switches[0].source, "A")
        self.assertEqual(result.evidence.switches[0].target, "B")

    def test_blocker_restatement_takes_the_fast_path(self) -> None:
        interp, llm = make_interpreter([])
        result = interp.interpret(
            text="The Escape Room still doesn't work for me — the fixed booking is a dealbreaker.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.CONCERN, reason="restate",
                              option_focus=["C"]),
        )
        self.assertTrue(result.fast_path)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            [(b.option_id, b.action) for b in result.evidence.blockers], [("C", "raised")]
        )
        self.assertEqual(
            [(c.option_id, c.severity) for c in result.evidence.concerns], [("C", "hard")]
        )

    def test_process_line_takes_the_fast_path(self) -> None:
        interp, llm = make_interpreter([])
        result = interp.interpret(
            text="Let's each name the one thing that would have to change here.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.PROCESS, reason="procedure"),
        )
        self.assertTrue(result.fast_path)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(result.evidence.commitments, [])
        self.assertEqual(result.evidence.supports, [])

    def test_plain_question_takes_the_fast_path(self) -> None:
        interp, llm = make_interpreter([])
        result = interp.interpret(
            text="Jonas, what do you think about the Museum?", speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.ASK, reason="ask",
                              option_focus=["A"], addressee_id="p2"),
        )
        self.assertTrue(result.fast_path)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            [(q.scope, q.addressee_id) for q in result.evidence.questions], [("direct", "p2")]
        )

    def test_premise_asserting_question_goes_to_the_validator(self) -> None:
        # "Isn't ... closed" embeds a factual premise: the claim audit runs.
        interp, llm = make_interpreter([{}])
        result = interp.interpret(
            text="Isn't the Museum closed on Sundays anyway?", speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.ASK, reason="ask",
                              option_focus=["A"]),
        )
        self.assertFalse(result.fast_path)
        self.assertEqual(llm.calls, 1)

    def test_mention_free_comment_takes_the_fast_path(self) -> None:
        interp, llm = make_interpreter([])
        result = interp.interpret(
            text="Fair point, let's keep this moving.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="light beat"),
        )
        self.assertTrue(result.fast_path)
        self.assertEqual(llm.calls, 0)

    def test_comment_naming_an_option_goes_to_the_validator(self) -> None:
        interp, llm = make_interpreter([{}])
        result = interp.interpret(
            text="The Museum keeps coming up, interesting.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="light beat"),
        )
        self.assertFalse(result.fast_path)
        self.assertEqual(llm.calls, 1)

    def test_clean_comparison_takes_the_fast_path(self) -> None:
        # Item 6: a digit-free two-option comparison whose only content beyond the
        # option names is comparison vocabulary needs no validator call.
        interp, llm = make_interpreter([])
        result = interp.interpret(
            text="The Museum is cheaper while the Bike Ride is longer.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.COMPARE, reason="cmp",
                              option_focus=["A", "B"]),
        )
        self.assertTrue(result.fast_path)
        self.assertEqual(llm.calls, 0)
        self.assertEqual([c.option_ids for c in result.evidence.comparisons], [["A", "B"]])

    def test_comparison_with_a_possible_capability_goes_to_the_validator(self) -> None:
        # A residual concrete noun (a potential invented capability) is not on
        # either card, so grounding still runs.
        interp, llm = make_interpreter([{}])
        result = interp.interpret(
            text="The Museum has a private gallery while the Bike Ride offers valet parking.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.COMPARE, reason="cmp",
                              option_focus=["A", "B"]),
        )
        self.assertFalse(result.fast_path)
        self.assertEqual(llm.calls, 1)

    def test_full_mode_disables_every_fast_path(self) -> None:
        interp, llm = make_interpreter([{}], mode="full")
        result = interp.interpret(
            text="I vote for the Bike Ride.", speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="vote",
                              option_focus=["B"]),
        )
        self.assertFalse(result.fast_path)
        self.assertEqual(llm.calls, 1)
        # The deterministic critical layer still catches the commitment.
        self.assertEqual(
            [(c.kind, c.option_id) for c in result.evidence.commitments], [("vote", "B")]
        )


class IntentSpecificPayload(unittest.TestCase):
    """Item 7: the validator is asked only for the categories the intended
    move needs; unrequested output is ignored, while the deterministic layer
    still catches critical commitments, blockers, and questions on any turn."""

    def test_unrequested_category_output_is_ignored(self) -> None:
        interp, _ = make_interpreter([{
            "supports": [{"option": "A", "strength": "weak", "span": "the Museum sounds fine"}],
            "proposals": [{"option": "A", "span": "the Museum sounds fine"}],  # not requested
        }])
        result = interp.interpret(
            text="Honestly, the Museum sounds fine for a calm day out.", speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support",
                              option_focus=["A"]),
        )
        self.assertEqual(result.requested_categories, ("supports", "concerns"))
        self.assertEqual(len(result.evidence.supports), 1)
        self.assertEqual(result.evidence.proposals, [])

    def test_deterministic_commitment_merged_on_non_vote_turn(self) -> None:
        # Digits force the validator path (grounding); the deterministic
        # layer still contributes the strict commitment on a SUPPORT turn.
        interp, _ = make_interpreter([{}])
        result = interp.interpret(
            text="After everything said today, I vote for the Bike Ride — the 12 euros price seals it.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support",
                              option_focus=["B"]),
        )
        self.assertFalse(result.fast_path)
        self.assertEqual(
            [(c.kind, c.option_id) for c in result.evidence.commitments], [("vote", "B")]
        )

    def test_deterministic_blocker_merged_without_validator_category(self) -> None:
        interp, _ = make_interpreter([{}])
        result = interp.interpret(
            text="The Escape Room is a dealbreaker for me, the booking lock kills it.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.CONCERN, reason="object",
                              option_focus=["C"]),
        )
        self.assertEqual(
            [(b.option_id, b.action) for b in result.evidence.blockers], [("C", "raised")]
        )

    def test_deterministic_question_detected_on_any_turn(self) -> None:
        interp, _ = make_interpreter([{}])
        result = interp.interpret(
            text="Jonas, does the Museum really fill a whole Saturday for us?",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="light beat"),
        )
        self.assertEqual(
            [(q.scope, q.addressee_id) for q in result.evidence.questions], [("direct", "p2")]
        )

    def test_deterministic_comparison_fills_a_missing_validator_comparison(self) -> None:
        # Validator returns no comparison for a clearly comparative line; the
        # deterministic merge supplies the two-option pair (item 3), so a real
        # COMPARE turn is not falsely flagged COMPARISON_MISSES_OPTIONS.
        interp, _ = make_interpreter([{"primary_act": "compare"}])
        result = interp.interpret(
            text="The Museum is cheaper, while the Bike Ride has more variety.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.COMPARE, reason="cmp",
                              option_focus=["A", "B"]),
        )
        self.assertEqual([c.option_ids for c in result.evidence.comparisons], [["A", "B"]])

    def test_two_option_mention_without_comparison_stays_empty(self) -> None:
        interp, _ = make_interpreter([{}])
        result = interp.interpret(
            text="I like the Museum and the Bike Ride.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.COMMENT, reason="c"),
        )
        self.assertEqual(result.evidence.comparisons, [])

    def test_validator_comparison_is_not_overwritten(self) -> None:
        # When the validator already returned a comparison, the deterministic
        # merge leaves it alone (it owns subtle direction/dimension).
        interp, _ = make_interpreter([{
            "comparisons": [{"options": ["B", "A"], "favored": "B",
                             "span": "the Bike Ride beats the Museum on energy"}],
            "primary_act": "compare",
        }])
        result = interp.interpret(
            text="Honestly the Bike Ride beats the Museum on energy, no contest.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.COMPARE, reason="cmp",
                              option_focus=["A", "B"]),
        )
        self.assertEqual(len(result.evidence.comparisons), 1)
        self.assertEqual(result.evidence.comparisons[0].favored, "B")


class PromptContextReduction(unittest.TestCase):
    """Item 9: the validator prompt carries only the options the candidate
    can be about, and only the schema categories the intended move needs."""

    def test_irrelevant_option_cards_are_omitted(self) -> None:
        interp, llm = make_interpreter([{}])
        interp.interpret(
            text="The quiet pace makes the Museum feel genuinely relaxed.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support",
                              option_focus=["A"]),
        )
        prompt = llm.prompts[0]
        self.assertIn("Museum and Cafe Day", prompt)
        self.assertNotIn("Escape Room", prompt)
        self.assertNotIn("Lake Bike Ride", prompt)

    def test_context_candidates_keep_their_cards(self) -> None:
        interp, llm = make_interpreter([{}])
        interp.interpret(
            text="It is pricey though.", speaker_id="p1", context_candidates=("C",),
        )
        self.assertIn("Escape Room", llm.prompts[0])

    def test_schema_only_contains_requested_categories(self) -> None:
        interp, llm = make_interpreter([{}])
        interp.interpret(
            text="The quiet pace makes the Museum feel genuinely relaxed.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support",
                              option_focus=["A"]),
        )
        prompt = llm.prompts[0]
        self.assertIn('"supports"', prompt)
        self.assertIn('"concerns"', prompt)
        self.assertIn('"claims"', prompt)
        self.assertNotIn('"comparisons"', prompt)
        self.assertNotIn('"commitments"', prompt)
        self.assertNotIn('"primary_act"', prompt)
        self.assertNotIn("intended_move", prompt)
        self.assertNotIn("thread_relevant", prompt)  # no thread shown

    def test_thread_relevance_requested_only_with_thread(self) -> None:
        interp, llm = make_interpreter([{}])
        interp.interpret(
            text="The Museum keeps the pace calm for everyone involved.",
            speaker_id="p1",
            intent=MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="support",
                              option_focus=["A"]),
            thread_summary="concern thread about A (cost)",
        )
        self.assertIn("thread_relevant", llm.prompts[0])


class PrimaryActDerivation(unittest.TestCase):
    def test_empty_evidence_is_a_comment(self) -> None:
        self.assertEqual(derive_primary_act(VisibleEvidence()), ActType.COMMENT)


class ResultDefaults(unittest.TestCase):
    def test_result_flags_default_sane(self) -> None:
        result = InterpretationResult(evidence=VisibleEvidence())
        self.assertFalse(result.operational_failure)
        self.assertFalse(result.fast_path)
        self.assertEqual(result.verification_issues, [])


if __name__ == "__main__":
    unittest.main()
