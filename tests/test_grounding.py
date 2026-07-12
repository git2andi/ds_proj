"""Item 8 (todo_validation.md): claim-level grounding verification.

The FactTable verifies the exact option-attribute-value relationship — not
word membership. Listed facts pass; opinions, uncertainty, and qualified
inferences pass without becoming facts; reproducible arithmetic passes;
invented details and cross-option transfers fail; a concrete value that
conflicts with a card is a contradiction; and a soft label may not smuggle an
unreproducible number or a contradicting attribute value past verification.
"""

from __future__ import annotations

import unittest

from interpreter import FactTable, TurnInterpreter
from models import EvidenceSpan, GroundingClaim, OptionCard, Scenario
from tests.fixtures import make_resolver, make_scenario


def claim(
    span: str, kind: str, option: str | None = None, sources: list[str] | None = None,
    attribute: str | None = None, value: str | None = None,
) -> GroundingClaim:
    return GroundingClaim(
        span=EvidenceSpan(text=span, start=0), kind=kind, option_id=option,
        attribute=attribute, value=value, source_facts=list(sources or []),
    )


class ListedFacts(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = FactTable(make_scenario())

    def test_exact_listed_value_passes(self) -> None:
        supported, _ = self.facts.verify(claim("costs 24 euros", "listed_fact", "A"))
        self.assertTrue(supported)

    def test_number_word_matches_listed_digit(self) -> None:
        supported, _ = self.facts.verify(claim("takes about four hours", "listed_fact", "A"))
        self.assertTrue(supported)

    def test_value_from_another_option_fails_with_owner_named(self) -> None:
        supported, reason = self.facts.verify(claim("costs 32 euros", "listed_fact", "B"))
        self.assertFalse(supported)
        self.assertIn("belongs to option C", reason)

    def test_unlisted_value_fails(self) -> None:
        supported, reason = self.facts.verify(claim("costs 99 euros", "listed_fact", "A"))
        self.assertFalse(supported)
        self.assertIn("not listed", reason)

    def test_shared_context_fact_passes(self) -> None:
        supported, _ = self.facts.verify(claim("the 60 euro budget", "listed_fact", None))
        self.assertTrue(supported)

    def test_capacity_example_from_the_todo(self) -> None:
        scenario = Scenario(
            topic="Choose a venue",
            shared_context=["The whole department attends."],
            options=[
                OptionCard(id="A", name="Grand Hall", short_name="Grand Hall",
                           attrs={"capacity": "200 people"}),
                OptionCard(id="B", name="Studio Loft", short_name="Studio Loft",
                           attrs={"capacity": "100 people"}),
            ],
        )
        facts = FactTable(scenario)
        supported, reason = facts.verify(claim("B holds 200 people", "listed_fact", "B"))
        self.assertFalse(supported)
        self.assertIn("belongs to option A", reason)
        self.assertTrue(facts.verify(claim("B holds 100 people", "listed_fact", "B"))[0])


class OpinionsInferenceUncertainty(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = FactTable(make_scenario())

    def test_opinion_always_passes(self) -> None:
        self.assertTrue(self.facts.verify(claim("sounds the most fun to me", "opinion", "C"))[0])

    def test_uncertainty_always_passes(self) -> None:
        self.assertTrue(self.facts.verify(
            claim("not sure the budget stretches to dinner", "uncertainty", None)
        )[0])

    def test_opinion_with_listed_number_passes(self) -> None:
        # A subjective judgment that cites a genuinely listed value is fine.
        self.assertTrue(self.facts.verify(
            claim("32 euros feels steep to me", "opinion", "C")
        )[0])

    def test_opinion_label_cannot_smuggle_an_unlisted_number(self) -> None:
        # Item 10: a broad opinion label must not bypass an embedded concrete
        # premise — the invented "45 euros" fails even under kind "opinion".
        supported, reason = self.facts.verify(
            claim("45 euros just to get in seems unfair", "opinion", "C")
        )
        self.assertFalse(supported)
        self.assertIn("not reproducible", reason)

    def test_inference_with_unlisted_number_fails(self) -> None:
        supported, reason = self.facts.verify(claim(
            "so roughly 50 euros saved overall", "inference", "B", sources=["B.cost"]
        ))
        self.assertFalse(supported)
        self.assertIn("not reproducible", reason)

    def test_inference_with_valid_sources_passes(self) -> None:
        supported, _ = self.facts.verify(claim(
            "probably leaves the evening free", "inference", "C", sources=["C.duration"]
        ))
        self.assertTrue(supported)

    def test_qualified_inference_without_sources_passes(self) -> None:
        # Item 5: a hedged conclusion is not rejected merely because it does not
        # enumerate its source facts and its words are not verbatim on a card.
        self.assertTrue(self.facts.verify(claim("clearly the best value", "inference", "B"))[0])

    def test_qualified_inference_from_a_listed_cost_passes(self) -> None:
        # "the cheaper option may be easier on the budget" is a reasonable
        # qualified inference from listed cost facts.
        self.assertTrue(
            self.facts.verify(claim("may be easier on the budget", "inference", "B"))[0]
        )

    def test_opinion_cannot_smuggle_a_contradicting_attribute_value(self) -> None:
        # Item 5: an opinion label must not hide a concrete option-attribute
        # value that conflicts with the card. The Museum's (A) listed cost is 24;
        # asserting cost 32 (a real, derivable number, but C's) via the
        # structured fields is a direct contradiction, not a hedge.
        supported, reason = self.facts.verify(claim(
            "32 euros feels steep for the Museum", "opinion", "A",
            attribute="cost", value="32 euros",
        ))
        self.assertFalse(supported)
        self.assertIn("contradicts listed", reason)

    def test_context_source_bounds_are_checked(self) -> None:
        self.assertTrue(self.facts.source_exists("context:0"))
        self.assertTrue(self.facts.source_exists("context:1"))
        self.assertFalse(self.facts.source_exists("context:9"))
        self.assertFalse(self.facts.source_exists("Z.cost"))


class Arithmetic(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = FactTable(make_scenario())

    def test_reproducible_difference_passes(self) -> None:
        # 32 (C) - 12 (B) = 20
        self.assertTrue(self.facts.verify(claim("20 euros cheaper", "arithmetic", "B"))[0])

    def test_group_total_passes(self) -> None:
        # 3 people x 12 euros = 36
        self.assertTrue(self.facts.verify(claim("36 euros for all of us", "arithmetic", "B"))[0])

    def test_non_reproducible_number_fails(self) -> None:
        supported, reason = self.facts.verify(claim("saves 37 euros", "arithmetic", "B"))
        self.assertFalse(supported)
        self.assertIn("not reproducible", reason)


class AlwaysUnsupportedKinds(unittest.TestCase):
    def test_invented_cross_option_and_ungrounded_fail(self) -> None:
        facts = FactTable(make_scenario())
        for kind in ("invented_detail", "cross_option_transfer", "ungrounded_inference", "contradiction"):
            supported, reason = facts.verify(claim("free entry on Saturdays", kind, "A"))
            self.assertFalse(supported, kind)
            self.assertTrue(reason, kind)


class ContradictionDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = FactTable(make_scenario())

    def test_listed_fact_with_conflicting_attribute_value_is_a_contradiction(self) -> None:
        # A "listed_fact" that names cost but the WRONG number for that option is
        # a direct contradiction of the card, not a merely-absent value (item 5).
        supported, reason = self.facts.verify(claim(
            "the Museum runs 12 euros", "listed_fact", "A",
            attribute="cost", value="12 euros",
        ))
        self.assertFalse(supported)
        self.assertIn("contradicts listed", reason)

    def test_matching_attribute_value_still_passes(self) -> None:
        self.assertTrue(self.facts.verify(claim(
            "the Museum costs 24 euros", "listed_fact", "A",
            attribute="cost", value="24 euros",
        ))[0])


class GroundingRunsInsideInterpretation(unittest.TestCase):
    def test_interpret_marks_claims_supported_and_unsupported(self) -> None:
        class FakeValidator:
            last_tokens_in = last_tokens_out = 1

            def generate_json(self, prompt, *, profile="validator"):
                return {
                    "claims": [
                        {"span": "costs 24 euros", "kind": "listed_fact", "option": "A"},
                        {"span": "free entry on Saturdays", "kind": "invented_detail", "option": "A"},
                    ],
                    "primary_act": "comment",
                }

        scenario = make_scenario()
        interp = TurnInterpreter(FakeValidator(), make_resolver(scenario), scenario,
                                 {"p1": "Mira"})
        result = interp.interpret(
            text="The Museum costs 24 euros and has free entry on Saturdays.",
            speaker_id="p1",
        )
        by_span = {c.span.text: c for c in result.evidence.claims}
        self.assertTrue(by_span["costs 24 euros"].supported)
        self.assertFalse(by_span["free entry on Saturdays"].supported)
        self.assertTrue(by_span["free entry on Saturdays"].reason)


if __name__ == "__main__":
    unittest.main()
