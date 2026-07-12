"""Item 5 (todo_validation.md): consolidated deterministic reference resolution.

One canonical mention scan (option id + exact span + alias form + text order +
explicit/context status), conservative bare-label support, conservative
pronoun resolution against public context only, and single-owner addressee
resolution.
"""

from __future__ import annotations

import unittest

from parsing import OptionResolver, hybrid_blend_detected, resolve_addressee
from tests.fixtures import make_resolver

NAMES = {"p1": "Mira", "p2": "Jonas", "p3": "Lea"}


class MentionScan(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = make_resolver()

    def test_mentions_carry_spans_alias_forms_and_text_order(self) -> None:
        text = "The Bike Ride is cheaper, but the Museum is calmer."
        mentions = self.resolver.mentions(text)
        self.assertEqual([m.option_id for m in mentions], ["B", "A"])
        self.assertEqual([m.order for m in mentions], [0, 1])
        for mention in mentions:
            start = mention.span.start
            self.assertEqual(text[start:start + len(mention.span.text)], mention.span.text)
        self.assertEqual(mentions[0].alias_form, "Bike Ride")
        self.assertEqual(mentions[1].alias_form, "Museum")
        self.assertTrue(all(m.resolution == "explicit" for m in mentions))

    def test_ids_in_text_follows_visible_text_order(self) -> None:
        self.assertEqual(
            self.resolver.ids_in_text("Escape Room first, then the Museum, then the Bike Ride."),
            ["C", "A", "B"],
        )

    def test_option_label_forms_resolve(self) -> None:
        self.assertEqual(self.resolver.ids_in_text("I like Option A here."), ["A"])
        self.assertEqual(self.resolver.ids_in_text("Going with A) for me."), ["A"])
        self.assertEqual(self.resolver.ids_in_text("B has my vote."), ["B"])
        self.assertEqual(self.resolver.ids_in_text("I'm backing C."), ["C"])

    def test_bare_letter_stays_conservative(self) -> None:
        # Lowercase never matches a bare label; "A" is an article, "I" a pronoun.
        self.assertEqual(self.resolver.ids_in_text("b is fine i guess"), [])
        self.assertEqual(self.resolver.ids_in_text("A museum day sounds nice"), ["A"])  # via alias only
        mentions = self.resolver.mentions("A museum day sounds nice")
        self.assertEqual(mentions[0].alias_form.lower(), "museum")

    def test_overlapping_matches_prefer_longest_span(self) -> None:
        mentions = self.resolver.mentions("Option B still wins.")
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].alias_form, "Option B")

    def test_full_names_and_short_names_resolve(self) -> None:
        self.assertEqual(self.resolver.ids_in_text("the Museum and Cafe Day plan"), ["A"])
        self.assertEqual(self.resolver.ids_in_text("the Lake Bike Ride"), ["B"])

    def test_multiple_mentions_of_one_option_keep_all_spans(self) -> None:
        text = "Museum now, Museum later."
        mentions = self.resolver.mentions(text)
        self.assertEqual([m.option_id for m in mentions], ["A", "A"])
        self.assertNotEqual(mentions[0].span.start, mentions[1].span.start)


class ImplicitReferences(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = make_resolver()

    def test_single_public_candidate_resolves(self) -> None:
        option, ambiguous = self.resolver.resolve_reference(
            "It is the most expensive one, that's true.", context_candidates=["C"]
        )
        self.assertEqual(option, "C")
        self.assertFalse(ambiguous)

    def test_two_candidates_stay_ambiguous(self) -> None:
        option, ambiguous = self.resolver.resolve_reference(
            "I think it's the smarter pick.", context_candidates=["A", "B"]
        )
        self.assertIsNone(option)
        self.assertTrue(ambiguous)

    def test_no_candidates_stay_unresolved_without_ambiguity(self) -> None:
        option, ambiguous = self.resolver.resolve_reference(
            "I think it's the smarter pick.", context_candidates=[]
        )
        self.assertIsNone(option)
        self.assertFalse(ambiguous)

    def test_preceding_explicit_mention_wins_over_context(self) -> None:
        option, ambiguous = self.resolver.resolve_reference(
            "The Museum is calm, and it is cheap enough.", context_candidates=["C"]
        )
        self.assertEqual(option, "A")
        self.assertFalse(ambiguous)

    def test_former_and_latter_resolve_from_ordered_pair(self) -> None:
        text = "Between those two, the former seems safer."
        option, _ = self.resolver.resolve_reference(text, context_candidates=["A", "B"])
        self.assertEqual(option, "A")
        option, _ = self.resolver.resolve_reference(
            "The latter looks better to me.", context_candidates=["A", "B"]
        )
        self.assertEqual(option, "B")

    def test_no_implicit_reference_shape_resolves_nothing(self) -> None:
        option, ambiguous = self.resolver.resolve_reference(
            "I think that we should slow down.", context_candidates=["A"]
        )
        self.assertIsNone(option)
        self.assertFalse(ambiguous)

    def test_hidden_intent_is_not_a_source(self) -> None:
        # The API only accepts public candidates; a caller passing nothing
        # public gets no resolution even when a pronoun is present.
        option, ambiguous = self.resolver.resolve_reference(
            "That works for me.", context_candidates=[]
        )
        self.assertIsNone(option)
        self.assertFalse(ambiguous)


class AddresseeResolution(unittest.TestCase):
    def test_visible_name_resolves(self) -> None:
        self.assertEqual(
            resolve_addressee("Jonas, would that be too long?", "p1", NAMES), "p2"
        )

    def test_speaker_own_name_is_ignored(self) -> None:
        self.assertIsNone(resolve_addressee("Mira thinks it's fine.", "p1", NAMES))

    def test_no_name_no_addressee(self) -> None:
        self.assertIsNone(resolve_addressee("Could we all live with it?", "p1", NAMES))


class HybridBlendStillDetected(unittest.TestCase):
    def test_blend_and_non_blend(self) -> None:
        resolver = make_resolver()
        self.assertTrue(hybrid_blend_detected("The Museum and also the Bike Ride?", resolver))
        self.assertFalse(
            hybrid_blend_detected("The Museum is calmer than the Bike Ride.", resolver)
        )


class AliasSafety(unittest.TestCase):
    def test_colliding_aliases_resolve_to_nobody(self) -> None:
        from models import OptionCard
        options = [
            OptionCard(id="A", name="City Park Picnic", short_name="Picnic"),
            OptionCard(id="B", name="Beach Park Picnic", short_name="Beach Picnic"),
        ]
        resolver = OptionResolver(options)
        # "picnic" alone is owned by both options; only unique forms resolve.
        self.assertEqual(resolver.ids_in_text("a picnic sounds nice"), [])
        self.assertEqual(resolver.ids_in_text("the Beach Picnic then"), ["B"])


if __name__ == "__main__":
    unittest.main()
