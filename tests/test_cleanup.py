"""Generated-text cleanup: label/quote/metadata removal without length clipping.

todo_new item 2: word budgets are prompt-side generation targets. Cleanup must
never cut a returned utterance at a word boundary, produce a sliced fragment,
or append punctuation to make a slice look complete.
"""

from __future__ import annotations

import random
import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from controller.policy import PolicyMixin
from models import ActType, MoveIntent
from utils import clean_generated

from tests.fixtures import make_persona


class CleanupPreservesCompleteUtterances(unittest.TestCase):
    def test_over_target_complete_sentence_stays_complete(self):
        text = (
            "I really think the Museum and Cafe Day is the safest pick for all three "
            "of us because it stays inside the budget, keeps the afternoon flexible, "
            "and still leaves plenty of time to rest before the evening plans we made."
        )
        self.assertEqual(clean_generated(text, "Mira"), text)

    def test_long_unpunctuated_output_is_not_word_sliced(self):
        words = ["word%d" % i for i in range(80)]
        text = " ".join(words)
        cleaned = clean_generated(text, "Mira")
        self.assertEqual(cleaned.split(), words)

    def test_no_punctuation_is_appended_to_make_a_fragment_look_complete(self):
        text = "the bike ride is cheap and the museum is calm and the escape room is"
        cleaned = clean_generated(text, "Mira")
        self.assertEqual(cleaned, text)

    def test_multi_sentence_over_target_output_keeps_every_sentence(self):
        text = (
            "The Bike Ride is the cheapest option on the board. It also gets everyone "
            "outside for a few hours. I know one of us is tired, but we can keep the "
            "pace easy and stop at the lake cafe halfway through the loop."
        )
        self.assertEqual(clean_generated(text, "Jonas"), text)


class CleanupStillNormalizes(unittest.TestCase):
    def test_speaker_prefix_and_duplicated_prefix_are_stripped(self):
        self.assertEqual(clean_generated("Mira: fine by me.", "Mira"), "fine by me.")
        self.assertEqual(clean_generated("Mira: Mira: fine by me.", "Mira"), "fine by me.")

    def test_surrounding_quotes_and_metadata_trailer_are_removed(self):
        self.assertEqual(
            clean_generated('"The museum works for me. [act=support]"', "Mira"),
            "The museum works for me.",
        )

    def test_generic_filler_tail_is_removed(self):
        self.assertEqual(
            clean_generated("The museum keeps the day easy. What do you think?", "Mira"),
            "The museum keeps the day easy.",
        )

    def test_newlines_collapse_to_one_line(self):
        self.assertEqual(
            clean_generated("The museum\nworks for me.", "Mira"),
            "The museum works for me.",
        )


class WordBudgetsStaySoft(unittest.TestCase):
    """Verbosity shapes the requested budget distribution, never a hard cut."""

    def _budgets(self, verbosity: float, n: int = 400) -> list[int]:
        persona = make_persona("p1", "Mira", verbosity=verbosity)
        intent = MoveIntent(speaker_id="p1", act=ActType.SUPPORT, reason="say something")
        random.seed(11)
        return [PolicyMixin._word_bounds(intent, persona)[1] for _ in range(n)]

    def test_high_verbosity_still_draws_occasional_short_budgets(self):
        budgets = self._budgets(0.9)
        self.assertTrue(any(b <= 9 for b in budgets))
        self.assertGreater(sum(budgets) / len(budgets), 12)

    def test_low_verbosity_budgets_are_shorter_on_average_but_never_enforced(self):
        low = self._budgets(0.1)
        high = self._budgets(0.9)
        self.assertLess(sum(low) / len(low), sum(high) / len(high))
        # The budget is only ever a prompt target: cleanup keeps a complete
        # sentence far above any drawn budget.
        long_line = (
            "Even though I usually keep it short, this one time I want to explain "
            "properly why the Escape Room does not fit the plan we agreed on earlier."
        )
        self.assertEqual(clean_generated(long_line, "Mira"), long_line)


if __name__ == "__main__":
    unittest.main()
