"""Word-cap truncation must not leave mid-sentence fragments. No LLM calls."""

from __future__ import annotations

from utils import clean_generated


def test_short_text_unchanged():
    assert clean_generated("Fine by me.", "Sven", 20) == "Fine by me."


def test_chop_does_not_end_on_conjunction():
    text = "Dashlane's detailed activity logs mean we can track exactly who accessed what and more besides that"
    out = clean_generated(text, "Mina", 14)
    last = out.rstrip(".!?").split()[-1].lower()
    assert last not in {"and", "of", "the", "than", "with", "to"}
    assert out.endswith(".")


def test_chop_does_not_end_on_article_or_preposition():
    text = "LastPass is cheaper and supports Linux but its interface is clunky and past breaches worry me more than the competitors do"
    out = clean_generated(text, "Sven", 19)
    assert not out.rstrip(".").endswith(("the", "than", "of"))
    assert out.endswith(".")


def test_chop_prefers_existing_sentence_boundary():
    text = "We should keep costs low and pick the free tier now. Later we can revisit the paid plan if the team grows a lot"
    out = clean_generated(text, "Rina", 15)
    assert out.endswith(".")
    assert "revisit" not in out or out.endswith(".")


def test_chopped_question_keeps_question_mark():
    text = "If we want a unique challenge that really rewards planning, does the slower setup of Patchwork bother you more than the theme?"
    out = clean_generated(text, "Leo", 18)
    assert out.endswith("?")


def test_chopped_statement_still_ends_with_period():
    text = "The mountain lodge gives us privacy and quiet evenings plus easy trail access even though the drive there is quite long overall"
    out = clean_generated(text, "Rina", 16)
    assert out.endswith(".")


def test_broken_question_tail_removed_and_ends_as_statement():
    """#19: 'but what about those who [prefer ...]' cut mid-clause must not survive as '... who?'."""
    text = "We get fresh seafood and a chic vibe at Sushi Bar, but what about those who prefer their meals fully cooked tonight?"
    out = clean_generated(text, "Xena", 13)
    assert "what about" not in out.lower()
    assert not out.rstrip("?.").endswith(("who", "those", "about"))
    assert out.endswith(".")


def test_still_interrogative_stub_keeps_question_mark():
    text = "Given our weekday schedule and the parking situation downtown, can we handle a slightly longer wait time for the bistro without anyone getting annoyed?"
    out = clean_generated(text, "Thea", 18)
    assert out.endswith("?")


def test_anyone_stub_keeps_question_mark():
    text = "That tiki torch setup sounds a bit intense to manage on our own, anyone worried about the fire risk there?"
    out = clean_generated(text, "Diego", 14)
    assert out.endswith("?")
