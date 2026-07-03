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


# --- I14: complete sentences must survive; cuts happen at sentence bounds ---


def test_complete_sentence_mildly_over_budget_kept_whole():
    text = "Pavel's right that print clarity is key, especially with older machines, but Montserrat with Lora still looks fresh to me."
    out = clean_generated(text, "Faye", 18)
    assert out == text


def test_complete_question_mildly_over_budget_kept_whole():
    text = "Pavel, can you share what is still holding you back, or is there another option you would rather push for here?"
    out = clean_generated(text, "Moderator", 20)
    assert out == text


def test_far_over_budget_cut_at_last_sentence_boundary():
    two = (
        "The fixed desk is cheaper and has no motor to break. "
        "The electric one adjusts fast but needs a nearby outlet and costs more per unit overall, "
        "which with five people sharing adds up quickly over a year of heavy daily use."
    )
    out = clean_generated(two, "Nadia", 12)
    assert out == "The fixed desk is cheaper and has no motor to break."


def test_single_runaway_sentence_never_ends_on_modal_or_pronoun():
    text = (
        "I think we should weigh the warranty and the noise level before anything else "
        "because with five people sharing one desk every day the stability though we should"
    )
    out = clean_generated(text, "Priya", 20)
    last = out.rstrip(".!?").split()[-1].lower()
    assert last not in {"should", "we", "though", "you", "the"}
    assert out[-1] in ".!?"


def test_decimal_point_not_treated_as_sentence_end():
    text = (
        "The premium plan costs $4.50 per seat per month and includes the audit trail plus "
        "priority support which the free tier does not offer at all to anyone"
    )
    out = clean_generated(text, "Elif", 14)
    assert not out.rstrip(".").endswith("$4")
    assert "4.50 per" in out or "4.50" not in out
