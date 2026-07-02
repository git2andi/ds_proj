"""Deterministic tests for visible-text parsing and outcome logic (no LLM)."""

from __future__ import annotations

from parsing import OptionResolver, visible_commitment
from models import OptionCard


def _resolver() -> OptionResolver:
    return OptionResolver([
        OptionCard(id="A", name="Amazon Redshift Data Warehouse"),
        OptionCard(id="B", name="ClickHouse Open Source Analytics DB"),
        OptionCard(id="C", name="Google BigQuery Serverless Warehouse"),
        OptionCard(id="D", name="PostgreSQL with TimescaleDB Extension"),
    ])


def test_stopword_is_not_an_alias():
    """'with' (a stopword in option D's name) must not match arbitrary text."""
    r = _resolver()
    assert r.ids_in_text("queries with low maintenance and good support") == []


def test_proper_noun_short_token_matches():
    r = OptionResolver([
        OptionCard(id="C", name="Go 1.20 with Gin Web Framework"),
        OptionCard(id="A", name="Python 3.10 with Django Framework"),
    ])
    assert r.ids_in_text("Go with Gin offers top performance") == ["C"]
    assert r.ids_in_text("Django keeps prototyping fast") == ["A"]


def test_clear_vote_detected():
    r = _resolver()
    assert visible_commitment("I vote for ClickHouse, full stop.", r) == ("vote", "B")


def test_conditional_support_is_not_a_vote():
    r = _resolver()
    assert visible_commitment("I can support BigQuery, but only if costs stay flat.", r) is None


def test_hedged_support_is_not_a_vote():
    r = _resolver()
    assert visible_commitment("I'm leaning toward Redshift, maybe.", r) is None


def test_two_options_named_is_ambiguous():
    r = _resolver()
    assert visible_commitment("Between Redshift and BigQuery I prefer the cheaper one.", r) is None


def test_reject_detected():
    r = _resolver()
    assert visible_commitment("I can't support BigQuery at all.", r) == ("reject", "C")


def test_invalid_option_reference():
    r = _resolver()
    assert r.invalid_option_refs("Let's pick Option E instead") == ["E"]
    assert r.invalid_option_refs("Option B works") == []


def test_natural_vote_phrasings_detected():
    """#16: casual commitments count as direct votes so the moderator need not dictate a formula."""
    r = _resolver()
    assert visible_commitment("I'd go with ClickHouse.", r) == ("vote", "B")
    assert visible_commitment("I'll go with Redshift here.", r) == ("vote", "A")
    assert visible_commitment("I'm going with BigQuery.", r) == ("vote", "C")
    assert visible_commitment("My pick is ClickHouse.", r) == ("vote", "B")


def test_natural_vote_phrasing_still_blocked_by_conditions():
    r = _resolver()
    assert visible_commitment("I'd go with ClickHouse only if the setup is quick.", r) is None
    assert visible_commitment("Unless costs explode, I'd go with BigQuery.", r) is None


def test_all_in_for_counts_as_vote():
    """Observed 2026-07-02 (offsite run): 'I'm all in for X' was a clear commitment but went unparsed."""
    r = _resolver()
    assert visible_commitment("I'm all in for ClickHouse, it fits us.", r) == ("vote", "B")
    assert visible_commitment("Count me in for Redshift.", r) == ("vote", "A")
    assert visible_commitment("I'm in for BigQuery.", r) == ("vote", "C")


def _db_resolver() -> OptionResolver:
    """Option board from the 2026-07-02 unresolved run (logs/20260702_092804_559743)."""
    return OptionResolver([
        OptionCard(id="A", name="Amazon Redshift Data Warehouse", short_name="Redshift"),
        OptionCard(id="B", name="Google BigQuery Serverless Analytics", short_name="BigQuery"),
        OptionCard(id="C", name="PostgreSQL on Single VM", short_name="PostgreSQL VM"),
        OptionCard(id="D", name="ClickHouse Managed Cloud Service", short_name="ClickHouse Cloud"),
    ])


def test_generic_domain_words_are_not_aliases():
    """'analytics' (unique to option B's name) must not match ordinary sentences."""
    r = _db_resolver()
    assert r.ids_in_text("fast, low-maintenance analytics that fit our budget") == []
    assert r.ids_in_text("we need a solid database platform") == []


def test_vote_contraction_detected():
    """Observed 2026-07-02: 'My vote's on X' went unparsed and flipped a run to unresolved."""
    r = _db_resolver()
    assert visible_commitment("My vote's on ClickHouse for fast queries and low maintenance.", r) == ("vote", "D")
    assert visible_commitment("My vote's for Redshift here.", r) == ("vote", "A")
    assert visible_commitment("My vote goes to BigQuery.", r) == ("vote", "B")


def test_unresolved_run_votes_now_parse():
    """The exact lines that were dropped in logs/20260702_092804_559743."""
    r = _db_resolver()
    assert visible_commitment(
        "My vote is ClickHouse for fast, low-maintenance analytics that fit our $120 monthly budget and provide millisecond responses.",
        r,
    ) == ("vote", "D")


def test_grounding_prompt_contains_full_option_board():
    """#18: the fact-checker prompt must carry every option card, not just the focus."""
    import prompts
    from dialogue import initialise_state
    from models import Scenario
    from test_outcomes import _persona

    options = [
        OptionCard(id=x, name=f"Distinct{x} Venue", upside="u", tradeoff="t", concern="c", best_for="b",
                   attrs={"cost": f"${i}0"})
        for i, x in enumerate(["A", "B", "C", "D"])
    ]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    state = initialise_state(scenario, [_persona("p1", "A"), _persona("p2", "B")])
    prompt = prompts.grounding_check(utterance="DistinctA is cheaper than DistinctD.", state=state,
                                     focus_options=list(scenario.options))
    for o in options:
        assert o.name in prompt


def test_extended_commitment_phrasings_detected():
    """#17: common clear commitment forms must parse without a repair round."""
    r = _resolver()
    assert visible_commitment("ClickHouse gets my vote for the speed alone.", r) == ("vote", "B")
    assert visible_commitment("My top choice is Redshift.", r) == ("vote", "A")
    assert visible_commitment("My top pick is BigQuery.", r) == ("vote", "C")
    assert visible_commitment("I'm sold on ClickHouse.", r) == ("vote", "B")
    assert visible_commitment("I'm for Redshift.", r) == ("vote", "A")
    assert visible_commitment("Let's book ClickHouse and move on.", r) == ("vote", "B")
    assert visible_commitment("Let's do BigQuery.", r) == ("vote", "C")


def test_extended_phrasings_still_blocked_when_hedged():
    r = _resolver()
    assert visible_commitment("BigQuery probably gets my vote, but are we okay with the cost?", r) is None
    assert visible_commitment("I'm sold on ClickHouse only if setup stays simple.", r) is None


def test_second_wave_commitment_phrasings_detected():
    """#17 validation exposed more real forms (logs/20260702_1009*/1010*): all must parse now."""
    r = _resolver()
    assert visible_commitment("Redshift's easy access to our stack makes it my choice for a fun setup.", r) == ("vote", "A")
    assert visible_commitment("BigQuery's relaxed pricing is definitely my choice.", r) == ("vote", "C")
    assert visible_commitment("ClickHouse's fast queries get my vote.", r) == ("vote", "B")
    assert visible_commitment("I'm all for ClickHouse, it keeps us moving.", r) == ("vote", "B")
    assert visible_commitment("Redshift works best for me, quick and familiar.", r)[1] == "A"


def test_second_wave_phrasings_blocked_when_hedged():
    r = _resolver()
    assert visible_commitment("BigQuery is maybe my choice, not sure yet.", r) is None
    assert visible_commitment("Redshift works best for me unless the costs jump.", r) is None


def test_group_question_forms_are_genuine():
    """#19: group-directed question forms must create response obligations."""
    from parsing import _is_genuine_question
    assert _is_genuine_question("Are any of us okay with sushi's raw fish focus, or should we skip D?")
    assert _is_genuine_question("Should we worry about the limited menu?")
    assert _is_genuine_question("Is anyone against the earlier start?")
    assert _is_genuine_question("Does anyone actually want the rooftop option?")
    assert _is_genuine_question("Shall we lock in the cabin?")
    assert not _is_genuine_question("That's a great spot, right?")


def test_commitment_object_disambiguates_generic_token_noise():
    """#19 follow-up (logs/20260702_102817_241011): Isla's clear Garden vote was dropped
    because 'neighborhood'/'food' in the reason clause matched other options."""
    r = OptionResolver([
        OptionCard(id="A", name="Community Garden Bed Restoration"),
        OptionCard(id="B", name="Senior Center Tech Assistance"),
        OptionCard(id="C", name="Neighborhood Litter Cleanup Event"),
        OptionCard(id="D", name="Food Bank Sorting Shift"),
    ])
    assert visible_commitment(
        "I'd go with Community Garden Bed Restoration to improve our neighborhood green space and get some physical activity.", r
    ) == ("vote", "A")
    assert visible_commitment(
        "I'm backing Community Garden Bed Restoration for its hands-on impact on local food access and neighborhood green space.", r
    ) is None  # "I'm backing" is intentionally not a commitment phrase yet
    assert visible_commitment(
        "My vote is the litter cleanup for quick visible impact on our community garden street.", r
    ) == ("vote", "C")


def test_commitment_object_keeps_coordinated_pair_ambiguous():
    r = _resolver()
    assert visible_commitment("I'd go with either Redshift or BigQuery, both work.", r) is None
    assert visible_commitment("I'd go with Redshift or BigQuery, whichever is cheaper.", r) is None


def test_comparative_vote_resolves_to_committed_option():
    """'I'd go with X ... better than Y' is a clear X vote for any human reader."""
    r = _resolver()
    assert visible_commitment(
        "I'd go with ClickHouse for fast, affordable analytics, balancing performance and budget better than Redshift.", r
    ) == ("vote", "B")
    assert visible_commitment("ClickHouse gets my vote, it is faster than BigQuery for us.", r) == ("vote", "B")


def test_seems_like_a_pick_is_not_a_commitment():
    """#23: hedged lean from the podcast run must stay latent, not become a visible vote."""
    r = _resolver()
    assert visible_commitment("Redshift seems like a solid pick since it fits our time and keeps us informed.", r) is None
    assert visible_commitment("Sounds like BigQuery is the right pick for us.", r) is None
    # A direct vote with 'seems' elsewhere still counts.
    assert visible_commitment("The pricing seems fine; my vote is ClickHouse.", r) == ("vote", "B")


def test_used_commitment_phrases_detected():
    """#25: phrase families already used in a round are detected for avoidance."""
    from parsing import used_commitment_phrases
    texts = [
        "Count me in for the backyard setup—convenient for everyone.",
        "Keeping it simple matters most, so I'm going with backyard setup.",
        "We'd get the best scenery with the Garden, so that's my pick.",
    ]
    used = used_commitment_phrases(texts)
    assert "count me in for" in used
    assert "I'm going with" in used
    assert "my pick is" in used
    assert "gets my vote" not in used
    assert used_commitment_phrases([]) == []
