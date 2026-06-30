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
