from __future__ import annotations

from consensus import derive_narrowing_options, outcome_from_votes
from tests.fixtures import make_state


def _set_preferences(state, values):
    for pid, option in zip(state.runtimes, values):
        state.runtimes[pid].public_preference = option


def test_unique_public_leader():
    state = make_state(("A", "A", "B"))
    _set_preferences(state, ("A", "A", "B"))
    assert derive_narrowing_options(state) == ("A",)


def test_exact_top_pair():
    state = make_state(("A", "A", "B", "B"))
    _set_preferences(state, ("A", "A", "B", "B"))
    assert derive_narrowing_options(state) == ("A", "B")


def test_complete_tie_does_not_fabricate_finalists():
    state = make_state(("A", "B", "C", "D"))
    _set_preferences(state, ("A", "B", "C", "D"))
    assert derive_narrowing_options(state) == ()


def test_majority_acceptability_is_common_ground_fallback():
    state = make_state(("A", "B", "C"))
    _set_preferences(state, ("A", "B", "C"))
    for runtime in state.runtimes.values():
        runtime.public_acceptances.add("B")
    assert derive_narrowing_options(state) == ("B",)


def test_majority_closes_without_unanimity_repair():
    state = make_state(("A", "A", "B"))
    outcome = outcome_from_votes(state, {"p1": "A", "p2": "A", "p3": "B"}, allow_unresolved=False)
    assert outcome and outcome.status == "majority" and outcome.final_option == "A"


def test_no_majority_requires_revote_or_unresolved():
    state = make_state(("A", "B", "C"))
    assert outcome_from_votes(state, {"p1": "A", "p2": "B", "p3": "C"}, allow_unresolved=False) is None
    outcome = outcome_from_votes(state, {"p1": "A", "p2": "B", "p3": "C"}, allow_unresolved=True)
    assert outcome and outcome.status == "unresolved"
