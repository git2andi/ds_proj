from consensus import derive_narrowing_options, majority_threshold, outcome_from_votes
from eval import flat_metrics_for
from logger import metrics_for
from tests.fixtures import make_runner, make_state


def test_majority_threshold_is_strict():
    assert majority_threshold(3) == 2
    assert majority_threshold(4) == 3


def test_complete_tie_does_not_invent_leader():
    state = make_state(("A", "B", "C"))
    for runtime in state.runtimes.values():
        runtime.public_preference = runtime.preferred_option
    assert derive_narrowing_options(state) == ()


def test_vote_outcomes_are_deterministic():
    state = make_state(("A", "A", "B"))
    majority = outcome_from_votes(state, {"p1": "A", "p2": "A", "p3": "B"}, allow_unresolved=True)
    assert majority.status == "majority" and majority.final_option == "A"
    unresolved = outcome_from_votes(state, {"p1": "A", "p2": "B", "p3": "C"}, allow_unresolved=True)
    assert unresolved.status == "unresolved"


def test_core_metrics_and_flat_adapter():
    result = make_runner(("A", "A", "B"), seed=20).run()
    metrics = metrics_for(result.state, result.outcome)
    flat = flat_metrics_for(result.state, result.outcome)
    assert metrics["participant_turns"] == len(result.state.participant_turns)
    assert flat["outcome"] == result.outcome.status
    assert flat["vote_outcome_consistent"] is True
    assert result.state.public_point_counts
    assert len(result.state.recent_point_keys) <= 2
