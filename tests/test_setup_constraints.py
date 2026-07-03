"""I6: hard numeric caps in shared context are extracted and enforced against
option attributes. No LLM calls."""

from __future__ import annotations

from builders import enforce_shared_caps, shared_context_caps
from models import OptionCard, Scenario


def _scenario(shared_context, attrs_by_option):
    options = [
        OptionCard(id=oid, name=f"Option {oid} Name", attrs=dict(attrs))
        for oid, attrs in attrs_by_option.items()
    ]
    return Scenario(
        topic="t", decision_kind="generic_decision", opening_question="q",
        options=options, shared_context=list(shared_context),
    )


# --- extraction ---

def test_fixed_budget_cap_extracted():
    caps = shared_context_caps(["The budget per child for the activity week is fixed at $300"])
    assert len(caps) == 1
    assert caps[0]["kind"] == "money" and caps[0]["value"] == 300.0
    assert caps[0]["canon"] == 300.0 and caps[0]["per"] == "child"


def test_soft_budget_is_not_a_cap():
    assert shared_context_caps(["The painting budget is moderate, around $200 total."]) == []


def test_thousands_separator_parsed():
    caps = shared_context_caps(["The budget for the mural is capped at $10,000."])
    assert caps[0]["value"] == 10000.0 and caps[0]["per"] is None


def test_distance_cap_extracted():
    caps = shared_context_caps(["The venue must be within 10 miles of the office."])
    assert caps[0]["kind"] == "distance" and caps[0]["value"] == 10.0
    assert abs(caps[0]["canon"] - 16.09344) < 1e-6


def test_hour_cap_normalized_to_minutes():
    caps = shared_context_caps(["The movie must be under 2 hours."])
    assert caps[0]["kind"] == "duration" and caps[0]["canon"] == 120.0


# --- enforcement ---

def test_violating_cost_clamped_to_cap():
    scenario = _scenario(
        ["The budget per child is fixed at $300"],
        {"A": {"cost per child": "$300"}, "D": {"cost per child": "$320"}},
    )
    notes = enforce_shared_caps(scenario)
    assert scenario.option("D").attrs["cost per child"] == "$300"
    assert scenario.option("A").attrs["cost per child"] == "$300"  # untouched (already at cap)
    assert len(notes) == 1 and "D" in notes[0]


def test_per_basis_mismatch_is_skipped():
    scenario = _scenario(
        ["Budget is capped at $500 total for food"],
        {"A": {"cost per person": "$12"}, "B": {"cost per person": "$40"}},
    )
    assert enforce_shared_caps(scenario) == []
    assert scenario.option("B").attrs["cost per person"] == "$40"


def test_bare_number_cost_clamped():
    scenario = _scenario(
        ["Our budget for the synth is no more than $800"],
        {"A": {"cost": "900"}},
    )
    notes = enforce_shared_caps(scenario)
    assert scenario.option("A").attrs["cost"] == "800"
    assert notes


def test_distance_attr_clamped():
    scenario = _scenario(
        ["The spot must be within 10 miles of the office"],
        {"A": {"distance from center": "15 miles"}, "B": {"distance from center": "4 miles"}},
    )
    enforce_shared_caps(scenario)
    assert scenario.option("A").attrs["distance from center"] == "10 miles"
    assert scenario.option("B").attrs["distance from center"] == "4 miles"


def test_non_matching_kinds_untouched():
    scenario = _scenario(
        ["The budget is fixed at $300"],
        {"A": {"duration": "320 minutes", "rating": "4.8/5"}},
    )
    assert enforce_shared_caps(scenario) == []
    assert scenario.option("A").attrs["duration"] == "320 minutes"


# --- I15: unit normalization + report-only mode ---


def test_hour_cap_clamps_minutes_attr_with_unit_in_key():
    """The movie-night failure: 'under 2 hours' vs 'duration_minutes: 130'."""
    scenario = _scenario(
        ["The movie must be under 2 hours"],
        {"A": {"duration_minutes": "95"}, "D": {"duration_minutes": "130"}},
    )
    notes = enforce_shared_caps(scenario)
    assert scenario.option("D").attrs["duration_minutes"] == "120"
    assert scenario.option("A").attrs["duration_minutes"] == "95"
    assert len(notes) == 1 and "D" in notes[0]


def test_minute_cap_clamps_hours_attr():
    scenario = _scenario(
        ["Setup must take no more than 90 minutes"],
        {"A": {"setup time": "2 hours"}},
    )
    enforce_shared_caps(scenario)
    assert scenario.option("A").attrs["setup time"] == "1.5 hours"


def test_mile_cap_clamps_km_attr_floored():
    scenario = _scenario(
        ["The venue must be within 10 miles of the office"],
        {"A": {"distance": "20 km"}},
    )
    enforce_shared_caps(scenario)
    # 10 miles = 16.09 km; floored to one decimal so the clamp never exceeds the cap.
    assert scenario.option("A").attrs["distance"] == "16 km"


def test_scoped_walking_cap_ignores_wait_time():
    """Live false positive: a 15-minute *walking* cap must not clamp a wait time."""
    scenario = _scenario(
        ["We want a brunch spot within 15 minutes walking distance from the station"],
        {"A": {"average_wait_time": "20 minutes", "distance_from_station": "18 minutes walk"}},
    )
    notes = enforce_shared_caps(scenario)
    assert scenario.option("A").attrs["average_wait_time"] == "20 minutes"
    assert scenario.option("A").attrs["distance_from_station"] == "15 minutes walk"
    assert len(notes) == 1 and "distance_from_station" in notes[0]


def test_unscoped_duration_cap_still_binds_broadly():
    caps = shared_context_caps(["The movie must be under 2 hours"])
    assert caps[0]["scope"] is None


def test_report_only_mode_does_not_mutate():
    scenario = _scenario(
        ["The movie must be under 2 hours"],
        {"D": {"duration_minutes": "130"}},
    )
    notes = enforce_shared_caps(scenario, mutate=False)
    assert scenario.option("D").attrs["duration_minutes"] == "130"
    assert len(notes) == 1 and "violates cap" in notes[0]
