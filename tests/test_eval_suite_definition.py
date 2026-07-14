from __future__ import annotations

from eval.run_eval_suite import CASES, scenario_for


def test_evaluation_suite_has_fifteen_cases_ten_topics_and_all_group_sizes():
    assert len(CASES) == 15
    assert len({case.scenario_key for case in CASES}) == 10
    assert {len(case.preferences) for case in CASES} == {2, 3, 4, 5, 6, 7}


def test_evaluation_topics_expose_varied_public_attributes():
    attribute_names = {
        key
        for case in CASES
        for option in scenario_for(case.scenario_key).options
        for key in option.attrs
    }
    expected = {
        "accessibility",
        "weather exposure",
        "equipment",
        "privacy",
        "dietary coverage",
        "difficulty",
        "repairability",
        "refundability",
        "interaction",
        "technical dependence",
    }
    assert expected <= attribute_names
