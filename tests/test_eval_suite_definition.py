from __future__ import annotations

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from run_eval_suite import CASES, scenario_for  # noqa: E402


def test_evaluation_suite_has_seventeen_cases_ten_topics_and_all_group_sizes():
    assert len(CASES) == 17
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
        "collaboration",
        "capacity",
        "installation",
        "runtime",
        "repairability",
        "insurance",
        "interaction",
        "technical dependence",
    }
    assert expected <= attribute_names


def test_evaluation_suite_has_two_long_diagnostic_cases():
    long_cases = [case for case in CASES if case.deliberation_turn_range is not None]
    assert len(long_cases) == 2
    assert {len(case.preferences) for case in long_cases} == {3, 6}
    assert all(case.voluntary_budget_override is not None for case in long_cases)
