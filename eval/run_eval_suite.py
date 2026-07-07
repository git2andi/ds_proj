#!/usr/bin/env python3
r"""
Run a sequential evaluation suite for the option-grounded multi-user simulator.

Usage from the project root:

    py .\eval\run_eval_suite.py

The script temporarily overwrites config.yaml for each case, runs main.py, and
restores the original config.yaml at the end, even if a run fails or you stop it.

It writes all generated logs under eval/logs_eval_suite/ so they are separated from
normal interactive runs and kept with the evaluation scripts.
"""

from __future__ import annotations

import copy
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Child transcripts contain unicode (e.g. the "−" minus sign in option boards).
# When stdout is a cp1252 console or a pipe, re-printing them would crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Your project already uses yaml in config_loader.py, "
        "so run this inside the same virtualenv you use for the simulator."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MAIN_PATH = PROJECT_ROOT / "main.py"
SUITE_LOG_DIR = PROJECT_ROOT / "eval" / "logs_eval_suite"
SUMMARY_CSV = SUITE_LOG_DIR / "eval_suite_runs.csv"


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge patch into base and return base."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_config(data: dict[str, Any]) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)


def base_patch(
    *,
    seed: int,
    n: int,
    env_mode: str,
    participants_mode: str,
    forced_shape: str | None = None,
    moderator: dict[str, bool] | None = None,
    max_turns_per_participant: float | None = None,
    grounding_mode: str = "tripwire",
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "simulation": {
            "num_participants": n,
            "random_seed": seed,
        },
        "environment": {
            "mode": env_mode,
        },
        "participants": {
            "mode": participants_mode,
        },
        "personas": {
            "preference_distribution": {
                "forced_shape": forced_shape,
            }
        },
        "validation": {
            "grounding_check": True,
            "grounding_mode": grounding_mode,
        },
        "output": {
            "log_dir": "eval/logs_eval_suite",
            "write_prompts": False,
        },
        "limits": {
            "warn_total_input_tokens": 25000,
        },
    }
    if env_mode == "auto":
        patch["environment"]["manual"] = {}
    if participants_mode == "auto":
        patch["participants"]["profiles"] = []
    if moderator is not None:
        patch["moderator"] = moderator
    if max_turns_per_participant is not None:
        patch["conversation"] = {
            "min_discussion_turns_per_participant": max(2.0, max_turns_per_participant - 2.0),
            "target_discussion_turns_per_participant": max(3.0, max_turns_per_participant - 1.0),
            "max_discussion_turns_per_participant": max_turns_per_participant,
        }
    return patch


def option(
    option_id: str,
    name: str,
    attrs: dict[str, str],
    upside: str,
    tradeoff: str,
    concern: str,
    best_for: str,
    short_name: str | None = None,
) -> dict[str, Any]:
    data = {
        "id": option_id,
        "name": name,
        "attrs": attrs,
        "upside": upside,
        "tradeoff": tradeoff,
        "concern": concern,
        "best_for": best_for,
    }
    if short_name:
        data["short_name"] = short_name
    return data


RESTAURANT_ENV = {
    "topic": "Choose a restaurant for a mixed-preference group dinner",
    "decision_kind": "restaurant_choice",
    "opening_question": "Which restaurant gives the best compromise between price, dietary fit, travel time, and atmosphere?",
    "shared_context": [
        "The group wants dinner this Friday after work.",
        "The budget target is around 25 euros per person.",
        "One participant prefers vegetarian-friendly choices.",
    ],
    "options": [
        option(
            "A",
            "Corner Ramen",
            {"price": "18 euros", "travel": "10 minutes by tram", "vegetarian": "two vegetarian bowls"},
            "warm, quick, and easy to organize",
            "limited non-soup options",
            "may not feel special enough for everyone",
            "a low-effort dinner with predictable timing",
            "Ramen",
        ),
        option(
            "B",
            "La Piazza",
            {"price": "26 euros", "travel": "18 minutes by bus", "vegetarian": "several pasta and pizza options"},
            "broad menu and relaxed atmosphere",
            "slightly above the target budget",
            "can become noisy on Fridays",
            "a familiar compromise for mixed tastes",
            "Piazza",
        ),
        option(
            "C",
            "Green Table",
            {"price": "24 euros", "travel": "20 minutes walking", "vegetarian": "mostly vegetarian menu"},
            "best dietary fit and calm setting",
            "less appealing for people wanting meat dishes",
            "some may see it as too niche",
            "a dietary-safe and quieter dinner",
            "Green Table",
        ),
        option(
            "D",
            "Burger Cellar",
            {"price": "21 euros", "travel": "8 minutes walking", "vegetarian": "one vegetarian burger"},
            "closest and casual",
            "weakest vegetarian variety",
            "may not satisfy the dietary requirement well",
            "a cheap, easy, informal choice",
            "Burger Cellar",
        ),
    ],
}

WEEKEND_ENV = {
    "topic": "Choose a weekend activity for three friends with different energy levels",
    "decision_kind": "activity_choice",
    "opening_question": "Which activity balances cost, effort, travel, and enough flexibility for everyone?",
    "shared_context": [
        "The group only has Saturday available.",
        "Nobody wants to spend more than 60 euros.",
        "The plan should leave time to rest in the evening.",
    ],
    "options": [
        option(
            "A",
            "Museum and Cafe Day",
            {"cost": "24 euros", "travel": "15 minutes by subway", "duration": "4 hours"},
            "low effort and easy to adjust",
            "less exciting for active participants",
            "may feel too quiet",
            "a relaxed low-risk day",
            "Museum",
        ),
        option(
            "B",
            "Lake Bike Ride",
            {"cost": "12 euros", "travel": "25 minutes by train", "duration": "6 hours"},
            "active and inexpensive",
            "physically demanding",
            "bad fit for someone tired",
            "a cheap outdoor activity",
            "Bike Ride",
        ),
        option(
            "C",
            "Escape Room",
            {"cost": "32 euros", "travel": "20 minutes by tram", "duration": "2 hours"},
            "interactive and memorable",
            "shorter than a full-day plan",
            "less flexible once booked",
            "a focused group activity",
            "Escape Room",
        ),
        option(
            "D",
            "Home Cooking Night",
            {"cost": "18 euros", "travel": "none", "duration": "5 hours"},
            "cheapest and most flexible",
            "requires planning and cleanup",
            "may feel too ordinary",
            "a flexible low-cost option",
            "Cooking",
        ),
    ],
}

COFFEE_ENV = {
    "topic": "Choose a coffee machine for a small shared office kitchen",
    "decision_kind": "purchase_choice",
    "opening_question": "Which machine is the best fit for price, reliability, maintenance, and daily throughput?",
    "shared_context": [
        "The maximum budget is 320 euros.",
        "The kitchen counter is small.",
        "About ten people use the machine on office days.",
    ],
    "options": [
        option(
            "A",
            "Moccamaster KBG Select",
            {"cost": "299 euros", "type": "filter", "capacity": "10 cups"},
            "reliable for shared pots",
            "no espresso or milk drinks",
            "highest upfront cost",
            "teams that drink regular filter coffee",
            "Moccamaster",
        ),
        option(
            "B",
            "DeLonghi Dedica",
            {"cost": "179 euros", "type": "espresso", "capacity": "single shots"},
            "compact and good for espresso",
            "slower for many users",
            "requires more hands-on use",
            "small espresso-focused teams",
            "Dedica",
        ),
        option(
            "C",
            "Philips Senseo Switch",
            {"cost": "119 euros", "type": "pads and filter", "capacity": "7 cups"},
            "flexible and cheap",
            "pad waste and weaker taste",
            "less premium build",
            "mixed casual use on a budget",
            "Senseo",
        ),
        option(
            "D",
            "Ninja Filter Brewer",
            {"cost": "149 euros", "type": "filter", "capacity": "12 cups"},
            "large capacity for the price",
            "larger footprint",
            "less compact than the others",
            "high-volume budget coffee",
            "Ninja",
        ),
    ],
}

ROOMMATE_ENV = {
    "topic": "Choose whether two roommates should upgrade cleaning at home",
    "decision_kind": "purchase_choice",
    "opening_question": "Which purchase best addresses the shared cleaning problem without creating a new burden?",
    "shared_context": [
        "Two roommates share the apartment costs equally.",
        "The maximum budget is 450 euros.",
        "Both want less weekly cleaning friction.",
    ],
    "options": [
        option(
            "A",
            "Eufy Robot Vacuum",
            {"cost": "260 euros", "task": "daily floor cleaning", "space": "needs clear floor paths"},
            "reduces visible dust without manual effort",
            "does not help with dishes or kitchen cleanup",
            "can get stuck if the floor is cluttered",
            "someone who mainly worries about floors",
            "Robot Vacuum",
        ),
        option(
            "B",
            "Bosch Compact Dishwasher",
            {"cost": "430 euros", "task": "daily dishes", "space": "uses counter space"},
            "removes the most common kitchen chore",
            "near the top of the budget and takes space",
            "does not help with dust or floors",
            "someone who mainly worries about dishes",
            "Dishwasher",
        ),
        option(
            "C",
            "Monthly Cleaning Service Trial",
            {"cost": "80 euros per month", "task": "general cleaning", "duration": "three-month trial"},
            "covers several chores without buying equipment",
            "recurring cost instead of a one-time purchase",
            "depends on scheduling someone to come in",
            "testing whether outside help reduces conflict",
            "Cleaning Trial",
        ),
        option(
            "D",
            "Shared Cleaning Supplies Kit",
            {"cost": "70 euros", "task": "manual cleaning", "storage": "small closet box"},
            "cheap and easy to start immediately",
            "still requires both roommates to do the work",
            "may not change habits enough",
            "a low-cost reset before buying a device",
            "Supplies Kit",
        ),
    ],
}


def profiles_three_way() -> list[dict[str, Any]]:
    """Three distinct initial favorites; designed to catch premature split-vote closure."""
    return [
        {
            "name": "Mira",
            "description": "organized planner who cares about broad fit and avoiding awkward logistics",
            "private_goal": "wants the option that works for most people without needing extra coordination",
            "preferred_option": "A",
            "traits": {"openness": 3, "conscientiousness": 5, "extraversion": 3, "agreeableness": 3, "neuroticism": 2},
            "parameters": {
                "engagement": 0.55,
                "verbosity": 0.55,
                "initiative": 0.55,
                "responsiveness": 0.75,
                "stubbornness": 0.45,
                "directness": 0.50,
                "compromise_threshold": 0.45,
            },
        },
        {
            "name": "Jonas",
            "description": "practical budget-watcher who changes his mind when a concrete tradeoff is convincing",
            "private_goal": "wants the group to avoid overspending but can accept a better compromise",
            "preferred_option": "B",
            "traits": {"openness": 3, "conscientiousness": 4, "extraversion": 2, "agreeableness": 4, "neuroticism": 2},
            "parameters": {
                "engagement": 0.35,
                "verbosity": 0.35,
                "initiative": 0.30,
                "responsiveness": 0.80,
                "stubbornness": 0.35,
                "directness": 0.45,
                "compromise_threshold": 0.35,
            },
        },
        {
            "name": "Lea",
            "description": "high-energy participant who likes memorable choices and often drives the conversation forward",
            "private_goal": "wants the group to choose something that feels worth the effort",
            "preferred_option": "C",
            "traits": {"openness": 5, "conscientiousness": 3, "extraversion": 5, "agreeableness": 3, "neuroticism": 2},
            "parameters": {
                "engagement": 0.90,
                "verbosity": 0.80,
                "initiative": 0.90,
                "responsiveness": 0.60,
                "stubbornness": 0.55,
                "directness": 0.70,
                "compromise_threshold": 0.55,
            },
        },
    ]


def profiles_trait_spread_4() -> list[dict[str, Any]]:
    """One very active sim, one quiet sim, two middle sims."""
    return [
        {
            "name": "Nora",
            "description": "very engaged organizer who notices process problems and proposes next steps",
            "private_goal": "wants a clear decision and tends to keep the group moving",
            "preferred_option": "B",
            "traits": {"openness": 4, "conscientiousness": 4, "extraversion": 5, "agreeableness": 3, "neuroticism": 1},
            "parameters": {
                "engagement": 0.95,
                "verbosity": 0.85,
                "initiative": 0.95,
                "responsiveness": 0.70,
                "stubbornness": 0.35,
                "directness": 0.75,
                "compromise_threshold": 0.40,
            },
        },
        {
            "name": "Tarek",
            "description": "quiet participant who answers when asked but rarely pushes himself into the discussion",
            "private_goal": "wants the simplest acceptable choice and avoids long arguments",
            "preferred_option": "D",
            "traits": {"openness": 2, "conscientiousness": 3, "extraversion": 1, "agreeableness": 4, "neuroticism": 3},
            "parameters": {
                "engagement": 0.15,
                "verbosity": 0.25,
                "initiative": 0.10,
                "responsiveness": 0.65,
                "stubbornness": 0.30,
                "directness": 0.35,
                "compromise_threshold": 0.30,
            },
        },
        {
            "name": "Eva",
            "description": "balanced participant who weighs concrete constraints before moving position",
            "private_goal": "wants the option with the fewest hidden tradeoffs",
            "preferred_option": "A",
            "traits": {"openness": 3, "conscientiousness": 5, "extraversion": 3, "agreeableness": 3, "neuroticism": 2},
            "parameters": {
                "engagement": 0.55,
                "verbosity": 0.55,
                "initiative": 0.45,
                "responsiveness": 0.75,
                "stubbornness": 0.50,
                "directness": 0.55,
                "compromise_threshold": 0.45,
            },
        },
        {
            "name": "Sam",
            "description": "socially flexible participant who often bridges between opposing preferences",
            "private_goal": "wants the final choice to feel acceptable to everyone",
            "preferred_option": "C",
            "traits": {"openness": 4, "conscientiousness": 3, "extraversion": 3, "agreeableness": 5, "neuroticism": 2},
            "parameters": {
                "engagement": 0.50,
                "verbosity": 0.50,
                "initiative": 0.50,
                "responsiveness": 0.85,
                "stubbornness": 0.20,
                "directness": 0.40,
                "compromise_threshold": 0.25,
            },
        },
    ]


def profiles_hard_holdout_4() -> list[dict[str, Any]]:
    """One stubborn minority, useful for bounded reservation/compromise testing."""
    return [
        {
            "name": "Clara",
            "description": "detail-focused participant who will not accept weak dietary fit",
            "private_goal": "wants the option that clearly protects the dietary requirement",
            "preferred_option": "C",
            "traits": {"openness": 3, "conscientiousness": 5, "extraversion": 2, "agreeableness": 2, "neuroticism": 3},
            "parameters": {
                "engagement": 0.55,
                "verbosity": 0.55,
                "initiative": 0.40,
                "responsiveness": 0.70,
                "stubbornness": 0.85,
                "directness": 0.75,
                "compromise_threshold": 0.80,
            },
        },
        {
            "name": "Ben",
            "description": "cost-conscious participant who likes broad, familiar compromises",
            "private_goal": "wants a safe group choice that does not exceed the budget too much",
            "preferred_option": "B",
            "traits": {"openness": 3, "conscientiousness": 4, "extraversion": 3, "agreeableness": 4, "neuroticism": 2},
            "parameters": {
                "engagement": 0.60,
                "verbosity": 0.50,
                "initiative": 0.55,
                "responsiveness": 0.75,
                "stubbornness": 0.35,
                "directness": 0.50,
                "compromise_threshold": 0.40,
            },
        },
        {
            "name": "Iris",
            "description": "active social organizer who prefers easy logistics and broad menus",
            "private_goal": "wants the group to settle on a practical choice without dragging the debate out",
            "preferred_option": "B",
            "traits": {"openness": 4, "conscientiousness": 3, "extraversion": 5, "agreeableness": 4, "neuroticism": 1},
            "parameters": {
                "engagement": 0.85,
                "verbosity": 0.70,
                "initiative": 0.85,
                "responsiveness": 0.70,
                "stubbornness": 0.30,
                "directness": 0.65,
                "compromise_threshold": 0.35,
            },
        },
        {
            "name": "Omar",
            "description": "relaxed participant who usually follows a reasonable majority",
            "private_goal": "wants a choice that avoids obvious inconvenience",
            "preferred_option": "B",
            "traits": {"openness": 3, "conscientiousness": 3, "extraversion": 2, "agreeableness": 5, "neuroticism": 2},
            "parameters": {
                "engagement": 0.40,
                "verbosity": 0.35,
                "initiative": 0.25,
                "responsiveness": 0.80,
                "stubbornness": 0.20,
                "directness": 0.35,
                "compromise_threshold": 0.25,
            },
        },
    ]


def profiles_stubborn_deadlock_2() -> list[dict[str, Any]]:
    """Two stubborn opposing participants; designed to force the 1-1 protocol."""
    return [
        {
            "name": "Maja",
            "description": "stubborn roommate who thinks floors are the visible problem and dislikes spending the full budget",
            "private_goal": "wants the robot vacuum and does not want a counter-space appliance",
            "preferred_option": "A",
            "rejection": "B",
            "rejection_reason": "it is near the budget limit and uses counter space",
            "traits": {"openness": 2, "conscientiousness": 4, "extraversion": 2, "agreeableness": 1, "neuroticism": 4},
            "parameters": {
                "engagement": 0.55,
                "verbosity": 0.45,
                "initiative": 0.45,
                "responsiveness": 0.55,
                "stubbornness": 0.95,
                "directness": 0.80,
                "compromise_threshold": 0.90,
            },
        },
        {
            "name": "Felix",
            "description": "stubborn roommate who thinks dishes cause most conflict and distrusts partial floor-only fixes",
            "private_goal": "wants the dishwasher and does not want a device that ignores the kitchen mess",
            "preferred_option": "B",
            "rejection": "A",
            "rejection_reason": "it does not help with dishes or kitchen cleanup",
            "traits": {"openness": 2, "conscientiousness": 4, "extraversion": 3, "agreeableness": 1, "neuroticism": 3},
            "parameters": {
                "engagement": 0.60,
                "verbosity": 0.50,
                "initiative": 0.55,
                "responsiveness": 0.55,
                "stubbornness": 0.95,
                "directness": 0.85,
                "compromise_threshold": 0.90,
            },
        },
    ]


MOD_FULL = {
    "enabled": True,
    "opening": True,
    "mid_discussion_nudges": True,
    "final_vote_call": True,
    "closing": True,
}

MOD_NONE = {
    "enabled": False,
    "opening": False,
    "mid_discussion_nudges": False,
    "final_vote_call": False,
    "closing": False,
}

MOD_LIGHT = {
    "enabled": True,
    "opening": True,
    "mid_discussion_nudges": False,
    "final_vote_call": False,
    "closing": True,
}


CASES: list[dict[str, Any]] = [
    {
        "id": "q01_manual_manual_three_way_split",
        "suite": "quick",
        "why": "Catches the problem where A/B/C final votes close immediately without narrowing.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=101, n=3, env_mode="manual", participants_mode="manual", moderator=MOD_FULL),
            {
                "environment": {"manual": WEEKEND_ENV},
                "participants": {"profiles": profiles_three_way()},
            },
        ),
    },
    {
        "id": "q02_manual_manual_no_moderator_peer_process",
        "suite": "quick",
        "why": "Checks whether no-moderator mode can still narrow with participant probes/summaries and visible votes.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=102, n=4, env_mode="manual", participants_mode="manual", moderator=MOD_NONE),
            {
                "environment": {"manual": RESTAURANT_ENV},
                "participants": {"profiles": profiles_hard_holdout_4()},
            },
        ),
    },
    {
        "id": "q03_manual_manual_trait_spread",
        "suite": "quick",
        "why": "Checks whether high/low engagement changes turn share and word length visibly.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=103, n=4, env_mode="manual", participants_mode="manual", moderator=MOD_FULL),
            {
                "environment": {"manual": COFFEE_ENV},
                "participants": {"profiles": profiles_trait_spread_4()},
            },
        ),
    },
    {
        "id": "q04_manual_env_auto_participants_three_way",
        "suite": "quick",
        "why": "Manual option board with auto sims; tests controlled environment plus generated users.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=104, n=3, env_mode="manual", participants_mode="auto", forced_shape="1-1-1", moderator=MOD_FULL),
            {"environment": {"manual": RESTAURANT_ENV}},
        ),
    },
    {
        "id": "q05_auto_env_manual_participants",
        "suite": "quick",
        "why": "Auto option board with fixed sim traits; tests generated environment plus controlled users.",
        "topic": "Choose a team-building activity for a small software team with mixed energy levels",
        "patch": deep_merge(
            base_patch(seed=105, n=3, env_mode="auto", participants_mode="manual", moderator=MOD_FULL),
            {"participants": {"profiles": profiles_three_way()}},
        ),
    },
    {
        "id": "q06_auto_auto_baseline_n3",
        "suite": "quick",
        "why": "Normal default-style auto/auto run for regression comparison.",
        "topic": "Choose a birthday gift for a friend who likes practical things but already owns a lot",
        "patch": base_patch(seed=106, n=3, env_mode="auto", participants_mode="auto", forced_shape="1-1-1", moderator=MOD_FULL),
    },
    {
        "id": "f01_manual_manual_n2_stubborn_deadlock",
        "suite": "full",
        "why": "Forced two-person 1-1 split with stubborn manual participants; should trigger deadlock protocol.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=201, n=2, env_mode="manual", participants_mode="manual", forced_shape="1-1", moderator=MOD_FULL),
            {
                "environment": {"manual": ROOMMATE_ENV},
                "participants": {"profiles": profiles_stubborn_deadlock_2()},
            },
        ),
    },
    {
        "id": "f02_auto_auto_n5_trait_distribution",
        "suite": "full",
        "why": "Five-person auto/auto run; checks whether routing scales beyond n=3/4.",
        "topic": "Choose a venue for a student project celebration with different budgets and travel times",
        "patch": base_patch(seed=202, n=5, env_mode="auto", participants_mode="auto", forced_shape="2-1-1-1", moderator=MOD_FULL),
    },
    {
        "id": "f03_manual_manual_light_moderator",
        "suite": "full",
        "why": "Moderator opens/closes, but participants should handle final narrowing.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=203, n=4, env_mode="manual", participants_mode="manual", moderator=MOD_LIGHT),
            {
                "environment": {"manual": RESTAURANT_ENV},
                "participants": {"profiles": profiles_trait_spread_4()},
            },
        ),
    },
    {
        "id": "f04_manual_manual_grounding_tripwire",
        "suite": "full",
        "why": "Manual option board with limited facts; catches invented parking/weather/distance claims.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=204, n=3, env_mode="manual", participants_mode="manual", moderator=MOD_FULL),
            {
                "environment": {"manual": WEEKEND_ENV},
                "participants": {"profiles": profiles_three_way()},
            },
        ),
    },
    {
        "id": "f05_manual_env_auto_participants_n5",
        "suite": "full",
        "why": "Manual environment plus auto cast at n=5; checks bigger generated group over fixed facts.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=205, n=5, env_mode="manual", participants_mode="auto", forced_shape="2-1-1-1", moderator=MOD_FULL),
            {"environment": {"manual": COFFEE_ENV}},
        ),
    },
    {
        "id": "f06_auto_env_manual_participants_no_moderator",
        "suite": "full",
        "why": "Auto environment plus fixed participants without moderator; participant probes/summaries and visible votes should still work.",
        "topic": "Choose a weekend trip for friends where one person wants quiet and one wants something active",
        "patch": deep_merge(
            base_patch(seed=206, n=4, env_mode="auto", participants_mode="manual", moderator=MOD_NONE),
            {"participants": {"profiles": profiles_trait_spread_4()}},
        ),
    },
]


def selected_cases() -> list[dict[str, Any]]:
    """Return the full suite. The eval script intentionally has no quick/list modes."""
    return CASES



def run_case(case: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    deep_merge(cfg, case["patch"])

    # Keep case logs separate inside the suite folder by using run metadata in stdout;
    # the simulator itself creates timestamped subdirs under eval/logs_eval_suite/.
    write_config(cfg)

    cmd = [sys.executable, str(MAIN_PATH)]
    if cfg.get("environment", {}).get("mode") == "auto":
        cmd.append(case["topic"])

    started = datetime.now().isoformat(timespec="seconds")
    print(f"\n=== {case['id']} ===")
    print(case["why"])
    print("Command:", " ".join(str(x) for x in cmd))
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print("STDERR:")
        print(stderr)

    log_dir = ""
    for line in stdout.splitlines():
        if line.startswith("Logs:"):
            log_dir = line.split("Logs:", 1)[1].strip()

    metrics: dict[str, Any] = {}
    if log_dir:
        run_json = Path(log_dir) / "run.json"
        if run_json.exists():
            try:
                run_data = json.loads(run_json.read_text(encoding="utf-8"))
                metrics = run_data.get("metrics", {}) or {}
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                metrics = {"metrics_read_error": str(exc)}

    row = {
        "case_id": case["id"],
        "started": started,
        "returncode": proc.returncode,
        "log_dir": log_dir,
        "suite": case["suite"],
        "why": case["why"],
        "environment_mode": cfg.get("environment", {}).get("mode"),
        "participants_mode": cfg.get("participants", {}).get("mode"),
        "moderator_enabled": cfg.get("moderator", {}).get("enabled"),
        "final_vote_call": cfg.get("moderator", {}).get("final_vote_call"),
        "n": cfg.get("simulation", {}).get("num_participants"),
        "forced_shape": cfg.get("personas", {}).get("preference_distribution", {}).get("forced_shape"),
        "outcome": metrics.get("outcome_status") or metrics.get("outcome"),
        "final_option": metrics.get("final_option"),
        "engagement_behavior_correlation": metrics.get("engagement_behavior_correlation"),
        "discussion_lean_shifts": metrics.get("discussion_lean_shifts"),
        "participant_procedural_moves": metrics.get("participant_procedural_moves"),
        "peer_vote_call": metrics.get("peer_vote_call"),
        "split_reservation_exchanges": metrics.get("split_reservation_exchanges"),
        "two_person_deadlock_attempted": metrics.get("two_person_deadlock_attempted"),
        "unsupported_printed_turns": metrics.get("unsupported_printed_turns"),
        "invalid_printed_turn_count": metrics.get("invalid_printed_turn_count"),
        "repaired_turns": metrics.get("repaired_turns"),
        "fallback_turns": metrics.get("fallback_turns"),
        "visible_vote_count": metrics.get("visible_vote_count"),
        "final_support_fraction": metrics.get("final_support_fraction"),
        "final_blocker_violations": metrics.get("final_blocker_violations"),
        "direct_response_rate": metrics.get("direct_response_rate"),
        "concern_response_rate": metrics.get("concern_response_rate"),
        "participation_gini": metrics.get("participation_gini"),
        "repetition_score": metrics.get("repetition_score"),
        "avg_words_per_turn": metrics.get("avg_words_per_turn"),
        "short_turn_rate": metrics.get("short_turn_rate"),
        "tiny_turn_rate": metrics.get("tiny_turn_rate"),
        "total_tokens_in": metrics.get("total_tokens_in"),
        "total_tokens_out": metrics.get("total_tokens_out"),
    }
    append_summary(row)
    return row


def append_summary(row: dict[str, Any]) -> None:
    SUITE_LOG_DIR.mkdir(exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def remove_extra_metrics_csv() -> None:
    """Remove per-run metrics.csv files; keep only eval_suite_runs.csv as suite summary."""
    if not SUITE_LOG_DIR.exists():
        return
    for metrics_path in SUITE_LOG_DIR.rglob("metrics.csv"):
        if metrics_path.resolve() == SUMMARY_CSV.resolve():
            continue
        try:
            metrics_path.unlink()
        except OSError as exc:
            print(f"Warning: could not remove extra metrics CSV {metrics_path}: {exc}", file=sys.stderr)


def zip_suite_log_dir() -> Path:
    """Create eval/logs_eval_suite.zip containing the complete logs_eval_suite folder."""
    if not SUITE_LOG_DIR.exists():
        raise FileNotFoundError(f"Cannot zip missing suite log directory: {SUITE_LOG_DIR}")
    archive_base = SUITE_LOG_DIR.parent / SUITE_LOG_DIR.name
    archive_path = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=SUITE_LOG_DIR.parent,
        base_dir=SUITE_LOG_DIR.name,
    )
    return Path(archive_path)


def main() -> int:
    if not CONFIG_PATH.exists() or not MAIN_PATH.exists():
        print("Run this script from the project root, next to config.yaml and main.py.", file=sys.stderr)
        return 2

    cases = selected_cases()
    backup_path = CONFIG_PATH.with_suffix(".yaml.eval_backup")
    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    backup_path.write_text(original_text, encoding="utf-8")
    base_config = yaml.safe_load(original_text) or {}

    SUITE_LOG_DIR.mkdir(exist_ok=True)
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()

    print(f"Running {len(cases)} full eval cases. Logs: {SUITE_LOG_DIR}")
    print(f"Original config backup: {backup_path}")

    rows: list[dict[str, Any]] = []
    return_code = 0
    try:
        for case in cases:
            row = run_case(case, base_config)
            rows.append(row)
            if row["returncode"] != 0:
                return_code = row["returncode"]
                print(f"\nStopping because {case['id']} failed with return code {return_code}.")
                break
    finally:
        CONFIG_PATH.write_text(original_text, encoding="utf-8")
        print("\nRestored original config.yaml.")

    remove_extra_metrics_csv()
    zip_path = zip_suite_log_dir()

    print("\nSummary:")
    for row in rows:
        print(
            f"- {row['case_id']}: rc={row['returncode']}, "
            f"outcome={row.get('outcome')}, final={row.get('final_option')}, "
            f"lean_shifts={row.get('discussion_lean_shifts')}, "
            f"peer_proc={row.get('participant_procedural_moves')}, "
            f"split_pairs={row.get('split_reservation_exchanges')}, "
            f"n2_deadlock={row.get('two_person_deadlock_attempted')}, "
            f"unsupported={row.get('unsupported_printed_turns')}, "
            f"tokens_in={row.get('total_tokens_in')}"
        )
    print(f"\nSuite CSV: {SUMMARY_CSV}")
    print(f"Suite ZIP: {zip_path}")
    return return_code

if __name__ == "__main__":
    raise SystemExit(main())
