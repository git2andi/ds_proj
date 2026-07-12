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
) -> dict[str, Any]:
    # Validation/grounding is no longer configurable (todo_validation item 9):
    # semantic interpretation and claim-level grounding always run through the
    # validator LLM role, so the old validation: patch section is gone. The
    # token warning threshold accounts for the validator role's calls.
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
        "output": {
            "log_dir": "eval/logs_eval_suite",
            "write_prompts": False,
        },
        "limits": {
            "warn_total_input_tokens": 80000,
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
    concern: str,
    short_name: str,
) -> dict[str, Any]:
    return {
        "id": option_id,
        "name": name,
        "attrs": attrs,
        "upside": upside,
        "concern": concern,
        "short_name": short_name,
    }


RESTAURANT_ENV = {
    "topic": "Choose a restaurant for a mixed-preference group dinner",
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
            "may not feel special enough for everyone",
            "Ramen",
        ),
        option(
            "B",
            "La Piazza",
            {"price": "26 euros", "travel": "18 minutes by bus", "vegetarian": "several pasta and pizza options"},
            "broad menu and relaxed atmosphere",
            "can become noisy on Fridays",
            "Piazza",
        ),
        option(
            "C",
            "Green Table",
            {"price": "24 euros", "travel": "20 minutes walking", "vegetarian": "mostly vegetarian menu"},
            "best dietary fit and calm setting",
            "some may see it as too niche",
            "Green Table",
        ),
        option(
            "D",
            "Burger Cellar",
            {"price": "21 euros", "travel": "8 minutes walking", "vegetarian": "one vegetarian burger"},
            "closest and casual",
            "may not satisfy the dietary requirement well",
            "Burger Cellar",
        ),
    ],
}

WEEKEND_ENV = {
    "topic": "Choose a weekend activity for three friends with different energy levels",
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
            "may feel too quiet",
            "Museum",
        ),
        option(
            "B",
            "Lake Bike Ride",
            {"cost": "12 euros", "travel": "25 minutes by train", "duration": "6 hours"},
            "active and inexpensive",
            "bad fit for someone tired",
            "Bike Ride",
        ),
        option(
            "C",
            "Escape Room",
            {"cost": "32 euros", "travel": "20 minutes by tram", "duration": "2 hours"},
            "interactive and memorable",
            "less flexible once booked",
            "Escape Room",
        ),
        option(
            "D",
            "Home Cooking Night",
            {"cost": "18 euros", "travel": "none", "duration": "5 hours"},
            "cheapest and most flexible",
            "may feel too ordinary",
            "Cooking",
        ),
    ],
}

COFFEE_ENV = {
    "topic": "Choose a coffee machine for a small shared office kitchen",
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
            "highest upfront cost",
            "Moccamaster",
        ),
        option(
            "B",
            "DeLonghi Dedica",
            {"cost": "179 euros", "type": "espresso", "capacity": "single shots"},
            "compact and good for espresso",
            "requires more hands-on use",
            "Dedica",
        ),
        option(
            "C",
            "Philips Senseo Switch",
            {"cost": "119 euros", "type": "pads and filter", "capacity": "7 cups"},
            "flexible and cheap",
            "less premium build",
            "Senseo",
        ),
        option(
            "D",
            "Ninja Filter Brewer",
            {"cost": "149 euros", "type": "filter", "capacity": "12 cups"},
            "large capacity for the price",
            "less compact than the others",
            "Ninja",
        ),
    ],
}

ROOMMATE_ENV = {
    "topic": "Choose whether two roommates should upgrade cleaning at home",
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
            "Robot Vacuum",
        ),
        option(
            "B",
            "Bosch Compact Dishwasher",
            {"cost": "430 euros", "task": "daily dishes", "space": "uses counter space"},
            "removes the most common kitchen chore",
            "does not help with dust or floors",
            "Dishwasher",
        ),
        option(
            "C",
            "Monthly Cleaning Service Trial",
            {"cost": "80 euros per month", "task": "general cleaning", "duration": "three-month trial"},
            "covers several chores without buying equipment",
            "depends on scheduling someone to come in",
            "Cleaning Trial",
        ),
        option(
            "D",
            "Shared Cleaning Supplies Kit",
            {"cost": "70 euros", "task": "manual cleaning", "storage": "small closet box"},
            "cheap and easy to start immediately",
            "may not change habits enough",
            "Supplies Kit",
        ),
    ],
}


DEMO_ENV = {
    "topic": "Choose how a student software project should present its final demo",
    "shared_context": [
        "The demo slot is 15 minutes in front of the course staff and peers.",
        "The team has one week left before the presentation.",
        "A projector and reliable campus wifi are available in the room.",
    ],
    "options": [
        option(
            "A",
            "Live Coding Walkthrough",
            {"prep_time": "low", "risk": "high", "audience_engagement": "high"},
            "shows the app working in real time and feels authentic",
            "a live failure in front of staff is hard to recover from",
            "Live Coding",
        ),
        option(
            "B",
            "Recorded Screencast",
            {"prep_time": "medium", "risk": "low", "audience_engagement": "medium"},
            "safe and rehearsed, nothing can break on stage",
            "feels less lively and cannot answer follow-ups mid-play",
            "Screencast",
        ),
        option(
            "C",
            "Slide Deck With Screenshots",
            {"prep_time": "low", "risk": "low", "audience_engagement": "low"},
            "quickest to prepare and easy to keep on time",
            "least convincing that the software actually runs",
            "Slides",
        ),
        option(
            "D",
            "Interactive Audience Try-Out",
            {"prep_time": "high", "risk": "medium", "audience_engagement": "high"},
            "most memorable and lets the audience use the app",
            "needs the most setup and depends on wifi holding up",
            "Try-Out",
        ),
    ],
}

WORKSHOP_ENV = {
    "topic": "Choose a format and venue for a weekend community coding workshop",
    "shared_context": [
        "About thirty people from mixed skill levels have signed up.",
        "The organizing budget is 500 euros for the day.",
        "The workshop must fit into a single Saturday.",
    ],
    "options": [
        option(
            "A",
            "University Lab Hands-On Day",
            {"cost": "150 euros", "capacity": "40 seats", "setup": "computers provided"},
            "everyone gets a workstation and stable setup",
            "the campus location is farther for most attendees",
            "University Lab",
        ),
        option(
            "B",
            "Community Center Talks",
            {"cost": "200 euros", "capacity": "60 seats", "setup": "bring your own laptop"},
            "central and roomy with space for talks",
            "less hands-on and depends on attendees' own laptops",
            "Community Center",
        ),
        option(
            "C",
            "Online Live Sessions",
            {"cost": "40 euros", "capacity": "no seat limit", "setup": "video platform"},
            "cheapest and open to anyone regardless of travel",
            "harder to help beginners who get stuck at home",
            "Online",
        ),
        option(
            "D",
            "Cafe Meetup Workshop",
            {"cost": "120 euros", "capacity": "25 seats", "setup": "informal, limited power outlets"},
            "relaxed, social atmosphere that lowers the barrier",
            "tight on space and power for a full hands-on day",
            "Cafe Meetup",
        ),
    ],
}


def persona_profile(
    *,
    age: int,
    speech_style: str,
    description: str,
    private_goal: str,
    preferred_option: str,
    traits: dict[str, int],
    parameters: dict[str, float],
    name: str,
    rejection: str | None = None,
    rejection_reason: str = "",
) -> dict[str, Any]:
    """Create a manual persona profile with explicit age/speech_style fields.

    Eval personas intentionally keep behavior-driving parameters separate from
    surface wording: engagement/verbosity/directness/stubbornness control
    behavior; speech_style is only age-consistent register coloring.
    """
    profile: dict[str, Any] = {
        "name": name,
        "age": age,
        "speech_style": speech_style,
        "description": description,
        "private_goal": private_goal,
        "preferred_option": preferred_option,
        "traits": traits,
        "parameters": parameters,
    }
    if rejection:
        profile["rejection"] = rejection
        profile["rejection_reason"] = rejection_reason
    return profile


# The four compact age-band registers used by the builder (src/builders.py).
STYLE_YOUNG = "young casual wording"
STYLE_RELAXED = "relaxed practical wording"
STYLE_WORKPLACE = "direct workplace wording"
STYLE_TRADITIONAL = "measured traditional wording"


def profiles_three_way() -> list[dict[str, Any]]:
    """Three distinct initial favorites; designed to catch premature split-vote closure."""
    return [
        persona_profile(
            name="Mira",
            age=42,
            speech_style=STYLE_WORKPLACE,
            description="organized project coordinator who cares about broad fit and avoiding awkward logistics",
            private_goal="wants the option that works for most people without needing extra coordination",
            preferred_option="A",
            traits={"openness": 3, "conscientiousness": 5, "extraversion": 3, "agreeableness": 3, "neuroticism": 2},
            parameters={
                "engagement": 0.55,
                "verbosity": 0.55,
                "directness": 0.50,
                "stubbornness": 0.45,
                "switch_resistance": 0.40,
            },
        ),
        persona_profile(
            name="Jonas",
            age=24,
            speech_style=STYLE_YOUNG,
            description="early-career budget-watcher who rents a shared flat and avoids unnecessary spending",
            private_goal="wants the group to avoid overspending but can accept a better compromise",
            preferred_option="B",
            traits={"openness": 3, "conscientiousness": 4, "extraversion": 2, "agreeableness": 4, "neuroticism": 2},
            parameters={
                "engagement": 0.35,
                "verbosity": 0.35,
                "directness": 0.45,
                "stubbornness": 0.35,
                "switch_resistance": 0.30,
            },
        ),
        persona_profile(
            name="Lea",
            age=29,
            speech_style=STYLE_RELAXED,
            description="high-energy event planner who likes memorable choices and often drives the conversation forward",
            private_goal="wants the group to choose something that feels worth the effort",
            preferred_option="C",
            traits={"openness": 5, "conscientiousness": 3, "extraversion": 5, "agreeableness": 3, "neuroticism": 2},
            parameters={
                "engagement": 0.90,
                "verbosity": 0.80,
                "directness": 0.70,
                "stubbornness": 0.55,
                "switch_resistance": 0.50,
            },
        ),
    ]


def profiles_trait_spread_4() -> list[dict[str, Any]]:
    """One very active sim, one quiet sim, two middle sims."""
    return [
        persona_profile(
            name="Nora",
            age=37,
            speech_style=STYLE_WORKPLACE,
            description="very engaged product lead who notices process problems and proposes next steps",
            private_goal="wants a clear decision and tends to keep the group moving",
            preferred_option="B",
            traits={"openness": 4, "conscientiousness": 4, "extraversion": 5, "agreeableness": 3, "neuroticism": 1},
            parameters={
                "engagement": 0.95,
                "verbosity": 0.85,
                "directness": 0.75,
                "stubbornness": 0.35,
                "switch_resistance": 0.30,
            },
        ),
        persona_profile(
            name="Tarek",
            age=21,
            speech_style=STYLE_YOUNG,
            description="quiet university student who answers when asked but rarely pushes himself into the discussion",
            private_goal="wants the simplest acceptable choice and avoids long arguments",
            preferred_option="D",
            traits={"openness": 2, "conscientiousness": 3, "extraversion": 1, "agreeableness": 4, "neuroticism": 3},
            parameters={
                "engagement": 0.15,
                "verbosity": 0.25,
                "directness": 0.35,
                "stubbornness": 0.30,
                "switch_resistance": 0.30,
            },
        ),
        persona_profile(
            name="Eva",
            age=56,
            speech_style=STYLE_WORKPLACE,
            description="experienced office administrator who weighs concrete constraints before moving position",
            private_goal="wants the option with the fewest hidden tradeoffs",
            preferred_option="A",
            traits={"openness": 3, "conscientiousness": 5, "extraversion": 3, "agreeableness": 3, "neuroticism": 2},
            parameters={
                "engagement": 0.55,
                "verbosity": 0.55,
                "directness": 0.55,
                "stubbornness": 0.50,
                "switch_resistance": 0.55,
            },
        ),
        persona_profile(
            name="Sam",
            age=31,
            speech_style=STYLE_RELAXED,
            description="socially flexible UX designer who often bridges between opposing preferences",
            private_goal="wants the final choice to feel acceptable to everyone",
            preferred_option="C",
            traits={"openness": 4, "conscientiousness": 3, "extraversion": 3, "agreeableness": 5, "neuroticism": 2},
            parameters={
                "engagement": 0.50,
                "verbosity": 0.50,
                "directness": 0.40,
                "stubbornness": 0.20,
                "switch_resistance": 0.15,
            },
        ),
    ]


def profiles_hard_holdout_4() -> list[dict[str, Any]]:
    """One stubborn minority, useful for bounded reservation/compromise testing."""
    return [
        persona_profile(
            name="Clara",
            age=46,
            speech_style=STYLE_WORKPLACE,
            description="detail-focused operations specialist who will not accept weak dietary fit",
            private_goal="wants the option that clearly protects the dietary requirement",
            preferred_option="C",
            traits={"openness": 3, "conscientiousness": 5, "extraversion": 2, "agreeableness": 2, "neuroticism": 3},
            parameters={
                "engagement": 0.55,
                "verbosity": 0.55,
                "directness": 0.75,
                "stubbornness": 0.85,
                "switch_resistance": 0.90,
            },
        ),
        persona_profile(
            name="Ben",
            age=27,
            speech_style=STYLE_YOUNG,
            description="cost-conscious early-career employee who likes broad, familiar compromises",
            private_goal="wants a safe group choice that does not exceed the budget too much",
            preferred_option="B",
            traits={"openness": 3, "conscientiousness": 4, "extraversion": 3, "agreeableness": 4, "neuroticism": 2},
            parameters={
                "engagement": 0.60,
                "verbosity": 0.50,
                "directness": 0.50,
                "stubbornness": 0.35,
                "switch_resistance": 0.30,
            },
        ),
        persona_profile(
            name="Iris",
            age=34,
            speech_style=STYLE_RELAXED,
            description="active social organizer who prefers easy logistics and broad menus",
            private_goal="wants the group to settle on a practical choice without dragging the debate out",
            preferred_option="B",
            traits={"openness": 4, "conscientiousness": 3, "extraversion": 5, "agreeableness": 4, "neuroticism": 1},
            parameters={
                "engagement": 0.85,
                "verbosity": 0.70,
                "directness": 0.65,
                "stubbornness": 0.30,
                "switch_resistance": 0.25,
            },
        ),
        persona_profile(
            name="Omar",
            age=62,
            speech_style=STYLE_TRADITIONAL,
            description="relaxed retired teacher who usually follows a reasonable majority",
            private_goal="wants a choice that avoids obvious inconvenience",
            preferred_option="B",
            traits={"openness": 3, "conscientiousness": 3, "extraversion": 2, "agreeableness": 5, "neuroticism": 2},
            parameters={
                "engagement": 0.40,
                "verbosity": 0.35,
                "directness": 0.35,
                "stubbornness": 0.20,
                "switch_resistance": 0.15,
            },
        ),
    ]


def profiles_stubborn_deadlock_2() -> list[dict[str, Any]]:
    """Two stubborn opposing participants; designed to force the 1-1 protocol."""
    return [
        persona_profile(
            name="Maja",
            age=23,
            speech_style=STYLE_YOUNG,
            description="stubborn graduate student in a shared apartment who thinks floors are the visible problem",
            private_goal="wants the robot vacuum and does not want a counter-space appliance",
            preferred_option="A",
            rejection="B",
            rejection_reason="it is near the budget limit and uses counter space",
            traits={"openness": 2, "conscientiousness": 4, "extraversion": 2, "agreeableness": 1, "neuroticism": 4},
            parameters={
                "engagement": 0.55,
                "verbosity": 0.45,
                "directness": 0.80,
                "stubbornness": 0.95,
                "switch_resistance": 0.95,
            },
        ),
        persona_profile(
            name="Felix",
            age=58,
            speech_style=STYLE_WORKPLACE,
            description="stubborn long-time tenant who thinks dishes cause most conflict and distrusts partial floor-only fixes",
            private_goal="wants the dishwasher and does not want a device that ignores the kitchen mess",
            preferred_option="B",
            rejection="A",
            rejection_reason="it does not help with dishes or kitchen cleanup",
            traits={"openness": 2, "conscientiousness": 4, "extraversion": 3, "agreeableness": 1, "neuroticism": 3},
            parameters={
                "engagement": 0.60,
                "verbosity": 0.50,
                "directness": 0.85,
                "stubbornness": 0.95,
                "switch_resistance": 0.95,
            },
        ),
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
    # 1. Manual/manual, n=2, stubborn deadlock (shared-home upgrade regression).
    {
        "id": "c01_manual_manual_n2_stubborn_deadlock",
        "why": "Two-person opposing-preference deadlock; must attempt the deadlock protocol, not a false unanimity.",
        "topic": "",
        "expect": {"two_person_deadlock_attempted": True},
        "patch": deep_merge(
            base_patch(seed=201, n=2, env_mode="manual", participants_mode="manual",
                       forced_shape="1-1", moderator=MOD_FULL),
            {
                "environment": {"manual": ROOMMATE_ENV},
                "participants": {"profiles": profiles_stubborn_deadlock_2()},
            },
        ),
    },
    # 2. Manual/manual, n=3, three-way split and narrowing (Saturday plan).
    {
        "id": "c02_manual_manual_n3_three_way_split",
        "why": "A/B/C split must narrow before closing, not vote through immediately.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=202, n=3, env_mode="manual", participants_mode="manual", moderator=MOD_FULL),
            {
                "environment": {"manual": WEEKEND_ENV},
                "participants": {"profiles": profiles_three_way()},
            },
        ),
    },
    # 3. Manual/manual, n=4, strong trait spread, light moderator (student demo).
    {
        "id": "c03_manual_manual_n4_trait_spread_light_mod",
        "why": "High/low engagement should visibly change turn share and word length; light moderator only opens/closes.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=203, n=4, env_mode="manual", participants_mode="manual", moderator=MOD_LIGHT),
            {
                "environment": {"manual": DEMO_ENV},
                "participants": {"profiles": profiles_trait_spread_4()},
            },
        ),
    },
    # 4. Manual/manual, n=4, no moderator (group dinner with dietary/travel).
    {
        "id": "c04_manual_manual_n4_no_moderator",
        "why": "No-moderator mode must still narrow via participant probes/summaries and visible votes.",
        "topic": "",
        "expect": {"peer_process": True},
        "patch": deep_merge(
            base_patch(seed=204, n=4, env_mode="manual", participants_mode="manual", moderator=MOD_NONE),
            {
                "environment": {"manual": RESTAURANT_ENV},
                "participants": {"profiles": profiles_hard_holdout_4()},
            },
        ),
    },
    # 5. Manual/manual, n=3, grounding stress case (coffee machine).
    {
        "id": "c05_manual_manual_n3_grounding_coffee",
        "why": "Exposes invented product capabilities, inference over-rejection, and grounding/token cost.",
        "topic": "",
        "expect": {"zero_unsupported_printed": True},
        "patch": deep_merge(
            base_patch(seed=205, n=3, env_mode="manual", participants_mode="manual", moderator=MOD_FULL),
            {
                "environment": {"manual": COFFEE_ENV},
                "participants": {"profiles": profiles_three_way()},
            },
        ),
    },
    # 6. Manual environment / automatic participants, n=5 (community workshop).
    {
        "id": "c06_manual_env_auto_participants_n5",
        "why": "Fixed option board with a larger generated cast over fixed facts.",
        "topic": "",
        "patch": deep_merge(
            base_patch(seed=206, n=5, env_mode="manual", participants_mode="auto",
                       forced_shape="2-1-1-1", moderator=MOD_FULL),
            {"environment": {"manual": WORKSHOP_ENV}},
        ),
    },
    # 7. Automatic environment / manual participants, n=3 (alias-repair regression).
    {
        "id": "c07_auto_env_manual_participants_n3",
        "why": "Retains the automatic setup and alias-repair regression with controlled personas.",
        "topic": "Book a flight from Miami to Stockholm",
        "patch": deep_merge(
            base_patch(seed=207, n=3, env_mode="auto", participants_mode="manual", moderator=MOD_FULL),
            {"participants": {"profiles": profiles_three_way()}},
        ),
    },
    # 8. Automatic/automatic, n=3 baseline (volunteer scheduling).
    {
        "id": "c08_auto_auto_n3_baseline",
        "why": "Normal default-style auto/auto run for regression comparison.",
        "topic": "Choose a shared scheduling method for a volunteer group",
        "patch": base_patch(seed=208, n=3, env_mode="auto", participants_mode="auto",
                            forced_shape="1-1-1", moderator=MOD_FULL),
    },
    # 9. Automatic/automatic, n=5 scaling (one-day team retreat).
    {
        "id": "c09_auto_auto_n5_scaling",
        "why": "Checks routing/pacing scale beyond n=3/4 with mixed budgets and energy.",
        "topic": "Choose a one-day team retreat format with mixed budgets and energy levels",
        "patch": base_patch(seed=209, n=5, env_mode="auto", participants_mode="auto",
                            forced_shape="2-1-1-1", moderator=MOD_FULL),
    },
    # 10. Automatic/automatic, n=7 maximum size (student showcase), bounded turns.
    {
        "id": "c10_auto_auto_n7_max_size",
        "why": "Maximum group size; a bounded turn budget tests scale without dominating total cost.",
        "topic": "Choose a format for a student hackathon project showcase",
        "patch": base_patch(seed=210, n=7, env_mode="auto", participants_mode="auto",
                            forced_shape="3-2-1-1", moderator=MOD_FULL,
                            max_turns_per_participant=4.0),
    },
]


def selected_cases() -> list[dict[str, Any]]:
    """Return the full suite. The eval script intentionally has no quick/list modes."""
    return CASES


# Controller-facing phrasing that must never appear in a PRINTED participant
# line (item 10 leak detector). These are internal rationale/trace terms.
_LEAK_PHRASES = (
    "most defensible choice",
    "clearest visible support",
    "visible discussion",
    "route_source",
    "intended_move",
    "primary_act",
    "allowed_reason",
    "thread_id",
    "controller",
    "fallback",
    "required_vote",
    "option_focus",
    "coverage",
)


def _leak_hits(run_data: dict[str, Any]) -> list[str]:
    """Printed participant lines carrying controller-facing wording (item 10)."""
    hits: list[str] = []
    for turn in run_data.get("turns", []):
        if turn.get("speaker_id") == "moderator":
            continue
        text = (turn.get("text") or "")
        low = text.lower()
        for phrase in _LEAK_PHRASES:
            if phrase in low:
                hits.append(f"turn {turn.get('index')}: '{phrase}' in {text[:60]!r}")
    return hits


def case_flags(case: dict[str, Any], metrics: dict[str, Any], run_data: dict[str, Any]) -> list[str]:
    """Per-case acceptance checks — a case is more than returncode == 0 (item 9).

    Returns a list of human-readable flags for manual transcript review; an
    empty list means the case cleared every automatic check.
    """
    flags: list[str] = []
    pt = int(metrics.get("participant_turns", 0) or 0)

    def rate(key: str) -> float:
        return (int(metrics.get(key, 0) or 0) / pt) if pt else 0.0

    if int(metrics.get("invalid_printed_turn_count", 0) or 0) > 0:
        flags.append(f"invalid_printed_turns={metrics.get('invalid_printed_turn_count')}")
    if int(metrics.get("unsupported_printed_turns", 0) or 0) > 0:
        flags.append(f"unsupported_printed_turns={metrics.get('unsupported_printed_turns')}")
    if int(metrics.get("final_blocker_violations", 0) or 0) > 0:
        flags.append(f"blocker_violations={metrics.get('final_blocker_violations')}")
    if int(metrics.get("vote_state_consistency_failures", 0) or 0) > 0:
        flags.append(f"vote_state_consistency_failures={metrics.get('vote_state_consistency_failures')}")
    rr = float(metrics.get("repair_rate", 0.0) or 0.0)
    if rr > 0.25:
        flags.append(f"repair_rate={rr:.2f}>0.25")
    dr = rate("dropped_turn_count")
    if dr > 0.02:
        flags.append(f"drop_rate={dr:.2f}>0.02")
    leaks = _leak_hits(run_data)
    if leaks:
        flags.append(f"controller_language_leak x{len(leaks)}")

    # Case-specific expectations declared on the case.
    expect = case.get("expect", {}) or {}
    if expect.get("two_person_deadlock_attempted") and not metrics.get("two_person_deadlock_attempted"):
        flags.append("expected two_person_deadlock_attempted but none")
    if expect.get("peer_process") and int(metrics.get("participant_procedural_moves", 0) or 0) == 0:
        flags.append("expected peer procedural moves but none")
    if expect.get("zero_unsupported_printed") and int(metrics.get("unsupported_printed_turns", 0) or 0) > 0:
        flags.append("grounding case printed unsupported claims")
    return flags



def run_case(case: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    deep_merge(cfg, case["patch"])
    # Persist the suite case id so run.json/transcript metadata and the log
    # directory name all tie back to this case (item 9).
    cfg.setdefault("output", {})["case_id"] = case["id"]

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
    run_data: dict[str, Any] = {}
    if log_dir:
        run_json = Path(log_dir) / "run.json"
        if run_json.exists():
            try:
                run_data = json.loads(run_json.read_text(encoding="utf-8"))
                metrics = run_data.get("metrics", {}) or {}
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                metrics = {"metrics_read_error": str(exc)}

    # Tag the log directory with the case id (item 9) so the folder itself
    # identifies its case, not only run.json/CSV.
    if log_dir:
        src = Path(log_dir)
        if src.exists() and case["id"] not in src.name:
            dest = src.with_name(f"{src.name}__{case['id']}")
            try:
                src.rename(dest)
                log_dir = str(dest)
            except OSError:
                pass

    flags = case_flags(case, metrics, run_data)
    if flags:
        print("FLAGS:", "; ".join(flags))

    persona_age_style = "; ".join(
        f"{p.get('name', '?')}:{p.get('age', '?')}:{p.get('speech_style', '')}"
        for p in run_data.get("personas", [])
    )

    row = {
        "case_id": case["id"],
        "started": started,
        "returncode": proc.returncode,
        "flags": "; ".join(flags),
        "log_dir": log_dir,
        "why": case["why"],
        "environment_mode": cfg.get("environment", {}).get("mode"),
        "participants_mode": cfg.get("participants", {}).get("mode"),
        "moderator_enabled": cfg.get("moderator", {}).get("enabled"),
        "final_vote_call": cfg.get("moderator", {}).get("final_vote_call"),
        "n": cfg.get("simulation", {}).get("num_participants"),
        "forced_shape": cfg.get("personas", {}).get("preference_distribution", {}).get("forced_shape"),
        "persona_age_style": persona_age_style,
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
        # Evidence-contract health (todo_validation item 15).
        "intended_function_realized_rate": metrics.get("intended_function_realized_rate"),
        "repair_success_rate": metrics.get("repair_success_rate"),
        "dropped_turn_count": metrics.get("dropped_turn_count"),
        "validator_failure_turns": metrics.get("validator_failure_turns"),
        # Validator-cost surface (items 6/10): logical checks, API calls, the
        # per-accepted-turn ratio, token share, and fast-path rate.
        "participant_turns": metrics.get("participant_turns"),
        "validator_calls": metrics.get("validator_calls"),
        "validator_logical_checks": metrics.get("validator_logical_checks"),
        "validator_api_retries": metrics.get("validator_api_retries"),
        "validator_calls_per_accepted_turn": metrics.get("validator_calls_per_accepted_turn"),
        "validator_logical_checks_per_turn": metrics.get("validator_logical_checks_per_turn"),
        "validator_input_share": metrics.get("validator_input_share"),
        "validation_fast_path_rate": metrics.get("validation_fast_path_rate"),
        "accepted_metric_only": (metrics.get("assessment_action_counts") or {}).get("accept_with_metric"),
        "repair_rate": metrics.get("repair_rate"),
        "validator_tokens_in": metrics.get("validator_tokens_in"),
        "visible_vote_count": metrics.get("visible_vote_count"),
        "final_support_fraction": metrics.get("final_support_fraction"),
        "final_blocker_violations": metrics.get("final_blocker_violations"),
        "direct_response_rate": metrics.get("direct_response_rate"),
        "concern_response_rate": metrics.get("concern_response_rate"),
        # Cleanup-pass surfaces: thread-owned issue history and the actual
        # route-source mix (threads driving local moves vs coverage/normal).
        "settled_issue_keys": "; ".join(metrics.get("settled_issue_keys") or []),
        "route_source_distribution": json.dumps(metrics.get("route_source_distribution") or {}, sort_keys=True),
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
    # The safety copy lives under eval/ (not the project root), next to the
    # suite logs it belongs to.
    backup_path = PROJECT_ROOT / "eval" / "config.yaml.eval_backup"
    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    backup_path.write_text(original_text, encoding="utf-8")
    base_config = yaml.safe_load(original_text) or {}

    SUITE_LOG_DIR.mkdir(exist_ok=True)
    # Restart-safety (item 9): a fresh full run clears the previous summary AND
    # every prior per-run directory in one step, so an interrupted earlier run
    # never leaves orphaned run folders behind an up-to-date CSV. The suite
    # always runs all cases, so the end state is exactly len(cases) rows and
    # exactly len(cases) run directories.
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()
    for child in SUITE_LOG_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)

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

    print(f"\nSummary ({len(rows)} rows):")
    flagged = 0
    for row in rows:
        flags = row.get("flags") or ""
        if flags:
            flagged += 1
        print(
            f"- {row['case_id']}: rc={row['returncode']}, "
            f"outcome={row.get('outcome')}, final={row.get('final_option')}, "
            f"v/turn={row.get('validator_calls_per_accepted_turn')}, "
            f"v_share={row.get('validator_input_share')}, "
            f"repair={row.get('repair_rate')}, drops={row.get('dropped_turn_count')}, "
            f"unsupported={row.get('unsupported_printed_turns')}, "
            f"tokens_in={row.get('total_tokens_in')}"
            + (f"  ⚑ {flags}" if flags else "")
        )
    print(f"\n{flagged}/{len(rows)} cases flagged for manual review.")
    print(f"Suite CSV: {SUMMARY_CSV}")
    print(f"Suite ZIP: {zip_path}")
    return return_code

if __name__ == "__main__":
    raise SystemExit(main())
