"""Focused 15-case LLM-backed evaluation suite.

The suite tests the simplified runtime's most important end-to-end properties:
participant authority, conversation progression, issue handling, stance movement,
hard blockers, grounding, moderator-free operation, trait visibility, and bounded
re-voting. It uses ten varied topics and covers every supported group size from two through seven participants.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config_loader import cfg  # noqa: E402
from dialogue import DialogueRunner  # noqa: E402
from eval import flat_metrics_for  # noqa: E402
from logger import DialogueLogger, metrics_for  # noqa: E402
from llm_client import get_llm_client  # noqa: E402
from models import (  # noqa: E402
    ActionType,
    OptionCard,
    OptionStance,
    Persona,
    Phase,
    Scenario,
    SimulatorParameters,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
    VoteStatus,
)
from simulator import bid_probability, movement_probability  # noqa: E402


@dataclass(frozen=True)
class EvalCase:
    id: str
    why: str
    preferences: tuple[str, ...]
    seed: int
    moderator: bool = True
    scenario_key: str = "study"
    engagements: tuple[int, ...] | None = None
    verbosities: tuple[int, ...] | None = None
    directness: tuple[int, ...] | None = None
    stubbornness: tuple[int, ...] | None = None
    hard_blocker_index: int | None = None
    dislike_alternatives_for: tuple[int, ...] = ()
    expected_outcome: str | None = None
    min_switches: int = 0
    min_resolved_concerns: int = 0
    min_stale_issues: int = 0
    min_narrowing_turns: int = 0
    max_participant_turns: int = 60


CASES = (
    EvalCase(
        "two_person_weekend_consensus",
        "Two participants with the same preference should close efficiently without redundant debate.",
        ("A", "A"),
        301,
        scenario_key="weekend",
        expected_outcome="successful",
        max_participant_turns=20,
    ),
    EvalCase(
        "three_way_split_workspace",
        "A genuine high-stubbornness three-way split may remain unresolved without forcing compromise.",
        ("A", "B", "C"),
        102,
        scenario_key="study",
        stubbornness=(4, 4, 4),
        dislike_alternatives_for=(0, 1, 2),
        expected_outcome="unresolved",
        min_stale_issues=1,
    ),
    EvalCase(
        "compromise_presentation",
        "Low-stubbornness participants should be able to make a presentation format acceptable and move visibly.",
        ("A", "B", "B"),
        203,
        scenario_key="presentation",
        stubbornness=(1, 1, 2),
        min_switches=1,
        min_narrowing_turns=1,
    ),
    EvalCase(
        "majority_no_moderator_dinner",
        "Moderator-free discussion must progress and close without visible moderator turns.",
        ("A", "A", "A", "B"),
        204,
        moderator=False,
        scenario_key="restaurant",
        stubbornness=(2, 2, 2, 4),
    ),
    EvalCase(
        "hard_blocker_cleaning",
        "The sole hard blocker must never accept or vote for a nonpreferred cleaning option.",
        ("A", "A", "C"),
        205,
        scenario_key="cleaning",
        hard_blocker_index=2,
        expected_outcome="majority",
    ),
    EvalCase(
        "direct_question_hike",
        "A directly addressed hiking question must be answered next and remain grounded in route facts.",
        ("A", "C", "B"),
        206,
        scenario_key="hike",
        engagements=(5, 4, 3),
    ),
    EvalCase(
        "concern_resolution_book_club",
        "A non-hard concern should be answerable and may make another novel acceptable.",
        ("A", "B", "B"),
        180,
        scenario_key="book_club",
        engagements=(5, 5, 5),
        stubbornness=(1, 2, 2),
        dislike_alternatives_for=(0,),
        min_resolved_concerns=1,
        min_switches=1,
    ),
    EvalCase(
        "grounding_sensitive_flight_n4",
        "Baggage, refundability and transfer risk must remain grounded for four travellers.",
        ("B", "C", "A", "B"),
        308,
        scenario_key="flight",
    ),
    EvalCase(
        "engagement_spread_meeting_n5",
        "Five participants should show uneven voluntary participation without floor quotas.",
        ("A", "B", "C", "A", "D"),
        309,
        scenario_key="meeting",
        engagements=(5, 1, 2, 4, 3),
        max_participant_turns=48,
    ),
    EvalCase(
        "language_style_spread_laptop",
        "Verbosity, directness, age style and private priorities should affect visible language.",
        ("A", "B", "C", "D"),
        210,
        scenario_key="laptop",
        verbosities=(1, 5, 3, 4),
        directness=(1, 5, 3, 4),
        stubbornness=(2, 2, 2, 2),
    ),
    EvalCase(
        "six_person_weekend_participation",
        "A six-person activity decision should remain bounded while giving engagement room to appear.",
        ("A", "B", "C", "D", "A", "B"),
        311,
        scenario_key="weekend",
        engagements=(5, 1, 4, 2, 3, 5),
        max_participant_turns=55,
    ),
    EvalCase(
        "seven_person_workspace_scale",
        "The largest supported group must remain coherent and within absolute pacing caps.",
        ("A", "B", "C", "D", "A", "B", "A"),
        312,
        scenario_key="study",
        engagements=(5, 2, 3, 4, 1, 5, 3),
        max_participant_turns=62,
    ),
    EvalCase(
        "two_person_presentation_deadlock",
        "Two stubborn participants may legitimately remain split without a fabricated compromise.",
        ("A", "B"),
        313,
        scenario_key="presentation",
        stubbornness=(4, 4),
        dislike_alternatives_for=(0, 1),
        expected_outcome="unresolved",
        max_participant_turns=24,
    ),
    EvalCase(
        "five_person_restaurant_compromise",
        "Five diners with distinct priorities should have a genuine opportunity to propose common ground.",
        ("A", "B", "C", "D", "A"),
        314,
        scenario_key="restaurant",
        stubbornness=(1, 2, 2, 3, 2),
        max_participant_turns=48,
    ),
    EvalCase(
        "six_person_cleaning_mixed",
        "A six-person household decision should preserve a hard blocker while allowing majority progression.",
        ("A", "A", "B", "C", "D", "A"),
        315,
        scenario_key="cleaning",
        hard_blocker_index=3,
        max_participant_turns=55,
    ),
)


def scenario_for(key: str) -> Scenario:
    scenarios: dict[str, Scenario] = {
        "weekend": Scenario(
            "Choose a Saturday group activity",
            [
                OptionCard("A", "City Museum", {"accessibility": "step-free", "weather exposure": "none", "interaction": "self-paced"}, "easy to organize in any weather", "less active", "Museum"),
                OptionCard("B", "Lakeside Picnic", {"weather exposure": "high", "seating": "bring blankets", "interaction": "open social time"}, "relaxed outdoor time", "depends on dry weather", "Picnic"),
                OptionCard("C", "Escape Room", {"booking": "advance reservation", "teamwork": "high", "accessibility": "one narrow room"}, "interactive team challenge", "less flexible entry time", "Escape Room"),
                OptionCard("D", "Cinema Evening", {"schedule": "fixed screening", "interaction": "low during film", "accessibility": "step-free"}, "low planning effort", "little group interaction", "Cinema"),
            ],
            ["The group is free from 14:00 onward.", "Everyone wants one shared activity."],
        ),
        "study": Scenario(
            "Choose a Saturday study location",
            [
                OptionCard("A", "Central Library", {"noise level": "quiet", "privacy": "shared desks", "accessibility": "step-free"}, "quiet and predictable", "can become crowded", "Library"),
                OptionCard("B", "Riverside Cafe", {"noise level": "moderate", "seating": "informal tables", "food access": "on site"}, "relaxed atmosphere", "background noise", "Cafe"),
                OptionCard("C", "Engineering Lab", {"equipment": "specialist workstations", "privacy": "bookable room", "accessibility": "staff access required"}, "reliable technical equipment", "earlier closing time", "Lab"),
                OptionCard("D", "Online Session", {"travel": "none", "interaction": "video call", "equipment": "personal devices"}, "no travel", "less social interaction", "Online"),
            ],
            ["The group meets on Saturday.", "Everyone needs access to the same shared materials."],
        ),
        "presentation": Scenario(
            "Choose the format for a project presentation",
            [
                OptionCard("A", "Slide Presentation", {"rehearsal": "easy", "audience interaction": "questions at end", "technical dependence": "low"}, "predictable and easy to rehearse", "limited live demonstration", "Slides"),
                OptionCard("B", "Live Demonstration", {"audience interaction": "high", "technical dependence": "working prototype", "backup": "none specified"}, "shows the system directly", "technical failure risk", "Live Demo"),
                OptionCard("C", "Recorded Screencast", {"editing": "possible", "audience interaction": "low", "reliability": "playback file"}, "can be rehearsed and edited", "less audience interaction", "Screencast"),
                OptionCard("D", "Poster Session", {"material": "printed poster", "audience interaction": "informal", "mobility": "standing discussion"}, "supports informal questions", "requires printing and standing discussion", "Poster"),
            ],
            ["The audience consists of students and two instructors.", "Only one presentation format may be submitted."],
        ),
        "restaurant": Scenario(
            "Choose a restaurant for dinner",
            [
                OptionCard("A", "Green Table", {"dietary coverage": "broad", "seating": "one large indoor table", "reservation reliability": "confirmed"}, "broad dietary coverage", "limited outdoor seating", "Green Table"),
                OptionCard("B", "Harbor Grill", {"dietary coverage": "few vegetarian mains", "seating": "large group booths", "noise level": "lively"}, "large group tables", "few vegetarian mains", "Harbor Grill"),
                OptionCard("C", "Old Town Pasta", {"menu variety": "pasta-focused", "seating": "two adjacent tables", "reservation reliability": "walk-in only"}, "simple familiar menu", "group may be split across tables", "Pasta"),
                OptionCard("D", "Market Kitchen", {"menu variety": "small seasonal menu", "seating": "communal table", "dietary coverage": "moderate"}, "central location", "smaller menu", "Market"),
            ],
            ["The group meets at 19:00.", "Everyone wants to eat together."],
        ),
        "cleaning": Scenario(
            "Choose a household cleaning upgrade",
            [
                OptionCard("A", "Robot Vacuum", {"coverage": "floors", "maintenance": "empty dust bin", "storage": "charging dock"}, "reduces routine floor work", "does not clean dishes", "Robot"),
                OptionCard("B", "Weekly Cleaner", {"coverage": "several rooms", "schedule flexibility": "fixed weekly slot", "privacy": "external person enters home"}, "covers several rooms", "requires a fixed appointment", "Cleaner"),
                OptionCard("C", "Dishwasher Upgrade", {"coverage": "dishes", "installation": "kitchen fitting", "capacity": "12 place settings"}, "removes daily dishwashing", "does not clean floors", "Dishwasher"),
                OptionCard("D", "Shared Chore Plan", {"coverage": "all agreed chores", "coordination": "shared rota", "reliability": "depends on participation"}, "no purchase required", "requires consistent participation", "Chore Plan"),
            ],
            ["The household wants one primary upgrade.", "Storage space is limited."],
        ),
        "hike": Scenario(
            "Choose a day hike for the group",
            [
                OptionCard("A", "Lake Loop", {"difficulty": "easy", "terrain": "wide paths", "shade": "frequent"}, "manageable route with lake views", "less challenging", "Lake Loop"),
                OptionCard("B", "Ridge Trail", {"difficulty": "hard", "terrain": "steep rocky sections", "exposure": "open ridge"}, "wide mountain views", "steep sections", "Ridge"),
                OptionCard("C", "Wilderness Route", {"difficulty": "very hard", "navigation": "marked sparsely", "facilities": "none"}, "remote natural setting", "long and physically demanding", "Wilderness"),
                OptionCard("D", "Forest Path", {"difficulty": "easy", "terrain": "compact gravel", "accessibility": "most accessible"}, "short and accessible", "few panoramic views", "Forest"),
            ],
            ["The group has one full Saturday.", "Everyone must complete the same route."],
        ),
        "book_club": Scenario(
            "Choose the next book-club novel",
            [
                OptionCard("A", "The Silent Guest", {"genre": "mystery", "narrative style": "fast plot", "availability": "paperback"}, "suspenseful discussion material", "darker subject matter", "Silent Guest"),
                OptionCard("B", "Small Days", {"genre": "contemporary", "narrative style": "character-focused", "availability": "paperback and ebook"}, "short and character-focused", "slower pacing", "Small Days"),
                OptionCard("C", "Orbit of Ash", {"genre": "science fiction", "themes": "technology and identity", "availability": "ebook first"}, "rich speculative ideas", "longest book", "Orbit"),
                OptionCard("D", "The Garden Letters", {"genre": "historical", "narrative style": "letters", "availability": "paperback"}, "accessible historical setting", "less plot-driven", "Garden Letters"),
            ],
            ["The club has four weeks before the meeting.", "Members want one title available to everyone."],
        ),
        "laptop": Scenario(
            "Choose a shared project laptop",
            [
                OptionCard("A", "Performance Laptop", {"compute": "strongest", "repairability": "upgradeable memory", "ports": "full-size ports"}, "strongest compute performance", "highest weight and power use", "Performance"),
                OptionCard("B", "Battery Laptop", {"battery": "longest", "repairability": "sealed battery", "display": "standard brightness"}, "longest battery life", "lower graphics performance", "Battery"),
                OptionCard("C", "Budget Laptop", {"repairability": "replaceable storage", "memory": "least", "sustainability": "refurbished option"}, "lowest purchase price", "least memory", "Budget"),
                OptionCard("D", "Ultralight Laptop", {"weight": "lightest", "ports": "two compact ports", "display": "smallest"}, "easiest to carry", "smaller screen", "Ultralight"),
            ],
            ["The laptop will be shared for one academic year.", "It must run the project development tools."],
        ),
        "flight": Scenario(
            "Book a flight from Miami to Stockholm",
            [
                OptionCard("A", "Direct Premium Flight", {"transfers": "none", "baggage": "checked bag included", "refundability": "partial"}, "lowest transfer risk", "highest fare", "Direct"),
                OptionCard("B", "One-Stop Saver", {"transfers": "one", "baggage": "carry-on only", "refundability": "change fee"}, "balanced itinerary", "connection required", "Saver"),
                OptionCard("C", "Overnight Connection", {"transfers": "one overnight", "baggage": "checked bag included", "seat choice": "included"}, "overnight schedule", "connection required", "Overnight"),
                OptionCard("D", "Two-Stop Budget Flight", {"transfers": "two", "baggage": "personal item only", "refundability": "none"}, "lowest fare", "highest transfer risk", "Budget Flight"),
            ],
            ["All flights leave on the same date.", "Each traveller needs the same itinerary."],
        ),
        "meeting": Scenario(
            "Choose a format for a monthly team meeting",
            [
                OptionCard("A", "Office Meeting", {"interaction": "fully in person", "privacy": "private room", "equipment": "shared whiteboard"}, "strong face-to-face interaction", "commuting required", "Office"),
                OptionCard("B", "Hybrid Meeting", {"interaction": "mixed", "equipment": "conference camera", "accessibility": "remote or in person"}, "flexible attendance", "uneven remote participation", "Hybrid"),
                OptionCard("C", "Online Meeting", {"interaction": "video call", "privacy": "personal locations", "recording": "available"}, "no commute", "less informal interaction", "Online"),
                OptionCard("D", "Offsite Workshop", {"interaction": "facilitated exercises", "preparation": "agenda and materials", "accessibility": "travel required"}, "dedicated collaborative time", "largest preparation burden", "Offsite"),
            ],
            ["The team has eight members.", "The same format will be used for three months."],
        ),
    }
    return scenarios[key]


PREFERENCE_CONTEXT: dict[str, dict[str, tuple[str, str]]] = {
    "weekend": {
        "A": ("enjoys exhibitions and dislikes weather-dependent plans", "wants a calm activity that is easy to organize"),
        "B": ("spends most weekdays indoors", "wants relaxed outdoor time with the group"),
        "C": ("likes cooperative puzzles", "wants an activity with active group interaction"),
        "D": ("has had a tiring week", "wants a low-effort plan"),
    },
    "study": {
        "A": ("loses concentration in unpredictable spaces", "needs a quiet and stable place"),
        "B": ("works better in informal surroundings", "values a relaxed atmosphere and late access"),
        "C": ("works with hardware prototypes", "needs dependable technical equipment"),
        "D": ("has a long commute", "wants to avoid travel"),
    },
    "presentation": {
        "A": ("prefers carefully rehearsed delivery", "wants a predictable presentation format"),
        "B": ("built most of the working prototype", "wants the audience to see the system directly"),
        "C": ("is comfortable editing video", "wants to reduce live technical risk"),
        "D": ("enjoys informal one-to-one discussion", "wants room for audience questions"),
    },
    "restaurant": {
        "A": ("often coordinates mixed dietary needs", "wants everyone to have a suitable meal"),
        "B": ("expects a larger social group", "prioritizes comfortable group seating"),
        "C": ("is watching personal spending", "prioritizes the lowest meal cost"),
        "D": ("has limited time before another appointment", "prioritizes the shortest journey"),
    },
    "cleaning": {
        "A": ("does most of the routine vacuuming", "wants to reduce daily floor work"),
        "B": ("prefers professional cleaning", "wants several rooms handled at once"),
        "C": ("handles most of the dishes", "needs dishwashing removed from the routine"),
        "D": ("does not want another purchase", "prioritizes a no-cost solution"),
    },
    "hike": {
        "A": ("has moderate hiking experience", "wants a manageable route with good scenery"),
        "B": ("regularly hikes on weekends", "prioritizes mountain views"),
        "C": ("trains for long-distance hikes", "wants a demanding wilderness route"),
        "D": ("is recovering from a minor knee strain", "needs a short accessible route"),
    },
    "book_club": {
        "A": ("likes plot-driven mysteries", "wants a book that keeps the discussion lively"),
        "B": ("has limited reading time this month", "wants a shorter character-focused book"),
        "C": ("reads speculative fiction frequently", "wants rich ideas to debate"),
        "D": ("enjoys historical settings", "wants an accessible period story"),
    },
    "laptop": {
        "A": ("runs compute-heavy development tools", "prioritizes performance"),
        "B": ("often works away from power outlets", "prioritizes battery life"),
        "C": ("manages the project budget", "prioritizes the lowest sufficient cost"),
        "D": ("carries the laptop between campuses", "prioritizes low weight"),
    },
    "flight": {
        "A": ("has little tolerance for long travel days", "prioritizes the shortest trip"),
        "B": ("has a moderate travel budget", "wants a reasonable balance of price and duration"),
        "C": ("can sleep during overnight travel", "prefers the overnight schedule"),
        "D": ("has the tightest budget", "prioritizes the lowest fare"),
    },
    "meeting": {
        "A": ("values informal face-to-face discussion", "wants strong in-person interaction"),
        "B": ("coordinates colleagues in different locations", "prioritizes flexible attendance"),
        "C": ("has a long commute and many short meetings", "wants a concise no-travel format"),
        "D": ("facilitates collaborative planning", "wants dedicated workshop time"),
    },
}


def _speech_style(age: int) -> str:
    if age <= 27:
        return "young casual wording"
    if age <= 40:
        return "relaxed practical wording"
    if age <= 58:
        return "direct workplace wording"
    return "measured traditional wording"


def personas_for(case: EvalCase, scenario: Scenario) -> list[Persona]:
    names = ("Nora", "Ben", "Mira", "Omar", "Lea", "Tariq", "Sofia")
    ages = (24, 34, 49, 65, 29, 42, 57)
    contexts = PREFERENCE_CONTEXT[case.scenario_key]
    personas: list[Persona] = []

    for index, preferred in enumerate(case.preferences):
        hard = index == case.hard_blocker_index
        engagement = case.engagements[index] if case.engagements else 3
        verbosity = case.verbosities[index] if case.verbosities else 3
        directness = case.directness[index] if case.directness else 3
        stubbornness = 5 if hard else (case.stubbornness[index] if case.stubbornness else 2)
        background, goal = contexts[preferred]

        stances: dict[str, OptionStance] = {}
        for option in scenario.options:
            if option.id == preferred:
                stances[option.id] = OptionStance(option.id, STANCE_PREFERRED, option.upside or goal, option.concern)
            elif hard:
                stances[option.id] = OptionStance(option.id, STANCE_REJECTED, "", option.concern or "violates a non-negotiable requirement")
            elif index in case.dislike_alternatives_for:
                stances[option.id] = OptionStance(option.id, STANCE_DISLIKED, option.upside, option.concern or "does not fit the priority")
            else:
                stances[option.id] = OptionStance(option.id, STANCE_NEUTRAL, option.upside, option.concern)

        personas.append(Persona(
            id=f"p{index + 1}",
            name=names[index],
            sim_params=SimulatorParameters(engagement, verbosity, directness, stubbornness).validated(hard_blocker=hard),
            background=f"{names[index]} {background}.",
            private_goal=goal,
            preferred_options=[preferred],
            age=ages[index],
            speech_style=_speech_style(ages[index]),
            rejection=None,
            rejection_reason="only the preferred option satisfies the requirement" if hard else "",
            option_stances=stances,
            hard_blocker=hard,
        ))
    return personas


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.casefold()).split())


def _restricted_start_rate(state) -> float:
    names = [persona.name for persona in state.personas]
    option_names = [
        value
        for option in state.scenario.options
        for value in (option.name, option.short_name)
        if value
    ]
    relevant = [
        turn for turn in state.participant_turns
        if turn.phase in {Phase.DISCUSSION, Phase.NARROWING}
    ]
    if not relevant:
        return 0.0
    count = 0
    for turn in relevant:
        text = turn.text.lstrip(" \"'“”‘’")
        if re.match(r"^(?:I\b|I['’](?:m|d|ll|ve)\b|My\b|For me\b)", text, re.I):
            count += 1
            continue
        if any(re.match(rf"^{re.escape(name)}\b", text, re.I) for name in names):
            count += 1
            continue
        if any(re.match(rf"^{re.escape(name)}\b", text, re.I) for name in option_names):
            count += 1
    return round(count / len(relevant), 3)


def _same_speaker_repeat_count(state) -> int:
    previous: dict[str, str] = {}
    repeats = 0
    for turn in state.participant_turns:
        prior = previous.get(turn.speaker_id)
        if prior:
            similarity = SequenceMatcher(None, _normalized(prior), _normalized(turn.text)).ratio()
            if similarity >= 0.70:
                repeats += 1
        previous[turn.speaker_id] = turn.text
    return repeats


def evaluate_case(case: EvalCase, llm) -> dict[str, Any]:
    old_moderator = cfg.moderator.enabled
    cfg.moderator.enabled = case.moderator
    try:
        scenario = scenario_for(case.scenario_key)
        personas = personas_for(case, scenario)
        runner = DialogueRunner(
            "",
            scenario=scenario,
            personas=personas,
            llm=llm,
            logger=DialogueLogger(case.id),
            rng=random.Random(case.seed),
            seed=case.seed,
        )
        result = runner.run()
    finally:
        cfg.moderator.enabled = old_moderator

    state = result.state
    metrics = flat_metrics_for(state, result.outcome)
    detailed = metrics_for(state, result.outcome)
    participant_turns = state.participant_turns
    openings = sum(turn.action and turn.action.act is ActionType.OPENING for turn in participant_turns)
    narrowing_turns = sum(turn.phase is Phase.NARROWING for turn in participant_turns)

    direct_sequence_ok = True
    for index, turn in enumerate(participant_turns[:-1]):
        if turn.action and turn.action.act is ActionType.ASK and turn.action.addressee_id:
            next_turn = participant_turns[index + 1]
            direct_sequence_ok &= bool(
                next_turn.speaker_id == turn.action.addressee_id
                and next_turn.action
                and next_turn.action.act is ActionType.ANSWER
            )

    final_records = state.vote_records.get(state.vote_round, {})
    votes_valid = len(final_records) == len(personas) and all(
        record.status is VoteStatus.VALID for record in final_records.values()
    )
    hard_ok = all(
        state.votes.get(persona.id) == persona.preferred_option
        for persona in personas if persona.hard_blocker
    )
    moderator_ok = case.moderator == any(turn.moderator for turn in state.turns)
    expected_ok = case.expected_outcome is None or result.outcome.status == case.expected_outcome
    revote_has_movement = state.vote_round < 2 or metrics["narrowing_movements"] > 0
    repair_rate_ok = metrics["repairs"] / max(1, metrics["participant_turns"]) <= 0.25
    movement_commit_ok = (
        metrics["selected_movement_actions"] == metrics["committed_movement_actions"]
    )
    quality_ok = all((
        metrics["visible_switches"] >= case.min_switches,
        metrics["concerns_resolved"] >= case.min_resolved_concerns,
        metrics["issues_stale"] >= case.min_stale_issues,
        narrowing_turns >= case.min_narrowing_turns,
        revote_has_movement,
        metrics["unexplained_movements"] == 0,
        movement_commit_ok,
        repair_rate_ok,
    ))
    structural = all((
        state.phase.value == "CLOSED",
        openings == len(personas),
        direct_sequence_ok,
        votes_valid,
        not state.vote_protocol_degraded,
        hard_ok,
        moderator_ok,
        state.vote_round <= 2,
        movement_commit_ok,
        metrics["participant_turns"] <= case.max_participant_turns,
    ))

    participants = detailed["participants"]
    row: dict[str, Any] = {
        "case": case.id,
        "scenario": case.scenario_key,
        "participant_count": len(personas),
        "outcome": result.outcome.status,
        "final_option": result.outcome.final_option or "",
        "vote_round": state.vote_round,
        **metrics,
        "narrowing_turns": narrowing_turns,
        "restricted_start_rate": _restricted_start_rate(state),
        "same_speaker_repeats": _same_speaker_repeat_count(state),
        "openings": openings,
        "direct_sequence_ok": direct_sequence_ok,
        "hard_blocker_ok": hard_ok,
        "moderator_ok": moderator_ok,
        "expected_outcome_ok": expected_ok,
        "quality_expectations_ok": quality_ok,
        "revote_has_movement": revote_has_movement,
        "repair_rate_ok": repair_rate_ok,
        "movement_commit_ok": movement_commit_ok,
        "structural_pass": structural,
        "case_pass": structural and expected_ok and quality_ok,
        "avg_prompt_tokens": round(
            sum(turn.prompt_tokens for turn in participant_turns) / max(1, len(participant_turns)), 1
        ),
        "voluntary_by_participant": json.dumps({pid: data["voluntary"] for pid, data in participants.items()}),
        "avg_words_by_participant": json.dumps({pid: round(data["avg_words"], 2) for pid, data in participants.items()}),
        "log_dir": result.log_paths["dir"],
        "llm_provider": str(getattr(llm, "provider", cfg.llm.dialogue)),
        "llm_model": str(getattr(llm, "model_id", cfg.llm.models[str(cfg.llm.dialogue)])),
    }
    return row


def policy_calibration() -> dict[str, Any]:
    return {
        "bid_probability_by_engagement": {str(level): bid_probability(level) for level in range(1, 6)},
        "movement_probability_by_stubbornness": {str(level): movement_probability(level) for level in range(1, 6)},
        "conversation_budgets_n3": cfg.conversation_turn_budgets(3),
        "conversation_budgets_n4": cfg.conversation_turn_budgets(4),
        "conversation_budgets_n7": cfg.conversation_turn_budgets(7),
    }


def write_summary(rows: list[dict[str, Any]], root: Path) -> tuple[Path, Path, Path]:
    csv_path = root / "eval_suite_runs.csv"
    json_path = root / "eval_suite_summary.json"
    md_path = root / "eval_suite_summary.md"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps({"cases": rows, "policy_calibration": policy_calibration()}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Focused LLM-backed evaluation suite",
        "",
        "| Case | Outcome | Turns | Narrow | Move | Comp. | Concerns R/S | Re-vote | Repairs | Tokens | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['outcome']} | {row['participant_turns']} | {row['narrowing_turns']} | "
            f"{row['narrowing_movements']} | {row['compromise_proposals']}/{row['compromise_acceptances']} | "
            f"{row['concerns_resolved']}/{row['concerns_stale']} | {row['vote_round']} | "
            f"{row['repairs']} | {row['tokens_in']} | {'yes' if row['case_pass'] else 'NO'} |"
        )
    lines += [
        "",
        f"Structural passes: {sum(bool(row['structural_pass']) for row in rows)}/{len(rows)}",
        f"Quality-expectation passes: {sum(bool(row['quality_expectations_ok']) for row in rows)}/{len(rows)}",
        f"Case passes: {sum(bool(row['case_pass']) for row in rows)}/{len(rows)}",
        f"Total input tokens: {sum(int(row['tokens_in']) for row in rows)}",
        f"Mean input tokens per case: {round(sum(int(row['tokens_in']) for row in rows) / max(1, len(rows)), 1)}",
        "",
        "A second vote is permitted only when the preceding re-narrowing produced visible acceptance or switching.",
        "Comp. reports compromise proposals/acceptances; every selected movement must commit and carry a stored grounded reason; repair-rate quality threshold is 25%.",
        "",
        "## Policy calibration",
        "",
        "```json",
        json.dumps(policy_calibration(), indent=2),
        "```",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def zip_logs(root: Path) -> Path:
    target = root.parent / "logs_eval_suite.zip"
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent))
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list cases without running the LLM")
    parser.add_argument("--case", action="append", dest="cases", help="run only the named case; repeatable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [case for case in CASES if not args.cases or case.id in set(args.cases)]
    if args.list:
        for case in selected:
            print(f"{case.id}: {case.why}")
        return 0
    if not selected:
        print("No matching cases.", file=sys.stderr)
        return 2

    log_root = ROOT / "eval" / "logs_eval_suite"
    if log_root.exists():
        shutil.rmtree(log_root)
    log_root.mkdir(parents=True)
    old_log_dir = cfg.output.log_dir
    cfg.output.log_dir = "eval/logs_eval_suite"
    try:
        llm = get_llm_client()
        print(f"Using dialogue LLM: {llm.provider} / {llm.model_id}")
        rows: list[dict[str, Any]] = []
        for case in selected:
            print(f"\n=== {case.id} ===\n{case.why}")
            rows.append(evaluate_case(case, llm))
    finally:
        cfg.output.log_dir = old_log_dir

    csv_path, json_path, md_path = write_summary(rows, log_root)
    zip_path = zip_logs(log_root)
    print(f"\nCase passes: {sum(bool(row['case_pass']) for row in rows)}/{len(rows)}")
    print(f"Summary: {md_path}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Archive: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
