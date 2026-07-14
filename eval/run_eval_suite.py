"""LLM-backed end-to-end evaluation suite for the autonomous runtime.

The configured dialogue LLM realizes selected structured actions. Bidding,
action choice, state updates, validation, issues, voting, and outcomes remain
seeded Python responsibilities.
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config_loader import cfg  # noqa: E402
import prompts  # noqa: E402
from dialogue import DialogueRunner, initialise_state  # noqa: E402
from eval.eval import flat_metrics_for  # noqa: E402
from logger import DialogueLogger, metrics_for  # noqa: E402
from llm_client import get_llm_client  # noqa: E402
from models import (  # noqa: E402
    ActionType,
    OptionCard,
    OptionStance,
    Persona,
    Scenario,
    SimulatorParameters,
    STANCE_DISLIKED,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
    StanceUpdate,
    StanceUpdateKind,
    UserAction,
    VoteStatus,
)
from simulator import UserSimulator, switch_probability  # noqa: E402
from validation import validate_realization  # noqa: E402


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
    stubbornness: tuple[int, ...] | None = None
    hard_blocker_index: int | None = None
    dislike_alternatives_for: tuple[int, ...] = ()
    expected_outcome: str | None = None


CASES = (
    EvalCase("easy_agreement", "Shared initial preference should close early without repetitive filler.", ("A", "A", "A"), 101, expected_outcome="successful"),
    EvalCase("three_way_split", "Independent preferences should narrow without controller-authored concessions.", ("A", "B", "C"), 102),
    EvalCase("normal_compromise", "Low-stubbornness participants can visibly accept or switch.", ("A", "B", "B"), 5, stubbornness=(1, 1, 2)),
    EvalCase(
        "majority_holdout",
        "A valid majority must close without unanimity repair.",
        ("A", "A", "A", "B"),
        104,
        stubbornness=(2, 2, 2, 4),
        dislike_alternatives_for=(0, 1, 2, 3),
        expected_outcome="majority",
    ),
    EvalCase("hard_blocker", "The sole blocker never switches or votes away from its option.", ("A", "A", "C"), 105, scenario_key="cleaning", hard_blocker_index=2),
    EvalCase("direct_question_followup", "A direct question must be answered and receive a real issue outcome.", ("A", "B", "A"), 106, engagements=(5, 4, 3)),
    EvalCase(
        "unresolved_concern",
        "A concern owner may explicitly maintain an objection.",
        ("A", "B", "C"),
        107,
        stubbornness=(4, 4, 4),
        dislike_alternatives_for=(0, 1, 2),
    ),
    EvalCase(
        "concern_resolution",
        "A low-stubbornness concern owner can evaluate a mitigation, resolve the issue, and accept the option.",
        ("A", "B", "B"),
        1,
        engagements=(5, 4, 3),
        stubbornness=(1, 2, 2),
        dislike_alternatives_for=(0,),
    ),
    EvalCase("no_moderator", "Protocol must run with no visible moderator turns.", ("A", "B", "A"), 108, moderator=False, scenario_key="restaurant"),
    EvalCase("grounding_sensitive", "Prepared flight facts and comparisons must remain grounded.", ("B", "C", "A"), 109, scenario_key="flight"),
    EvalCase("engagement_spread", "Unequal engagement should yield unequal voluntary participation.", ("A", "B", "C", "A"), 110, engagements=(5, 1, 2, 4)),
    EvalCase("verbosity_spread", "Comparable voluntary contributions should reflect word-budget differences.", ("A", "B", "A"), 111, verbosities=(1, 5, 3)),
    EvalCase("visible_stance_switch", "A committed preference switch must be visible in the transcript.", ("A", "B", "B"), 1, stubbornness=(1, 1, 1)),
    EvalCase(
        "persona_distinctness",
        "Different personal priorities should produce different structured reasons and occasional relevant personal context.",
        ("C", "B", "A", "D"),
        112,
        scenario_key="study",
    ),
    EvalCase(
        "no_majority_revote",
        "A split may use one complete re-vote and then close unresolved.",
        ("A", "A", "B", "B"),
        113,
        stubbornness=(4, 4, 4, 4),
        dislike_alternatives_for=(0, 1, 2, 3),
        expected_outcome="unresolved",
    ),
)


def scenario_for(key: str) -> Scenario:
    if key == "flight":
        return Scenario(
            topic="Book a flight from Miami to Stockholm",
            shared_context=["All flights leave on the same date.", "Prices are per passenger."],
            options=[
                OptionCard("A", "Direct Premium Flight", {"price": "750 dollars", "duration": "10 hours", "stops": "none"}, "shortest travel time", "highest price", "Direct"),
                OptionCard("B", "One-Stop Saver", {"price": "520 dollars", "duration": "13 hours", "stops": "one"}, "lower price", "longer travel time", "Saver"),
                OptionCard("C", "Overnight Connection", {"price": "600 dollars", "duration": "12 hours", "stops": "one"}, "overnight timing", "connection required", "Overnight"),
                OptionCard("D", "Two-Stop Budget Flight", {"price": "430 dollars", "duration": "16 hours", "stops": "two"}, "lowest price", "longest travel time", "Budget"),
            ],
        )
    if key == "cleaning":
        return Scenario(
            topic="Choose a household cleaning upgrade",
            shared_context=["The household wants one primary upgrade.", "All listed prices are within the available budget."],
            options=[
                OptionCard("A", "Robot Vacuum", {"price": "260 euros", "task": "daily floor cleaning", "setup": "clear floor paths"}, "reduces routine floor work", "does not clean dishes", "Robot"),
                OptionCard("B", "Weekly Cleaner", {"price": "80 euros per visit", "task": "full weekly cleaning", "schedule": "Saturday morning"}, "covers several rooms", "requires a fixed appointment", "Cleaner"),
                OptionCard("C", "Dishwasher Upgrade", {"price": "480 euros", "task": "dish cleaning", "capacity": "12 place settings"}, "removes daily dishwashing", "does not clean floors", "Dishwasher"),
                OptionCard("D", "Shared Chore Plan", {"price": "free", "task": "manual rotation", "schedule": "three sessions per week"}, "no purchase cost", "requires consistent participation", "Chore plan"),
            ],
        )
    if key == "restaurant":
        return Scenario(
            topic="Choose a restaurant for dinner",
            shared_context=["The group will meet at 19:00.", "The target budget is 30 euros per person."],
            options=[
                OptionCard("A", "Green Table", {"price": "24 euros", "travel": "15 minutes", "menu": "mixed vegetarian"}, "broad dietary coverage", "limited outdoor seating", "Green Table"),
                OptionCard("B", "Harbor Grill", {"price": "29 euros", "travel": "20 minutes", "menu": "seafood and meat"}, "large group tables", "few vegetarian mains", "Harbor Grill"),
                OptionCard("C", "Old Town Pasta", {"price": "22 euros", "travel": "25 minutes", "menu": "Italian"}, "lowest meal price", "longest travel time", "Pasta"),
                OptionCard("D", "Market Kitchen", {"price": "27 euros", "travel": "10 minutes", "menu": "seasonal"}, "shortest travel time", "smaller menu", "Market"),
            ],
        )
    return Scenario(
        topic="Choose a Saturday study location",
        shared_context=["The group meets on Saturday.", "The budget is capped at 20 euros per person."],
        options=[
            OptionCard("A", "Central Library", {"cost": "free", "closing time": "20:00", "equipment": "standard desks"}, "quiet and predictable", "can become crowded", "Library"),
            OptionCard("B", "Riverside Cafe", {"cost": "8 euros", "closing time": "22:00", "noise": "moderate"}, "relaxed atmosphere", "background noise", "Cafe"),
            OptionCard("C", "Engineering Lab", {"cost": "free", "closing time": "19:00", "equipment": "specialist workstations"}, "reliable technical equipment", "earlier closing time", "Lab"),
            OptionCard("D", "Online Session", {"cost": "free", "travel": "none", "access": "from home"}, "no travel", "less social interaction", "Online"),
        ],
    )


def personas_for(case: EvalCase, scenario: Scenario) -> list[Persona]:
    names = ("Nora", "Ben", "Mira", "Omar", "Lea", "Tariq", "Sofia")
    goals = {
        "study": "needs a location that supports focused project work",
        "flight": "needs a practical balance of travel time and price",
        "cleaning": "wants the upgrade to remove the most frustrating recurring chore",
        "restaurant": "wants a convenient dinner that fits the group budget",
    }
    distinct_profiles = (
        ("works with hardware prototypes and needs the lab tools", "needs specialist equipment"),
        ("can only join after an evening work shift", "cannot arrive before 19:00"),
        ("loses concentration in busy rooms", "strongly avoids noisy environments"),
        ("has a long commute to campus", "prioritizes avoiding travel"),
    )
    result: list[Persona] = []
    for index, preferred in enumerate(case.preferences):
        hard = index == case.hard_blocker_index
        engagement = case.engagements[index] if case.engagements else 3
        verbosity = case.verbosities[index] if case.verbosities else 3
        stubbornness = case.stubbornness[index] if case.stubbornness else 2
        stances: dict[str, OptionStance] = {}
        for option in scenario.options:
            if option.id == preferred:
                stances[option.id] = OptionStance(option.id, STANCE_PREFERRED, option.upside, "")
            elif hard:
                stances[option.id] = OptionStance(option.id, STANCE_REJECTED, "", option.concern or "it violates the non-negotiable requirement")
            elif index in case.dislike_alternatives_for:
                stances[option.id] = OptionStance(option.id, STANCE_DISLIKED, option.upside, option.concern or "it does not fit the priority well enough")
            else:
                stances[option.id] = OptionStance(option.id, STANCE_NEUTRAL, option.upside, option.concern)
        result.append(Persona(
            id=f"p{index + 1}",
            name=names[index],
            sim_params=SimulatorParameters(
                engagement=engagement,
                verbosity=verbosity,
                directness=3 if case.id == "persona_distinctness" else 1 + (index * 2) % 5,
                stubbornness=5 if hard else stubbornness,
            ).validated(hard_blocker=hard),
            background=(
                f"{names[index]} {distinct_profiles[index][0]}."
                if case.id == "persona_distinctness"
                else f"{names[index]} is making this decision with the group."
            ),
            private_goal=(
                distinct_profiles[index][1]
                if case.id == "persona_distinctness"
                else goals[case.scenario_key]
            ),
            preferred_options=[preferred],
            age=22 + index * 9,
            speech_style=("young casual wording", "relaxed practical wording", "direct workplace wording", "measured traditional wording")[index % 4],
            rejection_reason="alternatives violate a non-negotiable personal requirement" if hard else "",
            option_stances=stances,
            hard_blocker=hard,
        ))
    return result


@contextmanager
def runtime_settings(*, moderator: bool) -> Iterator[None]:
    scalar_changes = {
        "min_voluntary_turns": 6,
        "soft_target_voluntary_turns": 12,
        "hard_max_voluntary_turns": 18,
        "narrowing_voluntary_turns": 3,
        "revote_narrowing_voluntary_turns": 2,
    }
    old_scalars = {key: getattr(cfg.conversation, key) for key in scalar_changes}
    old_log_dir = cfg.output.log_dir
    old_moderator_attr = cfg.moderator.enabled
    old_moderator_raw = cfg._raw["moderator"]["enabled"]
    try:
        for key, value in scalar_changes.items():
            setattr(cfg.conversation, key, value)
        cfg.output.log_dir = "eval/logs_eval_suite"
        cfg.moderator.enabled = moderator
        cfg._raw["moderator"]["enabled"] = moderator
        yield
    finally:
        for key, value in old_scalars.items():
            setattr(cfg.conversation, key, value)
        cfg.output.log_dir = old_log_dir
        cfg.moderator.enabled = old_moderator_attr
        cfg._raw["moderator"]["enabled"] = old_moderator_raw


def _policy_calibration() -> dict[str, Any]:
    sc = scenario_for("study")
    bid_counts: dict[int, int] = {}
    for engagement in (1, 3, 5):
        total = 0
        for seed in range(5):
            case = EvalCase("calibration", "", ("A",), seed, engagements=(engagement,))
            persona = personas_for(case, sc)[0]
            state = initialise_state(sc, [persona])
            simulator = UserSimulator(persona, random.Random(seed + 1000))
            total += sum(simulator.propose(state).wants_to_speak for _ in range(300))
        bid_counts[engagement] = total
    switch = {level: switch_probability(level, 0.8) for level in (1, 2, 3, 4)}

    diversity_case = EvalCase("policy_diversity", "", ("A", "B", "C"), 902)
    diversity_personas = personas_for(diversity_case, sc)
    diversity_state = initialise_state(sc, diversity_personas)
    diversity_state.runtimes["p2"].public_preference = "B"
    diversity_state.public_supporters["B"].add("p2")
    diversity_state.runtimes["p3"].public_preference = "C"
    diversity_state.public_concern_raisers["C"].add("p3")
    simulator = UserSimulator(diversity_personas[0], random.Random(903))
    runtime = diversity_state.runtimes["p1"]
    question_keys: list[str] = []
    for _ in range(8):
        action = simulator._ask_action(diversity_state, runtime)
        if action is None:
            break
        if action.question_key:
            question_keys.append(action.question_key)
            runtime.asked_question_keys.add(action.question_key)
    reason_candidates = simulator._positive_reason_candidates(diversity_state, runtime.preferred_option)

    return {
        "engagement_bid_counts": bid_counts,
        "engagement_monotonic": bid_counts[5] > bid_counts[3] > bid_counts[1],
        "switch_probabilities": switch,
        "stubbornness_monotonic": switch[1] > switch[2] > switch[3] > switch[4],
        "distinct_question_keys": question_keys,
        "question_key_diversity": len(set(question_keys)),
        "available_reason_sources": len(reason_candidates),
    }


def _realization_calibration(llm) -> dict[str, Any]:
    """Generate isolated language-trait diagnostics from one fixed action."""
    sc = scenario_for("study")
    base_case = EvalCase("realization_calibration", "", ("A",), 901)
    base = personas_for(base_case, sc)[0]
    action = UserAction(
        "p1", True, 0.7, ActionType.SUPPORT, ("A",),
        reason="quiet and predictable",
    )

    def realize(*, verbosity: int = 3, directness: int = 3, age: int = 35, style: str = "relaxed practical wording") -> dict[str, Any]:
        persona = Persona(
            id=base.id, name=base.name,
            sim_params=SimulatorParameters(3, verbosity, directness, 2).validated(),
            background=base.background, private_goal=base.private_goal,
            preferred_options=list(base.preferred_options), age=age,
            speech_style=style, option_stances=dict(base.option_stances),
        )
        state = initialise_state(sc, [persona])
        text = llm.generate(prompts.realization_prompt(state, persona, action), profile="dialogue").strip()
        return {"text": text, "word_count": len(text.split())}

    result = {
        "verbosity_1": realize(verbosity=1),
        "verbosity_5": realize(verbosity=5),
        "directness_1": realize(directness=1),
        "directness_5": realize(directness=5),
        "style_young": realize(age=22, style="young casual wording"),
        "style_measured": realize(age=65, style="measured traditional wording"),
    }
    result["verbosity_monotonic"] = (
        result["verbosity_5"]["word_count"] > result["verbosity_1"]["word_count"]
    )
    result["directness_qualified_terms"] = len(re.findall(
        r"\b(?:maybe|perhaps|might|could|somewhat|I think|I feel)\b",
        result["directness_1"]["text"], re.I,
    ))
    result["directness_explicit_terms"] = len(re.findall(
        r"\b(?:clearly|definitely|need|must|I prefer|I support)\b",
        result["directness_5"]["text"], re.I,
    ))

    switch_state = initialise_state(sc, [base])
    switch_runtime = switch_state.runtimes[base.id]
    switch_runtime.preferred_option = "B"
    switch_runtime.public_preference = "B"
    vote_action = UserAction(
        base.id, True, 1.0, ActionType.VOTE, ("A",),
        reason="the discussion changed the balance", vote_option="A",
        stance_update=StanceUpdate(
            StanceUpdateKind.SWITCH_PREFERRED, "A", previous_option_id="B"
        ),
    )
    vote_text = llm.generate(
        prompts.realization_prompt(switch_state, base, vote_action), profile="dialogue"
    ).strip()
    vote_validation = validate_realization(vote_text, switch_state, base, vote_action)
    result["formal_vote_switch"] = {
        "text": vote_text,
        "valid": vote_validation.ok,
        "errors": vote_validation.errors,
    }
    return result


def vote_protocol_flags(state, participant_count: int) -> tuple[bool, bool]:
    """Return (all required attempts recorded, final protocol valid).

    A generation failure or unclear validator result is a protocol degradation,
    not an intentional abstention. Explicit abstentions remain valid protocol
    outcomes even though the current simulator policy normally chooses a vote.
    """
    attempts_complete = bool(state.vote_records) and all(
        len(records) == participant_count for records in state.vote_records.values()
    )
    final_records = state.vote_records.get(state.vote_round, {})
    final_valid = (
        len(final_records) == participant_count
        and all(record.status in {VoteStatus.VALID, VoteStatus.ABSTAINED} for record in final_records.values())
    )
    return attempts_complete, final_valid


def _narrowing_before_direct_answer(state) -> int:
    violations = 0
    turns = state.turns
    for index, turn in enumerate(turns):
        action = turn.action
        if not action or action.act is not ActionType.ASK or not action.addressee_id:
            continue
        for later in turns[index + 1:]:
            if (
                later.action is not None
                and later.action.act is ActionType.ANSWER
                and later.speaker_id == action.addressee_id
            ):
                break
            if later.phase.value == "NARROWING":
                violations += 1
                break
    return violations


def _rapid_switch_count(state) -> int:
    by_speaker: dict[str, list[int]] = {}
    for turn in state.participant_turns:
        if (
            turn.stance_update is not None
            and turn.stance_update.kind is StanceUpdateKind.SWITCH_PREFERRED
        ):
            by_speaker.setdefault(turn.speaker_id, []).append(turn.index)
    return sum(
        current - previous < 3
        for indices in by_speaker.values()
        for previous, current in zip(indices, indices[1:])
    )


def _duplicate_structured_reasons(state) -> int:
    counts: dict[tuple[Any, ...], int] = {}
    for turn in state.participant_turns:
        action = turn.action
        if action is None or not action.reason:
            continue
        source = (
            action.reason_source.option_id,
            action.reason_source.attribute_name,
            action.reason_source.public_value,
        ) if action.reason_source else (action.reason.casefold().strip(),)
        key = (turn.speaker_id, action.act.value, *source)
        counts[key] = counts.get(key, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def _duplicate_question_keys(state) -> int:
    counts: dict[tuple[str, str], int] = {}
    for turn in state.participant_turns:
        if turn.action and turn.action.question_key:
            key = (turn.speaker_id, turn.action.question_key)
            counts[key] = counts.get(key, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def evaluate_case(case: EvalCase, llm) -> dict[str, Any]:
    llm.reset_session()
    sc = scenario_for(case.scenario_key)
    personas = personas_for(case, sc)
    with runtime_settings(moderator=case.moderator):
        runner = DialogueRunner(
            "",
            scenario=sc,
            personas=personas,
            llm=llm,
            logger=DialogueLogger(case.id),
            rng=random.Random(case.seed),
            seed=case.seed,
        )
        result = runner.run()
    detailed = metrics_for(result.state, result.outcome)
    row = {"case": case.id, "why": case.why, "scenario": case.scenario_key, **flat_metrics_for(result.state, result.outcome)}
    row["log_dir"] = result.log_paths["dir"]
    row["vote_round"] = result.state.vote_round
    row["direct_answers"] = sum(
        turn.action is not None and turn.action.act is ActionType.ANSWER and turn.mandatory
        for turn in result.state.participant_turns
    )
    row["hard_blocker_ok"] = all(
        result.state.votes.get(persona.id) == persona.preferred_option
        and result.state.runtimes[persona.id].preferred_option == persona.preferred_option
        for persona in result.state.personas if persona.hard_blocker
    )
    row["moderator_ok"] = case.moderator or row["moderator_turns"] == 0
    row["closed"] = result.state.phase.value == "CLOSED"
    row["max_one_revote"] = result.state.vote_round <= 2
    row["voluntary_by_id"] = json.dumps(detailed["turns"]["voluntary_turns_by_id"], sort_keys=True)
    row["avg_voluntary_words_by_id"] = json.dumps(detailed["turns"]["average_voluntary_words_by_id"], sort_keys=True)
    row["avg_comparable_voluntary_words_by_id"] = json.dumps(
        detailed["turns"]["average_comparable_voluntary_words_by_id"], sort_keys=True
    )
    row["action_counts"] = json.dumps(detailed["turns"]["action_counts"], sort_keys=True)
    vote_language = re.compile(r"\b(?:vote(?:d|s|ing)?|my\s+vote|ballot)\b", re.I)
    action_label_language = re.compile(
        r"\bi\s+(?:open\s+the\s+discussion(?:\s+by)?|acknowledge\b|compare\b)",
        re.I,
    )
    row["premature_vote_turns"] = sum(
        bool(turn.action)
        and turn.action.act is not ActionType.VOTE
        and bool(vote_language.search(turn.text))
        for turn in result.state.participant_turns
    )
    row["exposed_action_label_turns"] = sum(
        bool(action_label_language.search(turn.text))
        for turn in result.state.participant_turns
    )
    row["coverage_prompt_used"] = bool(result.state.coverage_prompt_used)
    row["question_followups"] = max(
        (issue.follow_up_count for issue in result.state.issue_history if issue.kind.value == "question"),
        default=0,
    )
    row["narrowing_before_direct_answer"] = _narrowing_before_direct_answer(result.state)
    row["rapid_switches"] = _rapid_switch_count(result.state)
    row["duplicate_structured_reasons"] = _duplicate_structured_reasons(result.state)
    row["duplicate_question_keys"] = _duplicate_question_keys(result.state)
    row["repair_rate"] = detailed["generation"]["repair_rate"]
    row["drop_rate"] = detailed["generation"]["drop_rate"]
    row["vote_switch_attempts"] = detailed["generation"]["vote_switch_attempts"]
    row["vote_switch_failures"] = detailed["generation"]["vote_switch_failures"]
    relevant_actions = [
        turn.action for turn in result.state.participant_turns
        if turn.action and turn.action.act in {
            ActionType.OPENING, ActionType.SUPPORT, ActionType.CONCERN,
            ActionType.ANSWER, ActionType.COMPARE, ActionType.COMPROMISE,
        }
    ]
    row["reason_source_rate"] = round(
        sum(action.reason_source is not None for action in relevant_actions) / max(1, len(relevant_actions)),
        3,
    )
    row["structured_reason_diversity"] = len({
        (action.reason_source.option_id, action.reason_source.attribute_name, action.reason_source.public_value)
        if action.reason_source else (action.reason.casefold().strip(),)
        for action in relevant_actions if action.reason
    })
    row["question_key_diversity"] = len({
        turn.action.question_key for turn in result.state.participant_turns
        if turn.action and turn.action.question_key
    })
    row["distinct_private_goals"] = len({persona.private_goal for persona in result.state.personas})
    row["relevant_concern_responders"] = detailed["issues"]["relevant_concern_responders"]
    row["llm_provider"] = str(getattr(llm, "provider", cfg.llm.dialogue))
    row["llm_model"] = str(getattr(llm, "model_id", cfg.llm.models[str(cfg.llm.dialogue)]))

    vote_round_attempts_complete, final_vote_protocol_valid = vote_protocol_flags(
        result.state, len(case.preferences)
    )
    final_records = result.state.vote_records.get(result.state.vote_round, {})
    row["vote_round_attempts_complete"] = vote_round_attempts_complete
    row["vote_round_valid"] = final_vote_protocol_valid
    # Retained aliases keep older analysis notebooks readable while the clearer
    # protocol terminology is used by the suite itself.
    row["vote_rounds_complete"] = vote_round_attempts_complete
    row["final_votes_all_valid"] = all(
        record.status is VoteStatus.VALID for record in final_records.values()
    ) and len(final_records) == len(case.preferences)
    row["vote_protocol_degraded"] = bool(
        result.state.vote_protocol_degraded
        or (vote_round_attempts_complete and not final_vote_protocol_valid)
    )
    row["structural_pass"] = all((
        row["closed"],
        row["openings"] == len(case.preferences),
        row["hard_blocker_ok"],
        row["moderator_ok"],
        row["max_one_revote"],
        row["repair_calls"] <= row["participant_turns"],
        vote_round_attempts_complete,
        final_vote_protocol_valid,
        not row["vote_protocol_degraded"],
        row["narrowing_focus_adherence"] >= 0.80,
        row["premature_vote_turns"] == 0,
        row["exposed_action_label_turns"] == 0,
        row["narrowing_before_direct_answer"] == 0,
        row["rapid_switches"] == 0,
        row["duplicate_question_keys"] == 0,
        row["vote_switch_failures"] == 0,
    ))

    voluntary = detailed["turns"]["voluntary_turns_by_id"]
    voluntary_words = detailed["turns"]["average_comparable_voluntary_words_by_id"]
    actions = detailed["turns"]["action_counts"]
    case_checks = {
        "easy_agreement": (
            result.outcome.status == "successful"
            and row["voluntary_turns"] <= 10
            and actions.get("compare", 0) <= 3
            and row["liveness_forced_turns"] <= 1
            and not row["coverage_prompt_used"]
            and row["premature_vote_turns"] == 0
        ),
        "normal_compromise": row["visible_switches"] > 0 or row["public_acceptances"] > 0,
        "majority_holdout": result.outcome.status == "majority" and result.state.vote_round == 1,
        "hard_blocker": row["hard_blocker_ok"],
        "direct_question_followup": (
            row["direct_answers"] >= 1
            and row["questions_answered"] >= 1
            and row["questions_resolved"] >= 1
            and row["question_followups"] >= 1
        ),
        "unresolved_concern": row["concerns_maintained"] >= 1,
        "concern_resolution": (
            row["concerns_resolved"] + row["concerns_partially_addressed"] >= 1
            and row["relevant_concern_responders"] >= 1
            and row["public_acceptances"] >= 1
            and row["vote_round_valid"]
        ),
        "no_moderator": row["moderator_turns"] == 0,
        "grounding_sensitive": (
            row["reason_source_rate"] >= 0.50
            and row["vote_round_valid"]
            and row["premature_vote_turns"] == 0
        ),
        "engagement_spread": voluntary.get("p1", 0) > voluntary.get("p2", 0),
        "verbosity_spread": (
            voluntary.get("p1", 0) > 0
            and voluntary.get("p2", 0) > 0
            and voluntary_words.get("p2", 0.0) > voluntary_words.get("p1", 0.0)
        ),
        "visible_stance_switch": row["visible_switches"] >= 1 and row["rapid_switches"] == 0,
        "persona_distinctness": (
            row["distinct_private_goals"] == len(case.preferences)
            and row["structured_reason_diversity"] >= len(case.preferences)
        ),
        "no_majority_revote": (
            result.outcome.status == "unresolved"
            and result.state.vote_round == 2
            and row["vote_round_valid"]
        ),
    }
    expected_outcome_ok = case.expected_outcome is None or result.outcome.status == case.expected_outcome
    row["case_pass"] = row["structural_pass"] and expected_outcome_ok and case_checks.get(case.id, True)
    return row


def write_summary(rows: list[dict[str, Any]], root: Path, calibration: dict[str, Any]) -> tuple[Path, Path, Path]:
    csv_path = root / "eval_suite_runs.csv"
    json_path = root / "eval_suite_summary.json"
    md_path = root / "eval_suite_summary.md"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"cases": rows, "policy_calibration": calibration}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# LLM-backed evaluation suite",
        "",
        f"Dialogue provider/model: {rows[0]['llm_provider']} / {rows[0]['llm_model']}" if rows else "",
        "The LLM realizes authoritative actions; seeded Python policies choose and commit them.",
        "",
        "| Case | Scenario | Outcome | Round | Turns | Voluntary | Repairs | Drops | Resolved issues | Switches | Vote protocol | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['scenario']} | {row['outcome']} | {row['vote_round']} | "
            f"{row['participant_turns']} | {row['voluntary_turns']} | {row['repairs']} | "
            f"{row['dropped_turns']} | {row['issues_resolved']} | {row['visible_switches']} | "
            f"{'DEGRADED' if row['vote_protocol_degraded'] else 'valid'} | "
            f"{'yes' if row['case_pass'] else 'NO'} |"
        )
    lines += [
        "",
        f"Structural passes: {sum(bool(row['structural_pass']) for row in rows)}/{len(rows)}",
        f"Case-specific passes: {sum(bool(row['case_pass']) for row in rows)}/{len(rows)}",
        f"Total repairs: {sum(int(row['repairs']) for row in rows)}",
        f"Total dropped turns: {sum(int(row['dropped_turns']) for row in rows)}",
        f"Vote-protocol degradations: {sum(bool(row['vote_protocol_degraded']) for row in rows)}",
        f"Rapid-switch violations: {sum(int(row['rapid_switches']) for row in rows)}",
        f"Direct-answer phase-boundary violations: {sum(int(row['narrowing_before_direct_answer']) for row in rows)}",
        "",
        "## Policy and isolated realization calibration",
        "",
        "```json",
        json.dumps(calibration, indent=2),
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


def main() -> int:
    log_root = ROOT / "eval" / "logs_eval_suite"
    if log_root.exists():
        shutil.rmtree(log_root)
    log_root.mkdir(parents=True)
    llm = get_llm_client()
    print(f"Using dialogue LLM: {llm.provider} / {llm.model_id}")
    calibration = _policy_calibration()
    calibration["realization_diagnostics"] = _realization_calibration(llm)
    rows: list[dict[str, Any]] = []
    for case in CASES:
        print(f"\n=== {case.id} ===\n{case.why}")
        rows.append(evaluate_case(case, llm))
    csv_path, json_path, md_path = write_summary(rows, log_root, calibration)
    zip_path = zip_logs(log_root)
    structural_passed = sum(bool(row["structural_pass"]) for row in rows)
    passed = sum(bool(row["case_pass"]) for row in rows)
    print(f"\nStructural passes: {structural_passed}/{len(rows)}")
    print(f"Case-specific passes: {passed}/{len(rows)}")
    print(f"Summary: {md_path}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Archive: {zip_path}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
