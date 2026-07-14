from __future__ import annotations

import json
import random
from typing import Any

from dialogue import DialogueRunner, initialise_state
from models import (
    ActionType,
    OptionCard,
    OptionStance,
    Persona,
    Scenario,
    SimulatorParameters,
    STANCE_NEUTRAL,
    STANCE_PREFERRED,
    STANCE_REJECTED,
)


def make_scenario() -> Scenario:
    return Scenario(
        topic="Choose a study location",
        shared_context=["The group meets on Saturday.", "The budget is capped at 20 euros per person."],
        options=[
            OptionCard(
                id="A", name="Central Library", short_name="Library",
                attrs={"cost": "free", "closing time": "20:00", "equipment": "standard desks"},
                upside="quiet and predictable", concern="can become crowded",
            ),
            OptionCard(
                id="B", name="Riverside Cafe", short_name="Cafe",
                attrs={"cost": "8 euros", "closing time": "22:00", "noise": "moderate"},
                upside="relaxed atmosphere", concern="background noise",
            ),
            OptionCard(
                id="C", name="Engineering Lab", short_name="Lab",
                attrs={"cost": "free", "closing time": "19:00", "equipment": "specialist workstations"},
                upside="reliable technical equipment", concern="earlier closing time",
            ),
            OptionCard(
                id="D", name="Online Session", short_name="Online",
                attrs={"cost": "free", "travel": "none", "access": "from home"},
                upside="no travel", concern="less social interaction",
            ),
        ],
    )


def make_persona(
    pid: str,
    name: str,
    preferred: str = "A",
    *,
    engagement: int = 3,
    verbosity: int = 3,
    directness: int = 3,
    stubbornness: int = 2,
    hard_blocker: bool = False,
    age: int = 32,
    speech_style: str = "relaxed practical wording",
) -> Persona:
    scenario = make_scenario()
    stances: dict[str, OptionStance] = {}
    for option_id in scenario.option_ids:
        if option_id == preferred:
            stances[option_id] = OptionStance(option_id, STANCE_PREFERRED, "fits my main priority", "")
        elif hard_blocker:
            stances[option_id] = OptionStance(option_id, STANCE_REJECTED, "", "conflicts with my non-negotiable requirement")
        else:
            stances[option_id] = OptionStance(option_id, STANCE_NEUTRAL, "", "")
    params = SimulatorParameters(
        engagement=engagement,
        verbosity=verbosity,
        directness=directness,
        stubbornness=5 if hard_blocker else stubbornness,
    ).validated(hard_blocker=hard_blocker)
    return Persona(
        id=pid,
        name=name,
        sim_params=params,
        background=f"{name} is working on a practical university project.",
        private_goal="needs a location that supports focused project work",
        preferred_options=[preferred],
        age=age,
        speech_style=speech_style,
        rejection=None,
        rejection_reason="will not accept an option that breaks the project requirement" if hard_blocker else "",
        option_stances=stances,
        hard_blocker=hard_blocker,
    )


def make_personas(preferences: tuple[str, ...] = ("A", "B", "C")) -> list[Persona]:
    names = ("Nora", "Ben", "Mira", "Omar", "Lea", "Tariq", "Sofia")
    return [make_persona(f"p{index + 1}", names[index], preference) for index, preference in enumerate(preferences)]


def make_state(preferences: tuple[str, ...] = ("A", "B", "C")):
    return initialise_state(make_scenario(), make_personas(preferences))


class NullLogger:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def write_prompt(self, prompt: str, kind: str) -> str:
        self.prompts.append((kind, prompt))
        return ""

    def write_run(self, *_args, **_kwargs) -> dict[str, str]:
        return {"dir": "", "transcript": "", "json": "", "metrics_csv": ""}


class ActionRendererLLM:
    """Offline renderer that follows the authoritative action in the prompt."""

    _modifiers = (
        "overall", "for my needs", "in practice", "on balance", "at this stage",
        "given the discussion", "from my perspective", "for the group", "as things stand",
        "after that point", "with the current trade-off", "for this decision",
    )

    def __init__(self, scripted: list[str] | None = None) -> None:
        self.scripted = list(scripted or [])
        self.prompts: list[str] = []
        self.profiles: list[str] = []
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        self.session_calls = 0
        self.calls = 0

    def reset_session(self) -> None:
        self.session_tokens_in = self.session_tokens_out = self.session_calls = 0

    def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
        self.prompts.append(prompt)
        self.profiles.append(profile)
        self.calls += 1
        if self.scripted:
            text = self.scripted.pop(0)
        else:
            action = self._action(prompt)
            text = self._render(action)
        self.last_tokens_in = max(1, len(prompt.split()))
        self.last_tokens_out = max(1, len(text.split()))
        self.session_tokens_in += self.last_tokens_in
        self.session_tokens_out += self.last_tokens_out
        self.session_calls += 1
        return text

    @staticmethod
    def _action(prompt: str) -> dict[str, Any]:
        if "AUTHORITATIVE ACTION:\n" in prompt:
            block = prompt.split("AUTHORITATIVE ACTION:\n", 1)[1].split("\nResolved addressee", 1)[0]
        else:
            block = prompt.split("Structured action: ", 1)[1].split("\nExact target", 1)[0]
        return json.loads(block)

    def _render(self, action: dict[str, Any]) -> str:
        act = action["act"]
        options = action.get("option_focus") or []
        update = action.get("stance_update")
        vote = action.get("vote_option")
        modifier = self._modifiers[self.calls % len(self._modifiers)]
        effect = action.get("issue_effect")
        label = lambda option_id: f"Option {option_id}"
        if effect == "maintain":
            return f"The concern about {label(options[0])} still matters to me {modifier}."
        if effect == "partial":
            return f"That helps somewhat, but the concern about {label(options[0])} is not fully solved {modifier}."
        if effect == "resolve" and act != ActionType.COMPROMISE.value:
            return f"That addresses the concern enough; {label(options[-1])} is workable for me {modifier}."
        joined = " and ".join(label(option_id) for option_id in options)
        if act == ActionType.OPENING.value:
            return f"Hi everyone. I prefer {label(options[0])} because it fits my main priority {modifier}."
        if act == ActionType.SUPPORT.value:
            return f"I support {label(options[0])}; it remains the strongest fit {modifier}."
        if act == ActionType.CONCERN.value:
            if len(options) > 1:
                return f"I still prefer {label(options[0])}; my concern about {label(options[1])} remains {modifier}."
            return f"I have a concern about {label(options[0])}; it does not fit my priority {modifier}."
        if act == ActionType.ASK.value:
            return f"What makes {label(options[0])} workable {modifier}?"
        if act == ActionType.ANSWER.value:
            return f"Yes, {label(options[0])} can work because the trade-off seems reasonable {modifier}."
        if act == ActionType.COMPARE.value:
            return f"{label(options[0])} fits my priority better than {label(options[1])} {modifier}."
        if act == ActionType.ACKNOWLEDGE.value:
            if update and update["kind"] == "make_acceptable":
                return f"That addresses my concern; {label(update['option_id'])} now seems workable and acceptable."
            return f"That point makes sense to me {modifier}."
        if act == ActionType.COMMENT.value:
            return f"I see the point, though I am still weighing it {modifier}."
        if act == ActionType.COMPROMISE.value:
            target = update["option_id"] if update else options[-1]
            if update and update["kind"] == "switch_preferred":
                old = update["previous_option_id"]
                return f"I preferred {label(old)}, but that changed my mind; I now prefer {label(target)} {modifier}."
            if update and update["kind"] == "make_acceptable":
                return f"{label(target)} now seems workable and acceptable to me {modifier}."
            return f"I could accept {joined} as a compromise {modifier}."
        if act == ActionType.VOTE.value:
            if update:
                return f"I preferred {label(update['previous_option_id'])}, but the discussion changed my mind, so I vote for {label(vote)} {modifier}."
            return f"I vote for {label(vote)} because it remains my best fit {modifier}."
        return f"That seems reasonable {modifier}."


def make_runner(
    preferences: tuple[str, ...] = ("A", "B", "C"),
    *,
    llm: ActionRendererLLM | None = None,
    seed: int = 7,
) -> DialogueRunner:
    return DialogueRunner(
        "",
        scenario=make_scenario(),
        personas=make_personas(preferences),
        llm=llm or ActionRendererLLM(),
        logger=NullLogger(),
        rng=random.Random(seed),
        seed=seed,
    )
