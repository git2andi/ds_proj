from __future__ import annotations

import random

from dialogue import DialogueRunner, initialise_state
from models import (
    OptionCard,
    OptionStance,
    Persona,
    Scenario,
    SimulatorParameters,
    STANCE_ACCEPTABLE,
    STANCE_DISLIKED,
    STANCE_PREFERRED,
    STANCE_REJECTED,
)


def make_scenario() -> Scenario:
    return Scenario(
        topic="Choose a study location",
        shared_context=["The group needs one location for a Saturday project session."],
        options=[
            OptionCard(id="A", name="Central Library", short_name="Library", attrs={"cost": "free", "closing time": "20:00", "equipment": "standard desks"}, upside="quiet and predictable", concern="can become crowded"),
            OptionCard(id="B", name="Riverside Cafe", short_name="Cafe", attrs={"cost": "8 euros", "closing time": "22:00", "noise": "moderate"}, upside="relaxed atmosphere", concern="background noise"),
            OptionCard(id="C", name="Engineering Lab", short_name="Lab", attrs={"cost": "free", "closing time": "19:00", "equipment": "specialist workstations"}, upside="reliable technical equipment", concern="earlier closing time"),
            OptionCard(id="D", name="Online Session", short_name="Online", attrs={"cost": "free", "travel": "none", "access": "from home"}, upside="no travel", concern="less social interaction"),
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
    alternatives_acceptable: bool = True,
) -> Persona:
    board = make_scenario()
    stances: dict[str, OptionStance] = {}
    for option in board.options:
        if option.id == preferred:
            stances[option.id] = OptionStance(option.id, STANCE_PREFERRED, option.upside, "")
        elif hard_blocker:
            stances[option.id] = OptionStance(option.id, STANCE_REJECTED, "", option.concern)
        elif alternatives_acceptable:
            stances[option.id] = OptionStance(option.id, STANCE_ACCEPTABLE, option.upside, option.concern)
        else:
            stances[option.id] = OptionStance(option.id, STANCE_DISLIKED, "", option.concern)
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
        age=30,
        speech_style="plain conversational wording",
        style_tendencies=("uses natural chat wording",),
        rejection=None,
        rejection_reason="will not accept another option" if hard_blocker else "",
        option_stances=stances,
        hard_blocker=hard_blocker,
    )


def make_personas(preferences: tuple[str, ...] = ("A", "B", "C"), **kwargs) -> list[Persona]:
    names = ("Nora", "Ben", "Mira", "Omar", "Lea", "Tariq", "Sofia")
    return [make_persona(f"p{i + 1}", names[i], pref, **kwargs) for i, pref in enumerate(preferences)]


def make_state(preferences: tuple[str, ...] = ("A", "B", "C")):
    return initialise_state(make_scenario(), make_personas(preferences))


class NullLogger:
    def write_prompt(self, prompt: str, kind: str) -> str:
        return ""

    def write_run(self, *_args, **_kwargs) -> dict[str, str]:
        return {"dir": "", "transcript": "", "json": "", "metrics": ""}


class ActionRendererLLM:
    """Deterministic offline renderer for the compact realization prompt."""

    def __init__(self, scripted: list[str] | None = None) -> None:
        self.scripted = list(scripted or [])
        self.prompts: list[str] = []
        self.profiles: list[str] = []
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        self.session_calls = 0
        self.provider = "offline"
        self.model_id = "action-renderer"

    def reset_session(self) -> None:
        self.session_tokens_in = self.session_tokens_out = self.session_calls = 0

    def generate(self, prompt: str, *, profile: str = "dialogue") -> str:
        self.prompts.append(prompt)
        self.profiles.append(profile)
        text = self.scripted.pop(0) if self.scripted else self._render(prompt)
        self.last_tokens_in = len(prompt.split())
        self.last_tokens_out = len(text.split())
        self.session_tokens_in += self.last_tokens_in
        self.session_tokens_out += self.last_tokens_out
        self.session_calls += 1
        return text

    @staticmethod
    def _instruction(prompt: str) -> str:
        marker = "Selected action:\n"
        if marker not in prompt:
            return ""
        return prompt.rsplit(marker, 1)[1].split("\n", 1)[0].strip()

    @staticmethod
    def _focus(instruction: str, before: str, after: str) -> str:
        return instruction.split(before, 1)[1].split(after, 1)[0].strip()

    def _render(self, prompt: str) -> str:
        instruction = self._instruction(prompt)
        lower = instruction.lower()
        if lower.startswith("give a short, natural first contribution"):
            option = instruction.split("choice is ", 1)[1].split("; explain", 1)[0]
            reason = instruction.split("with:", 1)[1].split(". A greeting", 1)[0]
            return f"Hi everyone, I prefer {option} because {reason}."
        focused_options = [
            line.split(": ", 1)[1]
            for line in prompt.splitlines()
            if line.startswith("- ")
            and ": " in line
            and "Fact for" not in line
            and "Grounded source" not in line
            and "Intended point" not in line
        ]
        option = focused_options[0] if focused_options else "that option"
        if lower.startswith("continue the exchange with why the focused choice suits you"):
            reason = instruction.split("grounded in:", 1)[1].split(". Continue", 1)[0]
            return f"That suits me because {reason}; {option} still works well."
        if lower.startswith("continue the exchange with a concern about the focused choice"):
            reason = instruction.split("grounded in:", 1)[1].split(". Do not", 1)[0]
            return f"That is difficult for me because {reason}, especially with {option}."
        if lower.startswith("respond to the previous point"):
            reason = instruction.split("using:", 1)[1].split(". Agree", 1)[0]
            return f"That point matters; {option} works for me because {reason}."
        if lower.startswith("contrast the two focused options"):
            facts = [
                line.split(": ", 1)[1]
                for line in prompt.splitlines()
                if line.startswith("- Fact for ")
            ]
            first, second = (facts + ["a different value", "another value"])[:2]
            names = " and ".join(focused_options[:2]) or "the two options"
            return f"Looking at {names}, I prefer the difference between {first} and {second}."
        if lower.startswith("ask "):
            target = instruction.split("Ask ", 1)[1].split(" one natural", 1)[0]
            reason = instruction.split(":", 1)[1].split(". Connect", 1)[0]
            prefix = "Everyone" if target == "the group" else target
            return f"{prefix}, does {reason} change whether {option} works for you?"
        if lower.startswith("reply naturally and directly to this question"):
            block = prompt.split("Selected action:\n", 1)[1].split("\n\nRelevant public", 1)[0]
            option_line = next(line for line in block.splitlines() if line.startswith("Use this grounded point"))
            reason = option_line.split(":", 1)[1].split(". A forced", 1)[0]
            option = next(
                line.split(": ", 1)[1].split(",", 1)[0]
                for line in prompt.splitlines()
                if line.startswith("- ") and ": " in line and "Fact for" not in line
            )
            return f"{option} still works for me because {reason}."
        if lower.startswith("show naturally that you now prefer") or lower.startswith("show naturally that you could now accept"):
            marker = "now prefer " if "now prefer " in instruction else "could now accept "
            option = instruction.split(marker, 1)[1].split(" rather than", 1)[0]
            reason = instruction.split("this point:", 1)[1].split(". It may", 1)[0]
            return f"That changes my view; I can go with {option} because {reason}."
        return "That point matters for my decision."


def make_runner(
    preferences: tuple[str, ...] = ("A", "B", "C"),
    *,
    llm: ActionRendererLLM | None = None,
    seed: int = 7,
    **persona_kwargs,
) -> DialogueRunner:
    return DialogueRunner(
        "",
        scenario=make_scenario(),
        personas=make_personas(preferences, **persona_kwargs),
        llm=llm or ActionRendererLLM(),
        logger=NullLogger(),
        rng=random.Random(seed),
        seed=seed,
    )
