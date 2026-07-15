from __future__ import annotations

import random

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
    """Offline renderer that follows the compact plain-language action prompt."""

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
            action = self._action_line(prompt)
            text = self._render(action)
        self.last_tokens_in = max(1, len(prompt.split()))
        self.last_tokens_out = max(1, len(text.split()))
        self.session_tokens_in += self.last_tokens_in
        self.session_tokens_out += self.last_tokens_out
        self.session_calls += 1
        return text

    @staticmethod
    def _action_line(prompt: str) -> str:
        marker = "Selected action:\n"
        if marker not in prompt:
            return "Make one relevant contribution."
        return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()

    def _render(self, action: str) -> str:
        suffix = ("Overall", "For me", "At this point", "Given that", "In practice")[self.calls % 5]
        lower = action.casefold()

        if lower.startswith("start the discussion naturally"):
            option = action.split("prefer ", 1)[1].split(", and give", 1)[0].strip()
            reason = action.split("reason:", 1)[1].strip(" .")
            return f"Hi everyone. {option} seems best because {reason}."
        if lower.startswith("join the opening naturally. another participant"):
            option = action.split("prefers ", 1)[1].split("; align", 1)[0].strip()
            reason = action.split("reason:", 1)[1].split(". A greeting", 1)[0].strip(" .")
            return f"Same here—{option} fits me because {reason}."
        if lower.startswith("join the opening naturally with a different preference"):
            option = action.split("preference:", 1)[1].split(". Give", 1)[0].strip()
            reason = action.split("reason:", 1)[1].split(". A greeting", 1)[0].strip(" .")
            return f"I’d rather take {option} because {reason}."
        if lower.startswith("add one useful supporting point"):
            option = action.split("for ", 1)[1].split(":", 1)[0].strip()
            reason = action.split(":", 1)[1].strip(" .")
            return f"{suffix}, {option} works best for me because {reason}."
        if lower.startswith("raise this concrete concern"):
            option = action.split("about ", 1)[1].split(":", 1)[0].strip()
            reason = action.split(":", 1)[1].strip(" .")
            return f"My concern with {option} is that {reason}."
        if lower.startswith("react to the response"):
            option = action.split("about ", 1)[1].split(" still matters", 1)[0].strip()
            return f"That helps, but my concern about {option} still remains."
        if lower.startswith("briefly state that this unresolved concern"):
            option = action.split("about ", 1)[1].split(" still blocks", 1)[0].strip()
            reason = action.rsplit(":", 1)[1].strip(" .")
            return f"The {reason} still keeps me from accepting {option}."
        if lower.startswith("ask "):
            target = action.split("Ask ", 1)[1].split(" ", 1)[0]
            if "which factor matters more" in lower:
                option = action.split("for ", 1)[1].split(".", 1)[0].strip()
                return f"{target}, which factor matters more for {option}?"
            if "known benefit of" in lower and "known concern" in lower:
                option = action.split("known benefit of ", 1)[1].split(" is enough", 1)[0].strip()
                return f"{target}, does the known benefit make the concern about {option} acceptable to you?"
            if "whether any known condition" in lower:
                option = action.split("make ", 1)[1].split(" workable", 1)[0].strip()
                return f"{target}, is there anything known that would make {option} workable?"
            option = action.split("choice of ", 1)[1].split(":", 1)[0].strip()
            return f"{target}, does that concern change your choice of {option}?"
        if lower.startswith("answer "):
            if "available information is insufficient" in lower:
                return "I’m not sure; we don’t have enough information to say."
            if "concern still affects your choice" in lower:
                reason = action.rsplit("Concern:", 1)[1].strip(" .")
                return f"It still affects my choice because {reason}."
            if "recognize the concern and still prefer" in lower:
                option = action.split("still prefer ", 1)[1].split(".", 1)[0].strip()
                reason = action.rsplit("Decisive reason:", 1)[1].split(".", 1)[0].strip()
                return f"I still prefer {option}; {reason} matters more to me."
            if "known information that addresses" in lower:
                return f"{suffix}, the known information addresses it."
            answer = action.split("Actual position:", 1)[1].strip(" .") if "Actual position:" in action else "that still works for me"
            return f"{suffix}, {answer}."
        if lower.startswith("compare "):
            names = action.split("Compare ", 1)[1].split(" using", 1)[0]
            tradeoff = action.split("trade-off:", 1)[1].split(". A useful", 1)[0].strip(" .")
            return f"Between {names}, {tradeoff}."
        if lower.startswith("react briefly"):
            option = action.split("around ", 1)[1].split(".", 1)[0].strip()
            return f"That works for me too; {option} is reasonable."
        if lower.startswith("respond to the concern"):
            reason = action.rsplit("Decisive reason:", 1)[1].split(".", 1)[0].strip(" .")
            return f"I still support it; {reason} matters more to me."
        if lower.startswith("acknowledge this response"):
            acknowledged = action.split(":", 1)[1].split(". Then", 1)[0].strip(" .")
            concern = action.rsplit(":", 1)[1].strip(" .")
            return f"I see the point about {acknowledged}, but {concern} still worries me."
        if lower.startswith("respond that this concern"):
            reason = action.rsplit(":", 1)[1].strip(" .")
            return f"That concern still matters to me because {reason}."
        if lower.startswith("respond to the active issue") or lower.startswith("continue the current exchange"):
            reason = action.rsplit(":", 1)[1].strip(" .")
            return f"That matters here; {reason}."
        if lower.startswith("clearly state that you are moving to"):
            tail = action.split("moving to ", 1)[1]
            option = tail.split(".", 1)[0].split(" because", 1)[0].strip()
            if "concrete reason:" in action:
                reason = action.split("concrete reason:", 1)[1].split(".", 1)[0].strip()
                return f"I’m moving to {option} because {reason}."
            return f"That changed my mind; I’m moving to {option}."
        if lower.startswith("say that the response addressed your concern"):
            option = action.split("make ", 1)[1].split(" visibly", 1)[0].strip()
            reason = action.split("concrete reason:", 1)[1].split(".", 1)[0].strip()
            return f"That addresses my concern; I can accept {option} because {reason}."
        if lower.startswith("make ") and "visibly acceptable" in lower:
            option = action.split("Make ", 1)[1].split(" visibly", 1)[0].strip()
            reason = action.split("concrete reason:", 1)[1].split(".", 1)[0].strip()
            return f"{option} isn’t my first choice, but I can accept it because {reason}."
        if lower.startswith("briefly state that you are staying with"):
            option = action.split("staying with ", 1)[1].split(".", 1)[0].strip()
            return f"I’m staying with {option}."
        if lower.startswith("state only one short"):
            option = action.split("choice for ", 1)[1].split(".", 1)[0].strip()
            return f"{option} for me."
        if lower.startswith("state one short, clear vote"):
            option = action.split("for ", 1)[1].split(" and indicate", 1)[0].strip()
            return f"I’m moving to {option}; that gets my vote."
        if lower.startswith("state one clear vote"):
            option = action.split("for ", 1)[1].split(", indicate", 1)[0].strip()
            reason = action.split("concrete reason:", 1)[1].split(".", 1)[0].strip()
            return f"I’m moving to {option} because {reason}; that gets my vote."
        return "That point matters for my decision."

    @staticmethod
    def _option_after(action: str, markers: tuple[str, ...]) -> str:
        for marker in markers:
            if marker in action:
                tail = action.split(marker, 1)[1]
                for sep in (", and explain", ". Main", ". Explain", ".", ":"):
                    if sep in tail:
                        tail = tail.split(sep, 1)[0]
                return tail.strip()
        return "Option A"


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
