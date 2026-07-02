"""I12: per-turn length jitter and reason-clause echo suppression. No LLM calls."""

from __future__ import annotations

import random

from dialogue import DialogueRunner
from models import ActType, MoveIntent, Persona, SimulatorParameters, TraitProfile
from parsing import round_reason_snippets


def _persona(verbosity: float) -> Persona:
    return Persona(
        id="p1",
        name="P1",
        traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=SimulatorParameters(0.5, verbosity, 0.5, 0.5, 0.5, 0.5, 0.5),
        background="b",
        private_goal="g",
        preferred_options=["A"],
    )


def test_word_bounds_vary_per_turn_but_keep_trait_ordering():
    random.seed(9)
    intent = MoveIntent(speaker_id="p1", act=ActType.BUILD, reason="r")
    terse = [DialogueRunner._word_bounds(intent, _persona(0.1))[1] for _ in range(60)]
    chatty = [DialogueRunner._word_bounds(intent, _persona(0.9))[1] for _ in range(60)]
    assert len(set(terse)) > 1, "same-persona budgets should jitter across turns"
    assert max(terse) < min(chatty), "verbosity ordering must survive the jitter"


def test_switch_headroom_survives_jitter():
    random.seed(10)
    persona = _persona(0.5)
    plain = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r")
    switch = MoveIntent(speaker_id="p1", act=ActType.VOTE, reason="r", allow_vote_change=True)
    for _ in range(60):
        _, plain_max = DialogueRunner._word_bounds(plain, persona)
        _, switch_max = DialogueRunner._word_bounds(switch, persona)
        assert switch_max > plain_max


def test_round_reason_snippets_extracts_justifications():
    texts = [
        "Count me in for World Music for its unique and culturally rich atmosphere.",
        "I'd go with Sailing because it builds confidence on the water.",
        "My pick is Arts.",  # no reason clause
    ]
    snippets = round_reason_snippets(texts)
    assert len(snippets) == 2
    assert snippets[0].startswith("for its unique and culturally rich")
    assert snippets[1].startswith("because it builds confidence")


def test_prompt_forbids_used_reasons():
    import prompts
    from models import DialogueState, OptionCard, Scenario
    from dialogue import initialise_state

    options = [OptionCard(id="A", name="Option A Name", attrs={"cost": "$5"})]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    persona = _persona(0.5)
    state = initialise_state(scenario, [persona])
    intent = MoveIntent(
        speaker_id="p1", act=ActType.VOTE, reason="r", option_focus=["A"],
        avoid_reasons=["for its unique and culturally rich atmosphere"],
    )
    prompt = prompts.sim_utterance(
        persona=persona, state=state, intent=intent, recent_lines=[],
        focus_options=options, addressee_name=None, max_words=20, min_words=6,
    )
    assert "culturally rich atmosphere" in prompt and "DIFFERENT reason" in prompt
