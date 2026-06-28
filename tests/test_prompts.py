"""Prompt-size contracts for turn generation and focused repairs."""

from __future__ import annotations

from conftest import _make_persona, make_intent, make_turn
from config_loader import cfg
from models import ActType
from dialogue import max_words_for, recent_lines_for_prompt
from prompts import repair_utterance, runtime_speaker_card, sim_utterance


def test_recent_context_does_not_duplicate_exact_response_target(state):
    for index, (pid, name) in enumerate(
        [("p1", "Alice"), ("p2", "Bob"), ("p3", "Carol"), ("p2", "Bob")],
        start=1,
    ):
        state.turns.append(make_turn(index, pid, name, f"message number {index}"))

    intent = make_intent("p1", ActType.ANSWER, ["B"])
    intent.respond_to_turn = 4

    recent = recent_lines_for_prompt(state, intent)

    assert all("message number 4" not in line for line in recent)


def test_runtime_speaker_card_omits_raw_traits_and_opener_history(state):
    state.runtimes["p1"].already_said = [
        "I think Mountain Retreat is affordable.",
        "I still prefer Mountain Retreat for the cost.",
    ]
    state.turns.extend(
        [
            make_turn(1, "p1", "Alice", state.runtimes["p1"].already_said[0]),
            make_turn(2, "p1", "Alice", state.runtimes["p1"].already_said[1]),
        ]
    )

    card = runtime_speaker_card(state.personas[0], state, make_intent("p1", ActType.REACT, ["A"]))

    assert "Traits:" not in card
    assert "Your last openers:" not in card
    assert state.personas[0].role not in card
    assert state.personas[0].speech_style not in card


def test_runtime_speaker_card_describes_behavior_without_persona_labels(state):
    persona = _make_persona(
        "p1",
        "Alice",
        conscientiousness=5,
        neuroticism=3,
        agreeableness=3,
        directness=0.7,
    )
    state.personas[0] = persona

    card = runtime_speaker_card(persona, state, make_intent("p1", ActType.OBJECT, ["A"]))

    assert "behavior:" in card
    assert "check one concrete constraint" in card
    assert not any(label in card for label in ("worrier", "blunt", "methodical", "collaborator"))


def test_direct_response_prompt_makes_the_exact_target_the_local_job(state):
    target = make_turn(4, "p2", "Bob", "The beach is fun, but the long drive worries me.")
    state.turns.append(target)
    intent = make_intent("p1", ActType.PUSH_BACK, ["B"])
    intent.addressee_id = "p2"
    intent.respond_to_turn = target.index

    prompt = sim_utterance(
        persona=state.personas[0],
        state=state,
        recent_lines=recent_lines_for_prompt(state, intent),
        intent=intent,
        focus_options=[state.scenario.option("B")],
        addressee_name="Bob",
        max_words=30,
    )

    assert "Local job: respond to this exact message first" in prompt
    assert prompt.count(target.text) == 1
    assert "Leans:" not in prompt


def test_response_length_budgets_are_monotonic_and_chat_bounded():
    intent = make_intent("p1", ActType.REACT, ["A"], length_hint="long")
    budgets = [
        max_words_for(intent, _make_persona("p1", "Alice", response_length=level))
        for level in range(1, 6)
    ]

    assert budgets == sorted(budgets)
    assert len(set(budgets)) == 5
    assert budgets[-1] <= int(cfg.utterances.max_chat_words)


def test_long_response_prompt_requires_simple_chat_sentences(state):
    persona = _make_persona("p1", "Alice", response_length=5)
    state.personas[0] = persona
    intent = make_intent("p1", ActType.SUPPORT, ["A"], length_hint="long")

    prompt = sim_utterance(
        persona=persona,
        state=state,
        recent_lines=[],
        intent=intent,
        focus_options=[state.scenario.option("A")],
        addressee_name=None,
        max_words=max_words_for(intent, persona),
    )

    assert "at most two short sentences" in prompt


def test_vote_prompt_requires_visible_commitment_grammar(state):
    intent = make_intent("p1", ActType.VOTE, ["A"])
    prompt = sim_utterance(
        persona=state.personas[0],
        state=state,
        recent_lines=[],
        intent=intent,
        focus_options=[state.scenario.option("A")],
        addressee_name=None,
        max_words=max_words_for(intent, state.personas[0]),
    )

    assert "first-person commitment verb" in prompt
    assert "shortest recognizable option name" in prompt


def test_vote_repair_names_target_and_requires_selection_now(state):
    intent = make_intent("p1", ActType.VOTE, ["A"])
    prompt = repair_utterance(
        original_text="Mountain Retreat has the best scenery.",
        issue_codes=["UNCLEAR_VOTE"],
        persona=state.personas[0],
        state=state,
        recent_lines=[],
        intent=intent,
        max_words=24,
    )

    assert "Mountain Retreat" in prompt
    assert "selecting it now" in prompt
    assert "trailer is metadata" in prompt


def test_grounding_repair_uses_only_focused_options_and_no_recent_chat(state):
    intent = make_intent("p1", ActType.OBJECT, ["A"])
    prompt = repair_utterance(
        original_text="Mountain Retreat definitely has free parking.",
        issue_codes=["INVENTED_OPTION_ATTRIBUTE"],
        persona=state.personas[0],
        state=state,
        recent_lines=["Bob: Beach Resort is crowded.", "Carol: City Tour sounds tiring."],
        intent=intent,
        max_words=30,
    )

    assert "Mountain Retreat" in prompt
    assert "Beach Resort" not in prompt
    assert "City Tour" not in prompt
    assert "Recent chat:" not in prompt
