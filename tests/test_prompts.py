"""Prompt-size contracts for turn generation and focused repairs."""

from __future__ import annotations

from conftest import make_intent, make_turn
from models import ActType
from dialogue import recent_lines_for_prompt
from prompts import repair_utterance, runtime_speaker_card


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
