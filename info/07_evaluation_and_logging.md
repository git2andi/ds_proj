# 07 — Evaluation and logging

Each run writes readable and structured artifacts.

## Outputs

- `transcript.md`: human-readable setup, transcript, outcome, and metrics.
- `run.json`: structured scenario, personas, turns, runtime stance ranks, and outcome.
- `metrics.csv`: append-only summary rows.
- optional prompt dump if `output.write_prompts` is enabled.

## Persona logging

Logs include the hidden OCEAN traits, the four simulator parameters (engagement, verbosity, directness, stubbornness), age/speech_style, profile/background, private goal, initial preference, and initial option ranks. This is important for checking whether behavior comes from the parameters while wording variation comes from speech_style.

The eval suite also records a compact `persona_age_style` summary so manual eval casts can be checked quickly.

## Parameter meanings in eval

```text
engagement   -> turn share / free-discussion turn share
verbosity    -> average words per participant turn
directness   -> optional/manual or heuristic wording signal
stubbornness -> fewer/later switches and stronger stance defense
speech_style -> manual qualitative check only
```

Engagement realization is measured against the same expected-turn-share function the router uses (`simulator.expected_turn_share`). Verbosity realization is measured against the controller's own word-budget formula (`0.45 + 0.85 * verbosity` on the act's base budget, short beats folded in).

## Important metrics to inspect

- outcome status and final option;
- visible votes;
- stance-rank distribution;
- runtime preferred option by rank;
- split reservation exchanges;
- two-person deadlock protocol use;
- question/answer completion;
- turn share and engagement correlation;
- average words by persona and act;
- switch explanation / bridge rate;
- unsupported fact flags and printed unsupported turns;
- repair/fallback counts;
- token usage by call type;
- persona age/speech_style/profile plausibility.

Metrics are useful but not sufficient. Manual transcript review is required, especially after stance-rank, agenda, or speech-style changes.
