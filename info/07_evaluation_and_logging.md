# 07 — Evaluation and logging

Each run writes readable and structured artifacts.

## Outputs

- `transcript.md`: human-readable setup, transcript, outcome, and metrics.
- `run.json`: structured scenario, personas, turns, runtime stance ranks, and outcome.
- `metrics.csv`: append-only summary rows.
- optional prompt dump if `output.write_prompts` is enabled.

## Persona logging

Logs include participant traits, simulator parameters, age/style, profile/background, private goal, initial preference, and initial option ranks. This is important for checking whether behavior comes from traits while wording variation comes from style.

The eval suite also records a compact `persona_age_style` summary so manual eval casts can be checked quickly.

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
- persona age/style/profile plausibility.

Metrics are useful but not sufficient. Manual transcript review is required, especially after stance-rank, agenda, or persona-style changes.
