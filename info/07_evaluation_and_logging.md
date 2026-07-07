# 07 — Evaluation and logging

Each run writes readable and structured artifacts.

## Outputs

- `transcript.md`: human-readable setup, transcript, outcome, and metrics.
- `run.json`: structured scenario, personas, turns, runtime stance ranks, and outcome.
- `metrics.csv`: append-only summary rows.
- optional prompt dump if `output.write_prompts` is enabled.

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
- token usage by call type.

Metrics are useful but not sufficient. Manual transcript review is required, especially after stance-rank changes.
