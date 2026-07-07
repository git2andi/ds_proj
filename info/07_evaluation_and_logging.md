# 07 — Evaluation and logging

Each run writes readable and structured artifacts.

## Outputs

- `transcript.md`: human-readable setup, transcript, outcome, and metrics.
- `run.json`: structured scenario, personas, turns, state metadata, and outcome.
- `metrics.csv`: append-only summary rows.
- optional prompt dump if `output.write_prompts` is enabled.

## Important metrics to inspect

- outcome status and final option;
- visible votes;
- split reservation exchanges;
- two-person deadlock protocol use;
- participant procedural moves;
- question/answer completion;
- turn share and engagement correlation;
- average words by persona and act;
- switch explanation / bridge rate;
- unsupported fact flags and printed unsupported turns;
- repair/fallback counts, which should stay low because decision prompts and validation are parser-aligned;
- token usage by call type.

## v3 validation focus

After running the quick/full suite, inspect transcripts for:

- whether split votes are narrowed instead of closing abruptly;
- whether successful outcomes are earned rather than forced;
- whether holdouts visibly switch or stay with a concrete reason;
- whether no-moderator procedural turns sound participant-owned;
- whether local threads continue before new issues open;
- whether traits influence behavior without creating obvious templates.

Metrics are useful but not sufficient. Manual transcript review is required.
