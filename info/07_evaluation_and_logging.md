# Evaluation and logging

## Normal transcript

`transcript.md` contains:

- public option board;
- compact participant trait table;
- visible conversation;
- outcome;
- compact run summary;
- compact participant statistics.

It does not contain a large nested JSON metric dump.

## Structured output

`run.json` stores scenario, personas, runtime state, turns, issues, votes, outcome, and compact metrics. Deep generation attempts and validation diagnostics are written only when `output.debug_metrics` is enabled.

`metrics.csv` contains one flat row per run for comparison.

## Deterministic tests

The focused test suite verifies the architecture without requiring an LLM endpoint.

## LLM-backed suite

`eval/run_eval_suite.py` runs 15 focused manual scenarios with varied topics. It reports structure, adaptive narrowing participation, question/concern resolution and staleness, visible switches, repetitive starts, same-speaker repetition, outcomes, token use, and prompt size. The suite is diagnostic and always completes all selected cases.


Compact quality diagnostics include compromise proposals/acceptances, selected versus committed movement actions, movement realization failures and fallbacks, grounded versus unexplained movements, narrowing movements, whether a re-vote was skipped for no movement, semantic reason reuse, and aggregate repair causes. The suite fails when any selected movement does not commit, or when a second vote occurs without preceding movement.
