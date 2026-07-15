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

`eval/run_eval_suite.py` runs 17 focused manual scenarios across ten varied domains, including technical infrastructure, sustainability, media, community planning, logistics, leisure, and teamwork. Fifteen use normal pacing and two `long_*` cases use isolated stress overrides. It reports structure, adaptive narrowing participation, question/concern resolution and staleness, visible switches, repetitive starts, same-speaker repetition, outcomes, token use, and prompt size. The suite is diagnostic and always completes all selected cases.


Compact quality diagnostics include compromise proposals/acceptances, selected versus committed movement actions, movement realization failures and fallbacks, grounded versus unexplained movements, narrowing movements, whether a re-vote was skipped for no movement, semantic reason reuse, and aggregate repair causes. The suite fails when any selected movement does not commit, or when a second vote occurs without preceding movement.


## Turn terminology

A self-selected (legacy metric name: `voluntary`) turn means that the simulator chose to enter the floor. Openings, direct-answer obligations, required narrowing positions, and votes are not self-selected. After a required answer, another participant who independently reacts through the ordinary reaction policy produces a self-selected turn. The question then closes after that one optional continuation or immediately when no reaction is selected.

The long diagnostic cases report deliberation turns separately from openings and voting. They temporarily permit semantic reason reuse so the logs reveal repetition and coordination problems that emerge under deliberately extended pacing. This override never changes normal runtime defaults.

Grounding diagnostics distinguish unsupported concrete values from a narrow set of qualitative strengthening errors. The latter catches only high-risk words such as unsupported superlatives or intensifiers; it is not a general semantic validator.
