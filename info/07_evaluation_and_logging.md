# Evaluation and logging

## Run artifacts

Each run writes a human-readable transcript and structured `run.json`. The JSON contains:

- scenario, generated aliases, and complete persona cards;
- actual seed, provider/model, sampling profiles, configuration hash, Python version, and Git revision when available;
- visible turns and optional structured action traces;
- raw, repaired, dropped, fallback, and deterministic vote records;
- public preferences, point counts, recent point keys, final votes, outcome, and review flags;
- compact process and token counters.

Invalid voluntary text is not inserted into the transcript, but its attempts remain available for debugging.

## Deterministic evaluation

The report-facing summary keeps only metrics that support concise claims:

- setup completion and failures;
- outcome distribution;
- participant and moderator turns;
- protocol pass and vote/outcome consistency;
- visible preference changes;
- repair, drop, fallback, and response-failure rates;
- LLM calls and token use;
- engagement versus voluntary participation;
- verbosity versus words per turn;
- optional directness hedge-rate proxy.

Detailed thread, point-reuse, alias, and generation evidence remains in `run.json` rather than expanding the report tables.

## LLM transcript judging

`judge_transcripts.py` is a separate post-hoc evaluator. Independent referee roles receive the same visible transcript, option board, persona cards, votes, and outcome. No referee sees another referee’s result. The judge is never called during dialogue generation.
