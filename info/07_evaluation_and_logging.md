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

`run.json` stores scenario, personas, runtime state, turns, issues, votes, outcome, compact metrics, and compact failed-generation records. Deep generation attempts and validation diagnostics are written only when `output.debug_metrics` is enabled; full per-turn structured actions only when `output.write_action_trace` is enabled.

`metrics.csv` contains one flat row per run for comparison. The flattened schema is produced by `src/eval.py` (`flat_metrics_for`), which lives in `src/` because it exposes the runtime's own metrics; all evaluation scripts consume it.

## Deterministic tests

The focused pytest suite (154 tests, no LLM endpoint required) verifies configuration validation, simulator authority, floor arbitration, question/concern protocol, movement and vote fallbacks, hard blockers, logging, bounded pacing for 2–7 participants, and the evaluation-script definitions (scenario-file format and sweep-variant validity).

## LLM-backed evaluation scripts

Everything below is diagnostic; none of it influences the runtime, and every script overrides configuration in memory only (`eval/experiment_common.py`), so `config.yaml` is never modified by an experiment. All batch scripts write incremental CSV plus Markdown summaries, so interrupted batches keep partial results.

### Focused case suite — `eval/run_eval_suite.py`

Seventeen hand-built cases across ten varied domains (technical infrastructure, sustainability, media, community planning, logistics, leisure, teamwork) covering every group size 2–7 with pinned personas, seeds, and expectations. Fifteen use normal pacing; two `long_*` cases use isolated stress overrides that temporarily allow semantic reason reuse so long-range repetition becomes observable. Reported checks include structure (phase closure, opening count, direct-answer ordering, valid votes, hard-blocker integrity, bounded re-voting, movement commitment) and quality expectations (minimum switches, resolved concerns, stale issues, repair rate ≤ 25%, deliberation-turn ranges). Outputs land in `eval/logs_eval_suite/` and are zipped.

### Scenario batch — `eval/run_scenarios.py`

Runs each `participant_count | topic` line of `eval/scenarios.txt` as one complete automatic run (scenario generation → persona generation → dialogue). The file holds 102 deliberately diverse everyday decision topics with balanced counts 2–7; deterministic tests assert the balance, topic uniqueness, and that no topic names a contradicting group size. Supports `--list`, `--limit`, `--start`, `--counts`, and `--seed` for reproducible batches; failed runs are recorded as `outcome=error` rows instead of aborting. Summary tables aggregate outcomes, turns, and token use per group size.

### Config sensitivity sweep — `eval/run_config_sweep.py`

For every numeric value under `conversation:`, `simulator:`, and `language:`, the sweep runs the same topic with a smaller value, the shared current-config baseline, and a larger value — exactly one knob changed per variant, default three runs per variant with identical seed sets, so differences between variants come from the knob rather than policy randomness. Level mappings (bid/movement probabilities, verbosity word budgets, action word caps) are varied as one knob by shifting or scaling all levels while preserving monotonicity and bounds. Derived values always satisfy the config-validation constraints (a deterministic test applies every variant and re-runs full config validation). Knobs that cannot affect the chosen group size (large-group caps in a three-person run, small-group extras in a six-person run) are skipped with a printed note; sweep them with `--participants 6`.

### Transcript judge — `eval/judge_transcripts.py`

Post-hoc LLM judging modeled on ChatEval (Chan et al., ICLR 2024, arXiv:2308.07201), which found that multi-agent judge panels align better with human judgment when the judges have *diverse role personas* and communicate one-by-one, and that 2–4 judges with few rounds suffice. The script sends each `run.json` to up to three judge personas — conversation analyst (flow, turn-taking), behavioral scientist (persona consistency, believable dynamics), fact auditor (grounding in the option board) — where each later judge sees the earlier assessments. Scores (1–5) for naturalness, coherence, groundedness, persona consistency, and decision quality are averaged per run, following ChatEval's score aggregation. This complements the structural metrics: the batch runner and config sweep measure protocol behavior, the judge measures surface quality of the same logs.

The judge deliberately defaults to a different provider than the dialogue runtime — `uni`, the local Ollama endpoint configured in config.yaml — so the runtime model does not grade its own writing (self-preference bias). `--provider` and `--model` select any configured provider/model for judging.

## Compact quality diagnostics

The flat metrics row includes compromise proposals/acceptances, selected versus committed movement actions, movement realization failures and fallbacks, grounded versus unexplained movements, narrowing movements, whether a re-vote was skipped for no movement, semantic reason reuse, and aggregate repair causes. The case suite fails when any selected movement does not commit or when a second vote occurs without preceding movement.

## Turn terminology

A self-selected (legacy metric name: `voluntary`) turn means that the simulator chose to enter the floor. Openings, direct-answer obligations, required narrowing positions, and votes are not self-selected. After a required answer, another participant who independently reacts through the ordinary reaction policy produces a self-selected turn. The question then closes after that one optional continuation or immediately when no reaction is selected.

The long diagnostic cases report deliberation turns separately from openings and voting. Grounding diagnostics distinguish unsupported concrete values from a narrow set of qualitative strengthening errors; the latter catches only high-risk words such as unsupported superlatives or intensifiers and is not a general semantic validator.
