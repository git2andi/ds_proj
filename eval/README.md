# Evaluation scripts

All default paths are resolved relative to this folder. Commands work from the repository root or from inside `eval2/`. Scripts create their output directories and never modify `config.yaml` on disk.

## Main evaluation

```powershell
py .\eval\run_eval_suite.py
py .\eval\run_scenarios.py --limit 40
py .\eval\evaluate_runs.py
py .\eval\judge_transcripts.py
py .\eval\validate_judge.py
```

Default output folders:

- focused suite: `eval2/logs_eval_suite/`;
- scenario batch and deterministic analysis: `eval2/logs_scenarios/`;
- transcript judge: `eval2/logs_judge_scenarios/`;
- judge corruption validation: `eval2/logs_judge_validation/`.

`run_scenarios.py` creates runs. `evaluate_runs.py` reads `run.json` files and writes deterministic summaries. The two steps are separate so historical or partially completed batches can be analyzed without new LLM calls.

## Configuration experiments

```powershell
py .\eval2\run_config_sweep.py
py .\eval2\run_config_confirmation.py
```

The sweep tests only four settings connected to observed defects:

- semantic duplicate detection;
- issue follow-up depth;
- consecutive participant turns;
- small-group closure pacing.

For each replicate, the sweep generates the scenario and personas once and reuses that exact setup for the baseline and every variant. Each run stores a `setup_fingerprint` in its experiment metadata. This prevents setup variation from being mistaken for a configuration effect and reduces repeated setup calls.

The sweep writes `eval2/logs_config_sweep/sweep_selection.json`. The confirmation script reads it automatically and evaluates cumulative profiles on five topics from `scenarios.txt`. All profiles for one topic reuse the same setup.

To judge the confirmation transcripts:

```powershell
py .\eval2\judge_transcripts.py --logs .\eval2\logs_config_confirmation
```

The output folder is inferred as `eval2/logs_judge_config_confirmation/`.

## Files

```text
run_eval_suite.py          focused pinned LLM-backed cases
scenarios.txt              broader participant-count/topic batch
run_scenarios.py           batch producer
evaluate_runs.py           deterministic post-hoc metrics
judge_transcripts.py       rotated three-role LLM judge
validate_judge.py          controlled corruption validation
run_config_sweep.py        four-area paired sweep
run_config_confirmation.py matched multi-topic confirmation
evaluation_metrics.py      shared deterministic analysis helpers
experiment_common.py       in-memory overrides and paired run helpers
```

## Main output files

Deterministic evaluation:

- `deterministic_runs.csv`;
- `trait_participants.csv`;
- `trait_levels.csv`;
- `evaluation_summary.md`.

LLM judge:

- `judge_scores.csv`;
- `judge_scores_detailed.csv`;
- `judge_summary.md`;
- `judge_errors.csv` when calls still fail after retries.

Judge validation:

- `judge_validation_pairs.csv`;
- `judge_validation_detailed.csv`;
- `judge_validation_summary.md`;
- `judge_validation_errors.csv` when applicable.

The earlier `eval/` folder preserves historical outputs only. Current scripts can analyze them by passing an explicit input path.
