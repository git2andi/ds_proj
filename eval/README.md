# Evaluation scripts

All default paths are resolved relative to this `eval/` folder. The commands
work from the project root and do not modify `config.yaml`.

## Main evaluation

```powershell
py .\eval\run_eval_suite.py
py .\eval\run_scenarios.py --limit 40
py .\eval\evaluate_runs.py
py .\eval\judge_transcripts.py
py .\eval\validate_judge.py
```

Default folders:

- focused suite: `eval/logs_eval_suite/`
- scenario runs and deterministic results: `eval/logs_scenarios/`
- LLM transcript scores: `eval/logs_judge_scenarios/`
- LLM judge corruption check: `eval/logs_judge_validation/`

`run_scenarios.py` only creates runs. `evaluate_runs.py` reads the resulting
`run.json` files and writes compact deterministic summaries.

## Configuration experiments

```powershell
py .\eval\run_config_sweep.py
py .\eval\run_config_confirmation.py
```

The sweep tests only duplicate detection, issue follow-up depth, consecutive
turns, and small-group closure. It writes
`eval/logs_config_sweep/sweep_selection.json`. The confirmation script reads
that file automatically and tests cumulative profiles on five matched topics
from `scenarios.txt`.

To judge the confirmation transcripts:

```powershell
py .\eval\judge_transcripts.py --logs .\eval\logs_config_confirmation
```

The output folder is inferred as `eval/logs_judge_config_confirmation/`.

## Main output files

Deterministic evaluation:

- `deterministic_runs.csv`
- `trait_participants.csv`
- `trait_levels.csv`
- `evaluation_summary.md`

LLM judge:

- `judge_scores.csv`
- `judge_scores_detailed.csv`
- `judge_summary.md`
- `judge_errors.csv` when calls still fail after retries

Judge validation:

- `judge_validation_pairs.csv`
- `judge_validation_detailed.csv`
- `judge_validation_summary.md`
