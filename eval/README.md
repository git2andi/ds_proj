# Evaluation tools

The evaluation folder contains four entry points. The focused suite and scenario runner call the configured dialogue provider, the summarizer is deterministic, and the transcript judge is a separate post-hoc LLM evaluation. Configure the required provider key in the repository-root `.env` file before running LLM-backed tools.

## Focused development suite

```powershell
py .\eval\run_eval_suite.py
```

Runs pinned regression cases for direct and group threads, hard blockers, movement, majority and split outcomes, moderator-free operation, and voting. These cases are development checks rather than the main report dataset.

Default output:

```text
eval/logs_eval_suite/
```

## Broader scenario batch

```powershell
py .\eval\run_scenarios.py --seed 500 --clean
```

`scenarios.txt` uses:

```text
participant_count | topic
```

Useful commands:

```powershell
py .\eval\run_scenarios.py --list
py .\eval\run_scenarios.py --limit 10 --seed 500 --clean
py .\eval\run_scenarios.py --counts 3,4 --seed 500 --clean
py .\eval\run_scenarios.py --output .\eval\logs_custom --clean
py .\eval\run_scenarios.py --workers 1 --limit 10 --seed 500 --clean
```

The runner uses two isolated scenario processes by default. The parent process writes `scenario_runs.csv` and `scenario_summary.md` incrementally, so completed rows remain available after interruption. A nonempty output directory is rejected unless `--clean` is supplied.

## Deterministic summary

```powershell
py .\eval\summarize_runs.py --logs .\eval\logs_scenarios
```

Outputs:

- `deterministic_runs.csv`: one compact reliability and process row per run;
- `trait_participants.csv`: one row per simulator;
- `trait_levels.csv`: aggregates by trait level;
- `evaluation_summary.md`: outcomes, protocol reliability, cost, and trait summary.

The main metrics are completion, outcomes, participant and moderator turns, voluntary participation, vote/outcome consistency, required-response failures, visible preference movement, repairs, dropped turns, fallbacks, LLM calls, token use, and trait realization. Deterministic vote lines are excluded from generated-language measurements.

## Independent transcript judges

```powershell
py .\eval\judge_transcripts.py --logs .\eval\logs_scenarios --judges 3 --provider uni
```

The three referee roles receive the same scenario, option board, complete persona cards, visible transcript, votes, and outcome. They score naturalness, coherence, groundedness, persona consistency, and deliberation quality. Moderator turns count toward the interaction-level dimensions but are excluded from persona consistency.

Useful options:

```powershell
py .\eval\judge_transcripts.py --logs .\eval\logs_scenarios --judges 3 --provider uni --workers 2
py .\eval\judge_transcripts.py --logs .\eval\logs_scenarios --limit 10
py .\eval\judge_transcripts.py --runs .\eval\logs_scenarios\RUN_DIR
```

Outputs are written incrementally to a judge directory inferred from the log-folder name unless `--output` is supplied:

- `judge_scores.csv`;
- `judge_scores_detailed.csv`;
- `judge_summary.md`;
- `judge_errors.csv`.

Existing complete panels with the same provider, model, judge count, and prompt version are skipped. Re-running the command therefore resumes incomplete work without deleting earlier scores. The judges are never part of the dialogue runtime.
