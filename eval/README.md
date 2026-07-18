# Evaluation tools

The evaluation folder contains four entry points. The focused suite and scenario runner call the configured dialogue provider; the summarizer is deterministic; the transcript judge is a separate post-hoc LLM evaluation.

## Focused development suite

```powershell
py .\eval\run_eval_suite.py
```

Runs a small set of pinned cases covering direct and group threads, hard blockers, movement, decisive majorities, splits, moderator-free operation, and voting. These cases are regression aids, not the main report dataset.

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
```

The runner refuses to use a nonempty output directory unless `--clean` is given. It writes its manifest incrementally so completed cases remain visible after interruption.

## Deterministic summary

```powershell
py .\eval\summarize_runs.py --logs .\eval\logs_scenarios
```

Outputs:

- `deterministic_runs.csv`: compact reliability/process row per run;
- `trait_participants.csv`: one row per simulator;
- `trait_levels.csv`: trait-level aggregates;
- `evaluation_summary.md`: concise outcomes, reliability, cost, and trait summary.

The main metrics are setup completion, outcomes, turns, moderator ratio, vote consistency, visible preference changes, generation failures, token use, engagement realization, and verbosity realization. Directness is included only as an optional lexical hedge-rate proxy.

## Independent transcript judges

```powershell
py .\eval\judge_transcripts.py --logs .\eval\logs_scenarios --judges 3 --provider uni
```

Independent referee roles receive the same scenario, persona cards, visible transcript, votes, and outcome. They score naturalness, coherence, groundedness, persona consistency, and deliberation quality. No referee receives another referee’s assessment.

Outputs:

- `judge_scores.csv`;
- `judge_scores_detailed.csv`;
- `judge_summary.md`;
- `judge_errors.csv` when needed.

The judges are never part of the runtime path.
