# 07 — Evaluation and logging

Each run writes `transcript.md`, a detailed `run.json`, and one concise `metrics.csv` row.

`run.json` contains the scenario, personas, visible turns, current runtime stance, threads,
repair history, controller trace, final outcome, grouped metrics, and mutually exclusive token
usage buckets. Detailed validation issue codes and state transitions remain in the trace rather
than becoming dozens of summary columns.

The metric schema is grouped into:

- run structure: participant/moderator turns, word lengths, question density;
- participation: expected engagement/share and realized counts/shares;
- traits: defensible verbosity and switch-opportunity measurements;
- interaction: question/concern threads, completion/response rates, repetition;
- decision behavior: visible votes, outcome, switches, lean shifts, coverage, compromise, blockers;
- validation/grounding: repairs, fallbacks, drops, critical grounding interventions, runtime validator calls;
- token usage: setup, participant, moderator, repair, runtime validation, and total.

Rates are null when their denominator is zero. Private rank preference is explicitly labelled
private and never reported as public support. The suite CSV keeps stable scalar aggregates and
flags excessive turns, post-vote loops, repairs, fallbacks, drops, vote inconsistency, blocker
violations, missing votes, repetition, poor thread completion, validator calls, and abnormal tokens.

Run deterministic tests first:

```powershell
py -m pytest -q
```

The ten-case live suite is:

```powershell
py .\eval\run_eval_suite.py
```


The metric schema version is `2.1`. Verbosity reporting separates configured verbosity, the average word budget actually assigned to each participant, realized average words, and budget adherence. `verbosity_budget_correlation` tests whether configuration reaches the controller budget; `verbosity_behavior_correlation` tests whether assigned budgets reach visible length.

`critical_grounding_interventions` counts accepted turn lifecycles whose original or repaired candidate triggered a high-confidence deterministic grounding check. It is not a claim that the final transcript has undergone exhaustive semantic fact auditing.
