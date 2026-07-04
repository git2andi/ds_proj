# 08 — Configuration and running

`config.yaml` is the main place for tunable behavior.

## Important sections

- `llm`: provider, model, endpoint, sampling, timeouts.
- `environment`: auto/manual environment setup.
- `participants`: auto/manual participant setup.
- `simulation`: participant count, random seed, repair count.
- `scenario`: option board shape and display limits.
- `personas`: trait sampling, hard blockers, initial preference distribution.
- `conversation`: pacing and vote turn caps.
- `moderator`: visible moderator behavior.
- `routing`: move weights and trait-weighted participation.
- `utterances`: recent context and word budgets.
- `validation`: turn validation and grounding checks.
- `output`: log paths and prompt dumping.

## Running normal simulations

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

For a topic file:

```powershell
py .\main.py scenarios.txt
```

Manual environment mode ignores CLI topics and uses `environment.manual`.

## Running evaluation

```powershell
py .\run_eval_suite.py --quick
py .\run_eval_suite.py --full
```

Use `--full` before claiming a behavioral issue is fixed.

## Required mode coverage

Across an implementation round, test:

```text
auto environment + auto participants
manual environment + auto participants
auto environment + manual participants
manual environment + manual participants
```

Also test:

- n=2 stubborn tie/deadlock;
- n=3 three-way split;
- n=4 trait spread;
- n=5 scaling;
- full moderator;
- no moderator;
- light moderator.

## Current open issue

The current eval suite needs a stronger forced `n=2` stubborn-deadlock case. The existing n=2 full-suite case converged before testing the deadlock protocol.
