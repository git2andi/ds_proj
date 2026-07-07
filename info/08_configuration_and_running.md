# 08 — Configuration and running

`config.yaml` is the main control surface.

## Important sections

- `llm`: provider, model, endpoint, timeout, and sampling.
- `environment`: auto/manual environment setup.
- `participants`: auto/manual participant setup.
- `simulation`: participant count, seed, retry/repair counts. Repairs are safeguards, not the normal decision-turn path.
- `scenario`: option-board shape and display limits.
- `personas`: trait sampling, hard-blocker probability, preference distribution.
- `conversation`: pacing, option coverage, and vote caps.
- `moderator`: visible moderator jobs.
- `routing`: move weights and participation policy.
- `style`: name/option/I/we opening suppression.
- `utterances`: recent context and word budgets.
- `validation`: grounding and turn validation.
- `output`: log paths and prompt dumping.

## Running

```powershell
py .\main.py "Choose a restaurant for a group dinner"
py .\main.py scenarios.txt
py .\eval\run_eval_suite.py --quick
py .\eval\run_eval_suite.py --full
py .\eval\run_eval_suite.py --list
```

Manual environment mode ignores CLI topics and uses `environment.manual`.

## Recommended validation sequence

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
py .\eval\run_eval_suite.py --quick
py .\eval\run_eval_suite.py --full
```

Run the full suite before treating v3 as stable.

## v3 tuning caution

Do not add new knobs before checking whether existing ones already express the desired behavior. The current important behavioral knobs are engagement, initiative, responsiveness, verbosity, stubbornness, directness, and compromise threshold. Final vote calls are moderator-owned again; peer self-closure was removed to keep the control flow explainable.
