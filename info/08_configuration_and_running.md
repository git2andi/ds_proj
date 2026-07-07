# 08 — Configuration and running

Important configuration sections:

- `llm`: provider and model selection.
- `simulation`: seeds and setup attempts.
- `environment`: auto/manual scenario setup.
- `participants`: auto/manual participant setup, hard-blocker probability, and preference distribution.
- `conversation`: pacing, vote rounds, and discussion length.
- `moderator`: opening, nudges, vote call, and closing behavior.
- `output`: log paths and prompt dumps.

Run one topic:

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

Run eval:

```powershell
py .\eval\run_eval_suite.py --quick
py .\eval\run_eval_suite.py --full
```

Static check:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```
