# 08 — Configuration and running

Important configuration sections:

- `llm`: provider and model selection.
- `simulation`: seeds and setup attempts.
- `environment`: auto/manual scenario setup.
- `participants`: auto/manual participant setup, hard-blocker probability, preference distribution, manual age/style/profile fields.
- `conversation`: pacing, vote rounds, and discussion length.
- `moderator`: opening, nudges, vote call, and closing behavior.
- `output`: log paths and prompt dumps.

## Manual participant fields

Manual participant profiles may include:

```yaml
name: "Mina"
description: "A 22-year-old design student who prefers cheap, flexible plans."
age: 22
style: "young casual style: concise, relaxed, lightly informal"
preferred_option: "B"
traits:
  openness: 4
  conscientiousness: 2
  extraversion: 3
  agreeableness: 4
  neuroticism: 3
```

Age must be plausible for the profile. Style should describe wording, not behavior.

## Run one topic

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

## Run eval

```powershell
py .\eval\run_eval_suite.py --quick
py .\eval\run_eval_suite.py --full
```

The eval suite patches config for controlled cases and includes manual personas with varied age/style/profile fields.

## Static check

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```
