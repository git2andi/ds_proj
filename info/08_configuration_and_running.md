# 08 — Configuration and running

Important configuration sections:

- `llm`: provider and model selection.
- `simulation`: seeds and setup attempts.
- `environment`: auto/manual scenario setup.
- `participants`: auto/manual participant setup, hard-blocker probability, preference distribution, manual age/speech_style/profile fields.
- `conversation`: pacing, vote rounds, and discussion length.
- `moderator`: opening, nudges, vote call, and closing behavior.
- `output`: log paths and prompt dumps.

## Manual participant fields

Manual participant profiles may include:

```yaml
name: "Mina"
description: "A 22-year-old design student who prefers cheap, flexible plans."
age: 22
speech_style: "young casual wording"
preferred_option: "B"
traits:
  openness: 4
  conscientiousness: 2
  extraversion: 3
  agreeableness: 4
  neuroticism: 3
parameters:
  engagement: 0.6
  verbosity: 0.4
  directness: 0.5
  stubbornness: 0.3
```

`parameters` accepts only `engagement`, `verbosity`, `directness`, and `stubbornness` (each 0-1, partial overrides allowed; unset values are derived from traits). Age must be plausible for the profile. speech_style should describe wording register only, not behavior; if omitted it is derived from age.

## Run one topic

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

## Run eval

```powershell
py .\eval\run_eval_suite.py
```

The eval suite patches config for controlled cases and includes manual personas with varied age/speech_style/profile fields. It always runs all cases.

## Static check

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```
