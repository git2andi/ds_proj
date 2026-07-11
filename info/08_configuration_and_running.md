# 08 — Configuration and running

Important configuration sections:

- `llm`: provider and model selection.
- `simulation`: seeds and setup attempts.
- `environment`: auto/manual scenario setup.
- `participants`: auto/manual participant setup, hard-blocker probability, preference distribution, manual age/speech_style/profile fields.
- `conversation`: pacing, vote rounds, and discussion length.
- `threads`: thread-engine timing — cooling window, stale timeouts (longer for hard blockers), per-thread turn caps, cooling-continuation probabilities, reactivation.
- `narrowing`: discussion→narrowing gates — discussion-support requirement, hot-hard-blocker gate, stable-top-pair window, the single narrowing→discussion fallback.
- `routing`: normal act weights (`support/concern/ask/compare/comment`) and speaker-share tuning.
- `moderator`: opening, nudges, vote call, and closing behavior.
- `validation`: grounding checks.
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
  switch_resistance: 0.25
```

`parameters` accepts only `engagement`, `verbosity`, `directness`, `stubbornness`, and `switch_resistance` (each 0-1, partial overrides allowed; unset values are derived from traits). Age must be plausible for the profile. speech_style should describe wording register only, not behavior; if omitted it is derived from age.

## Run one topic

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

## Run eval

```powershell
py .\eval\run_eval_suite.py
```

The eval suite patches config for controlled cases and includes manual personas with varied age/speech_style/profile fields. It always runs all cases.

## Deterministic tests and static check

```powershell
py -m unittest discover -s tests
py -m compileall -q main.py src eval tests
```

The deterministic controller tests run without any LLM access and must pass before stochastic evaluation.
