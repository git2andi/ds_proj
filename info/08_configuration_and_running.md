# 08 — Configuration and running

Important configuration sections:

- `llm`: two independently configurable roles — `dialogue` (all generative calls: setup, utterances, moderator, repair) and `validator` (structured semantic interpretation and claim grounding; runs cold, never writes public text). The same provider for both is valid; there is no third checker. The legacy single `llm.provider` key is a startup error.
- `simulation`: seeds and setup attempts.
- `environment`: auto/manual scenario setup.
- `participants`: auto/manual participant setup, hard-blocker probability, preference distribution, manual age/speech_style/profile fields.
- `conversation`: pacing, vote rounds, and discussion length.
- `threads`: thread-engine timing — cooling window, stale timeouts (longer for hard blockers), per-thread turn caps, cooling-continuation probabilities, reactivation.
- `narrowing`: discussion→narrowing gates — discussion-support requirement, hot-hard-blocker gate, stable-top-pair window, the single narrowing→discussion fallback.
- `routing`: normal act weights (`support/concern/ask/compare/comment`) and speaker-share tuning.
- `moderator`: opening, nudges, vote call, and closing behavior.
- `output`: log paths and prompt dumps.

- `validation`: the validation mode. `selective` (default) calls the validator LLM only when soft natural-language meaning can change state; deterministic fast paths (direct votes, sanctioned switches, blocker restatements, process/closing lines, mention-free light comments) skip the call, each skip traced with its reason. `full` interprets every candidate through the LLM (debug/eval). Safety-critical deterministic checks — strict commitments with post-checks, explicit blockers, claim-level grounding of accepted evidence — are always active in both modes and are not configurable.

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

## Running

```powershell
py .\main.py                                          # interactive prompt (auto) / configured environment (manual)
py .\main.py "Choose a restaurant for a group dinner" # explicit topic
py .\main.py topics.txt                               # batch file: one topic per line, # comments and blanks skipped
"Choose a restaurant" | py .\main.py                  # piped topic(s), one per line
```

Precedence:

1. An explicit CLI topic, topic file, or piped topic always requests automatic
   scenario generation for that topic — even when `environment.mode` is
   `manual`. Explicit input is never silently discarded.
2. With no explicit topic, `environment.mode: manual` runs the configured
   manual environment once; `auto` prompts interactively for a topic.
3. `participants.mode` is independent: manual profiles combine freely with an
   automatically generated (CLI) scenario.

Batch files run every topic in order; a failure identifies its topic and stops
the batch.

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
