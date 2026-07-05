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
- `style`: name/option/I/we opening suppression.
- `utterances`: recent context and word budgets.
- `validation`: turn validation and grounding checks.
- `output`: log paths and prompt dumping.

## LLM provider

For the next quality baseline, use:

```yaml
llm:
  provider: "gpt"
```

Do not compare quality across providers unless provider comparison is the explicit task. Provider differences affect style, grounding, repair behavior, and option parsing.

## Cost-related defaults

The current defaults intentionally keep participant prompts compact:

```yaml
scenario.option_prompt_max_words: 34
utterances.recent_turns_in_prompt: 4
validation.grounding_mode: tripwire
```

Do not raise these casually. Larger context usually increases cost faster than it improves dialogue quality. The next quality round should mostly reduce turn length and improve deterministic state/routing.

## Current tuning knobs

Settled in the 2026-07-06 round (change only with fresh evidence):

- `utterances.word_budgets`: opening 18, discussion 15, ask/answer 13, vote 11 — the controller scales these by verbosity/engagement and mixes in deterministic short beats (`policy._word_bounds`);
- `routing.direct_address_probability` (0.32) is additionally scaled down by group size in the policy (x0.15 at n=2, x0.6 at n=3);
- `style.name_prefix_max_fraction`: 0.3; n=2 suppresses non-functional name prefixes outright;
- `routing.trait_share_adaptation` 3.5, `max_share_overshoot` 0.16, softened anti-monopoly damp: trait-shaped dominance, never monologue;
- `personas.hard_blocker_probability`: hard blockers stay rare; manual profiles may pair a rejection with any explicit agreeableness;
- `validation.grounding_mode`: keep tripwire; the judge is scoped to the options a line actually mentions.

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
py .\run_eval_suite.py --list
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

- n=2 direct-address and deadlock behavior;
- n=3 three-way split;
- n=4 trait spread;
- n=5+ scaling and dominance;
- full moderator;
- no moderator;
- light moderator.

## Current validation focus

Inspect whether `gpt` runs produce shorter, less template-like, more trait-shaped discussions without increasing repair/grounding cost.
