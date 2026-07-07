# Option-grounded multi-user decision simulator

This repository generates configurable **multi-user decision discussions** with LLM-driven simulated participants.

The project scope is deliberately narrow:

```text
fixed option board + simulated participants + controller-routed discussion + visible decision outcome
```

It is not a generic chatbot, full society simulation, or full Generative-Agents-style memory system. The option board is the factual source of truth. Sims may compare options, ask questions, raise concerns, soften, resist, compromise, and vote, but they must not invent concrete facts outside the configured environment.

## Current v3 state

The current v3 line keeps the v1/v3 architecture but replaces the scattered private stance logic with one central per-sim/per-option rank table.

```text
4 = preferred
3 = acceptable
2 = neutral / untested
1 = disliked but negotiable
0 = rejected / hard blocked
```

Derived helpers such as `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` are computed from ranks. There are no separate runtime preference/rejection containers.

The persona setup may also provide a compact compatibility table for each sim and option:

```text
option id -> rank, short reason_for, short reason_against
```

Most options should remain neutral or acceptable. Strong dislikes and hard rejects should be rare and grounded.

## Controller / LLM separation

The controller owns the intended move:

```text
speaker + macro act + target option + reason + intended stance effect
```

The LLM renders one natural message. Validation checks whether the line visibly matches the intended move and stays grounded. State changes are applied through the rank table only after validation.

The compact macro act vocabulary is:

```text
opening, support, concern, ask, answer, compare, soften_toward, compromise, process, vote, closing
```

Only these macro acts are used by routing, prompts, and logs. Legacy act aliases were removed to avoid double control.

## Outcomes

A run ends in exactly one of three outcome states:

- `successful`: all visible final stances support the same option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement remains after bounded narrowing.

Outcomes are derived from visible transcript evidence only: explicit votes, acceptances, and parsed visible commitments. Private stance ranks guide routing, but they do not directly decide the final result.

## High-level pipeline

```text
CLI topic or manual environment
  -> scenario / option board
  -> automatic or manual simulated participants
  -> initial per-sim option ranks
  -> controller chooses speaker, target, macro act, and option focus
  -> LLM renders one utterance
  -> validation checks intent alignment and grounding
  -> observer updates rank/state views
  -> controller routes follow-ups, concerns, narrowing, votes, and closure
  -> consensus manager computes successful / majority / unresolved
  -> transcript.md, run.json, metrics.csv are written
```

## Main modules

- `main.py`: CLI entrypoint for one topic, a topic file, piped topics, or configured manual environment.
- `eval/run_eval_suite.py`: sequential regression suite for important mode combinations and edge cases.
- `config.yaml`: provider, environment, participant, pacing, routing, validation, and output settings.
- `src/builders.py`: builds automatic/manual scenarios and participants, including initial option-rank compatibility.
- `src/models.py`: dataclasses, compact macro acts, and per-option stance ranks.
- `src/simulator.py`: converts persona traits into operational simulator parameters and weak agenda hints.
- `src/dialogue.py`: orchestration loop for opening, discussion, voting, split narrowing, and closure.
- `src/policy.py`: speaker choice, macro-act choice, addressee choice, vote readiness, and procedural routing.
- `src/observer.py`: validated visible-state updates and rank movements from generated utterances.
- `src/parsing.py`: option references, commitments, votes, rejections, and parser-safe phrase families.
- `src/validation.py`: turn validation, parser/intent alignment, minimal fallback protection, and grounding checks.
- `src/prompts.py`: setup, utterance, moderator, repair, and grounding prompts.
- `src/consensus.py`: final outcome computation from visible evidence.
- `src/logger.py` / `eval/eval.py`: transcripts, structured traces, stance-rank metrics, and token diagnostics.

## Running

Activate the existing project environment, then run:

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

For eval cases:

```powershell
py .\eval\run_eval_suite.py --quick
py .\eval\run_eval_suite.py --full
```

Static check:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```
