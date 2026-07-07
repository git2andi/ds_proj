# Option-grounded multi-user decision simulator

This repository generates configurable **multi-user decision discussions** with LLM-driven simulated participants.

The project scope is deliberately narrow:

```text
fixed option board + simulated participants + controller-routed discussion + visible decision outcome
```

It is not a generic chatbot, full society simulation, or full Generative-Agents-style memory system. The option board is the factual source of truth. Sims may compare options, ask questions, raise concerns, soften, resist, compromise, and vote, but they must not invent concrete facts outside the configured environment.

## v3 state

v3 uses the clearer v1 codebase as the base and ports only selected v2 behavior fixes that directly support the project goal:

- controller-selected, LLM-rendered post-reservation switch/stay decisions after split votes and majority holdout checks;
- no downhill compromise: a sim should not switch from its own larger/equal visible camp into a weaker one just to force closure;
- bounded tie compromise for flexible sims when no option has a strict lead;
- unresolved acknowledgement before closure, so unresolved endings are socially legible rather than abrupt;
- split-summary self-answer avoidance in no-/low-moderator rounds;
- trait influence on routing: directness increases challenge tendency, compromise tendency increases bridge/soften moves;
- agenda is a weak hint only; active questions, answers, and unresolved concerns outrank private agenda items;
- vote wording is parser-safe, lightly trait-shaped, and validated against the controller-selected target;
- observer fixes prevent false blockers on a sim's own current favorite;
- final voting now avoids reverting to an old latent favorite that the same sim visibly objected to;
- vote calls are moderator-owned again; participant self-closure was removed to keep the phase transition simple and explainable.

v3 deliberately does **not** port v2's micro-reaction subsystem, friendliness parameter, personal anchors, or larger trait-colored wording subsystem. Those features added output texture but made the controller harder to explain.

## Outcomes

A run ends in exactly one of three outcome states:

- `successful`: all visible final stances support the same option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement remains after bounded narrowing.

Outcomes are derived from visible transcript evidence only: explicit votes, acceptances, and parsed visible commitments. Hidden latent preferences may guide routing, but they should not decide the final result directly.

## High-level pipeline

```text
CLI topic or manual environment
  -> scenario / option board
  -> automatic or manual simulated participants
  -> controller chooses speaker, target, act, and option focus
  -> LLM renders one utterance
  -> observer parses visible text and updates public state
  -> controller routes follow-ups, concerns, narrowing, votes, and closure
  -> consensus manager computes successful / majority / unresolved
  -> transcript.md, run.json, metrics.csv are written
```

## Main modules

- `main.py`: CLI entrypoint for one topic, a topic file, piped topics, or configured manual environment.
- `eval/run_eval_suite.py`: sequential regression suite for important mode combinations and edge cases.
- `config.yaml`: provider, environment, participant, pacing, routing, validation, and output settings.
- `src/builders.py`: builds automatic/manual scenarios and participants.
- `src/simulator.py`: converts persona traits into operational simulator parameters.
- `src/dialogue.py`: orchestration loop for opening, discussion, voting, split narrowing, and closure.
- `src/policy.py`: speaker choice, act choice, addressee choice, vote readiness, and procedural routing.
- `src/observer.py`: visible-state updates from generated utterances.
- `src/parsing.py`: option references, commitments, votes, rejections, and parser-safe phrase families.
- `src/validation.py`: turn validation, parser/intent alignment, minimal fallback protection, and grounding checks.
- `src/prompts.py`: setup, utterance, moderator, repair, and grounding prompts. Decision prompts deliberately use parser-friendly commitment wording so repair/fallback calls stay rare.
- `src/consensus.py`: final outcome computation from visible evidence.
- `src/logger.py` / `eval/eval.py`: transcripts, structured traces, metrics, and token diagnostics.

## Running

Activate the existing project environment, then run:

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

For a topic file:

```powershell
py .\main.py scenarios.txt
```

For the evaluation suite:

```powershell
py .\eval\run_eval_suite.py --quick
py .\eval\run_eval_suite.py --full
py .\eval\run_eval_suite.py --list
```

The suite temporarily patches `config.yaml`, writes logs under `eval/logs_eval_suite/`, writes `eval/logs_eval_suite/eval_suite_runs.csv`, and restores the original config afterward.

## Configuration modes

Two independent mode switches matter:

```yaml
environment:
  mode: auto | manual

participants:
  mode: auto | manual
```

This gives four important test modes:

```text
auto environment + auto participants
manual environment + auto participants
auto environment + manual participants
manual environment + manual participants
```

Manual environments are best for controlled tests. Manual participants are best for checking trait behavior, blockers, and specific split-vote shapes.

## LLM provider

The default quality baseline is the configured `gpt` provider:

```yaml
llm:
  provider: "gpt"
```

Provider comparisons should be explicit. Different providers change style, parsing reliability, grounding, and repair rates.

## Validation

Before claiming a behavioral fix, run at least:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
py .\eval\run_eval_suite.py --quick
```

Before treating a version as stable, run:

```powershell
py .\eval\run_eval_suite.py --full
```

Then inspect transcripts manually. Metrics alone are insufficient; a run can have a good outcome label while the discussion still feels forced or under-argued.

