# Option-grounded multi-user decision simulator

This repository is a university project for generating configurable **multi-user decision discussions** with LLM-driven simulated participants.

The intended scope is deliberately narrow:

```text
fixed option board + simulated group participants + visible discussion/voting outcome
```

It is **not** a generic chatbot, open-ended group-chat bot, agenda-based user simulator, or society simulation. The option board is the factual source of truth. Simulated users may compare options, raise concerns, soften their stance, propose bounded compromises, and vote, but they should not invent concrete facts outside the environment.

## Current project goal

Given either a one-line topic or a manual environment, the system should create a small group of 2-7 simulated users who discuss a set of option cards and reach one of three outcomes:

- `successful`: all visible final stances support the same option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement remains after bounded narrowing.

Do not introduce a fourth outcome for invalid consensus. If a participant cannot accept an option because of a hard blocker, hard constraint, or unresolved decisive concern, that should be visible in the transcript and the existing outcome logic should produce `majority` or `unresolved`.

The important claim is not merely that transcripts sound natural. The project should behave like a **configurable simulator**: participant parameters such as engagement, initiative, responsiveness, verbosity, stubbornness, directness, and compromise tendency should visibly affect turn-taking, answer behavior, stance movement, and willingness to compromise.

## Pipeline

A run follows this high-level pipeline:

```text
CLI topic or manual environment
  -> option-grounded scenario / option board
  -> automatic or manual simulated participants
  -> controller selects speaker, target, act, and option focus
  -> LLM renders one visible utterance
  -> observer parses visible text and updates state
  -> controller routes follow-ups, concerns, softening, votes, and narrowing
  -> outcome is computed from visible votes/acceptances only
  -> transcript, run.json, and metrics.csv are written
```

The main modules are:

- `main.py`: CLI entrypoint. Accepts a topic, a topic file, piped topics, or a manual environment from `config.yaml`.
- `src/config_loader.py`: validates the configuration and manual environment/participant modes.
- `src/builders.py`: builds scenarios, option boards, shared context, and personas.
- `src/simulator.py`: converts persona traits into operational simulator parameters.
- `src/dialogue.py`: orchestrates opening, discussion, vote rounds, narrowing, compromise, and closure.
- `src/policy.py`: speaker selection, act selection, routing, vote readiness, and procedural moves.
- `src/observer.py`: parses visible utterances into public state changes.
- `src/validation.py`: validates turns, repairs invalid output, and runs grounding checks.
- `src/prompts.py`: setup, participant, moderator, repair, and grounding prompts.
- `src/consensus.py`: computes `successful`, `majority`, or `unresolved` from visible evidence.
- `src/logger.py`: writes transcript, JSON trace, metrics, provider/model/mode metadata, and token diagnostics.
- `src/evaluation.py`: computes dialogue and simulator-realization metrics.
- `run_eval_suite.py`: sequential evaluation suite covering auto/manual environments, auto/manual participants, multiple group sizes, moderator modes, split votes, and grounding cases.

## Running

Activate the existing virtual environment for the project, then run:

```powershell
py .\main.py
```

For a single topic:

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

For a topic file:

```powershell
py .\main.py scenarios.txt
```

For the evaluation suite:

```powershell
py .\run_eval_suite.py --quick
py .\run_eval_suite.py --full
```

The suite temporarily overwrites `config.yaml`, runs cases sequentially, writes logs under `logs_eval_suite/`, writes `logs_eval_suite/eval_suite_runs.csv`, and restores the original config at the end.

## LLM provider

Use the `gpt` provider for the next dialogue-quality baseline unless explicitly testing provider differences.

```yaml
llm:
  provider: "gpt"
```

The project also contains other provider paths for compatibility, but quality evaluation should not mix providers casually because provider differences affect style, grounding, and parsing behavior.

## Configuration modes

Two independent mode switches matter:

```yaml
environment:
  mode: auto | manual

participants:
  mode: auto | manual
```

This creates four important test combinations:

```text
auto environment + auto participants
manual environment + auto participants
auto environment + manual participants
manual environment + manual participants
```

Manual environments define the option board and shared context deterministically. Manual participants define profiles, initial preferences, optional blockers, and optionally full parameter overrides. Fully manual environments plus complete manual profiles can skip setup LLM calls, but dialogue turns still use the LLM.

## Current quality focus

The project already has the main literature-shaped architecture: environment setup, simulated participants, controller routing, addressee targeting, visible-state observation, validation/repair, and evaluation. The next improvements should therefore not add a large new subsystem.

The 2026-07-06 behavioral round made discussions shorter, more causally coherent, more trait-shaped, and cheaper to run. The mechanisms now in place:

1. trait-scaled word budgets with a deterministic short-beat mixture (avg ~13-16 words/turn, short turns for every sim) and clause-boundary salvage instead of mid-sentence chops;
2. answer follow-ups develop the same thread instead of asking the next question; statement acts get a tail-question suppression flag;
3. direct addressing scales with group size (rare name prefixes in n=2);
4. dominance is judged on free discussion turns with softened anti-monopoly damping;
5. manual profiles may combine an explicit hard constraint with any agreeableness; normal auto personas get preference wording, never absolutes;
6. stance switches need net visible vote advantage or trait-level flexibility; a sim's own unanswered concerns add switch resistance;
7. an issue ledger stops repeated "we still don't know about parking" loops after one raise + one answer;
8. compromise proposals are pinned to one concrete option;
9. grounding runs on a narrowed tripwire with option-scoped judging.

Open work is maintained in `docs/todo.md` (currently: code-path simplification, grounding-judge false positives, combo monitoring, one metric artifact).

## Dialogue behavior principles

- Direct questions should usually be answered promptly by the addressed sim.
- A response should not routinely open a fresh unrelated topic.
- Speaking balance is not the target. Dominant sims may speak more if their traits support it.
- Quiet sims should still appear enough for their stance to be visible.
- Same-speaker continuations are allowed when they add new content rather than repeat.
- Direct names are useful but should be less frequent, especially in n=2 runs.
- Verbosity is an average tendency. All sims may have both short and longer turns.
- Sims may propose conditional compromises, but one concrete option should remain the final winner.
- Normal auto-generated sims should not receive categorical hard constraints unless the hard-blocker path is active.
- Explicit/manual constraints such as allergies, strict dietary needs, accessibility needs, or budget ceilings should be respected even if the participant is agreeable.

## Validation

Before claiming a behavioral fix, run:

```powershell
py -m py_compile main.py run_eval_suite.py src\*.py
py run_eval_suite.py --full
```

Inspect transcripts manually. Key questions:

- Are turns shorter and less summary-like?
- Does Q→A adjacency work without creating question churn?
- Is direct naming lower but still available when useful?
- Does speaking dominance follow traits on free discussion turns?
- Do stance switches have visible reasons?
- Do hard blockers prevent false unanimity?
- Do repeated unknown logistics disappear?
- Do repair/grounding token costs stay controlled?

Execution success alone is not enough. The transcript must read like a plausible option-grounded group decision.
