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

Manual environments define the option board and shared context deterministically. Manual participants define profiles, initial preferences, and optionally full parameter overrides. Fully manual environments plus complete manual profiles can skip setup LLM calls, but dialogue turns still use the LLM.

## Current status after latest full evaluation

The latest full evaluation run after the split/narrowing implementation showed clear improvement over the previous full suite:

```text
unresolved outcomes:        7/12 -> 4/12
mid-discussion lean shifts: 3 total -> 16 total
participant procedure:      now visible in no/light-moderator cases
split reservation exchanges: now present
transcript metadata:        provider/model/modes/seed/pacing visible
```

However, the system is not final. The remaining open issues are listed in `docs/todo.md`. The most important ones are:

1. split-vote candidate selection is still sometimes socially implausible;
2. post-reservation narrowing needs a clearer switch/stay/alternative step;
3. the `n=2` deadlock protocol still needs a forced validation case;
4. compromise prompts can still mix option-specific concerns;
5. token cost remains very high;
6. grounding still leaks occasional unsupported logistical claims;
7. trait routing is improved but should keep being monitored.

Do not add more broad features until these are addressed.
