# System overview

The system simulates a small group (2–7 configurable user simulators) choosing among a fixed set of public options. It is an option-grounded group-decision simulator, not a general social simulation: there are no coalitions, emotions, deception, or hidden hierarchies.

## Separation of responsibilities

The implementation separates:

- **setup** (`src/builders.py`): scenario and persona generation;
- **simulator policy** (`src/simulator.py`): participant-local structured decisions;
- **floor/environment** (`src/dialogue.py`): conversational obligations and phase control;
- **realization** (`src/prompts.py`, `src/llm_client.py`): one LLM call for the selected action;
- **validation** (`src/validation.py`, `src/aliases.py`): deterministic hard-correctness checks;
- **outcome** (`src/consensus.py`): public narrowing and clear visible votes;
- **logging/metrics** (`src/logger.py`, `src/eval.py`): transcript, structured state, flat metrics.

The key research object is the user simulator. Each simulator owns whether it wants to participate and what it wants to do. The environment does not assign ordinary support, concern, comparison, compromise, or switching behavior. The structured `UserAction` a simulator produces is authoritative: the LLM only turns it into one natural chat message, and no validator LLM exists.

## Runtime phases

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

One bounded `VOTING → NARROWING → VOTING` return is allowed when the first vote has no majority and the re-narrowing produces visible movement.

Discussion progression is deliberately simple: direct questions create obligations, one active issue supports short local exchanges, stagnation may expose one optional compromise opportunity, and adaptive narrowing schedules only participants whose position still matters.

## Evaluation stack

Deterministic tests (`tests/`, no LLM) assert ownership boundaries and protocol behavior. Four diagnostic LLM-backed tools live in `eval/`: a pinned 17-case suite, a diverse-topic batch runner over `scenarios.txt`, a one-knob-at-a-time config sensitivity sweep, and a ChatEval-style multi-judge transcript scorer. See `07_evaluation_and_logging.md`.
