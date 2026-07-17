# System overview

The project implements configurable user simulators for an option-grounded multi-user decision dialogue.

```text
topic
→ scenario and option board
→ persona cards and private stances
→ simulator-owned UserAction bids
→ categorical floor selection
→ LLM utterance realization
→ deterministic validation and grounding
→ public state and issue updates
→ narrowing and voting
→ outcome and logs
```

Main components:

- **setup** (`src/builders.py`): scenario, aliases, personas, traits, and initial stances;
- **simulators/floor** (`src/simulator.py`): participant-local decisions and intact bid selection;
- **environment** (`src/dialogue.py`): phase control, direct-answer obligations, one active issue, soft coverage, narrowing, and voting;
- **language** (`src/prompts.py`, `src/llm_client.py`): one realization call for the selected action and at most one focused repair;
- **validation** (`src/validation.py`, `src/aliases.py`): structured-action invariants and narrow deterministic grounding;
- **outcomes** (`src/consensus.py`): public narrowing and vote-derived results;
- **logging/evaluation** (`src/logger.py`, `src/eval.py`, `eval2/`): transcript, structured run state, compact metrics, batch evaluation, and LLM judging.

The simulator is the behavioral authority. It decides whether to bid, which action to take, what reason to use, whether to move, and how to vote. The environment manages only the shared interaction protocol. The LLM does not choose hidden behavior.

Runtime phases:

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

One bounded re-narrowing and re-vote is possible after a first round without a majority, but only when visible movement occurred.

Deterministic tests live in `tests/`. Active LLM-backed evaluation lives in `eval2/`. The old `eval/` folder preserves historical results, including the intentionally interrupted earlier scenario batch.
