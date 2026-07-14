# System overview

The system simulates a small group choosing among a fixed set of public options.

The implementation separates:

- **setup**: scenario and persona generation;
- **simulator policy**: participant-local structured decisions;
- **floor/environment**: conversational obligations and phase control;
- **realization**: one LLM call for the selected action;
- **validation**: deterministic hard-correctness checks;
- **outcome**: public narrowing and clear visible votes.

The key research object is the user simulator. Each simulator owns whether it wants to participate and what it wants to do. The environment does not assign ordinary support, concern, comparison, compromise, or switching behavior.


Discussion progression is deliberately simple: direct questions create obligations, one active issue supports short local exchanges, stagnation may expose one optional compromise opportunity, and adaptive narrowing schedules only participants whose position still matters.
