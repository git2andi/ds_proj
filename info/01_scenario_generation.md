# 01 — Scenario and environment generation

The environment defines the factual decision space. It should provide enough shared facts for sims to argue concretely without needing to invent details.

## Modes

`environment.mode = auto`:

- The CLI topic is sent to the setup LLM.
- The setup LLM creates the scenario, opening question, shared context, and option cards.
- Generated options are validated before the run starts.

`environment.mode = manual`:

- `environment.manual` in `config.yaml` defines the topic, shared context, and options.
- The CLI topic is ignored.
- The environment setup LLM call is skipped.
- This is the best mode for controlled tests.

## Option board

The option board is the source of truth. Each option should have:

- a stable ID, usually A-D;
- a concrete name;
- factual attributes;
- upside/tradeoff/concern/best-for notes where useful;
- a short alias for natural dialogue.

The option board should be factual enough to support comparison, but not overloaded. Too many facts increase prompt size and may worsen generation.

## Grounding rule

Sims may use:

- option facts;
- shared context;
- visible prior dialogue;
- explicit uncertainty.

Sims should not state new concrete facts as known. Hypothetical mitigations are allowed only when marked as uncertain.

Allowed:

```text
Maybe we could ask for a quieter table, but the board does not say whether that is possible.
```

Not allowed:

```text
They have a quieter table available.
```

## Repeated unknown logistics

Sparse option facts are useful for testing grounding, but they used to create repeated unknown-logistics talk. Since the 2026-07-06 round the observer keeps an issue ledger (parking, booking, weather, seating, availability, prep time, crowds): an unknown may be raised once and answered once; after that the sim prompts list it as settled-unknown and it must not be re-raised. `repeated_unknown_mentions` in the metrics tracks violations.
