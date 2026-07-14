# Scenario and option generation

Scenario generation remains separate from the dialogue runtime. `SetupBuilder` either reads a manual option board or asks the configured dialogue LLM to create one.

A `Scenario` contains:

- the public decision topic;
- a short shared public context;
- fixed option IDs;
- a full and short option name;
- topic-specific public attributes;
- a brief upside and concern.

All objective facts available during the discussion must appear in the shared context or option cards. There are no hidden option facts. The setup prompt therefore asks the model to choose attributes natural for the topic without prescribing a fixed schema.

Short names must be supplied and validated. The builder may make one alias-only repair call when names are invalid or duplicated; it does not derive clipped names from the full option name.

The runtime treats the resulting option board as the source of truth. Simulators can make subjective judgments about public facts, but the realization model may not introduce new prices, times, distances, capacities, facilities, or specifications.
