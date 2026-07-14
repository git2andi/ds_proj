# Scenario generation

A scenario contains:

- one topic;
- short shared public context;
- exactly the configured option IDs;
- one public card per option;
- topic-specific factual attributes;
- one short upside and concern;
- a unique short name.

The complete board is always printed once before the dialogue and stored in the transcript.

The Python runtime retains the full board in `DialogueState.scenario`. The LLM is stateless: each realization call receives only the facts relevant to the selected action.

Raw attributes remain available for factual questions and grounded comparisons, but they do not automatically create conversational reasons.
