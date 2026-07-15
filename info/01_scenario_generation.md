# Scenario generation

A scenario contains:

- one topic;
- one shared scenario description of one or two complete sentences;
- exactly the configured option IDs;
- one public card per option;
- topic-specific factual attributes;
- one short upside and concern;
- a unique short name.

The shared description is a paragraph, not a bullet list. It describes the common situation, constraints, and stakes that apply across the option board. It must not assign an option-specific duration, cost, availability, capacity, or outcome to the whole scenario. Hard numeric limits are checked against every option; contradictory generated boards are retried before the final deterministic cap repair.

For example, a shipping context may state that one fragile prototype must arrive complete and that delays or extra handling matter. It should not state that the journey takes twelve hours when the options have different transit times.

The complete context and board are printed once before the dialogue and stored in the transcript. The Python runtime retains them in `DialogueState.scenario`. The LLM is stateless: each realization call receives only the facts relevant to the selected action.

Raw attributes remain available for factual questions and grounded comparisons, but they do not automatically create conversational reasons.

If the setup LLM returns an invalid or duplicate `short_name`, the runtime makes two small alias-only repair attempts. If those still fail, it deterministically selects a short unique phrase from meaningful words already present in the full option name, such as `British Airways` or `Delta Airlines`. It never invents an abbreviation or external airport code.
