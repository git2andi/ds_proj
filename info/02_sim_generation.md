# Simulator and persona generation

After the scenario is valid, the builder samples participant parameters and asks the LLM for persona cards consistent with the assigned preferences and the first names fixed by the lightweight metadata call.

A persona contains:

- ID and unique first name;
- age and lexical speech style;
- background and private goal;
- engagement, verbosity, directness, and stubbornness;
- preferred option;
- stance and grounded reason for each option;
- optional hard-blocker status.

The Python runtime owns trait sampling, preference-shape sampling, ages, and hard-blocker selection through the run-local random generator. The persona LLM fills the descriptive card and reasons while respecting those assignments.

Stances use ranks from rejected to preferred. Normal participants may accept or switch when a public trigger and their stubbornness permit it. A hard blocker rejects every nonpreferred option and never moves.

An already validated scenario is preserved when persona generation retries. A missing, invalid, or duplicate metadata name receives a unique local fallback. The fixed name overrides any conflicting persona response, and occurrences of the discarded name are replaced in the background, private goal, rejection text, and stance reasons. Manual persona profiles can supply some or all fields through `participants.mode: manual`.

Action polarity is explicit: openings, support, and acceptance use `reason_for` or the option upside; objections and concern questions use `reason_against` or the option concern. Neutral card attributes are available for reactions and comparisons rather than being arbitrarily treated as positive or negative evidence.
