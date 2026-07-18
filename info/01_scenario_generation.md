# Scenario and alias generation

Automatic setup first asks the configured dialogue provider for a compact shared context and four public option cards. Each option contains:

- fixed ID `A`–`D`;
- full name;
- three to five public attributes;
- one upside;
- one concern.

Validation is structural and topic-general. It checks IDs, required fields, attribute count, unique full names, context length, and accidental generated-participant references. It does not infer whether arbitrary superlatives, attribute aliases, or missing real-world facts are semantically correct.

The scenario policy is:

```text
initial complete generation
→ full structural validation
→ up to two complete regenerations with the latest error message
→ fail and record a setup error if all three attempts are invalid
```

There is no partial semantic repair.

## Separate alias stage

After the option board is valid, one small LLM call proposes one or two natural aliases per fixed option name and one unique first name per participant. An accepted alias:

- uses words from the full name in the same order;
- contains at most the configured number of words;
- remains unique after case, punctuation, article, and accent normalization;
- does not collide with another option’s full name or accepted aliases;
- contains at least two words and no numbers;
- does not end in an incomplete connector.

For example, `Chicago City` may be accepted for `Chicago City Stay`. Single-word automatic aliases are not used.

Malformed or colliding aliases are discarded. The builder may derive one or two two-word candidates from the full name, then applies the same checks. Metadata-call failure never regenerates the valid scenario; derived aliases remain available and missing or invalid participant names receive unique local fallbacks. The first accepted natural alias is stored as `short_name` for concise prompts and deterministic protocol text.

Manual scenario configuration remains available through `environment.mode: manual` and may supply explicit aliases.
