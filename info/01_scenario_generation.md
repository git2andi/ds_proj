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
→ one complete regeneration with the error message
→ fail if still invalid
```

There is no partial semantic repair.

## Separate alias stage

After the option board is valid, one small LLM call proposes one to three natural aliases per fixed option name. An accepted alias:

- uses words from the full name in the same order;
- contains at most the configured number of words;
- remains unique after case, punctuation, article, and accent normalization;
- does not collide with another option’s full name or accepted aliases.

For example, `Chicago` may be accepted for `Chicago City Stay`. A generic reference such as `Hotel` is rejected when it could identify multiple options.

Malformed or colliding aliases are discarded. Alias failure never regenerates the valid scenario; the full name and `Option <ID>` remain valid fallbacks. The first accepted natural alias is stored as `short_name` for concise prompts and deterministic protocol fallbacks.

Manual scenario configuration remains available through `environment.mode: manual` and may supply explicit aliases.
