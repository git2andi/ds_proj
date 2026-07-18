# Scenario and alias generation

Automatic setup starts from a caller-supplied topic. The first LLM call generates a compact shared context and four public option cards. Each option contains:

- fixed ID `A`--`D`;
- full name;
- three to five public attributes;
- one upside;
- one concern.

Validation is structural and topic-general. It checks the fixed IDs, required fields, attribute counts, unique option names, context length, and accidental references to generated participants. It does not attempt to verify arbitrary real-world facts or infer missing topic semantics.

The scenario policy is:

```text
initial complete generation
→ structural validation
→ up to two complete regenerations with the latest error as feedback
→ setup failure if all configured attempts are invalid
```

Generated facts are not partially rewritten by deterministic code.

## Alias and name metadata

After the option board is valid, a separate lightweight LLM call proposes up to two aliases per option and one first name per participant. A generated alias must:

- be derived from words in the corresponding full name and preserve their order;
- remain within the configured word limit;
- contain no numbers;
- contain at least two words when the full name has more than one word;
- not end in an incomplete connector such as `to`, `of`, or `with`;
- identify only one option after normalization.

Malformed or colliding suggestions are discarded. The builder also tests deterministic two-word candidates derived from the full option name. The first accepted alias becomes `short_name`; if no alias is valid, the full name remains usable. A metadata-call failure never invalidates or regenerates an already valid scenario. Invalid or duplicate participant names receive unique local fallbacks.

Manual scenarios remain available through `environment.mode: manual` in `config.yaml`.
