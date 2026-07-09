# 02 — Sim generation and participant parameters

A sim is a configurable participant, not just a name inserted into a prompt. Traits and operational parameters influence observable behavior.

## Modes

`participants.mode = auto`:

- names, traits, age, style, backgrounds, goals, and option-rank compatibility are generated/sampled;
- useful for varied runs and demos.

`participants.mode = manual`:

- profiles are provided in `config.yaml`;
- manual profiles may include age and style;
- complete profiles can skip the persona setup LLM call;
- useful for controlled tests of traits, blockers, style visibility, and split votes.

## Option-rank compatibility

Each sim has one rank for every option:

```text
5 = preferred
4 = acceptable
3 = neutral / untested
2 = disliked but negotiable
1 = rejected / hard blocked
```

The setup call may provide short `reason_for` and `reason_against` fields. The builder normalizes the table so the assigned primary preference is rank 5, secondary preferences are acceptable at rank 4, neutral options are rank 3, disliked options are rank 2, and explicit rejections are rank 1.

Design rule:

- only give strong reasons for some options;
- leave many options neutral;
- keep reasons short;
- avoid hard rejects unless explicitly needed;
- allow rank movement during discussion.

## Operational parameters

The relevant simulator parameters are:

- `engagement`: tendency to participate;
- `verbosity`: average turn length;
- `initiative`: tendency to drive procedure or propose moves;
- `responsiveness`: tendency to answer and react;
- `stubbornness`: resistance to switching;
- `directness`: likelihood of explicit pushback;
- `compromise_threshold`: how much evidence/social pressure is needed before moving.

These parameters are derived from traits and are behavior-relevant.

## Age, profile, and style

Each persona has:

- `age`: integer in the configured valid range;
- `style`: concise speech-style instruction;
- `background`: short plausible profile/backstory;
- `private_goal`: decision motivation.

Age and style are not behavioral traits. They should affect wording only: formality, phrasing, sentence shape, and conversational flavor. They must not change the option-rank table, vote logic, compromise logic, or turn-taking weights.

The builder checks for obvious age/profile contradictions. Examples that should fail:

- a 20-year-old senior manager or executive;
- a 21-year-old with decades of experience;
- a very young participant described as a married parent with a mortgage;
- an older participant described as a high-school student.

The intended separation is:

```text
traits -> behavior
age/profile/style -> plausibility and wording
option stances -> decision preferences
```
