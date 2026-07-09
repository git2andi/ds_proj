# 02 — Sim generation and participant parameters

A sim is a configurable participant, not just a name inserted into a prompt. Traits and operational parameters influence observable behavior.

## Modes

`participants.mode = auto`:

- names, hidden OCEAN traits, age, backgrounds, goals, and option-rank compatibility are generated/sampled; speech_style is derived from age;
- useful for varied runs and demos.

`participants.mode = manual`:

- profiles are provided in `config.yaml`;
- manual profiles may include age and speech_style;
- complete profiles can skip the persona setup LLM call;
- useful for controlled tests of parameters, blockers, speech-style visibility, and split votes.

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

## OCEAN -> parameters -> attributes

OCEAN traits are hidden setup traits. They are sampled (or manually fixed), used once to derive the simulator parameters and to keep generated persona content plausible, and never passed into utterance prompts or routing.

The four simulator parameters are the only numeric behavior controls:

- `engagement`: expected speaker frequency / turn share;
- `verbosity`: average turn length, realized only as numeric word budgets;
- `directness`: blunt vs soft wording;
- `stubbornness`: resistance to changing stance and strength of stance defense.

Derivation (`src/simulator.py::derive_simulator_parameters`) uses normalized OCEAN values: extraversion (plus some conscientiousness) drives engagement; extraversion and openness drive verbosity; conscientiousness, extraversion, and low agreeableness drive directness; low agreeableness, neuroticism, low openness, and conscientiousness drive stubbornness. All parameters are clipped to `[0, 1]`. Manual profile `parameters` may override any of the four directly.

High stubbornness means very resistant but theoretically movable; a hard blocker comes only from `rejection` (option rank 1), never from stubbornness alone.

## Age, profile, and speech style

Each persona has:

- `age`: integer in the configured valid range (generated ages stay adult, 18-75);
- `speech_style`: compact age-band register, derived from age unless manually overridden:

```text
18-27 -> young casual wording
28-40 -> relaxed practical wording
41-58 -> direct workplace wording
59+   -> measured traditional wording
```

- `background`: short plausible profile/backstory;
- `private_goal`: decision motivation.

Age and speech_style are not behavioral controls. They affect wording only: formality, phrasing, sentence shape, and conversational flavor. They must not change the option-rank table, vote logic, compromise logic, or turn-taking weights, and no speech_style string encodes preferences, decision behavior, turn length, or directness.

The builder checks for obvious age/profile contradictions. Examples that should fail:

- a 20-year-old senior manager or executive;
- a 21-year-old with decades of experience;
- a very young participant described as a married parent with a mortgage;
- an older participant described as a high-school student.

The intended separation is:

```text
hidden OCEAN traits -> the four simulator parameters + plausible persona content
simulator parameters -> behavior
age/profile/speech_style -> plausibility and wording
option stances -> decision preferences
```
