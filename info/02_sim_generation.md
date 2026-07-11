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

The five simulator parameters are the only numeric behavior controls:

- `engagement`: expected speaker frequency / turn share;
- `verbosity`: average turn length, realized only as numeric word budgets (soft generation targets — accepted turns are never cut to length);
- `directness`: blunt vs soft wording;
- `stubbornness`: discussion-phase stance defense, concession, and willingness to soften — never final switching;
- `switch_resistance`: final movement only — switching, compromise acceptance, holdout behavior, and vote/repair resistance.

Beyond these five parameters and the per-option rank table there is no other persistent behavioral state: no commitment-strength or confidence float exists, and stance changes come only from accepted visible utterances.

Derivation (`src/simulator.py::derive_simulator_parameters`) uses normalized OCEAN values: extraversion (plus some conscientiousness) drives engagement; extraversion and openness drive verbosity; conscientiousness, extraversion, and low agreeableness drive directness; low agreeableness, neuroticism, low openness, and conscientiousness drive stubbornness. switch_resistance combines low agreeableness, conscientiousness, low openness, and neuroticism. All parameters are clipped to `[0, 1]`. Manual profile `parameters` may override any of the five directly.

High stubbornness means very resistant but theoretically movable; a hard blocker comes only from rank-1 rejections, never from stubbornness alone.

## Hard blockers

`personas.hard_blocker_probability` is a low **group-level** probability, sampled once per auto run. When it fires, exactly one participant becomes an exclusive hard blocker:

- exactly one preferred option (rank 5);
- every other option hard-rejected (rank 1) with a short grounded reason each;
- background and private goal state the one non-negotiable requirement that only the preferred option satisfies;
- agreeableness is pinned to 1, which derives high stubbornness and switch_resistance; engagement and verbosity stay free.

Multi-option rejection is represented by the rank table itself; the singular `rejection`/`rejection_reason` field remains the manual single-rejection input. The builder validates the exclusive contract after generation (one rank-5, all alternatives rank 1 with reasons) and retries the persona batch on violations; it also rejects a non-blocker persona that accidentally received the exclusive pattern. A blocker may speak politely, but its decision behavior consistently rejects every non-preferred option.

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
hidden OCEAN traits -> the five simulator parameters + plausible persona content
simulator parameters -> behavior
age/profile/speech_style -> plausibility and wording
option stances -> decision preferences
```
