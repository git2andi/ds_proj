# 02 — Sim generation and participant parameters

A sim is a configurable participant, not just a name inserted into a prompt. Traits and operational parameters influence observable behavior.

## Modes

`participants.mode = auto`:

- names, traits, initial preferences, backgrounds, goals, and option-rank compatibility are generated/sampled;
- useful for varied runs and demos.

`participants.mode = manual`:

- profiles are provided in `config.yaml`;
- complete profiles can skip the persona setup LLM call;
- useful for controlled tests of traits, blockers, and split votes.

## Option-rank compatibility

Each sim has one rank for every option:

```text
4 = preferred
3 = acceptable
2 = neutral / untested
1 = disliked but negotiable
0 = rejected / hard blocked
```

The setup call may provide short `reason_for` and `reason_against` fields. The builder normalizes the table so the assigned primary preference is rank 4, secondary preferences are acceptable, and explicit rejections are rank 0.

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

v3 avoids extra personality layers such as `friendliness` or personal anchors. Traits affect behavior first; wording variation is secondary.
