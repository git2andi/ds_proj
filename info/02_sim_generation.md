# 02 — Sim generation and participant parameters

A sim is a configurable participant, not just a name inserted into a prompt. Traits and operational parameters should influence observable behavior.

## Modes

`participants.mode = auto`:

- names, traits, initial preferences, backgrounds, and goals are generated/sampled;
- useful for varied runs and demos.

`participants.mode = manual`:

- profiles are provided in `config.yaml`;
- complete profiles can skip the persona setup LLM call;
- useful for controlled tests of traits, blockers, and split votes.

## Operational parameters

The relevant simulator parameters are:

- `engagement`: tendency to participate;
- `verbosity`: average turn length;
- `initiative`: tendency to drive procedure or propose moves;
- `responsiveness`: tendency to answer and react;
- `stubbornness`: resistance to switching;
- `directness`: likelihood of explicit pushback;
- `compromise_threshold`: how much evidence/social pressure is needed before moving.

## v3 stance on personality

v3 intentionally avoids adding extra personality layers such as `friendliness` or personal anchors. Traits should affect behavior first: who speaks, what act they choose, how resistant they are, and whether they can compromise. Wording variation is secondary and kept small.

## Hard blockers

Hard blockers should be rare in automatic generation. Manual profiles may explicitly define a rejection. A hard blocker should resist the blocked option but still participate normally in the discussion.

An agreeable participant can still have a real constraint if it is manually configured. Agreeableness affects tone and flexibility, not whether an absolute constraint exists.
