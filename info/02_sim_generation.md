# Simulator and persona generation

After the scenario is valid, the runtime fixes the experimental group before requesting persona text. Python-side seeded sampling determines:

- engagement, verbosity, directness, and ordinary stubbornness;
- whether one participant is a hard blocker;
- the group-size-specific preference-distribution shape;
- the preferred option assigned to each participant.

The persona LLM receives these assignments together with the validated option board and fixed participant names. It supplies a background, private goal, age, and option-specific stances. A stance contains a rank from 1 to 5 and grounded reasons for and against the option.

The runtime validates and normalizes the result. The assigned preferred option receives rank 5. For ordinary participants, an alternative rank of 5 is reduced to 4 and a rank of 1 is raised to 2; a hard blocker instead receives rank 1 for every nonpreferred option. Missing positive or negative reasons are replaced with the corresponding public upside or concern. Ages must fall within the supported range; a missing age receives a seeded fallback. Speech style and style tendencies are derived from age, verbosity, and directness unless manually configured.

A hard blocker receives stubbornness level 5, rejects every nonpreferred option, cannot accept or switch, and always votes for the preferred option. At most one hard blocker is present in a generated group.

The resulting private participant state contains the persona, traits, current preference, option ranks, acceptable and hard-rejected alternatives, grounded reasons, and used-point records. None of this private state becomes public unless an accepted utterance visibly expresses the corresponding preference, acceptance, or switch.

Persona generation may retry up to the configured setup-attempt limit without regenerating the already validated scenario. Manual participant profiles remain available through `participants.mode: manual`.
