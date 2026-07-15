# User simulator generation

Each persona contains:

- name;
- age, `speech_style`, and two stable realization tendencies;
- short background;
- private goal;
- direct traits: engagement, verbosity, directness, stubbornness;
- one `OptionStance` per option;
- optional hard-blocker state.

## Trait responsibilities

- engagement → probability of a voluntary bid;
- verbosity → maximum realization length;
- directness → wording instruction;
- stubbornness → movement probability after a concrete trigger. Movement may be acceptance or a preference switch.

Normal stubbornness is 1–4. A hard blocker uses 5, accepts only its preferred option, and never switches.

Persona reasons are the primary content source. Option upside/concern is fallback content. Private information becomes public only when the simulator says it visibly.


A non-hard-blocker can autonomously propose an acceptable alternative when the group is stagnant. The opportunity is probabilistic and participant-local; no controller target or switch is forced.

## Linguistic signature

Each persona receives two compact, deterministic style tendencies derived from its directness, verbosity, and speech register. Examples include leading with the conclusion, acknowledging another view before disagreeing, using conversational contractions, or preferring one compact sentence. These tendencies affect wording only and never change participation, stance, reasons, or vote choice.
