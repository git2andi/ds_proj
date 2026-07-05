# 02 — Sim generation and participant parameters

A sim is a configurable participant in the group decision. Sims are not just names in a prompt; their traits and parameters should influence observable behavior.

## Modes

`participants.mode = auto`:

- participants are sampled/generated automatically;
- traits, private goals, and initial preferences are assigned;
- useful for demos and varied runs.

`participants.mode = manual`:

- profiles are provided in `config.yaml`;
- complete profiles can skip the persona setup LLM call;
- this is the main mode for controlled behavior tests.

## Operational parameters

OCEAN/persona information is converted into operational parameters such as:

- engagement: how often the sim enters the discussion;
- verbosity: expected average utterance length;
- initiative: tendency to propose, summarize, call votes, or drive procedure;
- responsiveness: tendency to answer direct questions and react to others;
- stubbornness: resistance to switching;
- directness: explicitness of disagreement or preference;
- compromise threshold: how much evidence/social pressure is needed before moving.

## Hard blockers and constraints

`personas.hard_blocker_probability` controls the rare case where a sim is blocker-like by sampled traits. A hard blocker should resist compromise strongly, but should not sabotage the chat or refuse to interact.

Normal auto-generated sims should not routinely receive categorical hard constraints. They may start with preferences and goals, but those should usually be movable.

Manual profiles may explicitly define blockers. If a profile or generated description contains a genuinely absolute constraint such as strict dietary need, allergy, accessibility need, hard budget ceiling, or schedule impossibility, agreeableness should not erase the constraint. An agreeable participant can reject an option politely.

Since 2026-07-06 this is decoupled in the builder: a manual profile with a `rejection` keeps its explicitly configured agreeableness (the constraint binds regardless of personality); only a profile that leaves agreeableness unset falls back to the classic pinned-to-1 blocker persona. Normal auto personas are additionally instructed to phrase needs as preferences, never absolutes.

## Current behavior

Verbosity orders average turn length (~8 words for terse sims, ~16-18 for chatty ones) while every sim also produces genuinely short beats. Engagement shapes free-discussion turn share (correlation mostly 0.7-1.0 in the 2026-07-06 suite); opening/vote rounds stay intentionally uniform and are excluded from dominance judgments.

## Important design point

Do not implement a rigid agenda checklist. A sim may have goals, concerns, and commitments, but it should not mechanically execute an agenda item every turn. Persistent state should create pressure and continuity, while local dialogue decides the next act.

## Current open issue

Trait realization should be evaluated on free discussion turns, not on opening and final vote rounds. The next round should allow plausible dominance by high-engagement/high-initiative sims while preventing repetitive monologues and total disappearance of quieter sims.
