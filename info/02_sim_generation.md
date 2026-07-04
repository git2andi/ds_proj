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
- verbosity: expected utterance length;
- initiative: tendency to propose, summarize, call votes, or drive procedure;
- responsiveness: tendency to answer direct questions and react to others;
- stubbornness: resistance to switching;
- directness: explicitness of disagreement or preference;
- compromise threshold: how much evidence/social pressure is needed before moving.

## Current behavior

Trait-weighted participation is now much better than earlier versions. Manual trait-spread runs show visible turn/length differences. Low-engagement sims should be quieter but not invisible. High-engagement sims should be more active but not accidentally dominant.

## Important design point

Do not implement a rigid agenda checklist. A sim may have goals, concerns, and commitments, but it should not mechanically execute an agenda item every turn. Persistent state should create pressure and continuity, while local dialogue decides the next act.

## Current open issue

Trait routing should be monitored but is not the top priority. Some auto/auto runs still show weak engagement correlation, partly because opening and voting phases force everyone to speak. Do not overfit this until split-vote narrowing and token cost are improved.
