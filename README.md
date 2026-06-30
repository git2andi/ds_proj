# Group Discussion Simulator

A university project exploring how well LLMs can simulate a small group of people having a real discussion and arriving at a decision together.

Give it a topic ("pick a restaurant for the team dinner", "choose a board game for game night") and it generates a full multi-party chat: 2-7 personas with their own backstories, goals, and opinions talk it through, react to each other, change their minds (sometimes), and either land on a shared choice or fail to agree.

## Goal

The motivation is that LLM-driven "user simulators" are increasingly used to test dialogue systems and group-decision tools — but only if the simulated conversations actually *behave* like real group discussions. This project is an attempt to get closer to the real thing while staying fully topic-agnostic.

A good run reads like friends making a decision together: casual and plain-spoken without Gen-Z slang, corporate jargon, formal debate language, or mini-essays. People should respond to each other, show their persona traits through behavior and turn length, occasionally get persuaded, and either converge for understandable reasons or end without agreement. The text must stay grounded in the scenario's facts and avoid repetitive option pitches.

## What I've been doing

Iteratively improving discussion quality through multiple passes:

- **Structural fixes**: commitment gating, outcomes computed across visible support, supporter/holdout/missing-commitment moderator targeting, hard-blocker integrity, echo guards, robotic template detection
- **Naturalness fixes**: anti-card-reading (don't parrot option descriptions), self-narration detection ("I should consider..."), collective voice rewrite ("we prioritize" → "I prioritize"), opener variety enforcement, conversational register in prompts
- **Verification infrastructure**: structured run logs, metrics, manual transcript review, and known-failure tracking

Current state and remaining issues are tracked in `docs/known_failures.md`. Architecture and run instructions are synchronized in `AGENTS.md` and `CLAUDE.md`.
