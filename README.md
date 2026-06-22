# Group Discussion Simulator

A university project exploring how well LLMs can simulate a small group of people having a real discussion and arriving at a decision together.

Give it a topic ("pick a restaurant for the team dinner", "choose a board game for game night") and it generates a full multi-party chat: 2-7 personas with their own backstories, goals, and opinions talk it through, react to each other, change their minds (sometimes), and either land on a shared choice or fail to agree.

## Goal

The motivation is that LLM-driven "user simulators" are increasingly used to test dialogue systems and group-decision tools — but only if the simulated conversations actually *behave* like real group discussions. This project is an attempt to get closer to the real thing while staying fully topic-agnostic.

A good run reads like a casual group chat: people state a leaning, respond to each other, occasionally get persuaded, and the group either converges for understandable reasons or — when someone is a genuine holdout — ends without agreement. The text should stay grounded in the scenario's facts, avoid repeating itself, and sound like friends talking, not a formal debate.

## What I've been doing

Iteratively improving discussion quality through multiple passes:

- **Structural fixes**: commitment gating (votes only count during decision phases), hard-blocker integrity, echo guards, robotic template detection
- **Naturalness fixes**: anti-card-reading (don't parrot option descriptions), self-narration detection ("I should consider..."), collective voice rewrite ("we prioritize" → "I prioritize"), opener variety enforcement, conversational register in prompts
- **Evaluation infrastructure**: 110 unit tests, automated eval runner, known failures tracking, scenario spread for batch evaluation

Current state and remaining issues are tracked in `docs/known_failures.md`. Architecture and how to run/test are in `CLAUDE.md`.
