# CLAUDE.md

Use this file as project context for Claude Code or similar coding agents.

## What this project does

It generates small group-decision transcripts. A topic is turned into a controlled scenario with four factual options. Then 2-7 personas discuss the options and either reach a unanimous decision, a visible majority, or remain unresolved.

The important design choice is separation of responsibilities: the controller decides who speaks, what kind of conversational move is needed, and when voting happens; the LLM writes exactly one natural message for that move.

## How to run

```powershell
py .\main.py
py .\main.py scenarios.txt
"Choose a coffee machine for the office" | py .\main.py
```

Change participant count and provider settings in `config.yaml`.

## Current source layout

- `main.py`: entry point.
- `config.yaml`: tunable parameters.
- `src/prompts.py`: all LLM prompts and moderator templates.
- `src/dialogue.py`: compact discussion controller and consensus logic.
- `src/builders.py`: setup generation and persona parsing.
- `src/models.py`: typed state.
- `src/parsing.py`: option matching and `[act=...; opt=...; stance=...]` trailer parsing.
- `src/llm_client.py`: provider abstraction.
- `src/logger.py`: run logging.

## Current direction

Keep the codebase small. Avoid reintroducing an over-complex rule stack. Improve quality by changing the compact controller, the prompt, or a narrow validator only when a transcript shows a repeated issue.

The discussion should contain natural multi-party behavior: agreement, challenge, questions, answers, comparisons, invitations to quieter participants, and plausible compromise. It should not become a rigid debate template.

## Non-negotiable rules

- Never count hidden preference as final support.
- Never close before participants have a visible decision opportunity.
- Never let a hard blocker accept their rejected option through state mutation.
- Never add facts outside option cards/shared context.
- Keep the moderator sparse and neutral.
- Put all LLM-facing prose in `src/prompts.py`.
