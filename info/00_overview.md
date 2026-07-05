# 00 — Overview

This project is an option-grounded multi-user decision simulator. It generates group discussions between 2-7 simulated users over a fixed option board and logs the interaction as a structured trace.

The purpose is not to create arbitrary chat. The purpose is to define a controllable simulation environment where participant parameters influence observable dialogue behavior.

## Core flow

```text
topic or manual environment
  -> scenario + option board
  -> simulated participants
  -> discussion controller
  -> LLM utterance rendering
  -> visible-state observer
  -> narrowing / voting / outcome
  -> transcript + JSON + metrics
```

## Current components

- Environment generation can be automatic or manual.
- Participant generation can be automatic or manual.
- Sims have traits and operational parameters.
- The controller selects speakers, addressees, dialogue acts, and option focus.
- The LLM writes one visible utterance at a time.
- The observer updates state only from visible text.
- The moderator voice is configurable and can be reduced or disabled.
- Participant-owned procedural moves exist for low-/no-moderator modes.
- Final outcomes come from visible votes/acceptances only.
- Logs include transcripts, `run.json`, metrics, token usage, provider/model, mode settings, and pacing metadata.

## Current strengths

The architecture is mostly in the right shape. The system is already closer to the relevant literature than to a generic chatbot: it has an environment, option board, participant simulators, routing policy, visible-state observer, validation, and evaluation.

Current logs show that sims compare fixed options, react to others, raise tradeoffs, vote visibly, and can produce `successful`, `majority`, or `unresolved` outcomes.

## Current state (after the 2026-07-06 behavioral round)

The former quality weaknesses were addressed and verified with the full 12-case suite:

- turns average ~13-16 words with genuine short beats for every sim, while verbosity still orders the averages;
- answers develop the same thread; question density and tail questions dropped sharply;
- direct addressing scales with group size (no name prefixes in n=2 runs);
- dominance is trait-shaped and judged on free discussion turns (top share ~0.26-0.53 across the suite);
- stance switches need net visible support or trait-level flexibility; a sim's own open concerns resist switching;
- explicit constraints hold regardless of agreeableness, and no final tally contains a blocker violation;
- an issue ledger prevents repeated unknown-logistics loops (repeated_unknown_mentions 0 across the suite);
- grounding runs on a narrowed tripwire with option-scoped judging.

Remaining open work is the accumulated-code-path simplification plus small residuals. Use `docs/todo.md` as the authoritative current issue list.
