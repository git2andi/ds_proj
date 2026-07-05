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

## Current weaknesses

The current problem is not missing architecture. It is quality and control:

- utterances are too long and summary-like;
- questions open too many new topics before old points develop;
- names/direct addresses are overused;
- speaking distribution is still partly pulled toward balance instead of trait-shaped dominance;
- stance switches are sometimes too cheap;
- strict constraints or blockers can be overridden too easily;
- unresolved logistics such as parking/reservations can repeat;
- repair/grounding can become expensive and visible in the transcript;
- code paths have accumulated many local fixes and need cautious simplification.

Use `docs/todo.md` as the authoritative current issue list.
