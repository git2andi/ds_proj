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

The latest implementation improved major behavioral issues:

- split-vote reservation exchanges happen;
- mid-discussion stance movement is more visible;
- peer procedure appears in no-/light-moderator runs;
- transcripts include better metadata;
- unresolved outcomes are less frequent than before.

## Current weaknesses

The latest full evaluation still showed open problems:

- split-vote candidate selection can be socially implausible;
- post-reservation follow-up is still too shallow;
- `n=2` deadlock handling is not validated;
- compromise turns can confuse which option is being tested;
- token cost remains very high;
- grounding still leaks occasional unsupported logistical claims.

Use `docs/todo.md` as the authoritative current issue list.
