# 00 — Overview

The project is an option-grounded multi-user decision simulator. It generates group discussions between 2-7 simulated users over a fixed option board and logs both a readable transcript and a structured trace.

The important target is not arbitrary natural chat. The target is a configurable simulator: participant parameters such as engagement, initiative, responsiveness, verbosity, stubbornness, directness, and compromise tendency should visibly affect who speaks, how they react, how strongly they resist, and whether they can move toward consensus.

## Core flow

```text
topic or manual environment
  -> scenario + option board
  -> simulated participants
  -> controller routes speaker / act / target / focus
  -> LLM renders one visible utterance
  -> observer parses public state
  -> controller narrows / votes / repairs split
  -> consensus manager returns successful / majority / unresolved
```

## v3 design decision

v3 is a combination version. It uses v1 as the explainable base and ports only selected v2 features that directly improve outcomes or simulator validity.

Kept from v2:

- deterministic switch/stay decisions for holdouts after reservations;
- no downhill compromise;
- flexible tie compromise only when traits and resistance make it plausible;
- unresolved acknowledgement before closure;
- split-summary self-answer avoidance;
- active local threads outrank private agenda items;
- small trait influence on routing and vote phrase choice;
- observer fixes that prevent false blockers on a sim's own current favorite.

Not kept from v2:

- separate micro-reaction subsystem;
- friendliness parameter;
- personal anchors;
- broad trait-color wording subsystem;
- extra dynamic-pacing complexity.

## Current state

The core requirements are present: auto/manual environment, auto/manual participants, 2-7 sims, option-grounded dialogue, visible final outcomes, moderator/no-moderator modes, structured logs, and an eval suite. The next work should be simplification and validation, not feature expansion.
