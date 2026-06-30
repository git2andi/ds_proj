# System overview

These notes describe the intended final architecture of the project. They are meant as a compact mental model for development, not as a formal thesis chapter.

## Project scope

The project is an option-grounded multi-user dialogue simulator. Given a short topic, the system creates a small decision environment, generates several simulated users, lets them discuss the available options, and produces a transcript plus structured logs.

The project does not try to simulate arbitrary open-ended group chat. The chosen scope is small-group decision-making under a generated option board. This restriction is deliberate: it gives the simulated world a source of truth, reduces uncontrolled hallucination, and makes outcomes observable.

## Core idea

The simulator should not be understood as one prompt that asks an LLM to write a whole conversation. It should be understood as an environment/controller loop:

```text
one-line topic
  -> option-grounded scenario
  -> simulated users with private goals and parameters
  -> routing policy chooses who speaks, when, to whom, and why
  -> LLM realizes only the next visible utterance
  -> observer parses public transcript evidence
  -> state, agenda, coverage, and outcome are updated
```

The transcript is one output artifact. The real project object is the simulator framework behind that transcript.

## Major components

```text
Option generation
  Creates the shared fact base of the simulated world.

Sim generation
  Creates user simulators with OCEAN traits, explicit behavioral parameters, private goals, preferences, and agenda items.

Moderator / environment controller
  Handles state-aware facilitation, phase transitions, targeted questions, and bounded closure.

Turn-taking / routing
  Selects the next speaker, addressee, dialogue act, and option focus.

LLM realization
  Produces the next natural-language message only. It should not decide the whole conversation.

Observer / parser
  Extracts visible commitments, questions, option references, and public votes from transcript text.

Consensus / outcome logic
  Determines successful, majority, or unresolved from visible transcript evidence only.

Logging / evaluation
  Records transcript, structured state, metrics, and later evaluation signals.
```

## Relationship to MUCA and ConvLab-style ideas

MUCA is useful because multi-user interaction requires explicit control over what should be said, when a participant or assistant should speak, and who should be addressed. This project adapts that idea to simulated users: the controller must decide speaker, addressee, act, and timing.

ConvLab-style user simulation is useful because it separates goals, dialogue acts, policy, state, and evaluation. This project does not need a full ConvLab implementation, but it should keep the same spirit: simulated users should have internal goals and controllable behavior, not only decorative persona text.

## Design principles

1. Option facts are fictional when generated, but hard facts after generation.
2. Sims must not invent new concrete facts beyond the option board/context.
3. Internal simulator state may guide behavior, but only visible transcript text can decide the public outcome.
4. Prompt instructions should stay compact. Prefer controller/parser/state fixes over adding long prompt blocks.
5. Fixes must generalize across topics, group sizes, and option domains.
