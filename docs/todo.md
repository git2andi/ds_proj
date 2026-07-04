# TODO: Option-Grounded Multi-User Dialogue Simulator

This file is the active implementation guide for the next development round. It is not a changelog. Completed fixes, historical validation notes, and old issue descriptions should not be kept here.

## 0. Project framing

Treat the project as an **option-grounded multi-user decision simulator**, not as a generic transcript generator and not as a fully open-ended group-chat bot.

A run should model this pipeline:

```text
one-line topic or manual environment
  -> option-grounded decision environment
  -> 2-7 configurable simulated users
  -> controller selects speaker / addressee / dialogue act / option focus
  -> LLM writes one visible utterance
  -> observer updates public dialogue state from visible text
  -> discussion narrows through reactions, stance movement, reservations, and votes
  -> outcome = successful / majority / unresolved from visible votes only
```

The option board is a deliberate design choice. It gives sims concrete facts to reason about and reduces hallucination. The system should not claim to simulate arbitrary open-domain group chat unless the environment model is redesigned for that. The correct scope is:

> configurable simulated users in small-group, option-grounded decision discussions.

## 1. Implementation protocol

1. Work top to bottom through the open issues below.
2. Change one behavioral issue at a time.
3. Do not make the agenda more checklist-like. The goal is persistent stance/concern state, not forced agenda execution.
4. Keep options as the factual source of truth. Sims may express uncertainty, but must not invent concrete facts.
5. Validate each change with fresh generated chats, not only program execution. Inspect both transcript and `run.json`.
6. Test both auto and manual modes during implementation. Do not only run auto/auto demos. Across the implementation round, explicitly check combinations such as:
```text
auto environment + auto participants
manual environment + auto participants
auto environment + manual participants
manual environment + manual participants
```
7. Use varied group sizes, especially `n=3` and about three other runs for n=2-7.
8. For each implemented issue, update CLAUDE.md, this todo.md file and the affected `info/*.md` workflow files immediately afterwards so they describe the actual implemented behavior. Do not keep a separate documentation issue for this.

## Context
Keep aggregate evaluation secondary for now. First fix the visible discussion behavior.
For tests always use the "gpt" endpoint

## 2. Open issues

### Issue 8 (P2). Keep the option-grounded scope explicit and rename descriptions accordingly

**Problem.** Some descriptions can still sound like the system is a general multi-user chat simulator or a full agenda-based user simulator. That overstates the current project and creates confusion about whether open-ended group chat is expected.

**Where visible.**

- The project discussion around ConvLab3 and MUCA clarified that the real goal is simulator framework design, not simply nice transcripts.
- Current system design is centered on fixed options, visible votes, and decision outcomes.

**Correct behavior.**

README, `info/`, and future write-ups should consistently call the system an **option-grounded multi-user decision simulator** or **multi-user dialogue simulation framework for option-grounded group decisions**.

The wording should avoid claiming:

```text
general open-ended chat simulation
full agenda-based user simulation
human-realistic society simulation
```

unless those features are actually implemented.

### Issue 9 (P2). Transcript and run outputs should state the active LLM provider

**Problem.** Transcripts currently focus on the generated conversation and run metadata, but the human-readable transcript should also make clear which provider was used, for example `gpt`, `uni`, `groq`, or `gemini`. This matters because provider differences can change style, grounding, verbosity, and failure patterns.

**Where visible.**

- During manual review of generated transcripts, the provider is not immediately visible in the transcript header.
- The project uses multiple providers, so comparing logs without provider metadata is error-prone.

**Correct behavior.**

Each transcript should include the active provider near the top, together with existing run metadata. Example:

```text
Provider: gpt
Model: gpt-...
```

If the model identifier is already available, include it as well. The structured `run.json` should also keep provider/model metadata so failures can be compared across providers.

## 3. Explicit non-goals for this round

Do not prioritize these until the discussion behavior above is improved:

- aggregate evaluation experiments,
- large prompt rewrites,
- new research-paper integrations,
- more personality traits,
- broad open-domain chat support,
- cosmetic transcript polishing without state/control changes.

The critical requirement is simple: changing simulator parameters and conversational state must visibly change the group interaction.
