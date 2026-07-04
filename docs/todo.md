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

### Issue 4 (P1). Add bounded compromise and reservation negotiation around voting

**Problem.** Minority checks currently identify holdouts, which is good, but the follow-up is often too thin. A holdout usually either accepts the majority option immediately or repeats its own vote. There is not enough small-scale negotiation around what would make the majority option acceptable.

**Where visible.**

- `logs/20260704_212701_828451` (`Plan an office party...`): Jasper is asked about the majority option and accepts it in one turn. This is valid but shallow.
- `logs/20260704_211917_836818` (`Which Scifi tech should exist today`): split-vote compromise fails with repeated final votes rather than a meaningful reservation exchange.
- `logs/20260704_213235_315077` (`Plan a summer vacation with your families`): a three-way split remains unresolved without much visible negotiation of conditions.

**Correct behavior.**

When a majority exists but a minority sim has a meaningful reservation, the system should allow a bounded micro-negotiation:

```text
1. Majority or split is detected.
2. Holdout is asked what blocks agreement.
3. Holdout states a concrete reservation or condition.
4. One majority supporter responds to that reservation.
5. Optionally the holdout updates stance.
6. The system closes as successful, majority, or unresolved.
```

This should not run indefinitely. Usually one response and one optional update are enough. Compromise should also happen before voting sometimes, not only after final votes.

Example target behavior:

```text
Moderator or participant: Most are on B. What still makes you hesitate?
Holdout: I can live with B, but it does not really cover child-friendliness.
Supporter: Fair. The board does not prove that, but B's flexible timing at least helps families leave earlier.
Holdout: That is enough for me to accept B, with that caveat.
```

### Issue 5 (P1). Add participant-owned procedural moves, especially in low- or no-moderator runs

**Problem.** The moderator can now be disabled or reduced, but the hidden controller still carries most of the structure. Participants do not yet reliably perform enough group-management acts themselves.

**Where visible.**

- No-moderator and low-moderator modes can run, but participant-level narrowing is still weaker than the moderator-guided path.
- Current transcripts still often rely on explicit vote calls, holdout probes, or closure structure from the system/moderator.

**Correct behavior.**

High-initiative or high-engagement sims should sometimes perform procedural moves such as:

```text
suggest narrowing
summarize the current split
ask a quiet participant for their view
ask a holdout what blocks agreement
propose dropping a weak option
call for a final pick
check whether a compromise is acceptable
```

This should be more likely when the moderator is disabled, when the discussion is stalled, or when the group has already compared enough options.

### Issue 6 (P1). Allow rare intentional same-speaker continuations, but prevent duplicate consecutive turns

**Problem.** A hard ban on consecutive turns by the same sim would be too rigid. Real chats sometimes contain add-ons, repairs, afterthoughts, or self-corrections by the same person. However, accidental duplicate turns are still bad and make the routing look broken.

**Where visible.**

This issue comes from turn-taking design review rather than one specific log. It clarifies the earlier idea that sims should not simply repeat the same move after themselves.

Bad behavior to prevent:

```text
Anna: Tim, what do you think about Movie X?
Anna: How is Movie X for you, Tim?
```

Natural behavior to allow sometimes:

```text
Anna: Tim, what do you think about Movie X?
Anna: Oh, and do you guys think we should grab food before we go?
```

```text
Anna: Can someone help me figure out question 3?
Anna: Oh never mind, I just got it.
```

**Correct behavior.**

Consecutive same-speaker turns should be allowed only when the second turn is explicitly a continuation-type move:

```text
addendum
afterthought
self-repair
self-correction
clarification
self-resolution
short topic extension
```

Rules:

```text
no same-speaker chain longer than 3
additional turn must be short
additional turn must add, correct, clarify, or resolve something
additional turn must not repeat the same dialogue act, same question, same addressee, and same option framing
```

So the router should strongly penalize the previous speaker for normal turns, but allow a controlled continuation act when it is locally justified.

### Issue 7 (P1). Strengthen grounding: no invented concrete facts

**Problem.** The option board is supposed to be the factual source of truth, but recent logs still contain or flag unsupported concrete facts. The validator catches some of this, but unsupported content still appears often enough that grounding remains a behavioral issue.

**Where visible.**

- `logs/20260704_211704_067828` (`Book a flight to Stockholm`): `unsupported_fact_flags = 2`.
- `logs/20260704_211917_836818` (`Which Scifi tech should exist today`): `unsupported_fact_flags = 3`; the discussion adds claims such as privacy/safety/car-emission implications beyond the option facts.
- `logs/20260704_212701_828451` (`Plan an office party...`): `unsupported_fact_flags = 3`.
- `logs/20260704_213827_552087` (`Which movie to watch tonight`): `unsupported_fact_flags = 4`.
- `logs/20260704_214009_899789` (`Pick a restaurant for a date`): the transcript mentions parking problems for the sushi bar although the option card does not provide that fact.

**Correct behavior.**

Sims may reason from listed facts and may express uncertainty. They must not add new concrete facts as if known.

Allowed:

```text
"We do not know the parking situation, so I would not count that as a reason."
"B is lively, but the board says it is noisy, so conversation could be harder."
```

Not allowed:

```text
"Parking near B is bad."
"This venue has childcare."
"The flight includes baggage."
```

Grounding fixes should prefer environment-state checks, prompt constraints, and repair/fallback behavior over broad generic warnings.

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
