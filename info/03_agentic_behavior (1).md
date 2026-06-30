# Agentic behavior

## What changed conceptually

The system should not ask the LLM to generate a complete conversation. Instead, each turn is produced by a small agentic loop:

```text
read current environment state
select next speaker
select addressee if needed
select dialogue act
select option focus
ask LLM to realize only that turn
parse visible result
update state
repeat
```

This makes the LLM a language realizer, not the full conversation controller.

## Why this matters

A full-transcript prompt tends to produce fluent but artificial conversations. Participants often agree too easily, repeat templates, ignore direct questions, or reach decisions without visible support.

The agentic structure gives the system control over:

```text
who speaks
who must answer
which option still needs processing
when final voting is allowed
which public commitments count
when unresolved is justified
```

## Difference between topics

The conversation engine is the same for different topics. What changes is the generated environment.

For:

```text
Book a flight to Stockholm
```

The option board may contain flight options with cost, duration, departure time, layovers, and baggage constraints.

For:

```text
Plan a team meeting in the summer
```

The option board may contain meeting formats, dates, times, locations, duration, preparation effort, or attendance constraints.

The agentic loop does not fundamentally change. It still creates sims, assigns goals/preferences, routes turns, observes commitments, and computes an outcome.

## What should not change by topic

These mechanics should be topic-independent:

```text
option facts become source of truth
sims do not invent unsupported facts
direct questions create response obligations
moderator targets the right participant
votes require visible commitments
conditional support is not a final vote
unresolved must be procedurally justified
```

## What may change by topic

These elements can vary by topic:

```text
attribute types
option trade-offs
sim goals
practical constraints
typical objections
style of compromise
```

Example:

```text
Flight topic:
  compromise may balance cost, duration, and comfort.

Meeting topic:
  compromise may balance availability, preparation effort, and usefulness.
```

## Agentic boundary

The system is agentic in the sense that it controls state, routing, agenda, and repeated action selection. It is not autonomous in the sense of doing open-ended external research or inventing new world facts. The generated option board defines the world.
