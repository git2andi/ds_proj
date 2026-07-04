# Topic examples and generalization

The same simulator engine runs for any topic. What changes per topic is the generated
**environment** (`01`); the controller loop, routing, validation, and consensus logic
are topic-independent. This note shows a couple of examples and states what must
generalize.

## Same engine, different environment

You enter a short topic. The system builds an option-grounded world for it, then runs
the identical loop: create sims, assign preferences, route turns, observe commitments,
compute an outcome.

### Example: "Book a flight to Stockholm"

```text
Possible generated options:
  A direct morning flight
  B cheaper evening layover
  C balanced midday economy flight
  D cheapest red-eye low-cost flight

Likely discussion dimensions: cost, duration, layover risk, comfort,
baggage/arrival time (only if the cards list them).
```

### Example: "Pick a project management tool for the team"

```text
Possible generated options:
  A Asana Premium         B Trello Standard
  C Monday.com Basic      D Jira Software

Likely discussion dimensions: price, integrations, onboarding effort,
performance, ecosystem/plugins (only as the cards state).
```

In both, sims may compare and reason from the card facts and voice uncertainty, but
must not invent real product/airline facts (the grounding rule, `05`/`06`).

## What must generalize (never topic-specific)

Fixes and mechanics operate on abstract simulator concepts, never on "Stockholm",
"flights", or a specific option name:

```text
option reference        commitment / vote          question obligation
speaker + target        option coverage            unsupported-fact detection
latent lean vs vote     bridged switch             phase / outcome
hard blocker            visible-evidence narrowing
```

If a fix only works for one domain, it is wrong.

## Good topic inputs

Topics that describe a small decision with real trade-offs work best:

```text
Choose a movie for tonight.          Plan a birthday party.
Pick a book for next week's club.    Decide where to hold the team lunch.
Choose a database for analytics.     Pick a fitness activity for the office.
```

## Less suitable inputs

Pure open-ended debate or factual-research questions do not fit the option-grounded
group-decision frame unless first turned into concrete options:

```text
What is the meaning of life?     Explain quantum mechanics.     Who will win the election?
```

For a fully controlled experiment (fixed world + fixed cast), skip topic generation
entirely and author both sides via `environment: manual` and `participants: manual`
(`08`).
