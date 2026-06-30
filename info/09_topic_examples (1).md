# Topic examples and behavior

## Same simulator, different environment

The user enters a short topic. The system then creates a concrete option-grounded environment for that topic. The simulator loop stays the same across topics.

## Example: Book a flight to Stockholm

Input:

```text
Book a flight to Stockholm
```

Possible generated environment:

```text
A direct morning flight
B cheaper evening layover
C balanced midday economy flight
D cheapest red-eye low-cost flight
```

Likely discussion dimensions:

```text
cost
duration
layover risk
comfort
baggage constraints if listed
arrival/departure time if listed
```

The sims should not invent real airline facts. They can only use the generated option/context facts.

## Example: Plan a team meeting in the summer

Input:

```text
Plan a team meeting in the summer
```

Possible generated environment:

```text
A in-person half-day workshop
B short online sync
C hybrid afternoon session
D outdoor offsite meeting
```

Likely discussion dimensions:

```text
availability
preparation effort
travel burden
team engagement
weather risk if listed
cost if listed
```

The same simulator mechanics apply: sims get goals and preferences, discuss trade-offs, answer direct questions, narrow options, and either reach a visible outcome or close unresolved.

## What should generalize

Fixes should never depend on Stockholm, flights, meetings, or a specific option name. They should operate on abstract simulator concepts:

```text
option reference
commitment
question obligation
speaker target
coverage
unsupported fact
vote state
phase
```

## Good topic inputs

Good topics usually describe a decision task:

```text
Choose a movie for tonight.
Plan a birthday party.
Pick a book for next week's club reading.
Decide where to hold the team lunch.
Choose a coding project for beginners.
```

## Less suitable inputs

The current simulator is less suitable for pure open-ended debate or factual research questions:

```text
What is the meaning of life?
Explain quantum mechanics.
Who will win the next election?
```

Those topics do not naturally fit the option-grounded group-decision environment unless first converted into concrete options.
