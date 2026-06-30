# Option generation

## Purpose

Option generation turns a short user topic into a concrete decision environment. This is necessary because a one-line topic such as `Book a flight to Stockholm` or `Plan a team meeting in the summer` does not contain enough facts for grounded discussion.

The generated options are artificial, but once generated they become the hard facts of the simulated world.

## Input

```text
Topic: Book a flight to Stockholm
```

or:

```text
Topic: Plan a team meeting in the summer
```

The input topic should be broad enough to permit discussion, but narrow enough that reasonable options can be generated.

## Output

The option generator should produce:

```text
shared context
option A with attributes, positive trade-off, negative trade-off
option B with attributes, positive trade-off, negative trade-off
option C with attributes, positive trade-off, negative trade-off
option D with attributes, positive trade-off, negative trade-off
```

Example option fields:

```text
name
cost / time / effort / location / duration / capacity / risk / convenience
positive trade-off
negative trade-off
```

The exact attribute types depend on the topic. Flights may use cost, duration, departure time, layover. A team meeting may use date, format, duration, location, preparation effort, or availability assumptions.

## Source-of-truth rule

After generation, the option board and shared context are the source of truth. Sims may discuss, compare, and reason from these facts. They must not invent additional concrete facts.

Allowed:

```text
Option D is cheapest, but the red-eye sounds uncomfortable.
We do not know whether baggage is included, so I would not assume that.
```

Not allowed:

```text
The direct flight includes checked bags.
Customs will be faster at that time.
The meeting room already has catering.
```

unless those facts are explicitly present in the option board or context.

## Moderator framing

The moderator should make the generated world explicit:

```text
For this simulated decision, I’ll treat the following setup as the shared facts.
```

This avoids pretending that the generated details came from the user. They are generated assumptions, but within the run they are binding facts.

## Validation goal

The option generator should reject malformed options, but it should not be overly brittle. Cosmetic issues such as long option names should be cleaned, not cause the whole scenario to fail unless the option becomes ambiguous or unusable.
