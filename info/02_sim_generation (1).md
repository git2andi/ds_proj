# Sim generation

## Purpose

Sim generation creates the participants of the group discussion. A participant is not only a name and a persona sentence. It should be a small user simulator with private preferences, behavioral parameters, and pending communicative goals.

## Layers of a simulated user

A simulated user should have three layers:

```text
identity layer
  name, short background, social role if useful

personality/source layer
  OCEAN traits

operational simulator layer
  engagement, verbosity, initiative, responsiveness, stubbornness, directness, compromise_threshold
```

OCEAN traits are useful for generating plausible individual differences, but they are too abstract as the only control interface. The explicit simulator parameters are the variables that should actually influence routing and style.

## OCEAN to simulator parameters

The intended mapping is approximate and pragmatic:

```text
extraversion      -> engagement, initiative
agreeableness     -> responsiveness, compromise tendency, lower stubbornness
conscientiousness -> consistency, constraint focus, lower impulsive switching
openness          -> willingness to consider alternatives
neuroticism       -> risk sensitivity, uncertainty, caution
```

The exact mapping does not need to be psychologically perfect. It must be useful, stable, and observable in generated dialogue.

## Private state

Each sim should have private/internal state:

```text
initial preference
current lean
hard rejections
soft concerns
private goal
agenda items
simulator parameters
```

This state guides future behavior. It does not directly count as public agreement.

## Public state

Public state is what the transcript shows:

```text
visible option mentions
visible objections
visible questions
visible answers
visible commitments/votes
```

Outcome calculation must use public state, not private state.

## Agenda

Each sim should receive a small private agenda. Agenda items are pending communicative goals, not fixed text.

Examples:

```text
state initial preference
ask about a practical constraint
object to an option that violates a goal
answer a challenge
compare preferred option with compromise option
make final vote
```

Agenda items help sims remain consistent across turns. They also make the system closer to user-simulation work: each sim has goals and a simple policy, not only a persona description.

## Voice differentiation

Sims should sound different because their parameters affect utterance style.

Examples:

```text
high directness:
  "I would not choose D. The overnight travel is the problem."

low directness:
  "I see why D is attractive, but I am a bit worried about the overnight part."

high stubbornness:
  "I still think the cheap option matters most here."

high responsiveness:
  "Anton, on your baggage point, I think we should not assume anything beyond the listed facts."
```

The goal is not theatrical role-play. The goal is measurable variation in participation, wording, consistency, and compromise behavior.
