# 09 — Topic examples

Good topics for this simulator are small-group option-grounded decisions. They should invite tradeoffs but not require external facts.

## Good examples

```text
Choose a restaurant for a group dinner with mixed dietary preferences.
Choose a weekend activity for friends with different energy levels.
Pick a birthday gift for someone who already owns many things.
Choose a coffee machine for a shared office kitchen.
Decide where to hold a student project celebration.
Pick a team-building activity for a small software team.
Choose whether roommates should buy a robot vacuum or a better dishwasher.
Pick a movie for Friday night when people want different genres.
Choose a day trip plan with different budgets and travel tolerance.
Decide what app feature to build first in a small prototype.
```

## Less suitable examples

Avoid topics that require current external facts, open-ended politics, medical/legal advice, or unrestricted brainstorming without options.

Bad fit:

```text
Discuss the future of democracy.
Solve climate change.
What should Germany do next year?
Debate whether AI is good or bad.
```

Those can be made suitable by turning them into bounded option decisions.

## Useful stress-test topics

For split-vote narrowing:

```text
Choose a weekend activity where each participant values a different tradeoff.
```

For grounding and repeated unknowns:

```text
Choose between activities with sparse option facts and no external logistics.
```

For trait behavior and dominance:

```text
Choose a shared office purchase with one dominant organizer and one quiet participant.
```

For n=2 deadlock:

```text
Choose between two household purchases where both roommates are stubborn and prefer opposite options.
```

For short turns and direct addressing:

```text
Choose a Friday dinner plan between two friends with different budgets.
```

For hard-blocker/manual constraint testing:

```text
Choose a restaurant where one manual participant has an explicit dietary constraint.
```

## Topic design notes

Do not use topics that force every participant to reject one option from the start unless the goal is a blocker test. In normal runs, participants should start with preferences that can move through discussion.
