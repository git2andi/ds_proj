# Simulated users and direct traits

A persona contains:

```text
id, name
background, private_goal
age, speech_style
engagement, verbosity, directness, stubbornness
preferred option and option-specific stances
optional hard-blocker state and rejection reason
```

The project no longer uses OCEAN or a separate switch-resistance trait. Generated and manual personas use direct integer traits:

- `engagement: 1..5` controls only voluntary bid probability, urgency, and willingness to join a relevant issue;
- `verbosity: 1..5` controls only action-scaled soft realization targets (normal discussion is approximately 4–11, 7–16, 11–24, 16–32, and 22–44 words); acknowledgments, votes, and simple answers may be shorter, while comparisons and concern explanations may be longer;
- `directness: 1..5` controls only realization wording;
- `stubbornness: 1..4` controls defence, acceptance, and switching probability for normal simulators.

Age and `speech_style` provide small lexical guidance only. They do not alter participation, length, preferences, or stance-transition probabilities.

Option stances use ranks 1–5:

```text
5 preferred
4 acceptable
3 neutral
2 disliked
1 hard rejected
```

The runtime also keeps synchronized sets for acceptable, disliked, and hard-rejected options. A stance change occurs only through an accepted structured action that visibly communicates it.

## Hard blockers

At automatic setup time, the group-level `hard_blocker_probability` samples either zero or one hard blocker. Manual profiles may explicitly define one, but configuration rejects several.

A hard blocker has stubbornness 5, exactly one preferred option, rejects every alternative with a reason, never changes stance, and votes only for its preferred option. Hard blocking is never inferred from ordinary stubbornness.
