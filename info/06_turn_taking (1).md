# Turn-taking

## Purpose

Turn-taking decides who speaks next and why. It is one of the central parts of the simulator because multi-user realism depends on interaction order, not only on message content.

## Inputs to speaker selection

The router should consider:

```text
engagement
initiative
responsiveness
recent speaker history
turn count imbalance
pending direct questions
option coverage needs
phase of the discussion
agenda compatibility
```

## Balance vs dominance

Turn-taking should not be strict round-robin. Real groups are often imbalanced. A high-engagement sim may speak more than a low-engagement sim.

However, dominance should remain bounded unless the configuration explicitly allows stronger imbalance.

Useful checks:

```text
top speaker share
minimum turns per participant
unanswered direct questions
same-speaker repeats
```

## Consecutive turns

Consecutive turns by the same sim should be rare and interactionally justified.

Allowed:

```text
A sim answers a direct question and immediately adds a short clarification.
A sim repairs a misunderstood previous statement.
A sim gives a final vote after being directly prompted.
```

Not allowed:

```text
The same sim repeats the same preference twice.
The same sim continues without being addressed while another direct question is pending.
```

## Direct questions

A direct question creates a response obligation:

```text
Kenji -> Anton question
next relevant answer should be Anton
```

If another participant speaks first, the system should ensure the addressed participant still answers soon or the moderator explicitly abandons the question.

## Addressee selection

Not every utterance needs a name. Addressee names are useful when:

```text
answering a direct question
challenging a specific participant
clarifying a misunderstanding
explicitly inviting a quiet participant
```

Names should not be mechanically prefixed to every turn. Excessive `Name, ...` openings make the transcript sound templated.

## Style interaction with routing

Turn-taking and style should work together. A highly responsive sim may answer more direct questions. A highly initiative-driven sim may introduce comparisons or compromise options. A low-engagement sim should not vanish entirely, but should speak less and often more briefly.
