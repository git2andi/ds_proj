# Evaluation and logging

## Purpose

Logging records what happened. Evaluation measures whether the simulator behaved correctly. These should remain separate.

The current priority is not a complex final evaluation framework. The priority is to expose the failures that are visible in transcripts so they can be fixed systematically.

## Logs

Each run should produce:

```text
readable transcript
structured JSON trace
metrics summary
optional prompt/debug information
```

Logs should include enough information to inspect:

```text
participants and parameters
option board
turn sequence
moderator turns
visible votes
outcome
metrics
phase history
```

Old logs should be moved to `logs/archive/` before implementation passes so behavior changes can be compared cleanly.

## Basic metrics

Current stable metrics should include:

```text
participant turn counts
top speaker share
moderator ratio
average words per speaker
visible vote count
outcome status
option coverage
token usage
repair rate
```

## Failure-oriented metrics

The next useful metrics should directly reflect current transcript failures:

```text
unanswered direct-question count
moderator-target-miss count
name-prefix rate
repeated-opening-pattern count
vote-overwrite count
unsupported-fact warning count
split-vote closure count
```

These are more useful right now than abstract quality scores because they identify concrete bugs in the simulator loop.

## Implementation status (2026-07-03)

The stable metrics above are implemented, plus integrity counters:
`fallback_turns` (deterministic replacements of invalid turns) and
`invalid_printed_turn_count` (must stay 0). Per-sim `switch_events`
(from→to, has_reason) land in run.json. The deeper evaluation metrics below
are deliberately deferred until the discussion behavior is considered final
(user decision, 2026-07-03).

## Later evaluation ideas

Later, after behavior stabilizes, evaluation can include:

```text
participation Gini
direct response rate
question-answer completion rate
engagement realization error
repetition score
preference-shift visibility
option reason diversity
human rating of naturalness
```

## Validation principle

A run is not successful just because the program finishes. The transcript and metrics must be inspected. For each implementation pass, at least one `n=3` run and one additional different group-size run should be checked.
