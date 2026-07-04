# 07 — Evaluation and logging

The project writes human-readable and structured logs for every run.

## Outputs

Each run writes:

- `transcript.md`: readable transcript with setup metadata and summary metrics;
- `run.json`: full structured trace;
- `metrics.csv`: append-only row of summary metrics;
- optional prompts file if `output.write_prompts` is true.

## Transcript metadata

Transcripts should show enough metadata to inspect a run without opening JSON:

- provider and model;
- environment mode;
- participant mode;
- moderator flags;
- random seed;
- pacing caps;
- outcome;
- token summary.

## Important metrics

Inspect these after changes:

- `outcome_status` / final option;
- `visible_votes`;
- `discussion_lean_shifts`;
- `split_reservation_exchanges`;
- `two_person_deadlock_attempted`;
- `participant_procedural_moves`;
- `peer_vote_call`;
- `engagement_behavior_correlation`;
- `unsupported_fact_flags`;
- `unsupported_printed_turns`;
- token usage by call type.

## Latest evaluation summary

The latest full evaluation after the split/narrowing implementation improved behavior:

```text
unresolved outcomes:        7/12 -> 4/12
mid-discussion lean shifts: 3 total -> 16 total
participant procedure:      now visible
split reservation exchanges: now present
```

But token cost stayed high:

```text
approx. 460k input tokens across 12 runs
utterance calls: ~65% of input tokens
grounding calls: ~27%
repair calls: ~5%
```

## Current open issue

Evaluation is now informative enough to diagnose cost. The next implementation should reduce token cost without removing behavioral controls. In particular, reduce repeated utterance prompt context and unnecessary grounding calls.
