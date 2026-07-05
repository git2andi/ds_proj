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

## Latest known baseline

The uploaded latest full evaluation before this code round showed:

```text
unresolved outcomes:        7/12 -> 4/12
mid-discussion lean shifts: 3 total -> 16 total
participant procedure:      visible
split reservation exchanges: present
input tokens:               roughly 460k across 12 runs
utterance calls:            about 65% of input tokens
grounding calls:            about 27% of input tokens
```

## Current suite coverage

The full suite now includes a forced two-person stubborn deadlock case:

```text
f01_manual_manual_n2_stubborn_deadlock
```

This replaces the previous auto/auto n=2 edge case because that run converged before exercising the deadlock protocol.

## Current validation focus

After running `py run_eval_suite.py --full`, compare the new logs against the uploaded baseline. In particular:

```text
candidate choice in split sections;
post-reservation switch/stay/alternative lines;
two_person_deadlock_attempted in f01;
tokens_utterance_in and tokens_grounding_in;
unsupported_printed_turns and repair/fallback rates.
```
