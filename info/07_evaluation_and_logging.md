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

## Important existing metrics

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

## Needed diagnostic metrics

The next round should add or inspect metrics for the actual quality problems:

- average words per participant and per act;
- verbosity-to-average-length correlation;
- free-discussion turn share vs trait-derived expected share;
- top speaker share excluding opening/final votes;
- question rate and answer adjacency rate;
- repeated unknown issue mentions;
- name-prefix rate and direct-address rate by group size;
- same-speaker continuation count plus repeat/novelty check;
- stance switches with/without visible trigger;
- final hard-blocker/constraint violations;
- repair/grounding token cost.

## Metrics interpretation

Balanced participation is not automatically good. A low inequality score can mean the controller flattened trait behavior. Dominance is acceptable when it follows engagement/initiative and does not become repetitive.

Opening and final vote rounds should usually be excluded from trait-realization analysis because they intentionally give everyone a visible stance.

## Suite CSV caution

If metric schema changes, do not mix old and new `metrics.csv` rows without a clear header/version. Historical append-only CSVs can contain stale columns or repeated headers.

## Current validation focus

After running `py run_eval_suite.py --full`, compare transcripts manually against metrics. Do not claim a fix based only on an improved number.
