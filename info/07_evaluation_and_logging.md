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

## Diagnostic metrics added in the 2026-07-06 round

- `avg_words_by_act` and `short_turn_rate` (share of turns <= 10 words);
- `tail_question_rate`: questions tacked onto statement-type acts — the chaining signal (ask/invite/probe acts are exempt);
- `free_discussion_share`, `top_free_discussion_share`, `free_discussion_engagement_correlation`: dominance judged on free discussion turns only;
- `repeated_unknown_mentions` and `issue_ledger`: mentions of a practical unknown beyond its allowed raise+answer pair;
- `final_blocker_violations`: a hard blocker counted as supporting their rejected option in the final tally — must always be 0.

Long-standing metrics (`avg_words_by_persona`, `verbosity_behavior_correlation`, `question_answer_completion`, `switch_explanation_rate`/`switch_bridge_rate`, `name_prefix_rate`, `unsupported_fact_flags`/`unsupported_printed_turns`, per-call-type token counts) remain the regression baseline. Known artifact: `switch_explanation_rate` under-counts em-dash reason clauses (todo O4).

## Metrics interpretation

Balanced participation is not automatically good. A low inequality score can mean the controller flattened trait behavior. Dominance is acceptable when it follows engagement/initiative and does not become repetitive.

Opening and final vote rounds should usually be excluded from trait-realization analysis because they intentionally give everyone a visible stance.

## Suite CSV caution

If metric schema changes, do not mix old and new `metrics.csv` rows without a clear header/version. Historical append-only CSVs can contain stale columns or repeated headers.

## Current validation focus

After running `py run_eval_suite.py --full`, compare transcripts manually against metrics. Do not claim a fix based only on an improved number.
