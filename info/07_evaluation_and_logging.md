# Evaluation and logging

**Code:** `src/logger.py` (`DialogueLogger`), `src/evaluation.py`
(`metrics_for`, `flat_metrics_for`, `token_summary_for`).

Logging records **what happened**; evaluation measures **whether the simulators
behaved as configured**. They are kept separate. The point of the metrics is that the
project should not rely on "this transcript sounds natural" impressions — it should
produce structured per-run signals.

## What a run writes

Every run creates a directory under `logs/<run_id>/` plus a shared CSV:

```text
logs/<run_id>/transcript.md   human-readable: options, personas+parameters, turns,
                              outcome, and the full metrics block
logs/<run_id>/run.json        the structured trace (see below)
logs/<run_id>/prompts.jsonl   optional per-turn prompts (output.write_prompts)
logs/metrics.csv              one flat row per run (flat_metrics_for), appended
```

`run.json` is the analyzable trace. Top-level keys:

```text
run_id, topic
participants_mode, environment_mode, moderator_config   (which input modes were used)
scenario, personas                                       (the generated world + cast)
runtimes        per-sim visible state, incl. switch_events (from/to/has_reason/has_bridge)
turns           every turn: speaker, text, act, phase, validation_issues, repaired,
                used_fallback, tokens
outcome         status + final_option + reason
phase_history   the ordered phase trace (closure only on a resolved outcome — issue 6)
metrics, tokens
```

Old logs are moved to `logs/archive/` before a behavior change so runs can be
compared cleanly (never deleted).

## Metrics (all computed from existing state, no extra LLM calls)

`metrics_for` produces the per-run metrics; scalar ones are also flattened into
`logs/metrics.csv`. Grouped by what they tell you:

**Outcome & process**

```text
outcome_status, final_option, visible_vote_count, visible_votes
num_participants, participant_turns, hard_blocker_present, corpus_preset
min_discussion_turns / force_narrow_turns / hard_max_turns   (the derived pacing)
```

**Participation & dominance**

```text
participation_gini      inequality of turn share (0 = perfectly even)
top_speaker_share, turn_counts, avg_words_per_turn, avg_words_by_persona
moderator_turns, moderator_ratio
```

**Did configured behavior actually show up? (parameter realization)**

```text
engagement_realization_error / _by_persona / engagement_behavior_correlation
verbosity_realization_error   / _by_persona / verbosity_behavior_correlation
```

These compare each sim's configured engagement/verbosity against realized behavior
(measured against the controller's own `_word_bounds` target). Correlations need n≥3
and some variance. Known result: **verbosity is strongly realized** (words track the
parameter); **engagement is realized in who initiates, not in turn share**, because
the default router equalizes turn counts (`03`).

**Interaction quality**

```text
direct_response_rate           obligations answered / created
question_answer_completion      directed questions answered by the addressee in time
open_questions_at_end, unanswered_direct_questions
question_density, repetition_score (own-turn content-word overlap)
```

**Decision movement**

```text
switch_event_count
switch_explanation_rate   share of switches with any reason clause
switch_bridge_rate        share of switches that actually bridge old->new (issue 5)
compromise_success_rate   split-vote compromise resolved? (None if none ran)
final_support_fraction
```

**Surface style (templated-ness)**

```text
name_prefix_rate, option_opening_rate, i_opening_rate, we_opening_rate,
name_or_option_opening_rate, repeated_opening_patterns
```

**Integrity counters (must-watch)**

```text
repaired_turns / repair_rate / flagged_turns
fallback_turns              deterministic replacements of invalid turns
invalid_printed_turn_count  MUST stay 0 — an invalid line reaching the transcript
unsupported_fact_flags      grounding-judge hits
```

**Coverage, agenda, tokens**

```text
option_coverage (mentions / reasons / objections / acceptances per option)
agenda_status (pending / done / obsolete counts — expected to stay mostly pending, 02)
setup/dialogue/total tokens (in/out)
```

## How to read a run quickly

1. `outcome_status` + `visible_votes` — did it decide, and honestly?
2. `invalid_printed_turn_count` = 0 and low `fallback_turns` — was the text clean?
3. `switch_bridge_rate` = 1.0 — were all preference changes explained (issue 5)?
4. `phase_history` — closure only appears on a resolved outcome (issue 6)?
5. realization errors/correlations — did the configured parameters show up?

## Validation principle

A run is not successful just because the program finished. For each change, inspect at
least one `n=3` run and one other group size, reading both the transcript **and**
`run.json`/metrics. Successful execution ≠ successful dialogue quality.
