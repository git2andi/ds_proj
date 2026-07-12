# 07 — Evaluation and logging

Each run writes readable and structured artifacts.

## Outputs

- `transcript.md`: human-readable setup, transcript, outcome, and metrics.
- `run.json`: structured scenario, personas, turns, runtime stance ranks, threads, `repair_history`, the per-turn `controller_trace`, and outcome.
- `metrics.csv`: append-only summary rows.
- optional prompt dump if `output.write_prompts` is enabled.

## Controller trace

`run.json` contains an immutable per-turn trace: a pre-turn snapshot (phase, route source, selected speaker/act/addressee/focus, primary thread, coverage gaps, candidate, owed answers, no-progress count) and a post-turn result (realized act vs selected act, assessment action, intended-act/focus realization flags, accepted evidence kinds with option bindings, grounding-claim counts and exact unsupported spans with reasons, ambiguous references, validation issue codes, validation-repair attempts, fallback family, final option refs, formal vote realized, coverage realized, tokens). Phase transitions and repair objectives are separate trace entries with reasons. The trace explains why every turn was selected, what its accepted evidence says, and why every repair/fallback/drop happened. Route-level repair (`*_repair` route sources), semantic validation repair (`validation_repair_attempts`), and operational validator retries are never conflated.

Validation-path telemetry per final candidate (item 14): `validator_llm_used`, `validation_fast_path_reason` (why the validator LLM was skipped), `validator_categories` (the intent-specific payload requested when it ran), `validator_tokens_in/out`, and — for accepted turns — `state_changed` plus `state_changes`, the exact speaker-local/public fields the observer changed (`vote:p2: None -> 'B'`, `thread:t3: 'hot' -> 'cooling'`, `coverage:A: ... -> ...`, `lean:p1: 'A' -> 'B'`). The run header records `validation_mode` (selective | full).

## Persona logging

Logs include the hidden OCEAN traits, the five simulator parameters (engagement, verbosity, directness, stubbornness, switch_resistance), age/speech_style, profile/background, private goal, initial preference, and initial option ranks. This is important for checking whether behavior comes from the parameters while wording variation comes from speech_style.

The eval suite also records a compact `persona_age_style` summary so manual eval casts can be checked quickly.

## Parameter meanings in eval

```text
engagement        -> turn share / free-discussion turn share
verbosity         -> average words per participant turn
directness        -> optional/manual or heuristic wording signal
stubbornness      -> discussion-phase defense strength
switch_resistance -> fewer/later final switches, holdout persistence
speech_style      -> manual qualitative check only
```

Engagement realization is measured against the same expected-turn-share function the router uses (`simulator.expected_turn_share`). Verbosity realization is measured against the controller's own word-budget formula (`0.45 + 0.85 * verbosity` on the act's base budget, short beats folded in).

## Important metrics to inspect

- outcome status and final option;
- visible (formal) votes and vote clarity (`unclear_vote_repairs`);
- route-source distribution and selected-vs-realized act mismatch rate;
- thread counts by type/status, question response rate, concern response rate, unanswered question threads;
- coverage routes selected vs coverage turns realized;
- repairs run and their statuses (`repairs_run`, `repair_statuses`), reservation exchanges, deadlock use;
- stance-rank distribution (ranks 1–5) and runtime preferred option by rank;
- turn share and engagement correlation;
- average words by persona and act;
- switch explanation / bridge rate and `discussion_lean_shifts`;
- `intended_function_realized_rate` vs `act_mismatch_rate` — the latter is DIAGNOSTIC ONLY (a comparative question realizing a requested COMPARE counts as realized, never as a failure);
- `intended_focus_agreement_rate`, `ambiguous_reference_rate`, and `assessment_action_counts`;
- unsupported claim flags and `unsupported_printed_turns` (zero means the final accepted claims were verified against the fact table, not merely unflagged);
- `repair_success_rate`, `fallback_by_family`, `dropped_turn_count`, and `validator_failure_turns`;
- validation cost, with endpoint calls kept separate from logical checks: `validator_calls` (API hits incl. the ≤1 bounded retry), `validator_logical_checks`, `validator_api_retries`, `validator_calls_per_accepted_turn` and `validator_logical_checks_per_turn` (selectivity target < 0.80), `validation_fast_path_rate`, `validator_input_share` (target ≤ 0.80), and token usage by call type;
- consistency: `vote_state_consistency_failures` (public evidence vs observer state — must be 0) and `discussion_lean_shift_turns` (the source turn of every lean shift);
- persona age/speech_style/profile plausibility.

The full eval suite (`py .\eval\run_eval_suite.py`) holds exactly **10**
representative cases (`c01`–`c10`) spanning participant counts 2/3/4/5/7 and the
manual/auto and moderator combinations. Each run's `case_id` is written into
`run.json`, the log directory name, and the summary CSV; a fresh run clears the
prior summary and all prior run directories together, so it is restart-safe and
never leaves orphans. Beyond `returncode`, each case is scored by `case_flags`
(invalid/unsupported printed turns, blocker/vote-state violations, repair > 0.25,
drop > 0.02, per-case expectations, and a controller-language leak detector).
The suite runs against live LLM endpoints and is an explicitly approved costly
operation — verify the focused samples and the validator cost gate first.

Metrics are useful but not sufficient. Manual transcript review is required, especially after routing, thread, phase, or speech-style changes. Deterministic controller tests live in `tests/` (`py -m unittest discover -s tests`) and must pass before any LLM evaluation.
