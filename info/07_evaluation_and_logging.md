# 07 — Evaluation and logging

Each run writes readable and structured artifacts.

## Outputs

- `transcript.md`: human-readable setup, transcript, outcome, and metrics.
- `run.json`: structured scenario, personas, turns, runtime stance ranks, threads, `repair_history`, the per-turn `controller_trace`, and outcome.
- `metrics.csv`: append-only summary rows.
- optional prompt dump if `output.write_prompts` is enabled.

## Controller trace

`run.json` contains an immutable per-turn trace: a pre-turn snapshot (phase, route source, selected speaker/act/addressee/focus, primary thread, coverage gaps, candidate, owed answers, no-progress count) and a post-turn result (realized act vs selected act, validation issue codes, validation-repair attempts, fallback use, final option refs, formal vote realized, coverage realized, tokens). Phase transitions and repair objectives are separate trace entries with reasons. The trace explains why every participant turn was selected and what the final text actually realized. Route-level repair (`*_repair` route sources) and validation repair (`validation_repair_attempts`) are never conflated.

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
- unsupported fact flags and printed unsupported turns;
- repair/fallback counts;
- token usage by call type;
- persona age/speech_style/profile plausibility.

Metrics are useful but not sufficient. Manual transcript review is required, especially after routing, thread, phase, or speech-style changes. Deterministic controller tests live in `tests/` (`py -m unittest discover -s tests`) and must pass before any LLM evaluation.
