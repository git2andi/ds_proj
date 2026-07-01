# TODO: Option-Grounded Multi-User Simulator

This file lists only open issues. Completed items are intentionally not tracked here. Keep this file current after every implementation pass.

The current priority is to fix prominent failures visible in real generated transcripts before adding new features. Fixes must be general across arbitrary option-grounded topics. Do not solve a problem by adding large prompt blocks unless a smaller controller, parser, validator, state, or repair-policy change is not sufficient.

## Implementation protocol for each update

1. **Archive old logs first.** Before changing behavior, create `logs/archive/` if it does not exist and move all existing log files/directories from `logs/` into `logs/archive/`. Do not delete logs.
2. **Work on one issue at a time.** Pick exactly one open issue from this file unless the issue explicitly says that several small changes belong together.
3. **Apply the minimal fix.** Change the smallest amount of code/config needed to solve the selected issue. Prefer controller, parser, validator, state, or repair-policy fixes over simply making prompts longer.
4. **Validate with example runs.** After the fix, run at least:
   - one mandatory `n=3` run with a random topic,
   - at least one additional run with a different group size in the `n=2..7` range,
   - more runs only if the changed behavior is unstable or group-size-dependent.
5. **Inspect the transcript and metrics.** Do not rely only on successful execution. Check whether the transcript actually shows the intended behavior.
6. **Append newly observed issues.** If validation exposes a new problem, add it under `Newly observed issues` with log path/date, topic, group size, and the smallest description of the failure.
7. **End only after verification.** Finish the update only when the selected issue is implemented, the code compiles, and validation runs show the intended behavior or a clearly documented remaining limitation.

## Open issues, ordered by priority

### 7. Refine option coverage without forcing artificial discussion

Validation target is met (2026-07-01): the coverage nudge is bounded to one
attempt per option (`OptionCoverage.coverage_attempts`), so a missed/over-hard
requirement can no longer loop forever; the resolver matches distinctive
proper-noun tokens and ignores stopword aliases, so detection is reliable; the
coverage speaker uses a COMPARE move focused on the leading option. In recent
n=4 runs every option was processed before voting and no option was over-
discussed. `OptionCoverage` already records `reasons`/`objections`/`acceptances`
separately.

Deferred polish (only act on this if a transcript shows the problem again):
- Decide "coverage is enough" from meaningful processing type rather than a bare
  mention. The current mention-based gate already satisfies the target, and
  forcing deeper processing risks over-discussing unattractive options, so this
  is intentionally not implemented yet per the project's minimal-change rule.

### 12. Add optional corpus-inspired presets later

Current problem: corpus statistics such as Delidata-style turn length, group size, and speaker dominance are known but not yet represented as selectable presets.

Required behavior:
- Add optional presets later, not hard constraints.
- Example preset fields:
  - typical discussion length,
  - preferred group size,
  - expected top-speaker share,
  - dominance range,
  - participation imbalance tolerance.
- The simulator should still work without a corpus preset.

Implementation notes:
- Keep this lower priority until routing, votes, moderator targeting, and local surface naturalness work reliably.

Validation target:
- Presets should change runtime parameters measurably without requiring topic-specific hacks.

## Resolved this pass (2026-07-01)

- Issue #13 (naturalness cleanup): `src/style.py` now detects option-name openings
  (`leading_option`/`option_opener_terms`, incl. single brand words like "Trello")
  and repeated opening words (`repeated_opening_token`); the controller sets
  compact prompt flags (suppress name/option opening, vary opening) and word budgets
  are trait-driven (`_word_bounds`: verbosity/engagement give a real length spread,
  low floor for terse sims). `sim_utterance` length/tone guidance is now concrete
  and parameter-driven. Validated n=3/4/5: combined name+option opening rate ~0.18–0.26
  (target <1/3), repeated_opening_patterns ~1–2, and per-persona avg words clearly
  track verbosity (e.g. a v=0.26 sim at ~14 words vs chatty sims at ~21). New metrics:
  `option_opening_rate`, `name_or_option_opening_rate`.
- Issue #14 (dialogue.py bloat): outcome logic moved to `src/consensus.py`
  (`ConsensusManager`, `participant_turn_count`); text post-processing moved to
  `utils.py` (`clean_generated` + helpers); removed dead imports (`AgendaStatus`,
  `RunOutcome`, unused utils). dialogue.py 1286→1210 lines; no behavior change,
  tests green. Controller routing/moderator methods deliberately left in place to
  preserve cohesion and avoid regressions. No duplicated functions were found.
- Issue #1 (response obligations): `ResponseObligation` state + router consumption
  in both discussion and decision loops; direct questions detected from visible
  text regardless of routed act. Validated n=3/n=4/n=5: every named question is
  answered by the named participant within one turn.
- Issue #2/#3 (name-prefix + repetitive templates): `src/style.py` local tracker;
  deterministic name-prefix strip; routing bias away from concession/worry/
  trade-off streaks; stronger speaker-balance (deficit weighting + ping-pong
  penalty). Validated: name_prefix_rate ~0.18–0.33, top_speaker_share ~0.26–0.32.
- Issue #5 (vote stability): later vote rounds only re-prompt unclear/non-voters;
  `_set_vote` protects a clear vote from silent overwrite unless the text signals
  an explicit change. Deterministic tests added.
- Issue #10 (tests): `tests/` covers parsing, style, vote overwrite, and outcome
  logic (22 tests, no LLM). Run with `py -m pytest tests/ -q`.
- Critical parsing bug: stopwords such as "with" were used as standalone option
  aliases (option D = "PostgreSQL **with** TimescaleDB"), causing false multi-
  option matches that silently dropped clear votes and flipped a majority to
  unresolved. Aliases now exclude a stopword/generic list.
- Coverage loop (part of #7): a missed coverage detection forced the same option
  focus every turn until the hard cap, manufacturing long repetitive runs; the
  coverage nudge is now bounded to one attempt per option.
- Issue #4 (unsupported facts): GPT grounding endpoint (`prompts.grounding_check`,
  gated by `validation.grounding_check`) flags utterances that invent facts beyond
  the option board and drives one repair toward grounded text; opinions,
  derivations, and uncertainty are allowed. Validated functionally (catches
  "includes free checked bags / quiet airports", passes "$170 more saves 8 hours")
  and across n=2/n=3 runs with zero false positives. Metric:
  `unsupported_fact_flags`.
- Issue #6 (unresolved handling): when standard vote rounds leave no majority,
  `_maybe_split_vote_compromise` runs once — the moderator summarizes the split
  and names one candidate, and only participants who can move are invited to
  switch (with `allow_vote_change`) or restate. Validated n=4: a true 4-way split
  became a clean majority via one compromise pass ("majority after split-vote
  compromise"); the leader was not re-prompted and the debate did not restart.
- Issue #8 (agenda): richer parameter-tuned agenda (adds propose-compromise for
  cooperative sims, object-to-rival for stubborn sims); `AgendaStatus` gains
  BLOCKED/OBSOLETE; `refresh_agenda` retires items advocating an abandoned option
  and blocks compromise toward a rejected option. Agenda is logged in run.json and
  summarized by the `agenda_status` metric.
- Issue #9 (evaluation): added `unanswered_direct_questions`, `name_prefix_rate`,
  `repeated_opening_patterns`, `unsupported_fact_flags`, `agenda_status`, and a
  `planned_metrics` stub dict for future scoring. Evaluation stays separate from
  logging.
- Issue #10 (tests): `tests/` (22 tests, no LLM) covers parsing, style, vote
  overwrite, and outcome logic.
- Issue #11 (token bound): `limits.warn_total_input_tokens` (default 30000); main
  prints a warning when a run exceeds it. Typical n=3 runs are ~16k–21k input.
- Setup reliability: `setup_generation_attempts` raised 2→3 to absorb occasional
  malformed setup responses (too few attributes, preference mismatches).

## Newly observed issues

Add new validation findings here after each implementation pass. Include log path/date, topic, group size, and the smallest description of the failure.

- Setup flakiness: the persona LLM occasionally violates the required primary
  preference or gives a hard blocker two preferred options, forcing a setup
  retry. Mitigated by an aligned prompt line, but the retry loop still absorbs
  the occasional failure.
