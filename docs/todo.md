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

### 4. Prevent unsupported factual additions beyond the option board/context

Current problem: sims sometimes add plausible but unsupported facts. In the Stockholm run, a participant mentions quieter airports and customs, and another suggests the direct SAS flight includes checked bags, even though these facts are not part of the shared option/context board. The generated option facts are allowed to be artificial, but once generated they are the hard world facts of the simulation.

Required behavior:
- Sims may reason from provided facts, but they must not introduce new concrete logistical facts, included services, policies, locations, hidden fees, timing consequences, or operational assumptions unless those are present in the option board/context.
- Sims may express uncertainty as uncertainty: `we do not know whether checked bags are included` is allowed if the fact is absent.
- Option positives/negatives and attributes are the authoritative fact base.

Implementation notes:
- Add a compact `known_fact_terms` / `unsupported_fact_risk_terms` check for common concrete additions: included baggage, customs, visa, airport security, hotel, refund policy, seat availability, weather, exact arrival time, etc.
- Do not try to solve this with a huge prompt. Use a small validation warning/repair when unsupported concrete facts appear.
- Allow domain-generic reasoning only if it follows from listed attributes, e.g. red-eye + no checked baggage → discomfort / packing light.
- Consider adding `unknowns` to the scenario board later, but do not invent them during dialogue.

Validation target:
- In a travel run, participants should not invent new services/policies such as checked baggage being included unless listed.

### 6. Improve unresolved-handling before closure

Current problem: unresolved status can be correct, but the path to unresolved should feel socially and procedurally justified. In the Stockholm run, the final unresolved state is technically valid because votes split D/B/C, but the conversation contains missed answers and repeated vote prompts before closure.

Required behavior:
- Close as unresolved only after:
  - required response obligations are resolved or explicitly abandoned,
  - each participant has had a chance to clarify one final stance,
  - no unique majority is visible,
  - no obvious compromise option has pending discussion.
- If votes are split, the moderator should summarize the split once and either ask for one compromise attempt or close if no movement occurs.
- Avoid repeated final-vote prompts after a clear split.

Implementation notes:
- Add a small `closure_attempts` or `compromise_attempted` flag.
- If all participants voted for different options, trigger one `split_vote_compromise_prompt` before unresolved closure, unless hard max turns is reached.
- Keep this bounded so unresolved runs do not drag on.

Validation target:
- Split votes should produce either one bounded compromise attempt or a clean unresolved close, not repeated vote loops.

### 7. Refine option coverage without forcing artificial discussion

Partially addressed (2026-07-01): the coverage nudge is now bounded to one
attempt per option (`OptionCoverage.coverage_attempts`), so a missed/over-hard
coverage requirement can no longer loop forever. The resolver also matches
distinctive proper-noun tokens (e.g. "Gin", "Rails") and ignores stopword
aliases, so coverage detection is far more reliable.

Remaining work:
- Track meaningful processing, not only mention count: distinguish `reason`,
  `objection`, `comparison`, `explicit_skip` when deciding coverage is "enough".
- Prefer comparison prompts for compromise options.
- Do not over-discuss clearly unattractive options.

Validation target:
- In a four-option run, no option should remain completely untouched before
  voting unless the moderator or participants explicitly skip it.

### 8. Strengthen agenda-based simulator behavior

Current problem: agenda items are minimal. They help structure behavior, but they are not yet strong enough to make the sims look like persistent user simulators with goals and pending communicative tasks.

Required behavior:
- Each sim should have a small private agenda based on goal, initial preferences, blockers, and simulator parameters.
- Agenda items should include pending communicative acts such as:
  - state preference,
  - ask practical constraint,
  - object to option,
  - answer challenge,
  - propose compromise,
  - give final vote.
- Agenda items should have status: pending, completed, blocked, or obsolete.
- The router should prefer agenda-compatible moves without scripting exact text.

Implementation notes:
- Keep agenda simple; do not rebuild a full ConvLab-style policy yet.
- Use agenda for behavior selection, not hidden outcome evidence.
- Log agenda status for later debugging and evaluation.

Validation target:
- In a transcript, each sim should show continuity between their goal, earlier statements, later objections, and final vote.

### 9. Prepare evaluation layer, but keep it lightweight

Current problem: evaluation exists as a scaffold, but it should be organized so later work can expand it without touching generation logic.

Required behavior:
- Keep evaluation separate from logging.
- Include only stable basic metrics for now:
  - participant turn counts,
  - top speaker share,
  - moderator ratio,
  - visible vote count,
  - outcome status,
  - option coverage,
  - unanswered direct-question count,
  - name-prefix rate,
  - repeated-opening-pattern count.
- Prepare placeholders for later metrics without implementing complex scoring yet.

Already added (2026-07-01): `unanswered_direct_questions`, `name_prefix_rate`,
and `repeated_opening_patterns` are now in `evaluation.metrics_for`, alongside
the existing turn counts, top-speaker share, moderator ratio, vote count,
outcome, and option coverage.

Remaining work:
- Add TODO stubs for future metrics such as participation Gini, direct response
  rate, question-answer completion, repetition score, and engagement realization
  error.

Validation target:
- Metrics should expose the failures that are currently being manually spotted in transcripts.

### 11. Keep token usage bounded, but do not optimize prematurely

Current position: token usage around 5k-20k input tokens per typical `n=3` run is acceptable for now. Do not aggressively compress prompts if it worsens transcript quality.

Required behavior:
- Prevent token use from growing unbounded as group size increases.
- Keep per-turn prompt context intentional.
- Do not reintroduce extremely large transcripts such as 100k+ tokens per run.
- Revisit token optimization only after simulator behavior stabilizes.

Implementation notes:
- Log total setup/dialogue input and output tokens as already done.
- Add a warning threshold if a normal `n=3` run exceeds the configured upper range.
- Do not make token optimization a priority unless it starts harming iteration speed or cost.

Validation target:
- Normal `n=3` runs should stay below the configured warning threshold unless the transcript is intentionally long.

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

## Newly observed issues

Add new validation findings here after each implementation pass. Include log path/date, topic, group size, and the smallest description of the failure.

- Setup flakiness: the persona LLM occasionally violates the required primary
  preference or gives a hard blocker two preferred options, forcing a setup
  retry. Mitigated by an aligned prompt line, but the retry loop still absorbs
  the occasional failure.
