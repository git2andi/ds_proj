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

Validation target is met (2026-07-01, re-confirmed 2026-07-02 across four runs
at n=3/4/5: every option mentioned before voting, none over-discussed): the
coverage nudge is bounded to one attempt per option
(`OptionCoverage.coverage_attempts`), so a missed/over-hard requirement can no
longer loop forever; the resolver matches distinctive proper-noun tokens and
ignores stopword aliases, so detection is reliable; the coverage speaker uses a
COMPARE move focused on the leading option. `OptionCoverage` already records
`reasons`/`objections`/`acceptances` separately.

Deferred polish (only act on this if a transcript shows the problem again):
- Decide "coverage is enough" from meaningful processing type rather than a bare
  mention. The current mention-based gate already satisfies the target, and
  forcing deeper processing risks over-discussing unattractive options, so this
  is intentionally not implemented yet per the project's minimal-change rule.

## Resolved this pass (2026-07-02)

- Issue #25 (vote-phrase chorus, user-reported: "Count me in for" ×3 in one
  round): deterministic anti-chorus — `parsing.used_commitment_phrases` maps
  the round's turns (since the last moderator line) onto commitment-phrase
  families; `_apply_style_flags` sets `MoveIntent.avoid_phrases` for decision
  acts and both the first-pass vote prompt and the REPAIR prompt list them as
  forbidden/other-form-required (the chorus turned out to come mostly from
  repair outputs, which had a fixed example list led by "count me in for").
  Validated (gpt, n=5 farewell-gift run): five voters, five distinct phrase
  families in round 1 including three repaired turns; compromise beat also
  internally varied. Test: tests/test_parsing.py.
- Issue #26 (majority closed without engaging the minority, user-reported):
  `_minority_check` — when vote rounds produce a majority (not unanimity), the
  moderator acknowledges it and asks the holdouts whether they can live with
  it or what still holds them back; each dissenter takes one turn (movers may
  accept with `allow_vote_change` + bridge clause, others restate). Runs once,
  gated by the hard turn cap, skipped after the split-compromise pass (those
  dissenters were just asked); outcome re-finalized (can upgrade to
  unanimous). Validated (gpt): n=4 workshop run — Cleo was asked by name,
  switched with a reasoned bridge, outcome upgraded to successful; n=5 lunch
  run — the beat also mopped up an unclear conditional voter ("even if …")
  and a conditional acceptance correctly stayed a non-vote (majority 4/5);
  n=5 farewell run — correctly skipped after the compromise pass.

- Issue #24 (compromise switches read unmotivated, user-reported: sims changed
  their pick with no word on why — charity run,
  logs/archive/20260702_140808_537925): the mover's compromise intent now asks
  for a direct commitment PLUS a one-clause bridge naming what makes the
  compromise acceptable despite preferring <current option, by alias> (what
  they give up or what it still delivers), and switch turns
  (`allow_vote_change`) get +8 words budget headroom for that clause.
  Validated (gpt, n=4 coffee run with a live 2-2 split): "My pick is the Large
  Capacity Drip Coffee Maker since it serves everyone at once and keeps prep
  time low despite some flavor tradeoffs" — the trade-off he gives up is named;
  the second mover gave a reasoned (if not contrastive) switch; all switch
  votes parsed and the run closed unanimous. Residual: the concession clause
  is a prompt-level ask, so terse sims may switch with a plain reason instead
  of an explicit concession — acceptable variation, only revisit if bare
  unexplained switches reappear. Test: switch turns get budget headroom
  (tests/test_outcomes.py).

- Issue #23 (phantom split, user-reported: moderator announced a split right
  after both participants visibly voted the same option — podcast run,
  logs/archive/20260702_135727_501139, n=2). Cause chain: an opening "Daily
  News seems like a solid pick" parsed as a visible acceptance and set
  `explicit_vote`; the formal "gets my vote" round vote was then silently
  dropped by the overwrite protection; the stale vote sat outside the
  recent-chat window so the moderator said "another option". Fixes:
  (a) "seems/sounds like" added to `_HEDGE` — hedged leans stay latent;
  (b) `ParticipantRuntime.vote_stance` tracks how the vote was stated — a
  formal direct vote replaces an accept-derived commitment, while direct→
  direct stays protected (issue #5 tests untouched and green);
  (c) `_public_state_summary` names votes with short aliases so a real split
  is described concretely. Validated (gpt, n=2 + n=4): zero discussion-phase
  visible commitments, real splits named ("split between Born a Crime and The
  Martian"), both runs converged to unanimous through the dissenter beat.
  Tests: tests/test_outcomes.py (stance-override matrix),
  tests/test_parsing.py (seems/sounds hedges).

- Issue #20 ("We …" opening monotony, user-reported; 30–37% of turns after the
  #15 fix displaced "I" onto "we"): `style.leading_we`/`we_opening_fraction`
  mirror the I-tracker; above `style.we_opening_max_fraction` (0.30) the next
  non-decision turn gets `suppress_we_opening`; the suppression notes no longer
  suggest "we"/"I" as fallback openers. New metric `we_opening_rate`.
  Validated (gpt): n=3 run 0.056, n=5 run 0.148, n=2 run 0.0 (from 0.30–0.37),
  i-opening stayed 0.07–0.17.
- Issue #21 (traits audible in length but not register, user-reported):
  `_voice_guidance` rewritten to be contrastive with micro-examples at the
  extremes (blunt declaratives, clipped fragments "Games. Cheap, fun, done.",
  dig-in dismissals for stubbornness ≥0.8, flowing asides for verbosity ≥0.7)
  and the realizer style line now demands difference in register, not just
  content, with casual interjections allowed. Validated by reading n=3/n=5
  transcripts: a verbosity-0.26 sim speaks in keepsake-terse fragments while a
  verbosity-1.0/engagement-0.96 sim pushes energetic multi-clause opinions;
  who is speaking is recognizable without the name; avg-words spread persists
  (13.2 vs 25.6).
- Issue #22 (abrupt ending after failed split compromise, user-reported):
  `_maybe_split_vote_compromise` now gives EVERY dissenting voter one closing
  beat — movers may switch (`allow_vote_change`), non-movers briefly restate
  and react to the split. Validated live in an n=2 run where the
  vote-overwrite protection produced a momentary split: the dissenter took her
  beat and the run closed unanimous instead of cutting off.
- Chop stub polish: "anyone/anybody" now count as interrogative stubs so
  "…—anyone worried" keeps its question mark (seen once in the block-party
  run).

- Issue #19 (visible questions went unanswered; user-reported): four combined
  causes, all fixed —
  (a) `_GENUINE_QUESTION` missed common group-question forms ("should we",
  "are any of us", "is/does/can anyone", "shall we"), so those questions never
  created a response obligation; broadened.
  (b) `_obligation_intent` never set `respond_to_turn`, so the answering turn's
  prompt did not show WHICH question was owed and group-directed answers
  pivoted to unrelated points; the obligation's question turn is now linked.
  (c) The answer act had no guidance for questions unanswerable from the option
  board (weather, headcounts); the prompt now says to state plainly that we
  don't know, then give a take — instead of ignoring the question.
  (d) The word-cap chop could cut a question mid-clause and leave a nonsense
  stub ("… but what about those who?"); the chop now strips broken
  interrogative tails and keeps "?" only when the stub still reads as a
  question.
  Also found via the same run: another generic-alias vote loss ("neighborhood"
  / "food" tokens flipped a clear Garden vote to 3-way-ambiguous → unresolved).
  Two-part fix: more generic tokens excluded from standalone aliases
  (neighborhood, community, event, food, class, workshop, …) and — the general
  mechanism — `_commitment_object` in parsing.py resolves a multi-option vote
  line by the option nearest to the commitment phrase ("I'd go with X … better
  than Y" → X; "X gets my vote, faster than Y" → X), while coordinated pairs
  ("either X or Y") stay ambiguous. Validated (gpt, n=3 alias-hostile
  community-event topic + n=5 hiking topic): every visible question answered
  on-content in the next turn, zero truncated stubs, zero grounding flags,
  votes 3/3 and 5/5 recorded; the n=3 unresolved is a genuine three-way split.
  Tests: tests/test_parsing.py (question forms, disambiguation, coordination
  guards), tests/test_clean_generated.py (broken-tail handling).

- Issue #17 (vote turns burned repairs and came back as a "My vote is X"
  chorus): `_COMMIT`/`_DIRECT_VOTE` gained the clear commitment forms actually
  produced by sims across validation ("X gets/get my vote", "my top choice/pick
  is", "is/makes it my choice/pick", "I'm (all) for X", "I'm sold on X",
  "let's do/book/get X", "works (best) for me") — hedges/conditionals still
  block; the moderator's vote-time requested action now asks for a definite
  pick and explicitly avoids "leaning" wording; the first-pass vote instruction
  asks for a phrasing different from previous voters (variety), while the
  repair instruction prioritizes clarity over variety (a failed repair
  previously lost a compromise switch). Validated (gpt, two waves n=3/n=5):
  vote phrasing is varied with `repeated_opening_patterns=0`, all standard
  votes recorded (5/5, 3/3), repair rate roughly halved (0.30→0.15); remaining
  UNCLEAR repairs are single-shot and recover successfully. Tests:
  three phrasing waves in tests/test_parsing.py.
- Issue #18 (grounding false positives on cross-option comparisons):
  `_grounding_issue` now always passes the full option board to the checker
  (comparisons legitimately restate other options' card facts), and the
  checker prompt explicitly allows paraphrase and attribute-based comparisons
  for ANY option. Validated (gpt): comparison-heavy runs dropped from 4 false
  flags to 0 (n=5) / 1 borderline (n=3). Guard test asserts the prompt carries
  every option card. Residual: the LLM judge is still occasionally over-strict
  on loose paraphrase — flags are non-blocking and cost one repair, so this is
  accepted noise; revisit only if flags exceed ~2 per run.
  Follow-up (same day): the residual recurred at 4 flags in the summer-camp
  run (logs/archive/20260702_150822_781090) — all four were commonsense risk
  derived from an attribute plus explicit uncertainty ("outdoor → weather-
  dependent", "we don't know the forecast"). The checker prompt now names
  that shape as allowed. Re-validated on a deliberately weather-heavy picnic
  topic: 4 weather/uncertainty lines, 1 borderline flag (within tolerance).
- Minor fixes from the 2026-07-02 varied-topic analysis: the split-vote
  compromise ask no longer calls one side's pick a "middle ground" (renders as
  "could everyone live with X?" — confirmed in three validation runs); the
  word-cap chop preserves a trailing "?" when it truncates a question (so
  question/obligation detection still fires; tests in
  tests/test_clean_generated.py); the option-opening suppression note now asks
  sims to name the option mid-sentence instead of using a bare "this one"/"it"
  when switching focus from the previous turn.

- Vote-parsing gaps that flipped a majority to unresolved (observed in the #15/
  #16 regression run, logs/20260702_092804_559743, database topic, n=3: a
  2-vs-1 ClickHouse majority closed as unresolved): (a) "My vote's on X" /
  "my vote goes to X" were not commitment patterns, so two round-1 votes were
  silently missed; (b) "analytics" — generic English but unique to one option
  name ("Google BigQuery Serverless **Analytics**") — was a standalone alias,
  so "My vote is ClickHouse … analytics …" resolved to two options and was
  dropped as ambiguous (same bug class as the earlier "with"/"data" incident).
  `_ALIAS_STOPWORDS` gained common generic domain nouns (analytics, warehouse,
  database, serverless, managed, hosted, platform, solution, single, plan,
  suite, tool(s), package) and the vote patterns gained the contraction forms.
  Regression-tested with the exact dropped transcript lines
  (tests/test_parsing.py) and re-validated with a deliberately alias-hostile
  n=3 gpt run on an analytics-warehouse topic.
- Issue #15 (first-person opening monotony): 26–59% of participant turns in the
  2026-07-02 runs opened with "I …"; `repeated_opening_token` only catches
  identical consecutive openers, so alternating "I …" turns never triggered
  variation, and the prompt even suggested "I" as a fallback opener.
  `style.leading_first_person`/`first_person_opening_fraction` now track the
  recent-window share; above `style.i_opening_max_fraction` (0.35) the
  controller sets `MoveIntent.suppress_i_opening` on the next non-decision turn
  (vote/accept/reject exempt — "I'd go with" is natural and parser-relevant
  there), rendered as a concrete style note; "I" removed as a suggested
  fallback opener. New metric `i_opening_rate`. Validated (gpt): n=3 run 0.105,
  n=5 run 0.111 (down from 0.26–0.59), majorities reached, 0 unanswered
  questions.
- Issue #16 (scripted consensus calls): vote calls were stiff templates
  ("Everyone, please state your final choice clearly by saying, 'I vote
  for...'"). The controller's `requested_action` strings now ask naturally
  ("where everyone lands"), `moderator_nudge_prompt` forbids dictating quoted
  reply templates, `moderator_closure_prompt` asks for a conversational
  wrap-up, and participant vote instructions say "commit clearly in your own
  words". To keep vote clarity, `_DIRECT_VOTE`/`_COMMIT` gained natural forms:
  "I'd/I'll go with", "I'm going with", "my pick is", "I'm (all) in for",
  "count me in for" (hedges/conditionals still block). Validated (gpt) on the
  same n=3/n=5 runs as #15: moderator lines read naturally ("Hey everyone, can
  we quickly share which speaker you're leaning toward so far?", closure
  "Great — X it is, then."), votes in own words all parsed, majority within the
  normal round budget. One parser gap found and fixed during validation:
  "I'm all in for X" (offsite run, n=5) was a clear commitment but unparsed —
  now covered with tests. Residual watch item: the moderator sometimes asks
  what people are "leaning toward", which can invite hedged answers; the
  round-2 re-prompt absorbs this, but reword the vote-call action if unclear
  votes become common.

- Issue #12 (corpus-inspired presets): optional `corpus` config section with a
  `delidata` example preset; `apply_corpus_preset` (config_loader.py) folds
  preset fields into runtime parameters at load time — typical
  `turns_per_participant` onto the per-participant turn caps (min=0.75×t,
  target=t, max=1.5×t), `preferred_group_size` onto
  `simulation.num_participants` (clamped to the configured min/max) — and
  exposes dominance targets (`top_speaker_share`, `dominance_range`,
  `imbalance_tolerance`) via `cfg.corpus_active`. `_choose_speaker` uses
  share-aware dominance weighting (`utils.preset_dominance_weight`) only when a
  preset is active: the highest engagement+initiative persona is kept near the
  expected top share, others are rebalanced only outside the tolerance band
  around fair 1/n. With `preset: null` behavior is byte-identical to before.
  Metrics record `corpus_preset`. Validated (gpt): preset run forced n=5,
  pacing 17/25/36 vs baseline 9/14/20, designated dominant persona was the top
  speaker (share 0.267, spread 4–8 turns vs strict equalization); baseline
  n=3/n=4 runs with preset off unchanged and healthy. Known limitation:
  opening and vote rounds are evenly distributed by design (visible decision
  opportunity is non-negotiable), so at typical run lengths the realized top
  share stays slightly below a 0.30+ corpus dominance band; presets remain
  soft targets, not hard constraints. Tests: tests/test_corpus_preset.py.
- Setup flakiness (from `Newly observed issues`): persona rows that drop or
  reorder the controller-assigned primary preference, or give a hard blocker
  two preferred options, are now repaired deterministically in parsing
  (`builders.repair_preferred_options`) instead of burning a full persona
  retry; a rejection of the required option is a real contradiction and still
  retries. Validated: n=3 and n=4 gpt runs set up cleanly with zero retries.
  Tests: tests/test_setup_repair.py.
- Mid-sentence truncation (observed 2026-07-02, logs/20260702_090119_229907,
  password-manager topic, n=3: three turns ended on bare function words, e.g.
  "…track exactly who accessed what and."): the word-cap chop in
  `utils.clean_generated` now strips trailing `_DANGLING_TRAIL` function words
  like `compact_words` already did. Validated: n=3 and n=4 runs show zero
  dangling turn endings. Tests: tests/test_clean_generated.py.

## Resolved earlier (2026-07-01)

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

- Full-suite validation 2026-07-02 (six runs, logs/20260702_1503*–1513*,
  topics: office plant, brunch spot, store playlist, summer camp, carnival
  costume, picnic activity; n=3,3,4,5,6,3): outcomes 4× unanimous, 2× majority;
  all votes recorded (n/n) in every run; every visible question answered
  on-content next turn (incl. "we don't know" for off-board facts);
  unanswered_direct_questions=0; i-openings 0.07–0.11; we-openings 0.05–0.16;
  no commitment-phrase family repeated within any round; no phantom splits
  (all split announcements matched ≥3 distinct visible votes); minority checks
  named holdouts in all majority runs; one conditional acceptance ("as long
  as …") correctly stayed a non-vote; verbosity→length spread intact
  (v=0.33→12–13 words vs v=0.94→21–26). Only finding: the grounding-judge
  recurrence documented under #18's follow-up, fixed and re-validated the
  same day.

- Micro (2026-07-02, farewell-gift run logs/20260702_145824_457371, n=5): the
  word-cap chop can end on article+adjective ("…plus we can add a nice.") —
  trailing adjectives aren't in the dangling-word lists and can't be
  enumerated generally. Cosmetic and rare; only act if it becomes frequent
  (a POS-free heuristic would be to also strip a final word when the word
  before it is an article).
- Naturalness density (2026-07-02, family-reunion n=6 + hobby n=3 runs): the
  contrastive "We get X, but Y" (`tradeoff_but`) shape recurs in adjacent-turn
  pairs (4 pairs in 35 turns at n=6) — below the 3-in-a-row tracker threshold
  but visible when reading. Relates to resolved #3. Post-#17 note: the two
  2026-07-02 re-validation runs showed `repeated_opening_patterns=0`; only
  tighten if reading shows it dense again.
- Watch item (2026-07-02, winter-class run logs/20260702_101427_893321, n=3):
  a compromise-round *switch* can still be lost when a sim replies with support
  that never commits ("Photography fits our budget perfectly …") and the one
  repair also fails. The repair instruction now prioritizes clarity over
  variety, which should absorb this; conservative unresolved is acceptable by
  design (never count hidden preference), so only act if switches keep failing
  to register.
- Watch item (2026-07-02, logs/20260702_092804_559743, database topic, n=3):
  the split-vote compromise pass is skipped once `hard_max_turns` is reached
  (the run sat at exactly 20/20 when the split appeared), so a genuinely split
  vote at the turn cap closes unresolved without the one bounded compromise
  attempt. Harmless when parsing is correct (majorities form in the vote
  rounds); only act if unresolved-at-cap becomes a repeated pattern.
- Watch item (2026-07-02, logs/20260702_092542_143400, offsite topic, n=5):
  natural vote calls sometimes ask what people are "leaning toward", which can
  invite hedged answers the parser correctly refuses to count. The round-2
  re-prompt absorbed it (majority 4/5 reached); only act if unclear votes
  become common across runs.
