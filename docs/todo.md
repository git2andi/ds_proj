# TODO: Multi-User Dialogue Simulation Framework

Source of truth for open work. Rewritten 2026-07-03 to shift the project's framing from
"transcript generator" to "configurable multi-user simulator framework" and to replace
the closed-out fix log (I1-I20, all resolved/verified) with the current open plan.

Standing decisions:

- **Target model: `gpt` / `gpt-4.1-mini` only.** Never switch provider without explicit
  user instruction.
- **Git: one commit on `master` per verified issue.**
- **Validation topics: random and varied across domains** (free time, technical,
  fictional, household, work) — never reuse the same topic between runs.
- **No test-based validation plan for now.** Validate by running generated chats and
  inspecting transcripts / `run.json` / metrics directly (see section 5).

---

## 0. What this project actually is

The goal is **not** to produce nice-looking chat transcripts as an end in itself. The
goal is a configurable multi-user dialogue environment plus multiple tunable LLM-driven
user simulators that interact in group conversations, in the spirit of user-simulation
frameworks like ConvLab3 and the Multi-User Simulator idea from MUCA — but aimed at
generating the simulated users and their group interaction itself, not at building an
assistant that mediates between real users.

Concretely, a topic becomes a scenario with four factual options; 2-7 simulated
participants discuss the options; a controller decides speaker/act/target/voting while
an LLM renders one natural message per move; the run ends `successful`, `majority`, or
`unresolved` based on visible votes only.

### Honest current state

What exists and works reasonably well:

- Option-grounded scenario generation with hard-constraint enforcement on numeric caps.
- A controller that separates decision-making (who speaks, what move, when to vote) from
  text rendering (the LLM writes one message).
- Visible-evidence consensus logic: hidden preference never counts as final support.
- Structured per-turn validation, grounding tripwires, and a deterministic fallback path
  so invalid text is never printed.
- Deterministic style tracking that produces some length and phrasing variation across
  personas.

What is still weak or missing, and should not be overstated:

- **Participant generation is effectively random.** There is no way to specify a
  participant's traits or preferences directly; every run samples them via the LLM.
- **Environment generation is effectively one-shot.** A topic goes in, a full scenario
  (options, shared context, constraints) comes out; there is no way to pin down or reuse
  a fixed environment for controlled comparisons.
- **The per-persona agenda is a weak private hint list, not a real driver of behavior.**
  See issue 3 below.
- **The evaluation layer is a stub.** `src/evaluation.py` already returns real per-run
  counters (turn counts, repair rate, moderator ratio, etc.) but the metrics that would
  actually validate "do simulators behave according to their configured parameters" are
  declared as named placeholders (`_PLANNED_METRICS`) and return `None`.
- **The moderator does most of the structural work** (presents options, calls the vote,
  probes holdouts, closes), which is convenient but works against the framing of *user*
  simulation.
- **`src/dialogue.py` is a single ~1900-line file** covering routing, validation,
  grounding, phase control, moderator logic, and pacing — workable but not something
  that reads as a simulator framework's architecture.

---

## 1. Prioritized implementation plan

Work top to bottom; each item should land as its own commit once validated by live runs
(section 5). Do not start an item before the previous one is validated, unless they are
clearly independent.

1. ~~Explicit simulator-profile input mode (participants: auto | manual)~~ DONE 2026-07-03
2. ~~Explicit environment input mode (environment: auto | manual)~~ DONE 2026-07-03
3. Honest agenda documentation + framing (docs/model only, no behavior change forced)
4. Complete the planned evaluation layer
5. Fix sudden unexplained preference switches (bridge-clause enforcement)
6. Fix phase-history inconsistency (no false "closure" phase markers)
7. Reduce moderator dependency (configurable moderator behavior)
8. Split `dialogue.py` into policy / observer / validation modules

Items 1-2 come first because they are prerequisites for controlled experiments: most of
the other items (evaluation, moderator dependency) are much easier to judge once you can
fix the participants and environment and vary one thing at a time.

---

## 2. Open issues

### Issue 1 (P0). Add explicit simulator-profile input mode — DONE (2026-07-03)

Implemented as `participants: {mode: auto|manual, profiles: [...]}` in `config.yaml`
(schema documented there). Group size in manual mode = number of profiles;
`simulation.num_participants` is ignored. Profiles may be partial: missing names come
from the pool, missing traits are sampled from the configured ranges, missing
background/private_goal/preference are filled by the existing persona LLM call (which
is told to copy fixed texts verbatim). Direct `parameters:` overrides land on top of
the trait-derived values. A profile with `rejection` is a hard blocker (agreeableness
pinned to 1, `rejection_reason` required, never assigned its rejected option as
required primary). If every profile is complete, the persona LLM call is skipped and
the cast is fully deterministic. Config validation fails fast on unknown fields,
out-of-range traits/parameters, duplicate names, and blocker contradictions.
`run.json` records `participants_mode`.

Validated 2026-07-03 by live runs (n=3, varied topics):

- auto regression, coffee-subscription topic (`logs/20260703_212848_059078`):
  behavior unchanged, majority(A), 0 fallbacks, 0 unanswered obligations.
- manual partial, messaging-tool topic (`logs/20260703_213018_322145`): manual
  name/description/traits/parameter overrides and the blocker rejection appear
  verbatim in the trace; the empty third profile was auto-filled; majority(A).
- manual complete, autumn-weekend topic (`logs/20260703_213154_201229`): persona
  LLM call skipped (setup tokens 782-in vs ~1990-in), cast byte-identical to the
  profiles, and configured verbosity visible in realized behavior (verbosity 0.2 →
  12.5 avg words/turn vs 0.85 → 27.1); honest unresolved outcome for a stubborn
  three-way split.

### Issue 2 (P0). Add explicit environment input mode — DONE (2026-07-03)

Implemented as `environment: {mode: auto|manual, manual: {...}}` in `config.yaml`
(schema documented there). Manual defines topic, decision_kind, opening_question,
shared_context (constraints live here — hard numeric caps are parsed from it exactly
as in auto mode), and exactly `len(scenario.option_labels)` option cards (name + ≥1
factual attribute required; short_name/upside/tradeoff/concern/best_for optional).
The scenario LLM call is skipped; the authored cards are the factual source of truth
and are never rewritten — an option violating the manual caps is a startup config
error. No CLI topic is needed (a provided one is ignored with a stderr note).
Stopping/pacing intentionally stays under `conversation:`/`consensus:`. The auto
path's group-size contradiction guards (topic count, shared-context count) are not
applied to manual environments: they exist to catch the setup LLM disobeying the
requested world, and a manual fact like "25 colleagues will attend" describes the
scenario, not the deciding group (found live: the guard falsely rejected exactly
that fact). `run.json` records `environment_mode`.

Validated 2026-07-03 by live runs:

- manual env + auto participants, winter-party venue (`logs/20260703_213924_858347`):
  board rendered verbatim, discussion grounded in authored facts, no clamps,
  0 invalid printed turns; honest unresolved outcome on a persistent 1-1-1 split.
- manual env + complete manual cast (`logs/20260703_214050_076728`): **setup tokens
  0/0** — fully deterministic world+cast, only dialogue LLM calls; CLI topic ignored
  with note; majority(C) with the holdout named.
- auto/auto regression, board-game topic (`logs/20260703_214253_923700`):
  generation path unchanged, successful(D), 0 fallbacks.

### Issue 3 (P1). Be honest about the current agenda system

`models.py` defines `AgendaItem` and `simulator.build_initial_agenda`, and
`evaluation.py` already reports `agenda_status` counts. But recent run logs show most
agenda items never resolve:

- book-swap shelf run: agenda status `pending: 6, done: 1, obsolete: 1`
- charity run: agenda status `pending: 9, done: 1, obsolete: 2`

Interpretation: the agenda is not currently driving user behavior in any real sense.
It's closer to a weak private communicative-goal list / hint system than an
agenda-based user simulator.

Action: stop describing this as "agenda-based simulation" anywhere (docs, comments,
future write-ups). State plainly:

- **Current state:** weak private communicative-goal list.
- **Not yet:** agenda-based user simulator.

Possible future direction (not yet planned in detail): give each simulator a clearer
goal stack, where each turn consumes, defers, or updates one agenda item. This is a
future improvement, not something to implement opportunistically as part of this issue.

### Issue 4 (P1). Complete the planned evaluation layer

`src/evaluation.py` already has a `_PLANNED_METRICS` stub tuple returning `None` for
each name, plus a comment pointing at this exact gap. It needs real implementations,
using data already collected in `DialogueState`/`ParticipantRuntime` (turns, votes,
switch_events, coverage, style history) — no new logging should be required for most of
these.

Metrics to implement:

- participation balance / turn distribution (a proper inequality measure, not just
  `top_speaker_share`)
- engagement realization error (`|expected engagement - realized turn share|`)
- verbosity realization error (expected vs. realized word-count band per persona)
- direct response rate (share of response obligations actually answered)
- question-answer completion (adjacency-pair completion ratio)
- repetition score (lexical/semantic repetition across a persona's turns)
- compromise success rate (share of split-vote situations resolved via the compromise
  step)
- preference-switch explanation rate (share of `switch_events` with `has_reason=True`,
  cross-checked against issue 5 below)
- unresolved-question count (open response obligations at run end)
- speaker dominance (already partially covered by `top_speaker_share`; extend/replace
  as needed once the inequality measure above exists)
- a summary signal for whether configured behavioral parameters visibly affect
  generated behavior (i.e., do the realization-error metrics above actually correlate
  with the configured trait values across runs)

Goal: the project should not rely on manual "this transcript sounds natural"
impressions. It should produce structured per-run metrics that show whether simulators
behave according to their configured parameters.

### Issue 5 (P1). Fix sudden unexplained preference switches

Observed in a charity-run log: Leo argued for the senior-support option throughout the
discussion, then voted for Youth Arts with no visible bridge:

> "My pick is City Youth Arts Scholarships because they directly support creative
> education…"

This is internally consistent with the controller's state (a sanctioned switch was
presumably in effect) but socially under-explained in the transcript — nothing connects
the old stance to the new one.

Why this matters: the project is about simulated users, not just final vote tallies.
Preference movement must be visible, socially motivated, and grounded in the transcript,
not just legal at the state-machine level.

Needed rule: whenever `current_preference != vote_option` for a turn, the generated line
must contain (a) the old preference or an explicit concession, (b) the new accepted
option, and (c) a reason for the movement. Example of what's wanted:

> "I still like Senior Safety, but I can go with Youth Arts because it seems easier for
> the whole group to support."

`switch_events` already records from→to and whether a reason was given
(`has_reason`), but generation doesn't reliably force the bridge clause. This should be
enforced the same way other blocking issues are — as a validator/repair rule, not a
prompt-only request.

### Issue 6 (P1). Fix phase-history inconsistency

Observed in a book-swap run, the phase history read:

```
closure — all participants already gave a clear vote
closure — successful after split-vote compromise
closure — closed as successful
```

This is misleading: the system did not actually close after the first clear votes, it
continued into compromise handling. Structured logs must be trustworthy — they're part
of the analyzable simulation trace, not cosmetic.

Needed change: when all participants have voted but there is no majority or consensus
yet, do not record a final "closure" phase. Use an intermediate marker, e.g.:

```
narrowing — all participants voted but no majority; trying split compromise
```

Only record final closure once the outcome is actually resolved.

Additional evidence (2026-07-03, issue-1 validation runs): `logs/20260703_212848_059078`
(coffee subscription, n=3) logs `closure — all participants already gave a clear vote`
before continuing into the split-vote compromise, and `logs/20260703_213154_201229`
(autumn weekend, n=3) logs the same marker before ending `unresolved` — a "closure"
phase entry on a run that never closed on that vote state.

### Issue 7 (P2). Reduce moderator dependency

Currently the moderator carries most of the interaction structure: presents the option
board, calls the vote, probes holdouts, closes the decision. Useful for demos, but works
against the project's actual focus, which is simulated *users* in group interaction, not
a facilitator.

Needed change: make moderator behavior configurable.

```yaml
moderator:
  enabled: true
  opening: true
  mid_discussion_nudges: true
  final_vote_call: true
```

Then support lower-moderator or no-moderator modes where participants themselves narrow
the discussion and move toward a decision, so the system can produce peer-to-peer group
interaction and not only moderator-guided discussion. This will likely require pushing
some of the moderator's current responsibilities (option board framing, vote calling)
into participant-level acts when the moderator is disabled — scope that out concretely
before implementing.

### Issue 8 (P2). Split `dialogue.py` carefully

`src/dialogue.py` is ~1900 lines and mixes routing, validation, generation, grounding,
phase control, moderator logic, semantics, pacing, and output behavior. This makes the
project harder to maintain and harder to explain academically.

Needed change: extract a few coherent modules, not many tiny ones.

- `policy.py` — choose speaker, choose act, choose target, vote readiness, candidate
  selection.
- `observer.py` — parse generated text, apply semantics, update visible state, manage
  response obligations.
- `validation.py` — validate turn text, grounding tripwire, fallback text.
- `dialogue.py` stays as the orchestration loop.

Goal: the architecture should read as a simulator framework, not a patched
transcript-generation script. Do this last, once the behavior above has stabilized —
refactoring is much cheaper to validate when the discussion logic isn't also changing
underneath it.

---

## 3. Research-backed principles (kept from prior review)

Use paper insights only where they directly improve the simulator; never implement paper
architectures mechanically.

- **Turn-taking:** direct questions create response obligations; addressed speakers
  answer soon; self-selection is fine otherwise; no same speaker twice in a row;
  trait-driven but non-collapsing turn distribution.
- **Addressee selection:** multi-party dialogue needs *who speaks to whom about what*;
  not every turn targets the latest message; maintain active threads (open questions,
  objections, minority positions, unresolved constraints).
- **Moderation (MUCA-style):** decide what intervention, when, addressed to whom; nudge
  only when stalled, scattered, one-sided, or ready for visible narrowing; ask holdouts
  what blocks agreement instead of declaring consensus.
- **Decision emergence:** orientation → clarification/conflict → convergence as a
  tendency, not a script; closure requires visible narrowing evidence.
- **Personality/OCEAN:** traits create stable tendencies (verbosity, directness,
  initiative, compromise, stubbornness), never random contradiction; hard blockers rare
  and stable.

---

## 4. Non-negotiable rules

- Never count hidden preference as final support.
- Never close before participants have a visible decision opportunity.
- Never let a hard blocker accept their rejected option through state mutation.
- Never add facts outside option cards/shared context.
- Keep the moderator sparse and neutral (until issue 7 makes it configurable — even then,
  default behavior should stay sparse).
- Put all LLM-facing prose in `src/prompts.py`.
- Prefer controller/parser/validator/state fixes over enlarging prompts.
- No large theoretical prompt blocks; no mechanical paper implementations.
- No many tiny files before behavior is stable (see issue 8).
- Never optimize tokens by removing the visible evidence consensus needs.
- Successful execution ≠ successful dialogue quality.

---

## 5. Validation approach (no tests, for now)

Validation is done through live generated chats and structured-log inspection, not
automated tests:

1. Run several chats varying participant mode, environment mode, and group size.
2. Inspect the printed transcript for plausibility, bridge clauses on switches, and
   moderator-vs-participant balance.
3. Inspect `run.json` (phase history, switch_events, agenda_status, validation issues,
   fallback/repair counts).
4. Compare the intended behavioral parameters (engagement, verbosity, etc., manual or
   sampled) against what the metrics in section 2, issue 4 actually report.
5. Record observed failures back into this file (with log path/date, topic, group size,
   and the smallest description of the failure) and iterate.

### Observations from validation runs (2026-07-03, issues 1-2)

- **`switch_events` are not serialized.** `ParticipantRuntime.switch_events` is
  collected but `run.json` never writes runtimes, so switch analysis is impossible
  from the trace alone. Fix as part of issue 4 (the evaluation layer needs it) —
  issue 5's explanation-rate metric depends on it too.
- **Split probe claims "most support" on a pure tie.** Winter-party run
  (`logs/20260703_213924_858347`, n=3): after a 1-1-1 vote the moderator said votes
  were split "with Strike Lanes having the most support" — no option had more than
  one vote. `_split_probe_candidate`/nudge wording should not assert a lead that
  does not exist. Small, fold into issue 7's moderator work (or earlier if a run
  shows it misleads participants into false convergence).
- **More issue-5 evidence.** Board-game run (`logs/20260703_214253_923700`, n=3):
  Diego argued Ticket to Ride throughout, then voted "My vote goes to Azul because
  its simple rules..." with no bridge to his prior stance.
