# Moderator (the facilitator voice)

**Code:** `src/dialogue.py` (`_mod`, `_maybe_moderator_nudge`, `_moderator_vote_nudge`,
`_moderator_intervention_details`, `_minority_check`, `_maybe_split_vote_compromise`,
`_moderator_say`), `src/prompts.py` (`moderator_opening`, `moderator_nudge_prompt`,
`moderator_closure_prompt`).

The moderator is the environment's facilitator. It exists to prevent common
multi-user failures — ignored questions, premature voting, silent options, unclear
commitments, endless split votes — while staying sparse and neutral. Crucially, the
**controller policy is separate from the moderator voice**: the controller decides
who speaks and when a vote is ready regardless of the moderator; the moderator only
adds facilitation *turns*.

## Configurable moderator (issue 7)

The moderator's visible turns are gated by `moderator:` in `config.yaml`, checked via
`DialogueRunner._mod(part)`:

```yaml
moderator:
  enabled: true               # master switch; false = no moderator turns at all
  opening: true               # present the option board at the start
  mid_discussion_nudges: true # stall/circling nudges during discussion
  final_vote_call: true       # vote calls, holdout probes, split-compromise summary
  closing: true               # closing summary line
```

Because the flags touch only the moderator's turns, **lower- and no-moderator modes
still reach a decision**: the decision loop keeps emitting participant vote turns and
the participant-level narrowing acts (defend / compare / propose-compromise /
stagnation break in `03`) carry the discussion. This is how the system produces
genuine **peer-to-peer** interaction, not just facilitator-guided discussion.

- With `opening: false`, the board is still shown as plain setup scaffolding (console
  header + the transcript `## Options` section), just not as a moderator turn.
- Defaults reproduce the fully-moderated behavior. `run.json` records the resolved
  `moderator_config`.

## MUCA-style what / when / who

When it does act, the moderator decides three things
(`_moderator_intervention_details`):

```text
what   which intervention is needed?
when   is this the right moment (stalled / scattered / one-sided / ready to narrow)?
who    who should answer or be addressed?
```

A mid-discussion nudge (`_maybe_moderator_nudge`) only fires when the discussion has
stalled (`no_progress` window), respects a cooldown and a cap
(`moderator_max_interventions`), and picks the most concrete visible issue in
priority order:

```text
uncovered option  -> pending direct question  -> unresolved blocker on the candidate
(probed once)     -> visible split (weighed head-to-head)  -> single holdout
                  -> generic "strongest remaining concern" (last resort)
```

## What good vs bad moderation looks like

```text
Good (state-aware, targeted, one line):
  "Anton, Kenji asked about the no-checked-bag issue — is that a deal-breaker for you?"
  "Before we narrow, D hasn't come up — one reason to keep or drop it?"

Bad (generic, ignores state):
  "Can everyone share what feels best?"
```

The moderator never dictates a quoted vote formula, never repeats the option board,
and varies its phrasing. When it addresses a participant by name, that participant
owes the next answer (the response obligation in `03`).

## Vote calls are option-neutral

At finalization the moderator invites picks **without naming any option**, so the
current front-runner can't leak into the question and nudge a false consensus
("which Space Station option are you going with?" is exactly what it must avoid).
Later rounds re-prompt only the unclear/non-voters.

## Bounded closing beats

Two special facilitation passes, each at most once per run, keep closes honest:

- **Minority check** (`_minority_check`) — once a majority forms, the holdouts get
  one visible beat: accept the majority option *with a bridge clause* if they can
  move, or briefly restate what holds them back. May upgrade the outcome to unanimity.
- **Split-vote compromise** (`_maybe_split_vote_compromise`) — if votes split with no
  majority, the moderator summarizes the split and floats one candidate; movers may
  switch (with a bridge), the rest restate. The probe never targets an option with a
  visible unresolved dealbreaker, and it only claims a candidate "has the most
  support" when it is a **strict plurality** — a pure tie is announced as "evenly
  split with no option ahead".

Both passes embed a **reservation exchange** (issue 4, once per run, exactly two
turns): the most movable holdout states one concrete reservation about the
candidate (explicitly *not* a vote), and one supporter responds honestly — using
only card facts and conceding what the board can't prove. Only then come the
closing beats where the holdout may accept (bridged) or stay. With
`final_vote_call: false`, the holdout probe is asked by a high-initiative
*supporter* instead of the moderator — participant-owned procedure. The hard turn
cap forces the vote but never starves these bounded passes.

## Closures are status-aware

`moderator_closure_prompt` matches the real outcome (`06`): a `majority` close **names
the holdouts** and never implies full agreement; a `successful` close wraps up warmly;
an `unresolved` close presents **nothing** as chosen. (When `closing: false`, the
outcome is still computed and logged — only the moderator's summary line is skipped.)

## Current mismatch / intended correction

The moderator configuration exists, but lower-moderator and no-moderator modes still need stronger participant-owned structure. The hidden controller can currently carry the discussion even when the moderator is visually disabled. That works mechanically, but it weakens the claim that simulated users are managing a group interaction.

The intended behavior is that high-initiative or high-engagement sims sometimes perform procedural moves themselves: summarize the split, ask a quiet participant, suggest dropping an option, call for final picks, ask a holdout what blocks agreement, or check whether a compromise is acceptable. These moves should become more likely when moderator support is disabled or when the group is stalled.

