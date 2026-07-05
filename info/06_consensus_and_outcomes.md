# 06 — Consensus and outcomes

Outcomes are computed from visible text, not hidden preferences.

## Outcome types

- `successful`: all sims visibly support the same final option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no option reaches the required majority/unanimity after bounded discussion and narrowing.

## Visible evidence only

A sim counts as supporting an option only if the transcript contains a clear vote, acceptance, or sanctioned switch. Hidden `current_preference` is used for routing and simulation state, but it should not decide the final outcome directly.

## Vote stability

A clear vote should remain stable unless the sim explicitly switches. Ambiguous text should be repaired or re-asked rather than silently interpreted.

## Split-vote narrowing

No-majority votes should not close prematurely. The controller now ranks split candidates deterministically from visible votes and can test at most two candidates:

```text
first attempt: visible plurality, or best tied candidate by blockers/resistance/compromise fit;
second attempt: best remaining candidate if the first attempt fails;
then close unresolved if no majority/unanimity is visible.
```

Post-reservation decision turns should visibly do one of these:

```text
switch to the tested candidate;
stay with the original option and name the blocker;
name one concrete alternative candidate;
state that no acceptable compromise remains.
```

## n=2 deadlock

A 1-1 split requires special handling. The intended protocol is:

```text
1. each participant states the strongest blocker;
2. each proposes one condition or concession;
3. each makes a final switch/stay decision;
4. unresolved is valid if neither moves.
```

The full evaluation suite now includes `f01_manual_manual_n2_stubborn_deadlock`, a forced manual/manual case with two stubborn opposing participants. It should set `two_person_deadlock_attempted = true` when live validation is run.

## Current validation focus

Run `py run_eval_suite.py --full`, inspect `f01_manual_manual_n2_stubborn_deadlock`, and compare `visible_votes`, `outcome_status`, and `two_person_deadlock_attempted` in `run.json`.
