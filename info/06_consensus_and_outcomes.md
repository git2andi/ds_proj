# 06 — Consensus and outcomes

Outcomes are computed from visible text, not hidden preferences.

## Outcome types

- `successful`: all sims visibly support the same final option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no option reaches the required majority/unanimity after bounded discussion and narrowing.

There should be no fourth outcome label. Invalid unanimity should be prevented before closure: a participant with an unresolved blocker should visibly reject, stay elsewhere, or mark the option unacceptable, which naturally produces `majority` or `unresolved`.

## Visible evidence only

A sim counts as supporting an option only if the transcript contains a clear vote, acceptance, or sanctioned switch. Hidden `current_preference` is used for routing and simulation state, but it should not decide the final outcome directly.

## Vote stability

A clear vote should remain stable unless the sim explicitly switches. Ambiguous text should be repaired or re-asked rather than silently interpreted.

## Hard blockers and false unanimity

A hard blocker or explicit hard constraint should not be overridden by agreeableness. Agreeableness changes how politely the participant rejects or negotiates; it does not make an impossible option acceptable.

Normal auto-generated participants should usually have movable preferences, not categorical constraints. Manual profiles may define blockers explicitly.

## Split-vote narrowing

No-majority votes should not close prematurely. The controller ranks split candidates deterministically from visible votes and can test at most two candidates:

```text
first attempt: visible plurality, or best tied candidate by blockers/resistance/compromise fit;
second attempt: best remaining candidate if the first attempt fails;
then close unresolved if no majority/unanimity is visible.
```

Post-reservation decision turns use the `post_reservation_decision` act. The controller should ensure the visible line does one of these:

```text
switch to the tested candidate;
stay with the original/current option and name the blocker;
switch to one concrete alternative candidate;
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

## Current validation focus

Inspect final outcomes for plausibility, not just parseability. `successful` is valid only when all participants visibly and plausibly accept the same option. If one participant should not accept, the transcript should show a refusal and the result should become `majority` or `unresolved`.
