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

No-majority votes should not close prematurely. The system should run a bounded compromise/reservation sequence. Unresolved is valid only after relevant dissenters had a chance to switch, stay, or propose an alternative.

## n=2 deadlock

A 1-1 split requires special handling. The intended protocol is:

```text
1. each participant states the strongest blocker;
2. each proposes one condition or concession;
3. if neither moves, unresolved is valid.
```

## Current open issues

The latest full suite did not trigger the n=2 deadlock protocol, so this path is not validated. Add a forced stubborn 1-1 evaluation case and inspect whether `two_person_deadlock_attempted` becomes true and the transcript shows symmetric negotiation.
