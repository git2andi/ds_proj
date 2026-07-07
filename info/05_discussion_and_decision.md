# 05 — Discussion and decision flow

The discussion should move from initial preferences to comparison, challenge, possible softening, narrowing, moderator-owned vote call, voting, and closure.

## Phases

```text
opening -> discussion -> narrowing/voting -> closure
```

The opening round gives every participant a visible initial stance. The free discussion phase is where traits matter most. Narrowing and voting should be bounded and should not reopen unlimited debate. When the controller decides that voting is mature enough, a participant may first call for final picks so the chat can appear to close itself instead of relying on a moderator vote call.

## Split-vote handling

If final votes are split with no majority, the controller may run a bounded compromise attempt:

1. summarize the split;
2. test one plausible candidate;
3. ask holdouts for concrete reservations;
4. let supporters answer once;
5. make holdouts visibly switch, stay, or name one alternative;
6. optionally test a second candidate;
7. close as majority/successful/unresolved.

v3 lets the LLM render post-reservation decision lines, but only after the controller has selected the old preference, required final vote, and one grounded reason fragment. `MoveIntent.required_vote`, `old_preference`, and validation are aligned so normal LLM output should parse without repair. If a blocking issue still survives, the line is not allowed to become visible transcript evidence unless it is also countable by the state tracker.

## Majority holdout check

A majority should not always close immediately. Holdouts may get one bounded check. A holdout can accept the majority option only when `_can_shift_to` and `_should_switch_after_reservation` make that plausible.

## What must not happen

- no hidden consensus from latent preferences;
- no invented blended option;
- no forced successful outcome when a blocker remains;
- no endless negotiation loop.
