# Consensus and outcomes

Every participant provides one visible final choice. Natural forms such as `Lab gets my vote`, `I’m going with Riverside`, or `Library for me` are valid when exactly one intended option is visible.

Outcomes are derived from the final valid votes:

- `successful`: every participant selects the same option;
- `majority`: one option receives more than half of the votes;
- `unresolved`: no option reaches a majority after the bounded protocol.

A valid majority closes immediately. Holdouts are not pressured toward unanimity.

## Stance movement

A non-hard-blocker may make another option acceptable or switch preference after a concrete public trigger, such as a resolved concern, a newly relevant benefit, or a narrowing compromise opportunity. The simulator's stubbornness controls whether it takes that opportunity.

- rank 3 may be considered directly;
- rank 2 requires the participant's own concern to be resolved or softened;
- rank 1 is not eligible;
- a hard blocker accepts and votes only for its preferred option.

The simulator action stores the target and grounded movement reason. The visible text must express the required acceptance or switch before public stance state changes. Ordinary voluntary movement is not silently forced through a fallback: after failed generation and repair it is dropped and recorded. Mandatory movement statements and formal votes may use concise protocol fallbacks.

A changed vote is valid when the intended new option is unambiguous. A short explanatory bridge improves readability, but an otherwise clear vote is not rejected solely for lacking one fixed phrase.

## Re-voting

A second vote is allowed only when:

1. the first round has no majority; and
2. the intervening re-narrowing produces at least one visible acceptance or preference switch.

If nobody moves, the run closes unresolved instead of repeating the same votes.
