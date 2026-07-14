# Voting and outcomes

Every participant receives one formal vote obligation. Its simulator selects one valid option. The action and rendered text must identify the same option. A hard blocker always selects its sole preferred option.

When a vote differs from the latest public preference, the action includes a structured switch and the realization must visibly bridge the old preference to the new vote. Formal vote switches use this single vote-specific contract rather than also being forced through the ordinary discussion-switch rule. Natural old-to-new language and an explicit changed-mind statement are accepted when the prior public preference is known; multiple or contradictory vote targets remain invalid. An unclear vote remains invalid, and the runtime does not invent a replacement.

Outcomes are computed directly from structured valid votes:

```text
successful: every participant casts the same valid vote
majority:   one option reaches the configured majority threshold, but not unanimously
unresolved: no option reaches a majority after the single bounded re-vote
```

A valid majority closes immediately. There is no holdout pressure or majority-to-unanimity repair.

Only a first-round no-majority result returns briefly to narrowing. The runtime then requests exactly one re-vote. A second no-majority result closes unresolved.
