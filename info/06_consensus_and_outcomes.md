# 06 — Consensus and outcomes

The simulator has exactly three outcome labels:

- `successful`: all visible final stances support the same option;
- `majority`: enough visible support exists for one option, but not everyone supports it;
- `unresolved`: no sufficient agreement remains after bounded narrowing.

## Visible evidence rule

The final outcome is computed from visible transcript evidence. Public votes, acceptances, and parser-recognized commitments count. Private option ranks guide routing, but do not directly decide the outcome.

## Rank-aware compromise rule

The option ranks guide whether a switch is plausible:

- rank 4: the sim naturally supports it;
- rank 3: the sim can accept it as compromise;
- rank 2: the sim may move toward it if discussion gives a reason;
- rank 1: the sim should resist unless the concern is addressed;
- rank 0: the sim should not accept it.

This should allow consensus when earned, preserve majority outcomes when a holdout is coherent, and leave rare unresolved cases when no option is visibly acceptable enough.
