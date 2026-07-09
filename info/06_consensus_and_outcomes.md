# 06 — Consensus and outcomes

The simulator has exactly three outcome labels:

- `successful`: all visible final stances support the same option;
- `majority`: enough visible support exists for one option, but not everyone supports it;
- `unresolved`: no sufficient agreement remains after bounded narrowing.

## Visible evidence rule

The final outcome is computed from visible transcript evidence. Public votes, acceptances, and parser-recognized commitments count. Private option ranks guide routing, but do not directly decide the outcome.

## Rank-aware compromise rule

The option ranks guide whether a switch is plausible:

- rank 5: the sim naturally prefers it;
- rank 4: the sim can accept it as a compromise;
- rank 3: the sim is neutral or untested;
- rank 2: the sim dislikes it but may move if discussion gives a reason;
- rank 1: the sim should not accept it unless the hard concern is addressed.

This should allow consensus when earned, preserve majority outcomes when a holdout is coherent, and leave rare unresolved cases when no option is visibly acceptable enough.

## What does not decide the outcome

Age, speech_style, and background do not directly compute the outcome. They can influence the wording of visible commitments, but the outcome parser still relies on explicit transcript evidence.
