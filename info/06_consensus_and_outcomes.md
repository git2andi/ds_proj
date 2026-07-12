# 06 — Consensus and outcomes

The simulator has exactly three outcome labels:

- `successful`: all visible final stances support the same option;
- `majority`: enough visible support exists for one option, but not everyone supports it;
- `unresolved`: no sufficient agreement remains after bounded narrowing.

## Visible evidence rule

The final outcome is computed from visible transcript evidence, gated by phase: only clear commitments made during the formal `voting` and `compromise_repair` phases count. Vote evidence comes from each turn's validated visible commitment evidence (natural menu-less wording included), which has already passed the deterministic critical parser: one target, no unresolved prerequisite, no question masquerading as a vote, required-target alignment, and rejected-option protection. A repair-phase concession replaces an earlier formal vote; opening leans and discussion-phase acceptances update public stance but never silently become final votes. Private option ranks guide routing, but do not directly decide the outcome. Blocked (dropped) turns contribute no evidence.

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
