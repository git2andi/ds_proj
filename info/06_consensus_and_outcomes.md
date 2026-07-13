# 06 — Consensus and outcomes

The simulator has exactly three outcome labels:

- `successful`: all visible final stances support the same option;
- `majority`: enough visible support exists for one option, but not everyone supports it;
- `unresolved`: no sufficient agreement remains after bounded narrowing.

## Visible evidence rule

The final outcome is computed from visible transcript evidence, gated by phase: only clear commitments made during the formal `voting` and `compromise_repair` phases count. Vote evidence comes from each turn's validated visible commitment evidence (natural menu-less wording included), which has already passed the deterministic critical parser: one target, no unresolved prerequisite, no question masquerading as a vote, required-target alignment, and rejected-option protection. A repair-phase concession replaces an earlier formal vote; opening leans and discussion-phase acceptances update public stance but never silently become final votes. Private option ranks guide each simulator's own vote decision, but the framework never chooses a participant's vote to engineer consensus and does not directly decide the outcome. Blocked (dropped) turns contribute no evidence.

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

## Current commitment and split-candidate rules

Each participant has one active formal vote. A visible switch replaces the previous active vote; old votes remain trace history only and cannot continue backing the old option. Runtime state and the transcript-derived last-vote map are checked for consistency.

After equal formal vote counts, the framework selects which existing option to test from visible votes and evidence only: positive accepted discussion evidence and existing-option proposals increase priority, visible objection load lowers it, and stable option order resolves remaining ties. The framework schedules visible dissenters for one bounded reconsideration opportunity, but whether each moves is that simulator's own re-vote decision — the framework never computes the switch. A simulator's final repair commitment is restricted to the tested candidate or its own current vote, so the repair cannot manufacture a new third-option split. Movement reads only that simulator's own ranks, own live concerns, switch resistance, and its own visible openness toward the candidate; a hard blocker or rank-1 option can never switch. No unanimity is fabricated: majority and unresolved outcomes remain valid when simulators do not move.

A compromise success is recorded only when a no-majority split repair produces at least one visible formal switch and the resulting tally becomes majority or unanimous. Merely running repair logic is not a successful compromise.
