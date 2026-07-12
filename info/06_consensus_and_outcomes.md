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

## Current commitment and split-candidate rules

Each participant has one active formal vote. A visible switch replaces the previous active vote; old votes remain trace history only and cannot continue backing the old option. Runtime state and the transcript-derived last-vote map are checked for consistency.

After equal formal vote counts, the compromise candidate is selected from visible discussion history: the tied option with the most positive accepted discussion mentions is tested. Positive mentions include visible support, favored comparisons, explicit softening/acceptance, and existing-option proposals; pure mentions and concerns do not count. Only the minimum number of plausible movers needed for a majority are asked to reconsider.
 A mover's final repair commitment is restricted to the tested candidate or the mover's current vote, so the repair cannot manufacture a new third-option split. Plausibility uses private resistance only for routing and is improved by accepted visible openness toward the candidate; it never counts as a public vote.

A compromise success is recorded only when a no-majority split repair produces at least one visible formal switch and the resulting tally becomes majority or unanimous. Merely running repair logic is not a successful compromise.
