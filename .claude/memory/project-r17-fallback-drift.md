---
name: project-r17-fallback-drift
description: R17 fix — finalize() fallback used drifted leading_option instead of candidate_option
metadata:
  type: project
---

`finalize()` called `leading_candidate(state)` = `leading_option(state)` at closure time. After long confirmation phases with many PROPOSE_COMPROMISE turns, `current_preference` values drift as speakers propose different options — causing `leading_option` to return a low-support option instead of the actual vote candidate, triggering `unresolved` when 2/3 had clearly voted for the real candidate.

**Why:** `leading_option` is a live `option_support` score that reacts to `current_preference` changes during proposals. It wasn't intended to be used as the fallback target — `state.candidate_option` is the correct anchor (set when everyone voted in the narrowing round).

**How to apply:** The fix is one line: `candidate = state.candidate_option or self.leading_candidate(state)`. The confirmed candidate (from the vote round) takes priority for the fallback check; `leading_option` is only used if no candidate was set (group never reached narrowing).

Related: [[project-r16-question-cutoff]]
