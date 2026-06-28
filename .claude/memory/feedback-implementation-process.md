---
name: feedback-implementation-process
description: Must follow one-fix-at-a-time cycle with mandatory n=3 run between each fix; no bulk fixing
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

Follow the exact protocol in known_failures.md for every fix: one item at a time, pytest, n=3 run mandatory, read transcript, then next item. Do not chain multiple fixes in a single session without completing the validation cycle between each one.

**Why:** Multiple fixes done in sequence without validation introduced accumulating issues that were hard to attribute to specific changes. The user has stated this requirement several times.

**How to apply:** Before implementing any fix, confirm only one item is being worked. After implementation, run pytest, do the n=3 run, read the transcript. Only then move to the next item. Archive logs before starting a new batch.

Also: do not add mechanical/forced routing logic for naturalness goals (e.g. backchannel injection). Naturalness must emerge from traits and existing dynamics, not from hard probabilities or forced branching.

[[feedback-prompt-length]]
