---
name: project-r15-backtoback-routing
description: R15 fix — router back-to-back speaker at opening→answer boundary
metadata:
  type: project
---

A speaker could get two consecutive turns when they were next in the opening queue AND the router immediately routed them to ANSWER a question from another speaker's opening. After the last opening statement is given, the phase transitions to DISCUSSION and `next_intent()` checks `state.open_questions` — but had no guard for "target just spoke."

**Why:** `_best_answerer()` selects by option champion preference with no recency check; `next_intent()` returned the answer intent unconditionally.

**How to apply:** The fix is a 2-line guard in `router.py` `next_intent()` before the answer `MoveIntent` return: if `target == last_participant.speaker_id` and group size > 1, skip the answer this cycle (question stays queued, picked up next turn after someone else speaks).

Related: [[project-r13-question-echo-backstop]]
