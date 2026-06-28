---
name: project-r13-question-echo-backstop
description: R13 fix — deterministic repair backstop for question echoes; GROUP_REPETITION on question pairs escalated to QUESTION_ECHO repair (2026-06-26)
metadata:
  node_type: memory
  type: project
  originSessionId: laptop-session-20260626
---

R13 (2026-06-26): Prompt guidance alone (R12) prevented 3-turn ANSWER-routed echo chains but a 2-turn echo could still occur when the echoing turn was not ANSWER-routed (GROUP_REPETITION fired at warn level → no repair).

**Why:** GROUP_REPETITION is always warn-level, so it never triggers a repair call. Echoing a question is categorically wrong (you're returning a question for a question with no information) and always warrants repair, unlike echoing a statement where some repetition is natural.

**How to apply:** In `src/validation.py` `_check_repetition()`: when the jaccard-similarity check fires and both the current text and the matched recent turn contain "?", issue `QUESTION_ECHO` at `"repair"` level instead of `GROUP_REPETITION` at `"warn"`. The `QUESTION_ECHO` repair hint in `src/prompts.py` `_REPAIR_HINTS` says: "don't re-ask what was just asked — if the cards don't say, hedge ('not sure, we'd need to check') and move on". See [[project-r12-answer-echo]] for the companion prompt fix.
