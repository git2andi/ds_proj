---
name: project-r12-answer-echo
description: R12 fix — ANSWER act had no guidance; model echoed unanswerable questions back instead of hedging
metadata:
  node_type: memory
  type: project
  originSessionId: laptop-session-20260626
---

R12 (2026-06-26): When routed to ANSWER a question that the option cards don't cover, the model copied the question back verbatim — producing 3-turn echo chains (e.g. Tala→Diego→Tala all asking "do they have enough train cars?").

**Why:** `ActType.ANSWER` had no case in `_move_guidance()`, falling through to the default "Respond to the last point directly." With no information from the cards, the model re-asked the question.

**How to apply:** Fixed by adding an explicit ANSWER case in `src/prompts.py` `_move_guidance()`: "Answer if the option cards cover it. If they don't, say you're not sure and move on — don't repeat the question back." Residual: a 2-turn echo can still occur when the echoing turn is NOT ANSWER-routed (e.g. a spontaneous ASK about another speaker's option). This is mild (warn-level GROUP_REPETITION only) and not the 3-turn repair-triggering chain. See also [[project-r10-invented-context]].
