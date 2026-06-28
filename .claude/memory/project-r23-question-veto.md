---
name: project-r23-question-veto
description: "R23 fix — hard ASK veto in router when previous turn ended with \"?\" (2026-06-27)"
metadata: 
  node_type: memory
  type: project
  originSessionId: f147714a-bac6-4ee0-b308-a3b097235483
---

R23: Hard-zero `probs[ActType.ASK.value]` in `_select_act` in `router.py` when `recent[0].text` contains "?". This prevents ASK-routed back-to-back question turns from different speakers. The prior `ask_after_question_damping=0.40` was too soft.

**Why:** Two consecutive turns ending with "?" left questions unanswered and felt unnatural. The soft damping let it through; only a hard veto stops it reliably.

**How to apply:** If back-to-back question routing returns, check `_select_act` in `router.py`. Incidental questions in non-ASK turns (REACT/COMPARE) are caught downstream by the QUESTION_ECHO repair backstop (R13).
