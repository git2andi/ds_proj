---
name: project-r11-edges-ahead
description: R11 fix — "edges ahead" template phrase seeded by COMPARE guidance; eliminated 2026-06-26
metadata:
  node_type: memory
  type: project
  originSessionId: laptop-session-20260626
---

R11 (2026-06-26): "edges ahead" appeared 3× per run across different speakers (board game + restaurant topics). Same root cause as R9: `_move_guidance()` COMPARE case said "where yours edges ahead" — model copied verbatim.

**Why:** See [[project-r9-template-fix]]. Any phrase in guidance text (even descriptive, not quoted-example) gets treated as preferred phrasing by llama3.3.

**How to apply:** Fixed in three places — (1) COMPARE guidance in `src/prompts.py` rewritten to description only; (2) `re.compile(r"\bedges?\s+ahead\b", re.I)` added to `_ROBOTIC_TEMPLATES` in `src/validation.py`; (3) 'edges ahead' added to banned stock-phrases list in rule 3 of `sim_utterance`. When writing COMPARE guidance in future, describe the move ("acknowledge theirs, then give your reason") without using any competitive-comparison phrase.
