---
name: project-r20-repeated-start
description: R20 fix — opener feedback in speaker card + REPEATED_START escalated to repair + dynamic repair hint (2026-06-27)
metadata: 
  node_type: memory
  type: project
  originSessionId: f147714a-bac6-4ee0-b308-a3b097235483
---

R20: Three-part fix for repeated openers within a single speaker:
1. Last 2 turn openers shown in `runtime_speaker_card` with "start differently" instruction
2. REPEATED_START severity escalated from warn → repair (forces LLM rewrite)
3. Dynamic repair hint: names the exact repeated phrase instead of generic advice

**Why:** Generic "vary your opener" instruction gave no context; model drifted to same opener anyway. Naming the exact phrase ("don't start with 'Do they offer'") is concrete enough for llama3.3 to comply.

**How to apply:** REPEATED_START hits dropped from 11-25/batch (warn only) to 0-3 (repair triggered). If openers regress, check `runtime_speaker_card` opener logic and `_REPAIR_HINTS["REPEATED_START"]` in `prompts.py`.
