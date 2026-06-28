---
name: project-f71-token-cost
description: Token cost regression — n=3 went from 15k to 18-28k; root causes identified 2026-06-27
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

n=3 token cost went from ~15k to 18-28k depending on repair rate.

**Two root causes:**

1. **Per-prompt growth (~20% per call, +2,750 tokens/run for clean n=3):** accumulated from trait numbers in speaker card (F44), opener feedback (F42), shared context (F29), Rule 3 expansion. Clean runs (0-1 repairs) now cost ~18k — this is expected and largely unavoidable cost of better features.

2. **Repair call proliferation (main driver of 25k+ runs):** POSSESSIVE_SUBJECT (F41) and REPEATED_START (F42) were escalated from warn to repair. These fire 3-12 times per run. Each repair call costs ~1k tokens. Token data by repair bucket: 0-1 repairs → 18k, 5-9 repairs → 25k, 10+ repairs → 28k.

**Why:** Repair escalations were intended to stop specific patterns, but they fire so frequently that they dominate token cost.

**How to apply:** F71 is the next priority fix. Options: (a) down-escalate POSSESSIVE_SUBJECT back to warn — guidance in Rule 1 already bans it; the text is readable even with a possessive opener; (b) same for REPEATED_START; (c) improve guidance to reduce pattern frequency. Consider: is the quality improvement from repairing these worth ~1k tokens per occurrence?

[[project-r30-r31-fixes]]
