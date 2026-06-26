---
name: ds-proj-naturalness-plan
description: "Group Discussion Simulator — S1-S4 open issues identified 2026-06-25; code reverted, pending re-implementation with uni provider only"
metadata: 
  node_type: memory
  type: project
  originSessionId: 298c06fc-84d0-4d22-9f86-7d70671e7b6a
---

Group Discussion Simulator (ds_proj). Previous rounds (F1-F10, O1-O12, R1-R7) all resolved and committed.

**Fixed rounds (committed):**
- Rounds 1-3: Structural refactor, O1-O12 naturalness, follow-up fixes
- R1-R7: Stock phrase rewrite, farewell fix, alias generation, addressee routing, agreement-loop breaker, shared_context, diverse names

**S1-S4: identified 2026-06-25, NOT yet implemented.**
Session work was reverted because validation used groq instead of uni. See `docs/known_failures.md` for full issue descriptions.

- S1: `shared_context` lacks explicit unknowns and a do-not-invent guard — speakers invent missing facts or stay abstract
- S2: No proactive `PRACTICAL_CHECK` act type — discussions stay at preference/value level for most of a run
- S3: Conditional consensus not a distinct outcome — "I'll accept X if we do Y" counted as full consensus
- S4: Minority options (C, D) abandoned after 1-3 mentions — router should force one SUPPORT/COMPARE per underrepresented option

**How to apply:** Follow `docs/known_failures.md` process. One item at a time. Provider: always `uni` — see [[feedback-uni-provider]]. Never fall back to groq autonomously.
