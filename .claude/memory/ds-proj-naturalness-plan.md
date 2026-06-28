---
name: ds-proj-naturalness-plan
description: "Historical S1-S4 snapshot; current priorities and acceptance criteria live only in docs/known_failures.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: 298c06fc-84d0-4d22-9f86-7d70671e7b6a
---

Group Discussion Simulator (ds_proj). Previous rounds (F1-F10, O1-O12, R1-R7) all resolved and committed.

**Fixed rounds (committed):**
- Rounds 1-3: Structural refactor, O1-O12 naturalness, follow-up fixes
- R1-R7: Stock phrase rewrite, farewell fix, alias generation, addressee routing, agreement-loop breaker, shared_context, diverse names

**Historical snapshot:** S1-S4 were identified on 2026-06-25 and the attempted changes were reverted. These labels are not the current backlog. Use `docs/known_failures.md` as the only current priority list and do not implement an item from this file directly.

- S1: `shared_context` lacks explicit unknowns and a do-not-invent guard — speakers invent missing facts or stay abstract
- S2: No proactive `PRACTICAL_CHECK` act type — discussions stay at preference/value level for most of a run
- S3: Conditional consensus not a distinct outcome — "I'll accept X if we do Y" counted as full consensus
- S4: Minority options (C, D) abandoned after 1-3 mentions — router should force one SUPPORT/COMPARE per underrepresented option

**How to apply:** Follow the one-upgrade-at-a-time protocol in `docs/known_failures.md`. Use the provider explicitly authorized for the task; never substitute silently. The current top priority is the consolidated friend-chat, persona-expression, response-length, complexity, responsiveness, and repetition issue recorded there.
