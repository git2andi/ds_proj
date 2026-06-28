---
name: project-kf03-fix
description: KF03 resolved 2026-06-28 — no-trailer VOTE/ACCEPT now triggers repair and cannot ghost-commit to invisible options
metadata: 
  node_type: memory
  type: project
  originSessionId: 9a9b7b55-4d66-4264-ac3a-d9d8c971642c
---

KF03 resolved: Two-part fix for binding votes/accepts being too dependent on routing intent rather than visible text.

**Part A (parsing.py `_resolve_move`):** When no trailer is present (`not move.present`) and the intent is VOTE or ACCEPT, `intent.option_focus` is no longer used as the focus option. The option must come from `option_refs` (visible text). This prevents ghost-commits where routing routes to option A but the speaker never mentions it.

**Part B (validation.py `_check_decision_clarity`):** When no trailer on a VOTE/ACCEPT decision turn, UNCLEAR_VOTE/UNCLEAR_ACCEPT repair fires. The repair prompt asks the model to confirm explicitly with a trailer `[act=vote; opt=X; stance=vote]`. Non-binding acts (COMPARE, SUPPORT, etc.) are unaffected.

**Why:** The trailer IS the commitment signal per the architecture. Without a trailer, commitment should not be inferred silently from routing intent.

**How to apply:** Next P0 is cleared (no P0 items remain). P1 items start with KF06 (moderator targeting lone holdouts).
