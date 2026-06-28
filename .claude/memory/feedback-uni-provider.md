---
name: feedback-uni-provider
description: Use only the provider explicitly authorized for the current task; never substitute silently
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6ddd3417-1035-4682-a615-2e64385a7390
---

Use only the provider explicitly authorized in the current task. The standing default in `AGENTS.md` is `uni`, but an explicit current instruction may authorize another supported provider. The current upgrade sequence is explicitly authorized for `gpt`. Never switch endpoints because the selected provider is slow or unavailable.

**Why:** Switching to groq without asking invalidated an entire session of validation work — groq and uni run different models, so behavioral results from groq are not meaningful evidence for uni behavior. The user was explicit about this requirement.

**How to apply:** If the authorized provider is unreachable or timing out, stop and report the failure. Do not silently switch providers.
