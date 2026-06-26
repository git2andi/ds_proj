---
name: feedback-uni-provider
description: Always use uni provider in this project — never switch to groq or gemini without explicit user instruction
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6ddd3417-1035-4682-a615-2e64385a7390
---

Only ever use the `uni` provider (Bamberg Ollama endpoint) for validation runs in this project. Never autonomously switch to `groq` or `gemini`, even if uni is timing out or slow.

**Why:** Switching to groq without asking invalidated an entire session of validation work — groq and uni run different models, so behavioral results from groq are not meaningful evidence for uni behavior. The user was explicit about this requirement.

**How to apply:** If uni is unreachable or timing out, stop and tell the user. Do not silently switch providers. Do not run any validation run on any provider other than uni unless the user explicitly says so in the current message.
