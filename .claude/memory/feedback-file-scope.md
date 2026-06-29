---
name: feedback-file-scope
description: Only CLAUDE.md is in scope for edits; AGENTS.md and .agents/ must not be touched
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bc16d197-d922-4442-885f-d07980d8a535
---

Only ever read or edit `CLAUDE.md`. Never touch `AGENTS.md` or anything under `.agents/`.

**Why:** The user manages AGENTS.md and .agents/ separately (for Codex). Touching them without being asked causes unintended divergence.

**How to apply:** When updating project documentation or skills, only write to CLAUDE.md and `.claude/`. Leave AGENTS.md and .agents/ exactly as they are.
