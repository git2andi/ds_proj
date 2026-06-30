---
name: feedback-file-scope
description: AGENTS.md and .agents/ are Codex-owned — only touch them when the user explicitly asks for a sync
metadata: 
  node_node: memory
  type: feedback
  originSessionId: bc16d197-d922-4442-885f-d07980d8a535
---

Do not proactively edit `AGENTS.md` or anything under `.agents/` without explicit user instruction. When the user explicitly asks for a documentation sync, update AGENTS.md and the `.agents/` skill files to match CLAUDE.md and `.claude/skills/`.

**Why:** The user manages AGENTS.md and .agents/ separately (for Codex). Unsolicited edits cause unintended divergence; but when syncing is requested, all files must stay consistent.

**How to apply:** Routine upgrades → only write to CLAUDE.md and `.claude/`. When the user asks for a full sync or explicitly mentions AGENTS.md / Codex files → update those too.
