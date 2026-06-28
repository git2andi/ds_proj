---
name: feedback-implementation-process
description: One upgrade equals one independently verified issue, followed by synchronization of all active project guidance
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

Follow the exact protocol in `docs/known_failures.md` for every upgrade. One upgrade is one issue and one independently verifiable task unless the user explicitly groups issues. Include n=3 when live validation is required, and read every relevant transcript and `run.json`. Do not chain fixes without completing the full verification boundary between them.

**Why:** Multiple fixes done in sequence without validation introduced accumulating issues that were hard to attribute to specific changes. The user has stated this requirement several times.

**How to apply:** Before implementing a fix, confirm only one issue is in scope. After implementation and verification, audit and synchronize every applicable active information source: `AGENTS.md`, `CLAUDE.md`, both repository skills, active memory/index files, `docs/known_failures.md`, `README.md`, and other affected workflow docs. Historical per-fix records remain historical. Stop at the upgrade boundary unless automatic continuation was explicitly requested.

Also: do not add mechanical/forced routing logic for naturalness goals (e.g. backchannel injection). Naturalness must emerge from traits and existing dynamics, not from hard probabilities or forced branching.

Conversation quality means plain-spoken friends making a decision: casual without Gen-Z slang, corporate or academic register, mini-essays, or standalone option pitches. Traits and response-length settings must be visible without stereotypes or repeated catchphrases.

[[feedback-prompt-length]]
