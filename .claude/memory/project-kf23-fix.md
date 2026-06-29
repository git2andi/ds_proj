---
name: project-kf23-fix
description: KF23 setup reliability fix — minimal persona schema removes score/list contradiction failures
metadata: 
  node_type: memory
  type: project
  originSessionId: bc16d197-d922-4442-885f-d07980d8a535
---

KF23 resolved 2026-06-29. Persona model stripped to `preferred_options: list[str]`, `rejection: str | None`, `rejection_reason: str`, `background`, `private_goal`. All score-generation, acceptable-option lists, soft/hard rejection lists, role, speech_style, main_concern, reasons, reservation, reconsider_if fields removed from `Persona`. `_build_scores()` and `_postprocess_personas()` deleted entirely.

**Why:** LLM had to satisfy score/list coherence simultaneously across 3+ personas; ~50% GPT failure rate was blocking P0 validation. Minimal schema has no cross-field consistency requirements the LLM can violate.

**How to apply:** Persona refactor is complete. Runtime rejection initialized from `persona.rejection` in `initialise_state()`. `persona.preferred_option` property (returns `preferred_options[0]`) preserved for backward compat. GPT run `20260629_001500_154989` validated successfully first attempt.
