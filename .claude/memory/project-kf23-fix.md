---
name: project-kf23-fix
description: KF23 setup reliability — minimal persona schema plus deterministic concrete preference assignments
metadata: 
  node_type: memory
  type: project
  originSessionId: bc16d197-d922-4442-885f-d07980d8a535
---

KF23 resolved 2026-06-29. Persona model remains minimal: `preferred_options: list[str]`, `rejection: str | None`, `rejection_reason: str`, `background`, `private_goal`. All score-generation, acceptable-option lists, soft/hard rejection lists, role, speech_style, main_concern, reasons, reservation, and reconsideration fields remain removed.

Preference structure is now controller-owned through `personas.preference_distribution.shape_weights` for sizes 2–7, with optional `forced_shape` for controlled runs. The controller validates/samples a shape, generates the scenario, maps shape parts to concrete option IDs, and passes each participant one explicit required primary. The LLM only writes persona meaning around that row-local assignment. Scenario and persona retry loops are separate.

**Why:** Abstract camps required the LLM to coordinate relationships across rows, causing retries when same-camp rows diverged or different camps converged. Concrete assignments remove that relational generation task while retaining stochastic coalitions.

**Validation:** Seven deterministic tests passed, including 72,000 distribution samples, forced and invalid shapes, prompt composition, and isolated persona retry. Fourteen GPT (`gpt-4.1-mini`) setup-only generations passed on their first scenario and persona calls across sizes 2–7, including forced `2-1` and `2-2` splits. All 61 generated personas had coherent primary preferences, backgrounds, and private goals. Results: `evals/setup_eval_results.json`.
