---
name: project-r30-r31-fixes
description: R30/R31 fixes — ANSWER exhaustiveness rule prevents hallucinated card facts; len=N added to trait card (2026-06-27)
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**R30: ANSWER hallucination of non-card facts** — ANSWER guidance in `sim_utterance` now explicitly states that card attributes are exhaustive: "The card attributes are exhaustive — anything not listed is unknown. Answer only from what the card explicitly states ... never invent facts." Previous guidance just said "if the cards don't cover it, say you're not sure" — the model ignored this because it treated cards as partial, filling gaps from real-world knowledge. Validated: "can't confirm that", "unknown to me", "we'd have to look it up" appearing correctly; no invented service/facility claims in 7 follow-up runs.

**R31 historical note:** response length was originally sampled independently. It is now derived deterministically from extraversion, conscientiousness, and openness, and supplied to the runtime verbosity guidance without becoming a sixth generated trait.

**Why:** R30 prevents confident hallucinated answers to unanswerable questions. R31 gives the model clearer calibration for turn length.

**How to apply:** If ANSWER hallucination recurs, check the ANSWER grounding guidance. If verbosity calibration seems off, inspect the `TraitProfile.response_length` derivation and runtime verbosity guidance.
