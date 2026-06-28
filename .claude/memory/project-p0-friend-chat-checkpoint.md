---
name: project-p0-friend-chat-checkpoint
description: P0 prompt/routing checkpoint; broad GPT validation blocked by KF23 setup failures
metadata:
  node_type: memory
  type: project
---

On 2026-06-28, the P0 friend-chat upgrade added one-local-job prompts, exact routed response targets, behavior-based persona cues without generated role/style labels, monotonic response budgets capped at 48 words, two-sentence limits for long turns, and focused decision repairs. The visible-commitment cue set now accepts the unambiguous `select` verb family after GPT repeatedly produced visible selections that the parser rejected.

Trait follow-up: `TraitProfile` now stores only the five OCEAN traits. Response length and compromise willingness are derived; routing, pacing, and prompt behavior no longer depend on independently generated initiative, directness, or detail values.

Completed GPT runs: `20260628_214821_616924`, `20260628_215131_265947`, and `20260628_215601_400208`. These were iterative evidence, not final acceptance. No post-parser-fix dialogue completed because repeated score/list setup contradictions exhausted both retries. Keep P0 open; fix KF23 independently, then run a fresh mandatory n=3 plus n=2-7 spread and read every transcript/run.json before closing P0.
