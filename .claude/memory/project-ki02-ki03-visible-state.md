---
name: project-ki02-ki03-visible-state
description: KI02/KI03 visible outcomes and moderator targeting resolved 2026-06-30
metadata:
  node_type: memory
  type: project
---

KI02 and KI03 resolved as one explicitly grouped upgrade on 2026-06-30.

`ConsensusManager` now scans visible support across every option. The controller candidate, hidden preferences, and routing leans cannot select `successful`, `majority`, or `unresolved`. Successful requires unanimous visible support; majority uses the configured visible-support threshold; otherwise the outcome is unresolved.

Shared scoring helpers classify each participant from transcript evidence as a visible supporter, an actual alternative/objection holder, or someone missing an explicit commitment. Moderator prompts ask the last group only for confirmation. Discussion summaries use transcript-visible preferences rather than mutable controller leans. Majority/unresolved closures and majority farewells preserve the supporter/non-supporter distinction.

Controlled GPT validation covered n=2, n=3, n=5, and n=7. Three runs were successful, one majority, and one unresolved across 160 participant turns. Every outcome matched visible commitments. Live holdout prompts targeted only actual alternative voters; the missing-commitment branch is covered deterministically. Final tests: 31 passed plus 5 subtests.
