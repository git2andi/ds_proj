---
name: project-f61-f65-fixes
description: F61-F65 fixes in this session (2026-06-27); known_failures.md now has no open items
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**F61** UNCLEAR_VOTE guidance rewrite — "State your pick without 'I'm voting for X because'" was read as "be vague." Rewritten to "Say the option name out loud and commit to it." Repair hint sharpened to match. Committed d24ebc7.

**F62** Deterministic "Considering X," opener strip — `_strip_considering_opener()` added to `clean_generated()`. Removes the dependent clause before validation runs; prevents it surviving repair. Committed 4dd578d.

**F63** HARD_BLOCKER_WRONG_VOTE check — hard-blocker voted for non-preferred option: `UNCLEAR_VOTE` didn't fire (stance was "vote", only option letter was wrong). New validation check fires as repair with dynamic hint naming the correct option. Verified across 3 guaranteed-hard-blocker runs: 0 integrity failures. Committed 2c4dc5d.

**F64** R39: OBJECT guidance now says "from the option card — don't invent flaws not mentioned there." Closes gap where epistemic rule covered positive claims but not negative invented attributes in objection turns.

**F65** R40: "that's a great question / that is a good question" added to `_ROBOTIC_TEMPLATES`. "valid point" swapped for "great question" in sim_utterance rule 3 banned list (still 9 phrases). Committed 585e075.

**Why:** All remaining items from known_failures.md open section. 

**How to apply:** `known_failures.md` now states "No open items." The next issues to investigate would come from new validation batch runs. The `evals/scenarios.yaml` file now exists (20 scenarios, n=2–7) and `run_eval.py` venv path is correct (ds_proj). Batch results: 20 runs, 0 failures, 18 warnings (all warn-level robotic templates or zero-question-density in fast-converging groups).
