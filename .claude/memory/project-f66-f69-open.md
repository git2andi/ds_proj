---
name: project-f66-f69-open
description: F66-F70 status after fixes 2026-06-27; F66/F67/F70 resolved; F68 in progress; F69 logging added
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**F66** — FIXED. ACCEPT/VOTE guidance rewritten: explicit name + commit required in text. Face_work R9-class seeding phrase removed. Verified in n=3 concession run.

**F67** — FIXED. `OPTION_NAME_OPENER` warn-level check added to `MessageValidator`; Rule 2 updated: "Don't open with just an option name." VOTE/ACCEPT exempt. Tests added.

**F68** — REVERTED. The forced injection/backchannel repair approach was wrong — mechanical, not natural. Reverted all code. Known-failures entry rewritten: must emerge from low `response_length`/`detail` traits on REACT acts naturally, not from forced routing. Observe first before implementing.

**F69** — PARTIALLY FIXED. Diagnosed: INVENTED_OPTION_ATTRIBUTE dominates (model invents capacity/prices on PROPOSE_COMPROMISE despite epistemic rule). Fixed: "No inventing specific prices, sizes, or details" added to PROPOSE_COMPROMISE guidance. Secondary loop issue: repair-generated questions can echo across multiple speakers (separate sub-bug). repair_trigger_codes now in run.json for future diagnosis.

**F70** — FIXED. `[Name] brings/makes/raises a [adj] point/concern` pattern added to `_ROBOTIC_TEMPLATES` and Rule 3 banned list. Tests added.

**Setup failures on personal topics** — "Plan a weekend road trip route" and similar personal-preference topics cause setup failure: LLM assigns too-strong preferences, violating `non_blocker_min_acceptable=2`. Not filed as F yet; not critical.

**Why:** Found in 10-run batch after stubborn trait refactor. Documented 2026-06-27.

**How to apply:** F68 still needs a verification run. F69 will produce diagnostic data in next n≥5 run. After F68 is confirmed, investigate F69 repair codes and decide if fix is needed.
