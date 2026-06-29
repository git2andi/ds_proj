---
name: project-kf04-alias-contract
description: KF04 alias contract shared across builder/prompts/validation/parsing — resolved 2026-06-29
metadata: 
  node_type: memory
  type: project
  originSessionId: bc16d197-d922-4442-885f-d07980d8a535
---

KF04 (P1) alias contract unification resolved 2026-06-29.

**Why:** Builder previously accepted short names the parser couldn't resolve (e.g., "Spy" for "Codenames" — 3 chars, rejected by `OptionResolver`). Prompts showed the alias, parser ignored it, causing vote tracking failures.

**Fix:** `validated_short_alias()` in `src/aliases.py` is the single gate: words must appear in the option name, alias ≥ 4 chars, ≤ 3 words, last word not a stopword, not all generic. `deterministic_alias()` is the fallback. `short_alias_map()` handles collisions. All four consumers (builder, prompts, validation, parsing) use the same `short_alias_map()` output.

**How to apply:** If alias resolution fails, the root is always `aliases.py` — check `validated_short_alias` first. Never add a length filter outside that function.
