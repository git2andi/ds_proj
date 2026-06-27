---
name: project-r29-r32-open
description: Open issues R29, R32–R38 — Considering opener (priority), ANSWER gap, earned consensus, question drift, persona contradiction, hedge templates, thin acceptance, token regression (2026-06-27)
metadata:
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**R29:** `^\s*considering\b` in `_ROBOTIC_TEMPLATES` is warn-only. Appears 1-4x/run. Fix: escalate to repair (same as F41/F42). Hint: "open with the point itself, a reaction, or a question."

**R32:** ANSWER act not in `statement_only` so "?" on ANSWER turn passes silently and re-registers as OpenQuestion. Fix: add `ActType.ANSWER` to `statement_only`.

**R33:** Fallback outcomes phrased as clean wins; speakers jump from concern to accept in one sentence. Fix: verify visible accept before finalizing; fallback closure text should reflect partial agreement.

**R34:** Questions cleared after one hedge ANSWER even though still open. Fix: `hedged=True` flag on OpenQuestion; if hedged twice, route moderator to name it explicitly.

**R35:** Persona role/concern contradicts preferred option (quiet reader preferred high-noise arcade). Fix: constraint in belief-state prompt + post-generation check.

**R36:** Hedge phrases ("do they offer", "why that matters is") now repeat like old templates. Fix: add to `_ROBOTIC_TEMPLATES`; track raised-unknowns per option to suppress re-raising.

**R37:** Speaker switches from opposition to accept in one vague clause with no condition or acknowledgment of prior objection. Fix: bridge guidance must require named concern + trade-off condition.

**R38:** 20k+ tokens/run regression. Likely cause: opener feedback + trait block + shared_context + epistemic expansion all grew the card. Fix: profile `prompts.jsonl`, trim `already_said` to 1 claim, verify full cards only on COMPARE/VOTE.

**How to apply:** See full fix candidates in docs/known_failures.md. Always implement R29 first (appears in every run).
