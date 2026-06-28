---
name: project-r34-r35-fixes
description: R34 question lifecycle after hedge; R35 persona/option consistency constraint (2026-06-27)
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**R34:** Open questions are no longer cleared silently after one hedge answer. `hedge_count: int = 0` added to `OpenQuestion` (models.py). In `_update_questions` (dialogue.py): on ANSWER turn, if `_is_hedge_answer()` matches and `hedge_count == 0`, increment and skip the clear — question stays open for one more routing cycle. Second hedge or real answer clears it. `_is_hedge_answer()` regex covers "not sure", "can't confirm", "we'd need to check", "no idea", etc.

**R35:** `setup_personas` prompt (prompts.py) adds: "Persona consistency: a participant's role, main_concern, and preferred_option must be compatible. A 'quiet reader' prefers a calm low-key option; an adventure seeker prefers a high-energy option. Never assign someone a preferred option whose core attributes directly contradict their stated role and concern unless their backstory explicitly explains the contradiction."

**Why:** R34 prevents single-hedge answers from burying unanswered questions forever. R35 prevents setup generating internally contradictory personas (quiet reader wants arcade) which poison the whole discussion.

**How to apply:** Both committed in 09570b2 (2026-06-27). No further action needed unless similar issues surface.
