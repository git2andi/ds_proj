---
name: project-r14-still-pick
description: R14 fix — "still pick" template seeded by R11's COMPARE guidance replacement phrase
metadata:
  type: project
---

"I'd still pick X because..." appeared 3–4× per run in 4/6 new validation runs (farewell gift, sci-fi, framework, TV series). Root cause: R11's COMPARE guidance fix ("why you'd still pick yours") introduced the next template — same R9/R11 class of phrase-seeding.

**Why:** llama3.3 lifts exact verb phrases from guidance strings and reuses them verbatim across all speakers, making every participant sound identical.

**How to apply:** Any time COMPARE (or other act) guidance is rewritten to fix a template, inspect the new wording for liftable verb phrases before committing. Guidance strings must describe behavior without providing a sentence structure. Also add the new template to `_ROBOTIC_TEMPLATES` and the rule 3 banned list as a deterministic backstop.

Fixes: (1) COMPARE guidance → "One genuine strength of theirs; one concrete reason yours fits you better. No attribute lists, no templates."; (2) `re.compile(r"\bstill\s+pick\b", re.I)` added to `_ROBOTIC_TEMPLATES`; (3) 'still pick' added to banned phrases in rule 3.

Related: [[project-r9-template-fix]], [[project-r11-edges-ahead]]
