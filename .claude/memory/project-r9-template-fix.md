---
name: project-r9-template-fix
description: R9 fix — template leakage from guidance string examples; root cause and prevention rule
metadata: 
  node_type: memory
  type: project
  originSessionId: 61f5108c-d2c8-409d-b209-5586e337a8b2
---

R9 (2026-06-26): All repeating turn-templates traced to guidance strings literally containing the pattern as a quoted example. "Frame it as 'what if we...'" → every speaker said "What if we"; "a quick '+1' or 'works for me' is enough" → speakers said "works for me"; "Try: 'That only works if...', 'One option is...'" → two consecutive runs copied those exact phrases.

**Why:** llama3.3 treats anything in quotes inside a guidance string as a preferred phrasing and reproduces it across all topics and speakers.

**How to apply:** When editing prompts.py, never include quoted example phrases in guidance strings (not even as "try: '...'" hints). Describe the desired behavior only. If the model is overusing a phrase, add it to _ROBOTIC_TEMPLATES for detection and remove it from any guidance string. See [[feedback-prompt-length]].

Fixed locations (all in `src/prompts.py`):
- `_concession_bridge()`: rewrote to use persona-specific conditional bridges
- `_face_work()`: removed "Frame it as 'what if we...'"
- `_move_guidance()` PROPOSE_COMPROMISE: removed "Try: 'That only works if...', 'One option is...'"
- `_move_guidance()` ACCEPT/VOTE/REACT/COMPARE: removed seeded phrases
- `validation.py`: added `seems like a good fit` and `still beats/wins` to `_ROBOTIC_TEMPLATES`; added `what_if_opener` and `wait_what_about` to `_FRAME_PATTERNS`
