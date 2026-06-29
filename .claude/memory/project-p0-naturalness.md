---
name: project-p0-naturalness
description: P0 friend-chat naturalness improvements resolved 2026-06-29; key prompt changes
metadata: 
  node_type: memory
  type: project
  originSessionId: bc16d197-d922-4442-885f-d07980d8a535
---

P0 (KF08/KF09/KF14/KF24) naturalness improvements; extended fix 2026-06-29.

**Why:** Discussions read as option pitches / mini-essays; greetings formulaic; persona traits not distinguishable; moderator closure asking questions; farewells opening with "[Option name] it is."

**Key changes (prompts.py):**
- `runtime_speaker_card` includes `persona.background`, `persona.private_goal`, and a "Voice:" register note from `_voice_register()` (blunt/warm/cautious/firm/very-short — shows in sentence-level style)
- `sim_utterance` first line: "write like you'd text a friend, not like you're presenting"
- `_verbosity_note` now includes "contractions OK, no semicolons, no formal transitions" at every response_length level
- SUPPORT/OPENING/COMPARE guidance: "Start with 'I'/'My' — not the option name"
- `farewell_line`: "Don't open with the option name. Lead with your reaction."
- `moderator_agreement_prompt`: "Declarative sentence — no question, no 'should we'"
- `moderator_closure_prompt` (successful): "Declarative sentence — no question"

**Key changes (dialogue.py):**
- `_strip_body_semicolons()`: replaces `;` in body with ` —`, trailer-aware (protects `[act=...; opt=...; stance=...]`)
- `_surface_cleanup`: strips `^[A-Z]=` option-letter prefix

**How to apply:** If naturalness regresses, check `_voice_register`, `_verbosity_note`, and the farewell/agreement prompt instructions first. Background in speaker card + voice register is the biggest lever for trait differentiation.
