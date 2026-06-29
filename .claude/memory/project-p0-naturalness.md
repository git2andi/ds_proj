---
name: project-p0-naturalness
description: P0 friend-chat naturalness improvements resolved 2026-06-29; key prompt changes across two fix waves
metadata:
  node_type: memory
  type: project
  originSessionId: bc16d197-d922-4442-885f-d07980d8a535
---

P0 (KF08/KF09/KF14/KF24) naturalness improvements; extended fix and wave-2 sprint both 2026-06-29.

**Why:** Discussions read as option pitches / mini-essays; greetings formulaic; persona traits not distinguishable; moderator closure asking questions; farewells opening with "[Option name] it is."

**Key changes — initial fix (prompts.py):**
- `runtime_speaker_card` includes `persona.background`, `persona.private_goal`, and a "Voice:" register note from `_voice_register()` (blunt/warm/cautious/firm/very-short — shows in sentence-level style)
- `sim_utterance` first line: "write like you'd text a friend, not like you're presenting"
- `_verbosity_note` now includes "contractions OK, no semicolons, no formal transitions" at every response_length level
- SUPPORT/OPENING/COMPARE guidance: "Start with 'I'/'My' — not the option name"
- `farewell_line`: "Don't open with the option name. Lead with your reaction."
- `moderator_agreement_prompt`: "Declarative sentence — no question, no 'should we'"
- `moderator_closure_prompt` (successful): "Declarative sentence — no question"

**Key changes — initial fix (dialogue.py):**
- `_strip_body_semicolons()`: replaces `;` in body with ` —`, trailer-aware (protects `[act=...; opt=...; stance=...]`)
- `_surface_cleanup`: strips `^[A-Z]=` option-letter prefix

**Key changes — wave-2 sprint (2026-06-29):**
- `sim_utterance` opener ban moved to top of prompt: "Don't open with 'I get that', 'I hear you', or 'True, but'" — appears before option facts for LLM salience
- VOTE/ACCEPT guidance: "No trailing 'though', 'provided', 'if', or 'now'" — eliminates hedged votes and "I choose X now" procedural form
- `_VISIBLE_COMMITMENT_CUES` expanded in `validation.py` to catch "I'll take", "I'm in", "that works", "I'm going with"
- COMPARE guidance changed from two-sided to one-sided: "say the one thing your option has that matters more for YOUR situation — don't weigh both sides"
- OBJECT and PUSH_BACK guidance: explicit "no 'I get that' lead-in" added
- `_verbosity_note`: added "no 'we should', no 'we need to'" and "One thought — don't add a balancing clause"
- Grounding rule: "skip it entirely" instead of "say 'not sure'" — eliminates speculative questions
- `runtime_speaker_card` `already_said`: expanded to last 2 points to suppress self-repetition

**Post-wave-2 results** (8 validation runs):
- Repair rate: 0.04–0.14 (was 0.25–0.39)
- INVENTED_OPTION_ATTRIBUTE: ~0.2 per run (was 4+)
- UNCLEAR_VOTE: ~1.5 per run average (was 6+)
- "I get that" opener: zero
- "Hey everyone" greeting: zero

**How to apply:** If naturalness regresses, check `_voice_register`, `_verbosity_note`, and VOTE/ACCEPT/COMPARE/REACT guidance in `prompts.py` first. Background in speaker card + voice register is the biggest lever for trait differentiation. Keep opener ban at the TOP of `sim_utterance` for LLM salience.
