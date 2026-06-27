---
name: project-r27-r28-fixes
description: R27/R28 fixes — SELF_REPETITION now covers ACCEPT intents; epistemic phrase variety expanded to prevent chorus (2026-06-27)
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**R27: SELF_REPETITION skipped for ACCEPT/REJECT** — `_check_repetition` in `validation.py` returned early before consulting `already_said` when `intent.act in {ACCEPT, REJECT}`. The original comment said "confirmations are naturally similar" but the exemption was too broad: it allowed the exact same sentence to repeat 3× on ACCEPT-routed turns (Leo's "Taco Loco offers a unique twist." in restaurant run; `self_rep=0` in eval). Fixed: removed the early-return guard; SELF_REPETITION now checks `already_said` for all acts.

**R28: "we'd need to check" epistemic chorus** — Epistemic grounding guidance in `sim_utterance` (line 710 of `prompts.py`) listed exactly two alternatives ("I'm not sure" or "we'd need to check"). Model defaulted to "we'd need to check" on every uncertain turn; 7 consecutive turns in the n=6 fictional world run all ended with it. Fixed: expanded to 5 alternatives with "vary the phrasing" instruction: "say 'I'm not sure', 'can't say', 'we'd have to check', 'no idea', 'unknown to me' — vary the phrasing, never a confident claim."

**Why:** R27 let exact repeated sentences through on ACCEPT turns. R28 created a formulaic chorus that made all speakers sound identical when expressing uncertainty.

**How to apply:** If epistemic phrase variety regresses, check prompts.py line ~710 for the alternatives list. If SELF_REPETITION doesn't fire on near-identical ACCEPT turns, check that the early-return was not re-introduced in validation.py.
