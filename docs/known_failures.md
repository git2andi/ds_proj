# Known Failures — Open Issues Only

Last updated: 2026-06-28 (KF11–KF21 batch).
Scope of this file: only issues that still appear relevant after reading the currently supplied code files and the latest transcript observations. Fixed/history entries were removed. This is a working backlog, ordered by implementation priority.

The goal is not to add more knobs or more prompt text by default. Prefer small controller/parser/state fixes over additional prompt rules, because the current code already has many prompt-level guardrails.

---

## Validation process for each change

1. Move existing logs form logs/ into logs/archive/
2. Pick the highest-priority open item.
3. Implement the smallest change that should improve that item.
4. Validate with one mandatory n=3 run, then 5–6 additional runs across n=2–7 using random topics. Provider: `uni`.
5. Read transcripts, not only metrics. Check whether the fix improved the target failure without creating regressions.
6. Only close an issue when the visible dialogue improved and no obvious regression appeared.
7. If a change merely moves the problem elsewhere, keep the issue open and update the diagnosis.
8. Update `CLAUDE.md`, memory and relevant project notes after each successful implementation pass.

---


## P1 — Natural moderation and dialogue flow

### KF09 — Local conversation flow still needs more human uptake

**Problem:** Many turns still read as standalone argument cards. Sims often state their own position rather than directly answering, briefly acknowledging, challenging, or building on the previous turn. Backchannels and short reactions appear sometimes, but not as a stable part of the flow.

**Why it matters:** Naturalness depends less on longer prompts and more on local uptake: answering the previous question, naming the actual concern, making small concessions, and reacting in short turns when appropriate.

**Partial fix (2026-06-28):** ANSWER directive strengthened. When the responding-to turn ends with "?", `_responding_to_line()` now shows "Answer this (or say 'not sure'):" instead of "Responding to:". ANSWER guidance rewritten to "Address the question above first — say what the cards show, or 'not sure' if not covered." This produced a consistent 10–15pp improvement in `responsive` rate (n=3=37%, n=4=35%, n=6=40%, vs 24–25% pre-fix). "Not sure" responses to unanswerable questions now appear naturally.

**Remaining issues:**
- Act mismatch: router routes ASK but LLM generates a question in an OBJECT/SUPPORT trailer → question isn't registered as open → ANSWER never fires. Affects responsive rate in runs where the speaker produces incidental questions.
- Concern response: when the previous turn raises a concern (not a "?" question), no explicit routing/guidance to address it directly.
- Every non-REACT turn still tends toward a full reason + option pitch, even in discussion where a brief acknowledgment would suffice.

**Relevant code:** `src/router.py`: answer/response routing; `src/prompts.py`: `_responding_to_line()`, ANSWER guidance in `_move_guidance()`; `src/dialogue.py`: open-question tracking.

---

---

## P2 — Dialogue substance and grounding

### KF11 — Option coverage gate before narrowing (implemented 2026-06-28)

**Fix applied:** `DialogueController._can_start_narrowing()` now requires the leading option to have at least one substantive reason (`coverage[lead].reasons >= 1`) before natural narrowing. Forced narrowing (moderator/turn cap) and full early convergence still bypass the gate. `_progress_snapshot()` also extended (KF15) to include coverage, objections, and open-question count so the moderator stall window is less likely to fire prematurely.

**Status:** Implemented; validation runs pending (uni endpoint timing out). No regressions in tests (77/77).

---

### KF12 — Unsupported soft attributes (implemented 2026-06-28)

**Fix applied:** `_check_soft_attributes()` in `validation.py` warns when confident capacity/flexibility/availability claims appear without hedge words. Pattern covers "can accommodate", "has flexible/ample/dedicated", "offers flexible/customizable", "less risk of", "more seats/space/room/availability". Hedge bypass: if any of think/might/probably/etc. appears, the check is skipped.

**Status:** Implemented; warn-level only. Repair for this class of issue goes through KF13's existing "INVENTED_OPTION_ATTRIBUTE" path in `_REPAIR_HINTS`.

---

### KF13 — Repair prompts creating question cascades (implemented 2026-06-28)

**Fix applied:** `_REPAIR_HINTS["INVENTED_OPTION_ATTRIBUTE"]` changed from suggesting "do they...?" as a hedge form to "hedge with 'I think they might...' or 'not sure if they...' — never ask a question". `repair_utterance()` footer also changed to add "never ask a question" constraint. Validation-backed by `_check_question_chain()`.

---

### KF14 — Semantic repetition at slot level (implemented 2026-06-28)

**Fix applied:** `_check_repetition()` in `validation.py` now includes a slot-level self-repetition check: if the current speaker's turn shares ≥2 claim slots with a recent own turn that also referenced the same option, `SELF_REPETITION` (warn) fires. Complements the existing Jaccard check.

---

### KF16 — Token cost remains structurally high

**Problem:** The per-turn prompt contains full option names, context, speaker card, group lean state, recent chat, option facts, guidance, verbosity hints, frame hints, alias rules, and global rules. Repairs add extra calls. This may be necessary for quality, but it is still the main efficiency cost.

**Why it matters:** High input/output ratio makes larger n=5–7 runs expensive and slow. It also hides whether quality improvements are coming from better control or simply more prompt text.

**Relevant code:** `src/prompts.py`: `sim_utterance()`, `runtime_speaker_card()`, `repair_utterance()`; `config.yaml`: prompt windows and word budgets.

**Fix direction:** Defer heavy optimization until P0/P1 correctness is stable. Then reduce prompt size by shortening group state, avoiding repeated global rules, and making repair prompts stricter but smaller.

---

## P3 — Efficiency and repair pressure

### KF19 — Repair severity is not well aligned with quality cost

**Problem:** Some highly visible issues are warn-only and leak into transcripts, while some surface issues are repair-level and cause expensive retry calls. `max_repairs_per_turn=1` means one failed repair still leaks. `repair_on_warning=false` means warnings are mostly only logged.

**Why it matters:** Repair cost is high, but quality leaks remain. The system needs fewer but more meaningful repairs.

**Relevant code:** `src/validation.py`: issue severities; `src/dialogue.py`: `_needs_repair()`; `config.yaml`: `simulation.max_repairs_per_turn`, `validation.repair_on_warning`.

**Fix direction:** Reclassify only the most harmful issues as repair-level. Prefer deterministic cleanup for simple surface patterns. Do not globally enable repair-on-warning.

---
