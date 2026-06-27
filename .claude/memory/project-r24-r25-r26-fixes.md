---
name: project-r24-r25-r26-fixes
description: "R24/R25/R26 fixes — asker-answers-own-question guard, covered-slots hint activation, ANSWER echo loop blocker (2026-06-27)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**R24: Asker routed to answer their own question** — `_best_answerer()` in `router.py` now builds `other_askers = {oq.asked_by for oq in state.open_questions if oq.turn_id != q.turn_id}` and uses a two-tier eligibility filter (first: exclude both asked_by and other_askers; fallback: exclude just asked_by). Prevents the case where Sim A asks Q1, Sim B asks Q2, and the router routes Sim A back to answer Q2.

**R25: covered_slots_hint never fired** — `covered_slots_hint()` in `validation.py` received `text_slots=[]` always, so `repeated=[]` and the condition `not repeated` was always True. Removed the `text_slots` parameter entirely; function now fires when `len(covered) >= 3`. Call site in `sim_utterance` updated. Now gives "The group already argued cost, comfort… Try a new angle: time, effort" hints once an option is well-discussed.

**R26: ANSWER-turn echo loop (R13 gap)** — When an ANSWER-routed turn still echoed its question after repair (model resisted), `_update_questions` re-registered the echo as a new OpenQuestion, feeding another ANSWER cycle. Fixed in `StateTracker._update_questions()`: if intent.act==ANSWER and "QUESTION_ECHO" in validation_issues, suppress propagation. Eliminated 3-question echo loops (holiday party venue-cost case).

**Why:** R24 produces logically broken exchanges. R25 was dead code. R26 caused transcripts with 3+ identical back-to-back questions.

**How to apply:** If question echo loops reappear, check R26 guard in `_update_questions`. If covered-slots hint is not appearing for well-discussed options, verify the threshold is `len(covered) >= 3`.
