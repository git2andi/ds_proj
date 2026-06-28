# Known Failures — Open Issues Only

Last updated: 2026-06-28 (KF08 resolved).
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
8. Update `CLAUDE.md`, memoryand relevant project notes after each successful implementation pass.

---


## P1 — Natural moderation and dialogue flow

### KF09 — Local conversation flow still needs more human uptake

**Problem:** Many turns still read as standalone argument cards. Sims often state their own position rather than directly answering, briefly acknowledging, challenging, or building on the previous turn. Backchannels and short reactions appear sometimes, but not as a stable part of the flow.

**Why it matters:** Naturalness depends less on longer prompts and more on local uptake: answering the previous question, naming the actual concern, making small concessions, and reacting in short turns when appropriate.

**Relevant code:** `src/router.py`: answer/response routing; `src/prompts.py`: `sim_utterance()`, `intent_guidance()`, recent-chat prompt window; `src/dialogue.py`: open-question tracking and progress tracking.

**Fix direction:** Improve routing/context before adding more style rules:

- if the previous turn asked a question, route a true answer or explicit non-answer;
- if the previous turn raised a concern, route a response to that concern;
- allow short backchannel/support turns for low response-length personas;
- avoid every turn containing a full reason + option pitch.

---

### KF10 — Unresolved closures are still weak and can feel unfinished

**Problem:** The unresolved closure prompt asks for a concrete action, but the actual closure is still fully LLM-generated and unvalidated. It can produce unhelpful endings such as procedural dead ends (“flip a coin”) or vague tabling. Farewell prompts then reinforce the “no decision” state rather than producing a useful next step.

**Why it matters:** Honest unresolved outcomes are good, but the ending must still feel socially realistic: name the blocker and the next action.

**Relevant code:** `src/prompts.py`: `moderator_closure_prompt()`, `_identify_blocker()`, `farewell_line()`; `src/dialogue.py`: final closure generation.

**Fix direction:** For unresolved outcomes, prefer a deterministic closure skeleton with the blocker and next action filled from state, or validate the generated closure against banned weak endings.

---

## P2 — Dialogue substance and grounding

### KF11 — Option coverage is too shallow before narrowing/closure

**Problem:** `_coverage_gap_option()` only forces an option while `mentions == 0`. Once an option has one mention, it is considered covered. `min_options_touched_before_narrowing` is only 2, so viable options can remain almost unexplored before the group narrows or closes.

**Why it matters:** Final decisions can feel under-tested. Options like an obvious compromise can be mentioned once and then disappear.

**Relevant code:** `src/router.py`: `_coverage_gap_option()`, `_act_for_gap()`; `src/dialogue.py`: `DialogueController._can_start_narrowing()`; `config.yaml`: `conversation.min_options_touched_before_narrowing`.

**Fix direction:** Add a small coverage gate for the leading/candidate option and serious alternatives: at least one reason and one trade-off/objection should be visible before narrowing, unless the group explicitly rejects the option.

---

### KF12 — Unsupported soft attributes still leak through

**Problem:** Grounding validation mainly catches invented numbers. Non-numeric claims such as “likely has flexible check-in”, “less risk of room assignment issues”, “more seats”, or “better availability” can pass if no unsupported number appears. The prompt says to hedge unknown facts, but this is not deterministic.

**Why it matters:** The simulator still invents plausible real-world attributes, especially for real venues/products/tools.

**Relevant code:** `src/validation.py`: `_check_grounding_numbers()`; `src/prompts.py`: sim and repair grounding rules.

**Fix direction:** Do not add a large brittle fact ontology. Start with a compact denylist for common unsupported availability/logistics claims, or validate claims against option-card attribute keys for ANSWER/PROPOSE_COMPROMISE turns.

---

### KF13 — Repair prompts can bypass ASK/question controls and create question cascades

**Problem:** Initial act sampling dampens or vetoes ASK after recent questions, but repair generation is free-form. The repair prompt even allows unknown facts to become “do they...?” questions. `QUESTION_CHAIN` is warn-level and `repair_on_warning` is false, so repaired text can still ask a question after a question.

**Why it matters:** Named real-world topics can enter loops of unanswerable capacity/availability/group-rate questions.

**Relevant code:** `src/router.py`: `_sample_discussion_act()`; `src/prompts.py`: `repair_utterance()`, `_REPAIR_HINTS`; `src/validation.py`: `_check_question_chain()`, `_check_unwanted_question()`.

**Fix direction:** Carry “no new question” constraints into repair prompts when the prior participant turn was a question or when the original intent was not ASK. Remove “do they...?” as a recommended repair form for invented attributes.

---

### KF14 — Semantic repetition survives string-level repetition checks

**Problem:** Repetition checks use Jaccard similarity, shared phrase runs, repeated starts, and discourse frames. They still miss repeated claims that use different wording but the same semantic slot, for example repeatedly saying an option is broad, light, practical, comfortable, or better after a long day.

**Why it matters:** Dialogues can loop at the argument level while avoiding exact lexical repetition.

**Relevant code:** `src/validation.py`: `_check_repetition()`, discourse-frame checks; `src/prompts.py`: `classify_claim_slots()`, `covered_slots_hint()`; `src/dialogue.py`: `_update_progress()`.

**Fix direction:** Use the existing claim-slot tracking more directly. Track recent `(option, claim_slot, polarity)` patterns and nudge/validate when the same speaker or group repeats the same semantic move.

---

### KF15 — Progress detection is too coarse

**Problem:** `_progress_snapshot()` counts only stance changes and whether each option has at least one reason. It does not treat new claim slots, answered questions, objections, or cleared blockers as progress.

**Why it matters:** The moderator can interpret a discussion as stalled even when the group is adding new argument dimensions. This can trigger premature moderation or narrowing.

**Relevant code:** `src/dialogue.py`: `_update_progress()`, `_progress_snapshot()`; `src/validation.py`: `classify_claim_slots()`.

**Fix direction:** Include coverage slots, objections, and open-question changes in the progress snapshot. Keep it simple: do not use semantic embeddings.

---

## P3 — Efficiency, metrics, and repair pressure

### KF16 — Token cost remains structurally high

**Problem:** The per-turn prompt contains full option names, context, speaker card, group lean state, recent chat, option facts, guidance, verbosity hints, frame hints, alias rules, and global rules. Repairs add extra calls. This may be necessary for quality, but it is still the main efficiency cost.

**Why it matters:** High input/output ratio makes larger n=5–7 runs expensive and slow. It also hides whether quality improvements are coming from better control or simply more prompt text.

**Relevant code:** `src/prompts.py`: `sim_utterance()`, `runtime_speaker_card()`, `repair_utterance()`; `config.yaml`: prompt windows and word budgets.

**Fix direction:** Defer heavy optimization until P0/P1 correctness is stable. Then reduce prompt size by shortening group state, avoiding repeated global rules, and making repair prompts stricter but smaller.

---

### KF17 — Per-turn token logging undercounts repaired turns

**Problem:** `_generate_turn()` returns `self._llm.last_tokens_in/out`, which after repair refers only to the last generation attempt. Session totals include all attempts, but individual `TurnRecord.tokens_in/out` do not accumulate initial + repair costs.

**Why it matters:** Run totals are usable, but per-turn analysis underestimates which turns caused token spikes.

**Relevant code:** `src/dialogue.py`: `_generate_turn()`; `src/logger.py`: `_json_payload()`, `metrics_for()`.

**Fix direction:** Accumulate token counts across the initial generation and all repair attempts for the `TurnRecord`.

---

### KF18 — Metrics mix social turns with decision turns

**Problem:** `apply_social()` appends greetings/farewells as normal non-moderator turns, but does not update runtime turn counts. `metrics_for()` computes `participant_turns`, `question_density`, `avg_words_per_turn`, repair rate denominator, and moderator ratio from all non-moderator turns, including social greetings/farewells. This makes metrics inconsistent with `turn_counts` and decision quality.

**Why it matters:** Evaluation metrics are diluted by cosmetic lines. This can hide real question density, dialogue length, and repair rates for decision turns.

**Relevant code:** `src/dialogue.py`: `apply_social()`; `src/logger.py`: `metrics_for()`.

**Fix direction:** Add an explicit social/cosmetic flag or compute decision metrics from turns with real runtime impact only.

---

### KF19 — Repair severity is not well aligned with quality cost

**Problem:** Some highly visible issues are warn-only and leak into transcripts, while some surface issues are repair-level and cause expensive retry calls. `max_repairs_per_turn=1` means one failed repair still leaks. `repair_on_warning=false` means warnings are mostly only logged.

**Why it matters:** Repair cost is high, but quality leaks remain. The system needs fewer but more meaningful repairs.

**Relevant code:** `src/validation.py`: issue severities; `src/dialogue.py`: `_needs_repair()`; `config.yaml`: `simulation.max_repairs_per_turn`, `validation.repair_on_warning`.

**Fix direction:** Reclassify only the most harmful issues as repair-level. Prefer deterministic cleanup for simple surface patterns. Do not globally enable repair-on-warning.

---

## P4 — Low-priority surface cleanup

### KF20 — “Considering...” strip still misses single-clause cases

**Problem:** `_strip_considering_opener()` only strips `Considering X, Y` with a comma. A single-clause opener such as “Considering the budget is tight.” is not stripped by cleanup, though validation can still flag it.

**Relevant code:** `src/dialogue.py`: `_CONSIDERING_OPENER_STRIP`, `_strip_considering_opener()`; `src/validation.py`: `_CONSIDERING_OPENER`.

**Fix direction:** Extend the deterministic strip or accept this as a low-priority warning if it is rare.

---

### KF21 — Banned phrase handling is split across prompt, validation, and cleanup

**Problem:** Some stock phrases are rewritten deterministically, some are warn-only validation, and some are prompt-only bans. This makes it hard to predict whether a phrase will be repaired, logged, or silently fixed.

**Relevant code:** `src/prompts.py`: stock-phrase rule and repair hints; `src/validation.py`: `_ROBOTIC_TEMPLATES`, `_STOCK_PHRASE_REWRITES`; `src/dialogue.py`: `clean_generated()`.

**Fix direction:** Keep one small deterministic cleanup list for harmless replacements and one small repair-level list for severe robotic frames. Avoid growing the prompt ban list further.
