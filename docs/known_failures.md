# Known Failures

Last updated: 2026-06-27. F1–F52 fixed (F14, F18, F50 removed — contradicted by R37, R33, R36). Open: R29, R32–R38.

---

## Implementation process

1. Pick one item from the priority list.
2. Implement. Run `pytest tests/`.
3. Validate: one n=3 run mandatory, then 5-6 more from n=2–7 with random topics from `evals/topics.txt`. Provider: `uni` always.
4. Read transcripts. Did it improve? Any regressions?
5. New issues → add here before continuing.
6. Update CLAUDE.md, memories. Only then move to next item.

---

## Fixed (F1–F52)

**F1** Mid-discussion accepts counted as binding → commitment gating; ACCEPT/VOTE binding only in narrowing/confirmation.
**F2** Hard-blocker personas folded to non-preferred → only backs preferred; votes elsewhere ignored.
**F3** Self-narration in turns → banned from prompt.
**F4** Duplicate moderator lines → deduplication guard.
**F5** Setup instability (invalid worlds silently defaulted) → raise on unusable setup output.
**F6** Excessive repair → repair hints targeted; structural errors only trigger LLM repair.
**F7** No grounding to prior turns → `_responding_to_line` anchors each turn to most relevant prior.
**F8** Outcome categories too coarse → consensus / fallback / unresolved distinct.
**F9** No compromise mechanism → PROPOSE_COMPROMISE act with bridge guidance.
**F10** Argument-card recitation → discourse-frame/claim-slot tracking; situated reactions via prompt rewrite.
**F11** Option name overuse → alias instruction after opening; ~70% compliance.
**F12** Flat persona voice → behavioral speaking habits (12 branches covering all traits).
**F13** No face-work on objections → trait-based modifiers for OBJECT/PUSH_BACK.
**F15** Hedged accepts counted as binding → "still not sure", "not fully sold" clamped to neutral.
**F16** Moderator always directive → diagnostic questions; stall window scales with n.
**F17** Practical constraints never surface → stall-to-concrete routing; ASK doubled after 2+ no-progress turns.
**F19** Surface artifacts (space-before-punctuation etc.) → deterministic cleanup in `clean_generated`.
**F20** Turn-taking too balanced → extraversion/initiative weight; reduced catch-up boost for n>=5.
**F21** Large groups repeat known positions → skip-if-nothing-new for n>=4.
**F22** Room questions unrouted → registered as OpenQuestion; routed to option champion.
**F23** Moderator interrupts too early for large groups → stall window = base + max(0, n-3).
**F24** Stock phrases ("is a must for me", "major draw") → deterministic `fix_stock_phrases` in `clean_generated`.
**F25** Farewell lines stiff → formal closers banned in farewell prompt.
**F26** Option alias bad truncation ("Wine and", "Settlers of") → `short_name` LLM-generated (e.g. "Ticket to Ride"→"Ride"); `_clean_short_name` rejects stopword-ending; deterministic fallback upgraded.
**F27** Named-addressee rate low → address rule encourages name once per turn.
**F28** Repetitive agreement loops → when concentration==1.0 and no_progress>=2, force narrowing transition.
**F29** No shared situational context → `shared_context` generated at setup; injected into moderator opening and per-turn prompt.
**F30** Participant names repetitive → 48-name diverse pool pre-sampled in `builders.py`.
**F31** Seven turn-patterns seeded by guidance strings ("X seems like a good fit", "still beats", "What if we...", etc.) → guidance strings rewritten to describe behavior without quoting phrases; patterns added to `_ROBOTIC_TEMPLATES`.
**F32** Speakers assert real-world facts not in cards → epistemic constraint: "You know only what's in the option cards." Repair hints updated.
**F33** "edges ahead" phrase seeded by COMPARE guidance → guidance rewritten; phrase added to `_ROBOTIC_TEMPLATES` + stock-phrases ban.
**F34** Question echo in ANSWER turns → explicit ANSWER guidance: "don't repeat the question, hedge and move on."
**F35** 2-turn question echo when non-ANSWER-routed → GROUP_REPETITION on question pairs escalated to QUESTION_ECHO repair.
**F36** "still pick" phrase seeded by F33's COMPARE rewrite → guidance rewritten again (no verb phrases); "still pick" added to `_ROBOTIC_TEMPLATES`.
**F37** Back-to-back at opening→answer boundary → guard in `next_intent()` skips ANSWER routing if target just spoke.
**F38** Moderator closes mid-question in confirmation → (a) question stance override in `_resolve_move`; (b) confirmation timeout guards `if not state.open_questions`.
**F39** Fallback targets drifted `leading_option` → `finalize()` prefers `state.candidate_option` over live lean score.
**F40** Back-to-back votes in narrowing → last-speaker excluded from unvoted list; consecutive UNCLEAR_VOTE advances to CONFIRMATION.
**F41** POSSESSIVE_SUBJECT opener warn→repair → escalated; pattern covers `short_name` forms and names with parentheticals. Named rate 26–31%.
**F42** Repeated openers → opener feedback in speaker card (last 2 shown); REPEATED_START escalated to repair with dynamic hint naming exact repeated phrase.
**F43** Named-addressee rate low in ANSWER/REACT without explicit `addressee_id` → responding-to name injected as "use their name once" instruction.
**F44** Trait differences not perceptible → `Traits: extra=N agree=N neuro=N len=N` added to `runtime_speaker_card`.
**F45** Back-to-back questions from different speakers → hard-zero ASK weight when preceding participant turn contained "?".
**F46** Asker routed to answer own question → `_best_answerer` two-tier filter: prefer non-asker non-other-asker candidates first.
**F47** `covered_slots_hint` never fired (dead code) → broken `text_slots` param removed; fires when `len(covered) >= 3`.
**F48** ANSWER echo loop (F35 gap) → if ANSWER-routed turn has QUESTION_ECHO in issues, `_update_questions` suppresses propagation.
**F49** SELF_REPETITION skipped for ACCEPT/REJECT → early-return exemption removed; `already_said` checked for all act types.
**F51** ANSWER turns invent non-card facts → ANSWER guidance states card attributes are exhaustive; "never invent facts."
**F52** response_length absent from trait card → `len=N` added to Traits line.

---

## Open items (priority order)

### R29 — "Considering..." opener (TOP PRIORITY)
`^\s*considering\b` in `_ROBOTIC_TEMPLATES` is warn-only. Appears 1-4×/run despite prompt ban. Same escalation pattern as F41/F42.
**Fix:** Split from blanket warn → repair for this pattern only. Hint: "don't start with 'Considering' — open with the point itself, a reaction, or a question."

---

### R32 — ANSWER-routed "?" falls through validation
ANSWER not in `statement_only`, so a question on an ANSWER turn never fires UNWANTED_QUESTION. The "?" re-registers as OpenQuestion → another ANSWER cycle.
**Fix:** Add `ActType.ANSWER` to `statement_only`. Hint: "give an answer or say you can't confirm, don't re-ask."

---

### R33 — Earned consensus / acceptance gate
**Symptom:** Fallback outcomes phrased as clean wins (e.g. "Ruby wins" when support=0.667, outcome=fallback). Speakers jump from concern to accept in one sentence with no visible reasoning. System finalizes on hidden state tallies, not visible text quality.
**Fix candidate:** (a) Before finalizing, verify each participant has a visible accept/vote or explicit "I can work with X" this round — otherwise re-route one more narrowing turn. (b) Fallback moderator closure text should reflect partial agreement, not unanimous language.

---

### R34 — Open questions drift away unanswered
**Symptom:** "Can it handle five players?" raised and never answered or deferred. Routing clears the question after one hedge ANSWER even though the question is still open. Moderator doesn't name lingering unknowns before closure.
**Fix candidate:** Add `hedged=True` flag to OpenQuestion. If question hedged twice with no substantive answer, route moderator to name it explicitly ("We still don't know X — do we need this to decide?") rather than silently closing it.

---

### R35 — Persona/option setup contradiction
**Symptom:** "Quiet Observer, wants a quiet reading spot" assigned Cosmic Playground (high-noise arcade) as preferred option. Role, concern, and preference internally inconsistent — poisons the whole discussion.
**Fix candidate:** In `builders.py` belief-state prompt, add constraint: preferred option must plausibly match at least one stated concern. Post-generation check: if persona's concern cluster contradicts preferred option's attribute cluster, re-roll or raise.

---

### R36 — Hedge phrases became new templates
**Symptom:** "do they offer", "we'd have to check", "no idea if", "why that matters is" repeat across multiple speakers per run. F51/F50 fixed hallucination but replacement phrases are now the chorus. Same-unknown raised twice in same form by different speakers.
**Fix candidate:** (a) Add "do they offer", "why that matters is" to `_ROBOTIC_TEMPLATES`. (b) Track raised-unknowns per option in state; suppress re-raising already-hedged unknowns (or feed into `covered_slots_hint`). (c) Rewrite epistemic guidance to describe behavior without listing the exact phrases.

---

### R37 — Acceptance move too thin
**Symptom:** Speaker who actively opposed an option accepts it in one vague sentence with no condition, no acknowledgment of prior objection. Consensus feels unearned; character feels inconsistent.
**Fix candidate:** For ACCEPT on non-preferred option: bridge guidance must require (a) naming the concern that was addressed AND (b) a condition or trade-off making this workable — forces "I still prefer X, but since Y is handled, I'm okay with Z" over a single generic clause.

---

### R38 — Token usage regression (20k+ tokens/run)
**Symptom:** Recent runs show 20k+ tokens. Earlier runs were substantially lower. Likely grew with F42 opener feedback, F44 trait block, F50 epistemic expansion, F51 ANSWER guidance additions.
**Fix candidate:** (a) Profile `prompts.jsonl` token counts per turn. (b) Verify full option cards sent only on COMPARE/VOTE. (c) Trim `already_said` display to last 1 claim instead of 2. (d) Check if opener feedback + trait block + shared_context together bloat the card past useful size.
