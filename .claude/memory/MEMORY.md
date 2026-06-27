# Memory Index

- [ds-proj refactor status](ds-proj-naturalness-plan.md) — S1-S4 identified but NOT implemented; reverted 2026-06-25 (groq was used)
- [Prompt length feedback](feedback-prompt-length.md) — shorter prompts, not longer; llama3.3 ignores excess rules
- [Provider rule](feedback-uni-provider.md) — uni only; never switch to groq/gemini without explicit user instruction
- [R8: short_name alias fix](project-short-name-fix.md) — OptionCard.short_name LLM-generated alias replaces blind 2-word truncation (2026-06-26)
- [R9: template leakage fix](project-r9-template-fix.md) — never put quoted examples in guidance strings; llama3.3 copies them verbatim across all topics (2026-06-26)
- [R10: invented context fix](project-r10-invented-context.md) — epistemic grounding rule prevents confident invented facts about real-world named places (2026-06-26)
- [R11: edges ahead template fix](project-r11-edges-ahead.md) — "edges ahead" seeded by COMPARE guidance; removed from guidance + added to _ROBOTIC_TEMPLATES (2026-06-26)
- [R12: answer echo fix](project-r12-answer-echo.md) — ANSWER act now has explicit hedge guidance; eliminates 3-turn question echo chains (2026-06-26)
- [R13: question echo backstop](project-r13-question-echo-backstop.md) — GROUP_REPETITION on question pairs escalated to QUESTION_ECHO repair; closes 2-turn echo when non-ANSWER-routed (2026-06-26)
- [R14: still pick template](project-r14-still-pick.md) — "I'd still pick X" seeded by R11's COMPARE guidance rewrite; same R9/R11 class; guidance rewrite + robotic template backstop (2026-06-26)
- [R15: back-to-back routing](project-r15-backtoback-routing.md) — opening→answer boundary let same speaker go twice; guard in next_intent() skips if target just spoke (2026-06-26)
- [R16: question cutoff at closure](project-r16-question-cutoff.md) — question on ACCEPT-routed turn falsely credited as accept via fallback stance, triggering premature consensus; also confirmation timeout ignored open_questions (2026-06-26)
- [R17: fallback uses drifted leading option](project-r17-fallback-drift.md) — finalize() used live leading_option which drifts during proposals; fixed by preferring state.candidate_option (the vote result) as fallback target (2026-06-26)
- [R20: repeated opener fix](project-r20-repeated-start.md) — opener feedback in speaker card + REPEATED_START→repair + dynamic hint naming exact repeated phrase (2026-06-27)
- [R23: back-to-back question veto](project-r23-question-veto.md) — hard ASK=0 in _select_act when previous turn ends with "?"; soft damping was not enough (2026-06-27)
- [R22: trait numbers in speaker card](project-r22-trait-numbers.md) — Traits: extra/agree/neuro added to runtime_speaker_card; llama3.3 now has numeric calibration for persona voice (2026-06-27)
- [R24/R25/R26 fixes](project-r24-r25-r26-fixes.md) — asker-answers-own-question guard in _best_answerer; covered_slots_hint fixed (was dead code); ANSWER echo loop blocker in _update_questions (2026-06-27)
