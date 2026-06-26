# Memory Index

- [ds-proj refactor status](ds-proj-naturalness-plan.md) — S1-S4 identified but NOT implemented; reverted 2026-06-25 (groq was used)
- [Prompt length feedback](feedback-prompt-length.md) — shorter prompts, not longer; llama3.3 ignores excess rules
- [Provider rule](feedback-uni-provider.md) — uni only; never switch to groq/gemini without explicit user instruction
- [R8: short_name alias fix](project-short-name-fix.md) — OptionCard.short_name LLM-generated alias replaces blind 2-word truncation (2026-06-26)
- [R9: template leakage fix](project-r9-template-fix.md) — never put quoted examples in guidance strings; llama3.3 copies them verbatim across all topics (2026-06-26)
- [R10: invented context fix](project-r10-invented-context.md) — epistemic grounding rule prevents confident invented facts about real-world named places (2026-06-26)
