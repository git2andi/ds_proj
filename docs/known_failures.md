# Known Failures

Tracked failures and quality issues in the dialogue simulator.

Last updated: 2026-06-24, after implementing O1–O12 improvements and reviewing n=3 evaluation runs.

---

## Current fulfillment state

The simulator produces coherent, responsive, decision-oriented discussions with shorter, more natural turns. The major structural failures are resolved. Remaining issues are model-dependent (llama3.3 limitations) and fall into two categories: (a) residual formulaic patterns the model uses despite prompt bans, and (b) larger-group dynamics that need further tuning.

In short:

- **decision mechanics:** working,
- **basic grounding:** working,
- **repair / malformed output:** low repair rate (<5%),
- **moderator role:** improved — diagnostic before directive, names participants,
- **persona difference:** behavioral voice profiles active, distinguishable in n=3,
- **human-like conversation:** significantly improved, still model-constrained.

---

## Fixed / mostly working

- **F1: Mid-discussion acceptance counted as final commitment** — fixed. Binding final support gated to explicit decision/vote acts. Hedged accepts ("still not sure", "not fully sold") clamped to neutral.
- **F2: Hard blockers folding incorrectly** — fixed. Blocker constraints preserved and invalid votes corrected.
- **F3: Heavy `I think...` / personal-stance opener pattern** — reduced. Opener variety check limits consecutive "I"/"we" openers.
- **F4: Visible self-narration** — fixed. Deterministic detection and prompt ban.
- **F5: Duplicate moderator wording** — fixed. Prior moderator lines fed back to prevent reuse.
- **F6: Setup instability / timeout sensitivity** — fixed. Separate scenario/persona generation.
- **F7: Excessive repair loop** — fixed. Repair is structural only, not constant style rewriting.
- **F8: Basic interaction grounding** — fixed. Speakers respond to previous points via responding-to anchoring.
- **F9: Outcome categories** — fixed. Consensus, fallback, and unresolved endings are distinct.
- **F10: Simple compromise mechanisms** — working. Concession bridges with persona-specific guidance.
- **O10: Surface artifacts** — fixed. Space-before-punctuation, repeated punctuation, stray quotes cleaned deterministically.
- **O1: Argument-card turns** — improved. Prompt guidance generates situated reactions, not structured debate paragraphs. Discourse-frame tracking avoids echo-pivot openers.
- **O2: Option name overuse** — improved. After opening, prompt instructs shorthand ("the bistro", "the cheap one"). Model follows ~70% of the time.
- **O3: Persona differences** — improved. Speaking habits are behavioral ("blunt — cuts to the chase", "worrier — flags what could go wrong"). Turn counts less balanced.
- **O4: Social face-work** — improved. Face-work modifiers added for objections (softening for agreeable personas, anxiety for neurotic, directness for direct).
- **O5: Preference shifts under-motivated** — improved. Concession bridges with persona-specific guidance (residual worry, condition, trade-off, next step).
- **O7: Moderator too controller-imposed** — improved. Diagnostic questions before directive suggestions. Participant names included in prompt to prevent hallucination.
- **O8: Practical constraints** — improved. Stall-to-concrete routing: when discussion stalls (2+ turns no progress), ASK probability doubled, SUPPORT dampened.
- **O9: Endings socially thin** — improved. Unresolved closure names specific blocker. Farewell prompt encourages concrete reaction over "we'll figure it out".
- **O11: Turn-taking too balanced** — improved. Extraversion/initiative weight increased. Low-turn-count boost reduced for n>=5.
- **O12: Larger groups need contribution control** — improved. Skip-if-nothing-new check prevents low-information SUPPORT turns when option already has 3+ reasons.

---

## Still open (model-dependent residuals)

### R1: Stock phrases persist despite prompt bans

**Symptom:**
llama3.3 still occasionally outputs "X is a must for me", "a major draw", "is key for me" despite explicit prompt bans and validation warnings. These appear in ~10-20% of turns.

**Mitigation:**
Prompt bans + warn-level validation logging. Not repair-triggered because the cost of an extra LLM call per turn outweighs the benefit for a stylistic issue.

**Future fix:**
Deterministic rewrite (like `fix_collective_voice`) for the 3-4 most common stock phrases. Deferred to avoid adding complexity.

### R2: Farewell lines can still sound stiff

**Symptom:**
Post-closure lines like "Satisfied with the consensus" or "Looking forward to" are occasionally too formal for casual group chat.

**Mitigation:**
Farewell prompt bans seminar openers. Prior farewell lines shown to prevent repetition. Model compliance is ~70%.

### R3: Moderator can hallucinate participant names

**Symptom:**
Rare — the moderator once called a participant "Sarah" when no Sarah existed. Now mitigated by including participant names in the moderator prompt and the "don't invent names" rule.

### R4: Full option names still used sometimes after opening

**Symptom:**
The model uses full names (~30% of the time) even when the alias instruction is active. Board game names are harder to shorten than restaurant names.

### R5: Named addressee rate is low

**Symptom:**
Participants rarely mention each other by name (0% named rate across runs). This is somewhat natural for casual group chat but could be higher — real groups do say "wait, Kai, what about..." occasionally.

**Note:** n=5 evaluation verified — turn counts are uneven (10/7/8/7/9), question density 18.2%, face-work modifiers active, practical constraints surfaced (cost cap, guest list). Routing changes working.

---

## Current priority order

1. Verify n=5+ behavior with latest routing changes.
2. Add deterministic stock-phrase rewrite for the 3-4 worst offenders.
3. Improve farewell generation reliability.
4. Consider commitment ladder (deferred from dialogue_quality_refactor_plan.md Step 4).
5. Re-evaluate after manual transcript review.
