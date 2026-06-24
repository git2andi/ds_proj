# Known Failures

Tracked failures and quality issues in the dialogue simulator. This is the single file for tracking what needs fixing.

Last updated: 2026-06-24.

---

## How to implement fixes

Every change follows this process — no exceptions:

1. **Pick one item** from the priority list below.
2. **Implement the fix.** Run `pytest tests/` to verify nothing breaks.
3. **Validate with example runs.** One n=3 run is mandatory. Then pick 1-2 more from n=2–7, using a randomly selected topic from `evals/topics.txt`. Always use the `uni` provider.
4. **Read the transcripts.** Evaluate whether the change actually improved things. Check for regressions — did something else get worse?
5. **If new issues surface**, add them directly to this file before moving on.
6. **Update** CLAUDE.md, known_failures.md, memories and/or skills to be always up to date with the latest changes.
7. **Only when all runs are complete and reviewed**, move to the next item.

Do not batch multiple fixes before validating. One fix, validate, evaluate, then next.

---

## Current state

The simulator produces coherent, responsive discussions with natural turn lengths, persona-specific voice, and concrete concessions. Decision mechanics work. Remaining issues are model-dependent (llama3.3 compliance with prompt instructions).

- **decision mechanics:** working — consensus, fallback, unresolved outcomes distinct,
- **grounding:** working — responding-to anchoring, room-questions routed to option champions,
- **repair / malformed output:** low (<7%),
- **moderator:** diagnostic before directive, stall window scales with group size, topic-appropriate closures,
- **persona voice:** behavioral habits active, face-work modifiers for objections,
- **concessions:** option-specific with persona's reservation referenced,
- **option generation:** specific realistic names with concrete comparable attributes.

---

## Fixed

- **F1–F10:** Structural issues (mid-discussion accept counting, hard-blocker folding, self-narration, duplicate moderator lines, setup instability, excessive repair, interaction grounding, outcome categories, compromise mechanisms) — all resolved.
- **O1:** Argument-card turns → situated reactions via prompt rewrite and discourse-frame tracking.
- **O2:** Option name overuse → alias instruction after opening (~70% model compliance).
- **O3:** Persona differences → behavioral speaking habits ("blunt — cuts to the chase", "worrier — flags what could go wrong").
- **O4:** Social face-work → trait-based modifiers for objection/push-back acts.
- **O5:** Preference shifts → concrete concession bridges referencing option name and persona's reservation.
- **O6:** Consensus too coarse → hedged accept detection ("still not sure", "not fully sold") clamped to neutral.
- **O7:** Moderator imposed → diagnostic questions, participant names in prompts, stall window scales with n.
- **O8:** Practical constraints late → stall-to-concrete routing (ASK doubled after 2+ no-progress turns).
- **O9:** Endings thin → topic-appropriate closure (no generic "book it"), specific blocker naming.
- **O10:** Surface artifacts → deterministic cleanup of space-before-punctuation, repeated punctuation, stray quotes.
- **O11:** Turn-taking too balanced → extraversion/initiative weight, reduced catch-up boost for n>=5.
- **O12:** Large groups → skip-if-nothing-new for n>=4 when option already well-covered.
- **Room questions ignored** → genuine questions to the room now registered and routed to option champion.
- **Moderator too early for large groups** → stall window = base + max(0, n-3), minimum participant turns = n*2.
- **R6: No shared decision-situation context** → `shared_context` field added to Scenario dataclass, setup prompt, moderator opening, and per-turn prompt. 2-3 stable situational facts generated alongside options.
- **R7: Participant names lack variety** → 48-name diverse pool in `builders.py`, pre-sampled and enforced in parser.
- **R1: Stock phrases persist** → deterministic `fix_stock_phrases` rewrites "is a must for me" → "matters to me", "major draw" → "appeals to me" etc. in `clean_generated`. No LLM call.
- **R5: Repetitive agreement loops** → when concentration_score == 1.0 and no_progress_count >= 2, force narrowing transition.
- **R2: Farewell lines stiff** → explicit ban of formal closers in farewell prompt ("looking forward to", "confirmed and set", "satisfied with", "have a great day").
- **R3: Full option names after opening** → `_short_alias` generates concrete 2-word aliases from option names, shown in the alias instruction. Model shortens consistently.
- **R4: Named addressee rate low** → address rule now encourages using addressee's name once in the message.

---

## No open items

All tracked issues (F1–F10, O1–O12, R1–R7, R2–R4) are resolved. Remaining quality depends on model compliance (llama3.3). Re-evaluate after broader testing with diverse topics from `evals/topics.txt`.
