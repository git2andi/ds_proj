# Known Failures

Tracked failures and quality issues in the dialogue simulator.  
The fixed section is intentionally brief: it records what appears to be stable enough to avoid re-solving the same issue.  
The open section describes what still damages dialogue realism in the latest runs.

Last updated: 2026-06-24, after implementing O1–O9 and verifying with n=3 and n=4 test runs.

---

## Fixed / mostly fixed

- **F1: Mid-discussion accept counted as final commitment** — fixed by gating binding commitment to explicit decision acts.
- **F2: Hard blockers folding incorrectly** — fixed by keeping hard blockers tied to their permitted options and correcting invalid rejected votes.
- **F3: Wall-of-`I` openers / possessive opener tic** — mostly fixed. Recent transcripts no longer show the old heavy pattern of every turn starting with personal-stance templates.
- **F4: Self-narration** — mostly fixed. Earlier lines like `I should consider...` / `I must consider...` are no longer a dominant visible failure in the latest chats.
- **F5: Duplicate moderator lines** — improved by feeding prior facilitator lines back and requesting variation.
- **F6: Setup call timeouts** — improved by splitting scenario generation and persona generation.
- **F7: Excessive repair loop** — fixed as a performance issue by demoting style checks to warnings and keeping repair mainly for structural failures. Recent repair rates are low compared with earlier 30-52% repair rates.
- **F8: Basic interaction grounding** — improved. Recent chats contain more direct response behavior, e.g. speakers answer objections and sometimes address each other directly.
- **F9: Outcome-specific closure language** — improved. The system now distinguishes consensus, fallback, and no-decision outcomes better than before, although closure realism is still open.
- **O1: Formulaic social scaffolding** — improved. Discourse-frame tracking now detects repeated social templates (agreement prefaces, endorsement phrases, "given the..." openers) and nudges the model to vary phrasing via prompt hints. Template phrases per run reduced.
- **O2: Card-reading and option-attribute language** — improved. Non-compare/non-vote turns now receive only option names (not full attribute lists), with a prompt rule limiting turns to one concrete fact. Attribute density significantly reduced.
- **O3: Semantic repetition** — improved. Claim-slot tracking detects when the same dimension (cost, time, comfort, etc.) has been discussed for an option, and nudges toward a new angle. Less content repetition across speakers.
- **O4: Concession bridges** — improved. When accepting a non-preferred option, persona-specific bridge guidance now requires residual concern, a condition, a trade-off, or a practical next step. Concessions feel more earned.
- **O5: Moderator timing** — improved. When no clear holdout or leading option exists, the moderator now asks a diagnostic question instead of pushing a premature compromise.
- **O6: Closure realism** — improved. Moderator closure now includes concrete next steps (book, check availability, confirm) and fallback closures name remaining concerns.
- **O7: Consensus/fallback support consistency** — fixed. `support_fraction` now correctly prioritizes explicit votes over initial preferences and doesn't double-count. Added 9 unit tests for consensus/outcome state consistency.
- **O8: Persona voice differentiation** — improved. `_speaking_habit` now produces more operational, behavioral descriptions (e.g. "blunt and brief", "builds bridges", "curious — asks before judging") instead of generic labels.
- **O9: Large-group novelty control** — improved. For n>=5, when the focus option has many covered claim slots, the router now boosts COMPARE/ASK/OBJECT acts and dampens SUPPORT to avoid restating already-covered points.

---

## Still open

### O10: Token efficiency is better, but still too expensive

**Symptom:**  
The repair loop is no longer the main visible bottleneck, but input tokens are still high relative to output tokens. Large runs still send many thousands of input tokens for short generated turns.

**Why it matters:**  
Slow generation makes iteration painful, and large prompts encourage the model to copy option-card language.

**Current status:**  
Open as an optimization issue. Deferred until quality regressions from O1–O9 are stable.

**Regression signal:**  
Track per-turn: input tokens, output tokens, prompt section sizes, act type, repair count, whether full or brief option board was used.

---

### Residual: Some formulaic phrases still leak through

Despite discourse-frame tracking (O1), the model still occasionally produces "you make a good/great point" and "wins me over" — especially when the frame hint fires but the model ignores it. This is model-dependent and may improve with a different base model. Warn-level robotic template counts are typically 2–3 per run now (down from 4–10 pre-refactor).

---

### Residual: Occasional back-to-back same speaker

The router sometimes lets the same speaker go twice in a row (1-2 instances per ~5 runs). The existing same-speaker penalty in routing handles most cases but is not airtight during confirmation phases.

---

## Current priority order

1. Continue prompt/token reduction (O10) now that quality is stabilised.
2. Monitor residual formulaic phrases and back-to-back routing across more runs.
3. Verify O9 large-group novelty control with n=5-7 runs (currently untested due to uni endpoint timeout on large setups).
