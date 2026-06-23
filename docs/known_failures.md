# Known Failures

Tracked failures and quality issues in the dialogue simulator.  
The fixed section is intentionally brief: it records what appears to be stable enough to avoid re-solving the same issue.  
The open section describes what still damages dialogue realism in the latest runs.

Last updated: 2026-06-23, after reviewing runs 13-17 and transcripts 13-17.

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

---

## Still open

### O1: Formulaic social scaffolding

**Symptom:**  
The dialogues now sound more interactive, but many turns still rely on repeated social templates:

- `That's a fair point...`
- `You make a good point...`
- `X is a major draw...`
- `Y seals the deal...`
- `Z wins me over...`
- `What would it take...`

**Why it matters:**  
These phrases are individually plausible, but repeated across a run they create a generated-dialogue feel. The problem is no longer "no interaction"; it is "interaction expressed through a small template set."

**Current status:**  
Open. Regex-based robotic-template detection is not enough because the issue is partly semantic and distributional: too many turns share the same discourse shape even when the literal words differ.

**Regression signal:**  
Track repeated discourse frames across a run, not only exact repeated strings. Add counts for phrase families such as agreement-preface, concession-preface, option-seals-deal, and repeated compromise formulas.

---

### O2: Card-reading and option-attribute language still dominate

**Symptom:**  
Participants still talk too much like they are reading from the option board:

- exact prices, distances, playtimes, comfort labels, flexibility labels
- repeated phrases such as `medium comfort`, `high flexibility`, `60-minute playtime`, `multiple cuisines`
- repeated option summaries instead of situated reasons

**Why it matters:**  
Real participants do mention constraints, but they do not repeatedly restate the option table. This makes the dialogue feel like structured evaluation, not social discussion.

**Current status:**  
Open. Brief option cards helped, but the generated text still overuses option attributes, especially in larger group runs.

**Regression signal:**  
Measure option-attribute density per turn. Flag turns where more than one factual attribute is restated unless the act is explicitly `COMPARE`, `SUMMARIZE`, or `VOTE`.

---

### O3: Semantic repetition remains open

**Symptom:**  
The latest runs have fewer exact self-repetition flags, but the same idea is often restated in new words. Example patterns:

- "Speedster is good because six hours is short" repeated several times.
- "Mountain Adventure has scenic hiking trails and high flexibility" repeated across different speakers.
- "Ticket to Ride is easy to learn and works for groups" repeated as the main justification.

**Why it matters:**  
String repetition may be fixed while content repetition remains. The dialogue then appears locally varied but globally stagnant.

**Current status:**  
Open. Last-claim memory helps, but it does not prevent semantically equivalent claims.

**Regression signal:**  
Track per-option claim slots: cost, time, comfort, flexibility, group size, novelty, complexity. A speaker may reuse a slot only if they add a new detail, answer an objection, or change stance.

---

### O4: Concession bridges are visible but still too clean

**Symptom:**  
Participants now sometimes explain why they move from their preferred option to a compromise. However, the concession often feels too smooth and final. Real speakers usually show hesitation, residual concern, or a condition.

**Why it matters:**  
Consensus feels earned only if stance changes preserve the speaker's original concern. Otherwise the system appears to force convergence.

**Current status:**  
Partially fixed, still open. Concession bridges exist, but they are often too generic.

**Regression signal:**  
When a speaker accepts a non-preferred option, require at least one of:

- residual concern: `I'm still worried about X`
- condition: `as long as we do Y`
- trade-off statement: `I still prefer A, but B handles X better`
- practical next step: `let's check/confirm/book/limit Y`

---

### O5: Moderator sometimes pushes convergence too early

**Symptom:**  
The moderator now names conflicts better, but sometimes suggests a target compromise before the disagreement is actually resolved.

**Why it matters:**  
This makes the moderator feel like a controller forcing an outcome instead of a facilitator helping the group reason.

**Current status:**  
Open. Conflict-dimension detection improved, but intervention timing and intervention type still need work.

**Regression signal:**  
Before proposing a compromise, the moderator should identify:

1. the current leading option,
2. the holdout or unresolved concern,
3. what would need to change for the holdout to accept.

If these are missing, the moderator should ask a diagnostic question instead of proposing closure.

---

### O6: Closure realism remains weak

**Symptom:**  
Endings are functional but often abrupt:

- `Speedster works for everyone...`
- `Mountain Adventure is the way to go...`
- `Ticket to Ride it is...`

**Why it matters:**  
Real group decisions usually close with a concrete next step or a lingering condition. The current endings often mark a decision but do not feel socially complete.

**Current status:**  
Partially fixed, still open. Outcome-specific closure exists, but final turns need more grounded social closure.

**Regression signal:**  
Every final decision should include one of:

- action: book/check/reserve/confirm/buy
- unresolved condition: `if the cost works`
- assigned responsibility: `I'll check the reservation`
- explicit fallback framing: `not everyone's first choice, but...`

---

### O7: Consensus vs fallback support inconsistency

**Symptom:**  
Some runs report fallback even while support numbers appear inconsistent. In one reviewed birthday run, the outcome text referenced 0.80 support while metrics showed 0.60 support.

**Why it matters:**  
The dialogue may sound acceptable, but the state layer becomes unreliable. If outcome semantics are wrong, the moderator may close incorrectly.

**Current status:**  
Open. Needs state/outcome audit.

**Regression signal:**  
Add tests that compare:

- final explicit votes,
- final support fraction,
- outcome type,
- moderator closure wording,
- farewell tone.

These must agree.

---

### O8: Persona voices remain too similar

**Symptom:**  
Personas differ mainly by option preference and concern, not by stable linguistic behavior. A budget person talks about budget, a comfort person talks about comfort, but their actual speech rhythm and interaction style are often similar.

**Why it matters:**  
Realistic dialogue requires both stance diversity and voice diversity. Without voice diversity, participants feel like one model wearing different labels.

**Current status:**  
Open. Numeric traits and speech styles exist, but they are not strongly visible in the final wording.

**Regression signal:**  
Evaluate each persona over a run for stable differences in:

- average length,
- hedging,
- directness,
- question frequency,
- agreement style,
- emotional intensity,
- use of concrete examples.

---

### O9: Large-group scaling still works structurally but becomes verbose/repetitive

**Symptom:**  
The 6-person road-trip run stayed coherent, but it became long and repeated the same core claims across speakers.

**Why it matters:**  
For 5-7 participants, balance alone is not enough. The controller must avoid every speaker restating every option dimension.

**Current status:**  
Open. Speaker balancing is acceptable, but contribution selection needs stronger novelty control.

**Regression signal:**  
For n >= 5, each speaker should contribute one distinct function per phase:

- new criterion,
- objection,
- answer to objection,
- concession,
- summary,
- final vote.

Avoid routing a speaker just to repeat an already-covered option advantage.

---

### O10: Token efficiency is better, but still too expensive

**Symptom:**  
The repair loop is no longer the main visible bottleneck, but input tokens are still high relative to output tokens. Large runs still send many thousands of input tokens for short generated turns.

**Why it matters:**  
Slow generation makes iteration painful, and large prompts encourage the model to copy option-card language.

**Current status:**  
Open as an optimization issue. Performance is improved compared with the worst version, but still not lean.

**Regression signal:**  
Track per-turn:

- input tokens,
- output tokens,
- prompt section sizes,
- act type,
- repair count,
- whether full or brief option board was used.

Optimize the largest prompt sections first.

---

## Current priority order

1. Reduce formulaic social scaffolding.
2. Reduce card-reading / option-attribute repetition.
3. Add semantic claim-slot tracking.
4. Improve concession bridges with residual concern or conditions.
5. Audit consensus/fallback support consistency.
6. Improve moderator intervention timing.
7. Improve closure with concrete next steps.
8. Add stronger persona voice differentiation.
9. Add large-group novelty control.
10. Continue prompt/token reduction only after quality regressions are stable.
