# Dialogue Quality Refactor Plan

This plan replaces the previous completed refactor plan.  
The last refactor improved repair rate, prompt size, interaction grounding, and basic concession behavior. The remaining work is not to add more heavy repair layers, but to make the dialogue more socially realistic with simpler, more targeted control.

Core rule: **less is more**. Prefer small deterministic state checks and compact prompt guidance over large prompts, repeated repair calls, or many overlapping validators.

---

## Current diagnosis after latest runs

The simulator now produces more coherent and more responsive discussions. The old problems of obvious self-narration and high repair rates are mostly reduced.

The remaining quality problems are:

1. The dialogue uses repeated social templates.
2. Speakers still read too much from option cards.
3. Semantic repetition persists even when exact repetition is gone.
4. Concessions are visible but too smooth.
5. The moderator sometimes pushes convergence before the conflict is resolved.
6. Final closures are functional but socially thin.
7. Fallback / consensus state can become inconsistent.
8. Personas differ by preference more than by linguistic voice.
9. Larger groups stay coherent but become repetitive.
10. Token usage is still high relative to output.

The next work should target these issues in order.

---

## Step 1: Freeze the current good behavior

Before changing generation again, keep regression coverage for what improved.

Preserve:

- low repair rate,
- no obvious `I should consider...` self-narration,
- no same-speaker back-to-back unless structurally unavoidable,
- direct response behavior,
- concession bridges,
- outcome-specific closure,
- compact runtime speaker card.

Add or keep a small golden set of runs:

- n=2 restaurant / simple choice,
- n=3 birthday or university project,
- n=4 flight booking,
- n=5 birthday or board game,
- n=6 road trip.

Do not optimize one case while breaking another.

---

## Step 2: Add discourse-frame repetition tracking

Problem: the model repeats social scaffolding such as `fair point`, `good point`, `wins me over`, `seals the deal`.

Implementation direction:

- Add a lightweight `discourse_frame` classifier for generated turns.
- Start with simple regex / phrase-family buckets:
  - agreement-preface,
  - concession-preface,
  - option-endorsement,
  - compromise-acceptance,
  - moderator-diagnostic,
  - final-closure.
- Track counts per run and per speaker.
- If a frame was recently used, add a short prompt hint: `Do not use the same agreement/concession phrasing again. Respond directly.`

Avoid immediate LLM repair. This should be a routing/prompt nudge first.

Success signal:

- Fewer repeated phrases like `fair point`, `good point`, `seals the deal`.
- Same social function remains, but wording varies naturally.

---

## Step 3: Reduce card-reading further

Problem: turns still repeat option-table attributes too often.

Implementation direction:

- For most acts, do not expose full option attributes.
- Use full option details only for:
  - COMPARE,
  - explicit vote,
  - moderator summary,
  - first mention of an option.
- For normal response turns, provide only:
  - option name,
  - one relevant concern,
  - one relevant contrast.
- Add an `attribute_budget` per turn:
  - normal response: max 1 concrete attribute,
  - compare: max 2 concrete attributes,
  - moderator summary: max 3 concrete attributes.

Prompt rule:

> Use the option details as background. Do not restate the option card. Speak like someone discussing the choice, not reading a table.

Success signal:

- Fewer repeated numbers and labels.
- More situated language, e.g. `That sounds too cramped for Friday` instead of `limited seating and may be crowded`.

---

## Step 4: Add semantic claim-slot tracking

Problem: exact repetition is lower, but the same idea is repeated in different words.

Implementation direction:

Represent each option discussion as claim slots:

- cost,
- time/distance,
- comfort,
- flexibility,
- group size,
- novelty,
- risk,
- effort,
- quality,
- fairness.

For each turn, infer a rough claim slot from the intent and option focus. This can be deterministic and approximate at first.

Before generating a turn, pass:

- `already_said_for_option`,
- `unanswered_objections`,
- `new_angle_needed`.

If the speaker would repeat an already-covered slot, route them to one of:

- answer an objection,
- add a condition,
- compare against another option,
- concede,
- ask a clarifying question.

Success signal:

- The discussion progresses by adding new angles.
- Large groups do not repeat the same option benefit six times.

---

## Step 5: Strengthen concession bridges

Problem: stance changes are visible, but too clean.

Implementation direction:

When a participant accepts or votes for a non-preferred option, require a concession bridge with one of these forms:

1. residual concern:
   - `I still worry about X, but...`
2. condition:
   - `I can accept it if we...`
3. trade-off:
   - `I prefer A, but B handles X better.`
4. practical next step:
   - `Let's check X before finalizing.`

Add a `concession_type` field to the move intent.

Do not make every concession long. One short sentence is enough.

Success signal:

- Consensus feels earned.
- Holdouts do not suddenly become fully convinced without preserving their original concern.

---

## Step 6: Change moderator timing

Problem: the moderator sometimes proposes a compromise too early.

Implementation direction:

Before the moderator proposes a target option, require the state to contain:

- current leading option,
- holdout speaker(s),
- holdout concern,
- possible condition that would satisfy the holdout.

If not available, the moderator should ask a diagnostic question instead.

Moderator intervention types:

1. conflict diagnosis,
2. targeted holdout question,
3. option modification proposal,
4. vote call,
5. fallback framing,
6. closure.

Use the least forceful intervention that fits the state.

Success signal:

- The moderator no longer sounds like it is forcing an outcome.
- Holdout concerns are handled before closure.

---

## Step 7: Add option modification as a compromise mechanism

Problem: real groups often modify an option instead of merely choosing A/B/C/D.

Implementation direction:

Allow lightweight option modifications such as:

- Restaurant Party + fixed budget cap,
- Backyard BBQ + tent / backup plan,
- Mountain Adventure + carpooling,
- Ticket to Ride + short backup game,
- Speedster + seat check / baggage check.

Represent this as:

```python
proposal = {
    "base_option": "B",
    "condition": "if we can keep cost under $X",
    "modification": "carpool to reduce travel cost"
}
```

This should not create entirely new options. It should attach a condition or implementation detail to an existing option.

Success signal:

- Compromises feel practical.
- Fallback decisions can include the remaining condition instead of pretending everyone fully agrees.

---

## Step 8: Improve final closure

Problem: endings mark the decision but often do not feel socially complete.

Implementation direction:

Final closure should depend on outcome type.

Consensus:

- confirm the selected option,
- mention why it worked for the group,
- add one next step.

Fallback:

- state that it is a majority/fallback decision,
- mention the unresolved concern,
- add a condition or next step.

No decision:

- state what blocked agreement,
- suggest what information is missing.

Participant farewell should be optional and short. It does not always need a goodbye. Often a final practical acknowledgment is more natural.

Success signal:

- Endings sound like real decision closure, not just transcript termination.

---

## Step 9: Audit outcome-state consistency

Problem: support fraction, outcome type, votes, and closure wording can diverge.

Implementation direction:

Add tests for:

- explicit votes by participant,
- final lean by participant,
- support fraction,
- outcome type,
- closure prompt type,
- final moderator wording.

Rules:

- consensus requires the configured consensus threshold.
- fallback must be named as fallback/majority/working decision.
- no-decision must not receive consensus wording.
- support numbers in metrics and outcome must match.

Success signal:

- No run reports fallback with contradictory support fractions.
- Final wording matches actual state.

---

## Step 10: Add persona voice constraints without expanding prompts

Problem: personas differ in preference, but not enough in linguistic behavior.

Implementation direction:

Keep the compact runtime card, but make `speaking_habit` more operational.

Examples:

- direct speaker: short sentence, no long preface,
- cautious speaker: hedge once, mention risk,
- enthusiastic speaker: positive but not exaggerated,
- analytical speaker: compare one concrete trade-off,
- peacemaker: acknowledges two sides then asks/bridges.

Do not pass the full backstory every turn. Convert traits into one stable behavior instruction.

Success signal:

- Speakers become identifiable without reading their names.
- Voice difference does not become caricature.

---

## Step 11: Add large-group contribution roles

Problem: n=5-7 discussions remain coherent but repetitive.

Implementation direction:

For larger groups, assign temporary contribution roles per phase:

- first supporter,
- objector,
- evidence/detail provider,
- bridge-builder,
- holdout,
- summarizer,
- voter.

A speaker should not be selected only to restate an already-covered point. If no novel contribution exists, skip them until vote/closure.

Success signal:

- Larger groups do not require everyone to comment on every option.
- Turns become fewer but more purposeful.

---

## Step 12: Token and speed audit

Problem: performance improved after repair reduction, but input/output ratio is still high.

Implementation direction:

Instrument every LLM call with:

- act type,
- speaker,
- phase,
- input tokens,
- output tokens,
- prompt section sizes,
- option board mode: full / brief / none,
- repair triggered: yes/no,
- elapsed time.

Then rank the largest time/token sources.

Likely optimization order:

1. option board text,
2. runtime context history,
3. repeated group state,
4. moderator prompts,
5. repair calls.

Do not optimize by adding more post-processing first. Remove or shrink context before adding new logic.

Success signal:

- Lower input tokens per generated turn.
- No quality regression in the golden set.
- Fewer long prompts for simple response acts.

---

## Recommended implementation order

1. Freeze regression tests and golden examples.
2. Add discourse-frame repetition tracking.
3. Reduce option-card exposure and add attribute budgets.
4. Add semantic claim-slot tracking.
5. Strengthen concession bridge types.
6. Adjust moderator timing and intervention type.
7. Add option modification as compromise.
8. Improve closure with next steps / fallback framing.
9. Audit outcome-state consistency.
10. Improve compact persona voice behavior.
11. Add large-group contribution roles.
12. Run token/speed audit and simplify largest prompt sections.

---

## What not to do

- Do not re-enable broad style repair loops.
- Do not add many new validators that trigger extra LLM calls.
- Do not put full persona profiles back into every turn prompt.
- Do not solve realism by simply increasing turn count.
- Do not force every participant to speak in every phase.
- Do not treat majority fallback as consensus.
- Do not overfit to one transcript.

The goal is not more machinery. The goal is fewer, more purposeful turns with clearer social function.
