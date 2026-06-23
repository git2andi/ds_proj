# Dialogue Quality Refactor Plan

Completed 2026-06-23. This plan kept the explicit option board and targeted over-control in prompting, repair, routing, and closure behavior.

## Core diagnosis

The simulator produced valid decision transcripts, but not sufficiently interactive dialogue. Root causes were: too many style repairs, too much static persona dumped into every prompt, phase-first speaker selection, option-card language dominating every turn, stance shifts without visible persuasion, and generic moderator nudges.

## Design rule: keep full persona, prompt compact persona

Full persona profile (traits, goals, backstory, concerns, scores) stays in state for consistency. Per-turn prompt gets only a compact `runtime_speaker_card`: current lean, one concern, one speaking habit, last 2 prior claims, concession state. Numeric traits converted to behavioral descriptions.

## Implementation — all steps done

### Step 1: Repair policy ✓
10 style checks demoted from `repair` to `warn` severity in `validation.py`. Repair fires only for structural errors (missing trailer, invented option, malformed vote, multi-speaker, hard-blocker violation, invented numbers). Repair rate 30–52% → 0–4%. Config `repair_on_warning: false` can re-enable.

### Step 2: Runtime speaker card ✓
`runtime_speaker_card()` in `prompts.py` replaces `speaker_card()` in generation. Includes: name/role/voice, current lean, active concern, speaking habit (from traits), last 2 prior claims with "don't repeat" directive, concession state. `sim_utterance` rewritten: public board → one-line `_group_leans`, own_recent section removed, rules trimmed from 6 to 4. Token reduction ~50%.

### Step 3: Adjacency-first routing ✓
`_unanswered_challenge()` in `router.py` checks recent OBJECT/PUSH_BACK turns and routes the option champion to respond before gap/coverage logic. Uses `respond_to_turn` on MoveIntent. Combined with existing open-question handling, the router priority is: (1) answer question, (2) respond to challenge, (3) coverage gap, (4) weighted speaker selection.

### Step 4: Turn-response prompt ✓
`_responding_to_line()` finds the most relevant prior turn and includes it as "Responding to X: '...'" in the prompt. Full option cards only for COMPARE/VOTE; all other acts get brief attrs-only lines via `_option_brief()`. Anti-possessive rule: "Don't open with an option name." Opening guidance: "lead with your reason, not the option description."

### Step 5: Concession bridges ✓
`_move_guidance` for ACCEPT/VOTE checks if the speaker is changing from their preferred option. If so, guidance explicitly names the original preference ("You started out wanting X — say what convinced you"). Fires before chorus detection. `runtime_speaker_card` shows "Started with X, now leaning Y." Runs produce visible bridges.

### Step 6: Moderator improvements ✓
`_conflict_dimension()` extracts the top 2 persona concerns and names the trade-off. Stall prompt uses "Name the actual disagreement" instead of "we're repeating." Holdout prompt includes the holdout's `main_concern`. Runs produce "The disagreement is between comfort and trying something new."

### Step 7: Outcome-specific closure ✓
`moderator_closure_prompt` uses distinct language per outcome type. `farewell_line` sets persona-aware tone: got pick ("pleased but brief") / came around ("brief and genuine") / outvoted ("mild acceptance") / no decision ("oh well").

### Step 8: Interaction quality metrics ✓
`evals/run_eval.py` reports per run: `named` (turns mentioning another participant), `responsive` (name + "you"/"your" + formal anchor), `self_rep` (SELF_REPETITION count), `echoed` (ECHOED_PHRASE count). Typical values: named 0–20%, responsive 8–40%, self_rep 0, echoed 0–4.

## Results

| Metric | Before | After |
|--------|--------|-------|
| Dialogue tokens (n=3) | ~28,000 | 12,000–17,000 |
| Repair rate | 30–52% | 0–4% |
| POSSESSIVE_SUBJECT/run | 9–14 | 1–2 |
| Self-repetition | common | 0 |
| Responsive rate | unmeasured | 8–40% |
| Run time (n=3) | ~10–15 min | ~3–6 min |

## Remaining limitations

- Semantic repetition (same idea, different words) not caught automatically.
- Name-mention rate 0–20% — model prefers "you"/"your" over names (llama3.3 limitation).
- Farewell differentiation model-dependent — llama3.3 often defaults to generic positive closings.
- Cross-run moderator variety limited by model style.
- Same-speaker back-to-back still occurs occasionally when the only unvoted person just spoke (edge case in narrowing phase).
