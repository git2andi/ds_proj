# Known Failures

Failures and quality issues tracked across evaluation passes. Each entry records
the symptom, root cause, fix status, and how to detect a regression.

## Fixes tried so far

**Structural / controller fixes:**
Commitment gating (accept/vote only binding in narrowing/confirmation). Hard-blocker integrity (can only back preferred option, auto-correct on rejected vote, vote loop prevention). Same-speaker back-to-back prevention (consecutive-speaker counter, deferred last-speaker in opening/vote/confirmation, confirmation skips rejected personas). Turn balancing (equal-ish turn counts). Phase enforcement (opening → discussion → narrowing → confirmation → closure). Setup split into two LLM calls to reduce timeouts. Persuasion gating (min speaker turns before stance shift). Coalition/preference diversity enforcement.

**Prompt / language fixes:**
Prompt rules trimmed from 12 to 6. Explicit VOTE guidance. ASK probability raised and guidance rewritten. Conversational register instructions (fragments, fillers, reactions encouraged). Response-length trait mapped to word-count targets. Persona speech_style surfaced in rule 1.

**Validation / repair layer:**
Deterministic checks for: self-narration, robotic templates, possessive openers, collective voice, card-reading, echo guard, opener variety, repeated starts, question chains, invented numbers, incomplete turns, unclear votes/accepts/rejects. Repair: one LLM retry per flagged turn with targeted hints. Collective voice auto-rewrite (we→I). Hedged-accept detection prevents premature closure.

**Evaluation infrastructure:**
16 diverse eval scenarios across 6 domains, 30 run configs. Eval runner checks: back-to-back, question density, opener variety, hard-blocker integrity, duplicate moderator lines, outcome sanity, robotic template count.


## Current analysis note: keep options, reduce over-control

This project should keep explicit options. The option board is not the main quality problem: it gives the simulator a controlled candidate set, reduces hallucinated alternatives, and makes evaluation easier. The current failure is that the *discussion around the options* is over-controlled and over-conditioned.

Do **not** remove options A/B/C/D as a first fix. Instead, make the option board less dominant in each turn prompt. Participants should talk about the options through their concerns, objections, tradeoffs, and replies to other participants, not by repeatedly reading the option cards.

Also do **not** remove persona depth entirely. Traits, goals, backstory, priorities, and concerns are useful, but they should not all be copied into every single generation prompt. The current per-turn prompt gives the model too much static profile information and not enough local interaction pressure. The fix is persona compression:

- Keep the full persona profile in setup/state.
- Derive a compact runtime persona card per turn.
- Include only the fields relevant to the current turn: current lean, one active concern, one speaking habit, one relationship/interaction cue, and any recent concession or unresolved disagreement.
- Convert broad traits into observable speaking behavior. Example: instead of `agreeableness=4`, use `tends to soften disagreement but still names the practical blocker`.

The goal is not to add more research papers or more modules. The goal is to make the generated dialogue behave like actual interaction: one person responds to another person, disagreements have visible reasons, and compromises are earned instead of scheduled.

## Follow-up implementation plan

Use this plan before adding new features. Each step should be tested with the same eval scenarios and compared against the current logs.

### P1: Turn style repairs into diagnostics, not extra LLM calls ✓

**Problem linked to:** O1, O3, O8, performance concern. **Done.**

10 style checks demoted from `repair` to `warn` severity: DUPLICATE_TURN, ECHOED_PHRASE, SELF_REPETITION, REPEATED_START, REPETITIVE_OPENER, ROBOTIC_TEMPLATE, POSSESSIVE_SUBJECT, CARD_READING, SELF_NARRATION, QUESTION_CHAIN. Detection still runs and issues appear in logs/metrics, but no LLM repair call fires for them. Structural checks unchanged: EMPTY, MULTI_TURN_OUTPUT, SPEAKER_PREFIX, INVALID_OPTION_REFERENCE, UNGROUNDED_NUMERIC_FACT, INVENTED_OPTION_ATTRIBUTE, UNCLEAR_VOTE/ACCEPT/REJECT, QUESTION_IN_CONFIRMATION, UNWANTED_QUESTION, INCOMPLETE_TURN. Config `repair_on_warning: false` can be toggled to re-enable if needed.

### P2: Replace the large per-turn prompt with a compact runtime card ✓

**Problem linked to:** O2, O3, O4, O7. **Done.**

Implemented `runtime_speaker_card()` in `prompts.py`. Per-turn prompt now includes: speaker name/role/voice, current lean, active concern, speaking habit (derived from traits, not raw numbers), one prior claim ("Already said"), and concession state if lean shifted. Full `speaker_card()` retained for debugging. `sim_utterance` rewritten: removed public board dump (replaced with one-line `_group_leans`), removed `own_recent` section, added `_responding_to_line` (explicit prior turn to react to), brief option cards for non-compare/non-vote acts. Prompt rules trimmed from 6 to 4. Result: dialogue input tokens ~50% lower (28k→12-17k for n=3), POSSESSIVE_SUBJECT down from 9-14 to 1-4 per run.

### P3: Make routing adjacency-first instead of phase-first ✓

**Problem linked to:** O3, O7, O4. **Done.**

Added `_unanswered_challenge()` to `TurnRouter`: when a recent OBJECT/PUSH_BACK turn targets an option, the champion of that option (or explicit addressee) is routed to respond before gap/coverage/balance logic. Uses `respond_to_turn` to anchor the prompt's "Responding to" line. Combined with `_responding_to_line()` in prompts (always shows the most relevant prior turn), open-question handling (already existed), and `respond_to_turn` on MoveIntent, the router now prioritizes: (1) answer pending question, (2) respond to challenge, (3) gap/coverage, (4) weighted speaker selection.

### P4: Keep options, but stop making every utterance option-card-shaped ✓

**Problem linked to:** O2, O3. **Done (via P2 prompt rewrite).**

Full option cards (upside/tradeoff/concern/best_for) now only rendered for COMPARE and VOTE acts. All other acts get brief option lines (name + attrs only). Combined with the anti-possessive rule ("Don't open with an option name") and rewritten opening guidance ("lead with your reason, not the option description"), card-reading and possessive-opener patterns dropped sharply.
- Add a rule to prefer human references over option-card references: respond to `Kai's setup concern`, not `Option B's effort attribute`.

Example target:

```text
Not: "Bistro Bliss has reasonable cost and variety."
Better: "Kai, the price point is exactly why I keep coming back to Bistro Bliss. The only thing I still need to know is whether the menu is broad enough."
```

### P5: Require visible concession bridges for stance changes ✓ (prompt-level)

**Problem linked to:** O4, O5. **Partially done.**

Concession bridge logic added to `_move_guidance` in `prompts.py`: when a speaker ACCEPT/VOTE for an option different from their `preferred_option`, the guidance explicitly names their original preference and asks them to say what changed ("You started out wanting X — say what convinced you this one works instead"). This fires BEFORE chorus detection so it's never short-circuited. The `runtime_speaker_card` also shows concession state ("Started with X, now leaning Y"). Full state-level enforcement (blocking stance changes without a triggering argument) is deferred — the prompt-level fix produces visible concession bridges in practice.

### P6: Reduce moderator dependence

**Problem linked to:** O5, O6, O8. **Partially done.**

The moderator already has distinct intervention types (stall/agreement/holdout), rate limiting (cooldown + cap), outcome-specific closure (consensus/fallback/unresolved use different language), and split diagnosis via `_camp_split`. Remaining: more specific conflict diagnosis and participant-led closure for some runs.

### P7: Add evaluation metrics that measure interaction, not only validity ✓

**Problem linked to:** O2, O3, O4, O7. **Done.**

Added to `evals/run_eval.py`: `named` (turns mentioning another participant by name), `responsive` (turns with name-mention, "you"/"your", or formal `respond_to_turn`), `self_rep` (SELF_REPETITION flagged count), `echoed` (ECHOED_PHRASE count). Displayed as INFO line per run. Typical values after refactor: named 0-6%, responsive 8-21%, self_rep 0, echoed 0-4.

## Fixed

### F1: Mid-discussion "accept" counted as binding commitment
- **Symptom**: Premature fake-unanimous outcomes while chat is still arguing (12/19 accepts in an n=7 run happened mid-discussion).
- **Root cause**: The model liberally tagged ordinary discussion lines `stance=accept`; these were recorded as real acceptances.
- **Fix**: Commitment only honoured on routed decision turns (narrowing vote / confirmation). `parsing._resolve_move`, gated by `_DECISION_ACTS`.
- **Regression signal**: `run.json` turns in discussion phase with `act_type=accept` that have `explicit_vote` set.

### F2: Hard blockers could be talked into accepting non-preferred options
- **Symptom**: A hard-blocker who preferred A "accepted" C then B, turning a deadlock into fake consensus. Also: hard blocker stuck in vote loop.
- **Root cause**: No guard preventing hard-blocker stance shifts; rejected votes left the persona "unvoted" causing infinite re-pick.
- **Fix**: Hard blocker only ever backs their preferred option; any vote/accept elsewhere ignored. Auto-corrects rejected votes to preferred option. `_vote_intent` forces hard blocker focus to preferred option.
- **Regression signal**: `evals/run_eval.py` `hard_blocker_integrity` check.

### F3: "Wall of I" — every turn opens with self-anchored stance
- **Symptom**: "I'm drawn to X because...", "I worry that Y...", "I prefer Z..." — parallel monologues.
- **Fix**: Prompt rewrite (react-first), `validation._check_opener_variety` / `REPETITIVE_OPENER`, `REPEATED_START` promoted to repair.
- **Regression signal**: `evals/run_eval.py` `opener_variety` check.

### F4: Possessive opener tic ("X's `<feature>`")
- **Symptom**: ~40% of turns opened with `<OptionName>'s <attribute>`.
- **Fix**: `validation._check_robotic_phrasing` / `_possessive_openers`.
- **Regression signal**: `POSSESSIVE_SUBJECT` in validation issues.

### F5: Hedged acceptance closing the chat
- **Symptom**: "might be okay if there's free time" recorded as firm accept.
- **Fix**: `parsing._HEDGED_ACCEPT` — tentative/conditional acceptance stays neutral.
- **Regression signal**: `test_parsing.py::test_hedged_accept_*`.

### F6: Moderator repeating stock lines verbatim
- **Symptom**: Two identical moderator nudges in one run.
- **Fix**: Prior facilitator lines fed back into moderator prompt with "say it differently" rider.
- **Regression signal**: `evals/run_eval.py` duplicate moderator lines check.

### F7: Formulaic templates leaking through
- **Symptom**: "X outweighs Y", "makes me think...", "Given the discussion...", etc.
- **Fix**: `validation._ROBOTIC_TEMPLATES` deterministic catch + repair.
- **Regression signal**: `ROBOTIC_TEMPLATE` in validation issues.

### F8: Collective "we" for personal stances
- **Symptom**: "We consider...", "We prioritize..." — individual sounds like a committee.
- **Fix**: `validation.fix_collective_voice` deterministic rewrite (we→I) + expanded cognition verbs (evaluate/weigh/assess/factor).
- **Regression signal**: `test_validation.py::test_collective_voice_*`.

### F9: Card-reading — turns parrot option card text verbatim
- **Symptom**: Upside/tradeoff prose quoted word-for-word from the card.
- **Fix**: `validation._check_card_reading` flags 4+ word verbatim matches.
- **Regression signal**: `CARD_READING` in validation issues.

### F10: Self-narration ("I should consider...")
- **Symptom**: "I should consider...", "I should prioritize...", "I must consider..." — internal reasoning exposed as dialogue.
- **Fix**: `validation._check_self_narration` flags the pattern. Prompt bans narrating own thinking.
- **Note**: Detection works but repair often reproduces the pattern (see O1). Analysis recommends treating as hard rejection.
- **Regression signal**: `SELF_NARRATION` in validation issues.

### F11: Same-speaker back-to-back turns
- **Symptom**: Same participant speaks 2+ times consecutively, especially in narrowing/confirmation.
- **Fix**: `_consecutive_same_speaker` helper excludes persona after 2+ consecutive turns. Confirmation skips rejected personas. All phase intents defer last speaker.
- **Regression signal**: `evals/run_eval.py` `same_speaker_back_to_back` check.

### F12: Zero questions in discussions
- **Symptom**: question_density = 0.0 despite ask probability.
- **Fix**: Raised ask base to 0.20, reduced damping to 0.40. ASK guidance rewritten.
- **Regression signal**: `evals/run_eval.py` `zero_question_density` check.

### F13: Unclear votes — model ignores vote instruction
- **Symptom**: Model writes discussion-style turn instead of naming a pick.
- **Fix**: Separate VOTE branch in `_move_guidance`.
- **Regression signal**: `UNCLEAR_VOTE` in validation issues.

### F14: Collective voice missing cognition verbs
- **Symptom**: "We should evaluate/weigh/assess/factor" not caught.
- **Fix**: Added verbs to `_COLLECTIVE_VOICE` regex.
- **Regression signal**: `test_validation.py::TestCollectiveVoice`.

### F15: Setup call timeouts on uni endpoint
- **Symptom**: Single large setup call times out at 240s under load.
- **Fix**: Split into two calls: (1) scenario/options, (2) personas given those options. Each ~2700 chars vs ~5000 combined.
- **Regression signal**: `RuntimeError: Scenario setup failed` with timeout.

## Open

### O1: Repair layer ineffective on persistent patterns
- **Symptom**: Self-narration ("I should consider..."), "Considering..." openers, and possessive subjects are detected but the single repair attempt reproduces the same pattern. Repair rate is 30–52% across runs, meaning 8–17 extra LLM calls per run, yet flagged patterns still appear in the final transcript. The sci-fi run has repair_rate=0.52 and 10 flagged turns remaining after repair.
- **Impact**: Each repair is a full extra LLM call (~800–1200 tokens). With 30–50% of turns triggering repair, this roughly doubles the time per flagged turn — a likely main cause of the overall slowdown from ~2 min to ~10–15 min per run.
- **Status**: Addressed by P1 — all 10 style checks demoted to warn-level diagnostics. Repair now fires only for structural errors. Expected repair rate drops from 30–52% to near 0% for most runs. Style patterns still detected and logged for analysis.
- **Detection**: Compare `repaired_turns` vs `flagged_turns` in metrics — repaired_turns should now be near 0; flagged_turns will increase (style issues are still detected, just not repaired).

### O2: Repetition and content recycling
- **Symptom**: Same speaker repeats their own point; different speakers chain the same noun phrase; one option dominates discussion; confirmation devolves into restating the same justification.
- **Status**: Significantly improved. Runtime card now shows last 2 prior claims with "don't repeat" directive. Brief option cards for non-compare acts reduce card-reading. Self-repetition count is 0 across recent runs. Echoed phrases down to 0-4 per run. Semantic repetition (same idea, different words) still not caught automatically.
- **Detection**: `SELF_REPETITION` count, `ECHOED_PHRASE` count, `option_coverage` imbalance. `evals/run_eval.py` interaction metrics.

### O3: Slogan-like utterances — no interactional grounding
- **Symptom**: Turns are free-floating preference statements rather than responses to specific points.
- **Status**: Significantly improved. `_responding_to_line` anchors each turn to a specific prior turn. Challenge-response routing (`_unanswered_challenge`) creates natural adjacency pairs. Responsive rate (turns with name/you/anchor) 8-21% in recent runs. Possessive openers down from 9-14 to 1-2 per run. Model still defaults to formal declarative sentences (llama3.3 limitation).
- **Detection**: `evals/run_eval.py` `responsive` metric. Manual review.

### O4: Under-motivated stance changes
- **Symptom**: Participants accept without visible concession bridge.
- **Status**: Improved. ACCEPT/VOTE guidance now explicitly names the speaker's original preference when they're accepting/voting for a different option ("You started out wanting X — say what convinced you"). Concession check fires before chorus detection. Runtime card shows concession state. Recent runs show visible bridges ("switching from X since Y's variety won me over").
- **Detection**: `run.json` turns where `current_preference` changes. Manual review of acceptance language.

### O5: Consensus vs fallback closure confusion
- **Symptom**: Fallback outcomes use closure language that sounds like consensus.
- **Status**: Addressed. `moderator_closure_prompt` already uses outcome-specific language: consensus gets "wrap it up warmly," fallback gets "acknowledging it wasn't unanimous," unresolved gets "name the actual split and a concrete next step."
- **Detection**: Compare `outcome_status` with the tone of the final moderator/closure lines.

### O6: Mechanical moderator
- **Symptom**: Moderator sounds generic across runs ("we're going in circles").
- **Status**: Improved. Stall prompt now includes `_conflict_dimension` (names the actual concern trade-off, e.g. "The real tension is cost vs comfort"). Holdout prompt includes the holdout's `main_concern` so the moderator can address it directly. Recent runs produce moderator lines like "The disagreement is between comfort and trying something new" and "Are you opposed to Python because it's too familiar?". Cross-run variety still limited by model style.
- **Detection**: Manual review. Compare moderator lines across runs.

### O7: Weak addressee handling
- **Symptom**: Speakers rarely address a specific person by name.
- **Status**: Partially addressed. Challenge-response routing directs responses to specific people. `_responding_to_line` names the person being responded to. Model uses "you"/"your" (responsive rate 8-21%) more than names (named rate 0-6%). Name usage is model-dependent (llama3.3 prefers indirect references).
- **Detection**: `evals/run_eval.py` `named` and `responsive` metrics.

### O8: Awkward closings
- **Symptom**: Closings are formulaic ("Sounds good, looking forward to X").
- **Status**: Improved. `farewell_line` prompt now sets persona-aware tone based on outcome relationship: "You got your top pick — be pleased but brief" vs "This wasn't your first choice — show mild acceptance" vs "You came around — brief and genuine" vs "No decision — mild 'oh well'". Some runs show differentiation ("I'm a happy girl" for the winner). Model compliance varies (llama3.3 often defaults to generic positive closings regardless of tone).
- **Detection**: Manual review of closure turns.

## Performance concern — resolved

Dialogue generation was ~10–15 min per run. Now ~3–6 min for n=3 and ~8 min for n=5.

| Fix | Impact |
|-----|--------|
| **Repair rate 30-52% → 0-4%** | Biggest factor. Eliminates 8-17 extra LLM calls per run. |
| **Prompt size ~50% smaller** | Runtime card + brief options + no public board → fewer input tokens per call (~500 vs ~900). |
| **Turn count unchanged** | 15-30 participant turns, same as before. |
