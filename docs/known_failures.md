# Known Failures

Tracked failures and quality issues. Each fixed entry notes what was wrong and the regression signal. Open entries describe the remaining symptom and current status.

## Fixed

- **F1: Mid-discussion accept as binding** — `parsing._resolve_move` gates commitment to decision acts only. Signal: `run_eval.py` `mid_discussion_accepts`.
- **F2: Hard blockers folding** — hard blocker only ever backs preferred option; auto-corrects rejected votes. Signal: `run_eval.py` `hard_blocker_integrity`.
- **F3: "Wall of I" openers** — prompt rewrite (react-first) + `_check_opener_variety`. Signal: `run_eval.py` `opener_variety`.
- **F4: Possessive opener tic** — `_possessive_openers` regex + anti-possessive prompt rule. Signal: `POSSESSIVE_SUBJECT` in issues.
- **F5: Hedged accept closing chat** — `parsing._HEDGED_ACCEPT` clamps tentative acceptance to neutral. Signal: `test_parsing.py::test_hedged_accept_*`.
- **F6: Duplicate moderator lines** — prior facilitator lines fed back with "say it differently" rider. Signal: `run_eval.py` duplicate check.
- **F7: Formulaic templates** — `_ROBOTIC_TEMPLATES` regex detection. Signal: `ROBOTIC_TEMPLATE` in issues.
- **F8: Collective "we" for personal stances** — `fix_collective_voice` deterministic rewrite (we→I). Signal: `test_validation.py::TestCollectiveVoice`.
- **F9: Card-reading** — `_check_card_reading` flags 4+ word verbatim matches from option prose. Signal: `CARD_READING` in issues.
- **F10: Self-narration** — `_check_self_narration` regex flags "I should consider/prioritize/think about". Signal: `SELF_NARRATION` in issues.
- **F11: Same-speaker back-to-back** — `_consecutive_same_speaker` helper + deferred last-speaker in all phase intents. Signal: `run_eval.py` `same_speaker_back_to_back`.
- **F12: Zero questions** — ask base raised to 0.20, ASK guidance includes "end with '?'". Signal: `run_eval.py` `zero_question_density`.
- **F13: Unclear votes** — separate VOTE branch in `_move_guidance`. Signal: `UNCLEAR_VOTE` in issues.
- **F14: Collective voice missing verbs** — added evaluate/weigh/assess/factor to `_COLLECTIVE_VOICE`. Signal: `test_validation.py::TestCollectiveVoice`.
- **F15: Setup call timeouts** — split into two LLM calls (scenario, then personas). Signal: `RuntimeError: Scenario setup failed`.

## Open

### O1: Repair layer was ineffective — resolved
Style checks (10) demoted to warn-only diagnostics (P1). Repair fires only for structural errors. Repair rate 30–52% → 0–4%. Performance recovered (~3–6 min for n=3 vs ~10–15 min before).

### O2: Repetition and content recycling — mostly resolved
Runtime card shows last 2 prior claims with "don't repeat" directive. Brief option cards for non-compare acts. Self-repetition count 0 across all recent runs. Echoed phrases 0–4/run. Semantic repetition (same idea, different words) still not caught automatically.

### O3: Slogan-like utterances — mostly resolved
`_responding_to_line` anchors each turn to a prior turn. `_unanswered_challenge` routing creates adjacency pairs. Anti-possessive prompt rule. POSSESSIVE_SUBJECT down from 9–14 to 1–2/run. Responsive rate 8–40%. Model still defaults to formal register (llama3.3 limitation).

### O4: Under-motivated stance changes — mostly resolved
ACCEPT/VOTE guidance names the original preference when changing stance. Concession check fires before chorus detection. Runtime card shows concession state. Runs show visible bridges ("switching from X since Y's variety won me over"). Full state-level enforcement (blocking changes without a triggering argument) not implemented.

### O5: Closure confusion — resolved
`moderator_closure_prompt` uses outcome-specific language. `farewell_line` sets persona-aware tone (got pick / came around / outvoted / no decision).

### O6: Mechanical moderator — improved
`_conflict_dimension` names the concern trade-off (e.g. "cost vs comfort"). Holdout prompt includes `main_concern`. Runs produce lines like "The disagreement is between comfort and trying something new." Cross-run variety still limited by model style.

### O7: Weak addressee handling — partially addressed
Challenge-response routing + `_responding_to_line`. Model uses "you"/"your" (responsive 8–40%) more than names (named 0–20%). Name usage is model-dependent.

### O8: Awkward closings — improved
Persona-aware farewell tone based on outcome relationship. Some runs show differentiation. Model compliance varies with llama3.3.

## Performance — resolved

| Fix | Impact |
|-----|--------|
| Repair rate 30–52% → 0–4% | Eliminates 8–17 extra LLM calls per run |
| Prompt size ~50% smaller | ~500 vs ~900 input tokens per call |
| Turn count unchanged | 15–30 participant turns |

Runs now take ~3–6 min for n=3, ~8 min for n=5 (was ~10–15 min).
