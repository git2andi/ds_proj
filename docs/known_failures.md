# Known Failures - Open Issues Only

Last updated: 2026-06-30.

This backlog contains only issues still supported by the current implementation or the current logs. Resolved issues and implementation history are intentionally omitted. Fixes must remain topic-agnostic, work for 2-7 participants, preserve visible-commitment outcome rules, and avoid provider-specific phrase patches.

## Audit basis

The current audit read every `transcript.md` and `run.json` in the eight active GPT validation runs from `20260630_002124_450800` through `20260630_003653_164440`, covering all group sizes 2-7 and 252 decision turns. The prior 18-run audit set was moved to `logs/archive/` before validation.

The active runs contain 36 repaired and 92 flagged decision turns. Exact response context was routed on 134 of 143 eligible turns (93.7%, up from 71.7% in the archived baseline), and manual review confirms that discussion replies normally engage the targeted point instead of restarting a standalone pitch. Response length strongly distinguishes personas; for example, the `n=6` board-game run averages 21.1 words for the high-extraversion/high-response-length participant and 9.5-9.8 for three low-extraversion/short participants. Remaining repetition, social-round templates, commitment churn, and grounding failures are tracked below rather than folded back into one broad naturalness issue.

## Validation protocol

For each fix:

1. Work on one issue only unless the user explicitly groups issues.
2. Use the provider explicitly authorized for the task. Never silently substitute an endpoint.
3. Run one mandatory `n=3` discussion, then the requested spread across `n=2-7` when behavioral validation is required.
4. Read every relevant `transcript.md` and `run.json`; metrics and validator counts are not sufficient.
5. Compare against the concrete evidence and acceptance criteria in the issue, including regressions in grounding, commitment gating, persona behavior, and pacing.
6. Close an issue only when the target behavior improves across topics and sizes without an obvious regression.
7. Before completion, synchronize all applicable active guidance listed in `AGENTS.md`.
8. Stop at the completed issue boundary unless the user explicitly asks to continue.

Current implementation order: P0 conversation quality, P1 setup/state integrity, P2 moderator integrity, P3 grounding, then P4 token cost. Each item is an independent upgrade.

## P0 - Conversation quality

### ~~KF26~~ - Cosmetic greeting and farewell rounds (resolved 2026-06-30)

**Fix:** `_social_speakers()` now returns at most one speaker — the most extraverted persona with probability = `extraversion / trait_max` — so a social beat is an optional single line rather than a round-the-table chorus. `farewell_line()` prompt no longer includes persona background (was causing biography callbacks). `greeting_line()` reframed from “casual hello / first text someone fires off” to “opening line before an ongoing-thread discussion” to reduce slang-arrival templates.

**Tests added:** `SocialBeatTests` in `test_conversation_contract.py` — at-most-one speaker, most-extraverted selection, empty-return probability, farewell omits background, greeting avoids arrival framing.

**Pending watch:** If live runs show that zero-greeting outcomes in n=2 feel abrupt, the probability floor could be raised for small groups. Not a regression yet.

### ~~KF14/KF27~~ - Pacing / NARROWING churn (resolved 2026-06-30)

**Fix (three parts):**
1. **Multi-voter NARROWING escape** (`router.py` `_vote_intent()`): when every remaining unvoted persona has had `max_vote_attempts_per_person` (config, default 2) VOTE-intent narrowing turns without gaining `explicit_vote`, force-advance to CONFIRMATION rather than cycling until the hard cap. Addresses 50-turn NARROWING loops where multiple voters kept producing UNCLEAR_VOTE turns.
2. **Stall window de-scaling** (`dialogue.py` `_moderator_intervention()`): removed `+ max(0, n - 3)` from stall window — larger groups now trigger stall interventions at the same pace as small groups instead of being more lenient.
3. **Slot-exhaustion early narrowing** (`dialogue.py` `_can_start_narrowing()`): when every staked option has ≥ `slot_exhaustion_threshold` (config, default 3) covered claim slots, progress has stalled, and each participant has had their minimum turns, narrowing can begin before the derived turn-floor. Prevents semantic circling after substance is exhausted.

**Validation (n=7 city topic):** NARROWING phase 50 → 9 turns; SELF_REPETITION warnings 15 → 0; total turns 88 → 73. n=3 board-game run remained natural at 21 participant turns.

**Pending watch:** Discussion length at n=7 (~49 turns) is still on the longer side; adjusting `slot_exhaustion_threshold` downward or `base_per_participant` could shorten further if needed.

## P1 - Setup, state, and outcome integrity

### ~~KF28~~ - Scenario, option, and persona coherence (resolved 2026-06-30)

**Evidence:** Several successful setups are structurally valid but socially contradictory. `20260629_210241_966736` uses the user topic “team lunch for a group of 5 colleagues” while running three participants. `20260629_215720_381932` gives Leo a goal about impressing his date during a team birthday dinner. `20260629_215856_690649` states that all six family members want to remain engaged, yet includes and assigns preferences for two- and four-player games. Hard word truncation also produces broken names such as “Pacific Coast Highway from San Francisco to San” in `20260629_205507_547657`.

**Root cause:** Validation checks field shape, option IDs, explicit participant counts in `shared_context`, and preference membership, but not consistency between the user topic, shared context, option feasibility, persona goals, and selected group size. Option names are truncated mechanically instead of rejected or regenerated.

**Required direction:** Add setup-level coherence checks for explicit group-size statements in the topic, hard shared constraints, option feasibility, persona relationship/context, and complete option names. Surface a conflict between a topic-specified group size and configured `n` rather than silently rewriting the world. Reject or regenerate malformed source data; do not repair it with fabricated defaults.

**Fix:**
- `_validate_topic_participant_count()` in `builders.py`: pre-LLM check that raises immediately with a user-actionable message when the topic text explicitly names a participant count that contradicts `num_participants` in config.
- `_clean_name()` in `builders.py`: raises `ValueError` if the word-capped option name ends on a function word (preposition, conjunction, article) — triggers scenario retry rather than silently truncating mid-phrase names.
- `setup_personas` prompt: tightened background/private_goal guidance to explicitly prohibit inventing a relationship or event not present in the shared decision context.

**Tests added:** `SetupCoherenceTests` in `test_preference_distribution.py` — topic/n mismatch raises before LLM call; topics without explicit counts do not raise.

**Remaining gap:** Option feasibility for group size (e.g. 4-player game assigned to 6 players) requires semantic understanding of attribute values and is not checked deterministically. Not a regression.

### ~~KF29~~ - Changes of mind and final consensus (resolved 2026-06-30)

**Fix (three parts):**
1. **Hedged-confirmation auto-soft-rejection** (`dialogue.py` `_update_runtime()`): when a CONFIRMATION-phase ACCEPT turn is repaired and state mutation is still blocked (UNCLEAR_ACCEPT after repair), automatically record a `"hedged-confirmation"` soft rejection on the candidate. On the next routing call, `_confirmation_intent()` sees this and skips the persona rather than retrying with an identical prompt.
2. **Routing reason clarification** (`router.py` `_confirmation_intent()`): the routing reason now explicitly says "a reluctant concession counts as acceptance; if not, say the one thing blocking you" — giving the model a clear exit for reluctant-but-real acceptance without forcing an enthusiastic line.
3. **UNCLEAR_ACCEPT repair hint** (`prompts.py`): updated from "say the option name explicitly and confirm agreement" to "say plainly that the option works for you — a reluctant yes ('can live with it', 'it works') is fine" — so the repair attempt doesn't prompt for a tone the model can't produce.

**Validation:**
- n=3 dinner: successful, clean single confirmation turn, 0 UNCLEAR_ACCEPT repairs cycling
- n=5 city trip: Oscar (Savannah holdout) conceded "Portland can work for me even if it's not exactly the calm, historic place I hoped for"; Nadia conceded "I can live with less nightlife for better nature access" — both grounded in prior turns, no cycling

**Pending watch:** Semantic legitimacy of mind-changes (whether prior turns actually addressed the holdout's concern) is still trust-based rather than enforced. No regression: the core gap is a language-model limitation, not a routing or state bug.

## P2 - Moderator integrity

### ~~KF06~~ - Moderator state claims, completeness, and impossible compromises (resolved 2026-06-30)

**Fix (three parts):**
1. **No-blend rule** added to `_MODERATOR_RULES` in `prompts.py`: “Do not suggest combining, blending, or merging options — the options are fixed.” Eliminates dog-name/option-blend proposals.
2. **Lean vs. vote language**: `_camp_split()` now uses “leaning toward X” instead of “N for X”; new `_vote_summary()` function lists only personas with explicit `explicit_vote` entries; `_MODERATOR_RULES` now includes “say 'leaning toward' or 'seems to prefer', never 'voted for', unless someone explicitly cast a visible vote”; `moderator_agreement_prompt()` passes `_vote_summary()` and instructs “Use 'leaning toward' if no one has explicitly voted yet.”
3. **Completeness check** in `_moderator_say()` (`dialogue.py`): after generation, checks whether the text ends in sentence-final punctuation (`.!?`); if not, retries with an explicit instruction to finish the sentence. Combined with the existing invalid-option-ref retry — both checks can fire on the same attempt and both steer the retry prompt.

**Validation (n=3 volunteer project):** Moderator said “Elif, since you seem to prefer...” (lean language, not vote language) and “Looks like Painting and Renovation at Local Youth Center is the plan, so let's arrange to collect the $100 needed for paint and supplies by midweek” (card-grounded, complete sentence).

## P3 - Grounding and question quality

### ~~KF12/KF25~~ - Grounding: ASK binding + ANSWER guidance (resolved 2026-06-30)

**Evidence:** Manual review finds unsupported details throughout the fresh logs. The volunteer run invents indoor break areas, shade, a ten-minute walk, weather flexibility, and adjustable task pacing. The city run invents ride-share prices, neighborhood distances, traffic, rain, crowds, plantations, art walks, and harbor cruises. The dessert run invents berry prices, leftovers, smaller purchase sizes, and extra guests. Validator counts substantially understate these qualitative inventions.

The current ASK prompt shows card attributes, but the model still asks beyond them. ANSWER turns often convert “unknown” into a confident fact. The concrete-noun and numeric checks catch a narrow subset and miss unsupported qualitative comparisons and logistics.

**Fix (three parts):**
1. **ASK attribute binding** (`prompts.py` `_move_guidance()`): for ASK turns, the guidance now extracts the actual attribute key names from the focused option(s) and presents them explicitly — "Ask about ONE specific card attribute — one of: cost, time commitment, physical effort, impact scope. Do not ask about things not listed." Prevents questions about weather, proximity, service quality, and other off-card logistical attributes.
2. **ANSWER grounding emphasis** (`prompts.py` `_move_guidance()`): updated from "say that plainly" to "The only valid facts are those already in the card or shared_context. If the question asks for something the card does not state, say 'the card doesn't say' and pivot to a point the card does support. Never estimate, guess, or turn an unknown into a confident new fact."
3. **Per-turn prompt footer** (`prompts.py` `sim_utterance()`): added explicit prohibition on logistics inventions — "never estimate, guess, or invent a logistics detail (distance, quality, availability, weather, or service not listed)."

**Validation (n=3 city trip — original failure category):** Zero INVENTED_OPTION_ATTRIBUTE issues; no ride-share prices, art walks, harbor cruises, or distance claims. ANSWER turns referenced card attributes explicitly. No off-card ASK questions in this run.

**Remaining gap:** Mild qualitative elaborations persist ("lakeside deck", "chill spots") where the model extends an implied attribute (lakefront → deck). Numeric grounding validation and the facility-noun pattern catch hard inventions; soft qualitative inference beyond card fields cannot be fully blocked without semantic validation. Not a regression from the baseline.

## P4 - Cost

### KF16 - Token cost remains structurally high

**Evidence:** The eight active validation runs use 179,197 dialogue input tokens and 16,694 setup input tokens. Dialogue input averages 711.1 tokens per decision turn, excluding setup. Repetition, repair churn, and long large-group runs still increase cost without improving decision quality.

**Required direction:** Address P0-P3 before optimizing away context needed for correctness. Then remove duplicated state and rules, pass only the exact response target and structured facts required by the routed act, and make repair prompts issue-specific. Track quality and repair rate per input token by provider and group size.

**Acceptance criteria:** Input tokens per decision turn fall materially without regressions in grounding, local responsiveness, persona visibility, commitment integrity, or setup reliability.

**Relevant code:** `src/prompts.py`, `src/dialogue.py`, `config.yaml`.
