# TODO: Natural Option-Grounded Multi-User Discussion Simulator

Source of truth for open work. Restructured 2026-07-02 after a full repository + log review. Updated 2026-07-03 after reviewing the post-update logs in `log.zip` (`log/20260703_094030_633172` through `log/20260703_094959_231864`). Earlier issue groups I1-I13 are mostly verified as improved/resolved; the open work below reflects the new failure layer visible after the refactor.

Standing decisions (agreed 2026-07-02):

- **Target model: `gpt` / `gpt-4.1-mini` only.** Prompts and validators may assume gpt-level instruction following; llama3.3-safety constraints no longer apply. Never switch provider without explicit user instruction.
- **Git: one commit on `master` per verified issue.**
- **Validation topics: random and varied across domains** (free time, technical, fictional, household, work) — never reuse the same topic between runs.

The transcript is the product. A fix is only successful if the generated discussion becomes more plausible and internally consistent. Metrics are useful, but secondary.

---

## 0. Project goal and target behavior

Given any reasonable one-line topic, the system generates a natural option-grounded group discussion between 2-7 simulated participants.

A good run shows:

- real interaction, not isolated opinion statements;
- sims asking and answering each other, with follow-ups (not one shallow Q/A beat);
- visible agreement, disagreement, clarification, challenge, persuasion, and compromise;
- socially plausible turn-taking; no same-speaker repetition without a repair reason;
- trait-dependent but varied response lengths;
- rare but stable hard blockers;
- a moderator that guides only when useful;
- an outcome inferable from visible transcript evidence.

Valid end states:

- `successful`: every sim visibly accepts the same option.
- `majority`: a unique majority visibly accepts the winning option, at least one sim does not.
- `unresolved`: no clear visible consensus or majority within limits.

Hidden preferences may guide generation. They must never be treated as final evidence — and (the dual, see I1/I2) the transcript must never visibly say something the recorded state contradicts.

---

## 1. Implementation protocol for every update

1. **Archive old logs first.** Move existing log files/directories from `logs/` into `logs/archive/` before changing behavior. Never delete logs. (Done for the 2026-07-02 baseline.)
2. **Work on one issue at a time**, unless the issue explicitly bundles small changes.
3. **Apply the minimal fix.** Prefer controller, parser, validator, state, or repair-policy fixes over longer prompts.
4. **Keep sim generation stable unless proven otherwise.** Persona generation mostly works; redesign only on concrete log evidence (see I6 for the one known setup defect).
5. **Do not over-split the architecture.** One cohesive extraction (e.g. `src/policy.py`) only if it clearly simplifies `src/dialogue.py`.
6. **Validate with example runs** on the `gpt` provider: one mandatory `n=3` run with a fresh random topic, at least one more run with a different group size in 2..7, more only if behavior is unstable or group-size-dependent.
7. **Inspect transcript and metrics.** Read the transcript; check the intended behavior is visible. Successful execution is not successful dialogue quality.
8. **Add no-LLM tests** for parser/state/controller invariants.
9. **Append newly observed issues** with log path/date, topic, group size, and the smallest description of the failure.
10. **End only after verification**, then commit the issue as one commit on `master`.

---

## 2. Research-backed but practical principles

Use paper insights only where they directly improve the simulator; never implement paper architectures mechanically.

- **Turn-taking:** direct questions create response obligations; addressed speakers answer soon; self-selection is fine otherwise; no same speaker twice in a row; trait-driven but non-collapsing turn distribution.
- **Addressee selection:** multi-party dialogue needs *who speaks to whom about what*; not every turn targets the latest message; maintain active threads (open questions, objections, minority positions, unresolved constraints).
- **Moderation (MUCA-style):** decide what intervention, when, addressed to whom; nudge only when stalled, scattered, one-sided, or ready for visible narrowing; ask holdouts what blocks agreement instead of declaring consensus.
- **Decision emergence:** orientation → clarification/conflict → convergence as a tendency, not a script; closure requires visible narrowing evidence.
- **Personality/OCEAN:** traits create stable tendencies (verbosity, directness, initiative, compromise, stubbornness), never random contradiction; hard blockers rare and stable.

---

## 3. Open issues after the 2026-07-03 post-update log review

Current verdict from the new logs: the previous severe architecture failures are mostly gone. Across the ten reviewed runs, `invalid_printed_turn_count` is 0, visible votes generally match outcome metadata, hard blockers are not forced into accepting their rejected option, turn distribution is balanced enough, moderator ratio is no longer dominant, and n=3 token usage is back inside the rough target range (`20260703_094030_633172`: 19.0k input tokens). Do **not** reopen I1-I5 or I8-I12 globally unless a new regression reproduces the old failure.

However, the generated transcripts now show a new set of visible quality and integrity problems. These are ordered by priority.

#### I19 (P1). Final-vote and fallback/repair phrasing is still too repetitive

**Evidence from new logs:**

- Many final vote turns are repaired from `UNCLEAR_VISIBLE_COMMITMENT`, producing repeated forms: "Count me in for...", "My pick is...", "X gets my vote", "I'd go with...".
- `log/20260703_094959_231864`, standing desk, n=5: Cleo says the exact same fallback line twice: "Fixed Desk gets my vote."
- `log/20260703_094133_016853`, caterer, n=7: seven vote turns in a row all become highly similar one-sentence formulas.

**Why this matters:** this is less severe than factual/state errors, but it weakens naturalness. The vote round should be clear but not all participants should sound generated from the same slot template.

#### I20 (P2). Local interaction improved, but some threads remain shallow or semantically distorted

**Evidence from new logs:**

- Good improvement: `log/20260703_094505_308949` hiking and `log/20260703_094307_629872` mascot show multi-turn threads around slippery falls / handler risk.
- Remaining issue: `log/20260703_094800_722511` garden n=2 repeats the same local-history vs pollinator framing without much actual movement and even distorts "Pollinator’s Patch" as appealing to "history lovers".
- Remaining issue: several runs still lean on card paraphrase + single tradeoff rather than lived social reasoning.

**Why this matters:** this is not a blocker for a working version, but it is the next quality layer after integrity bugs.


#### I7 (P2, deferred by user decision 2026-07-03). Implement the planned metrics and the new integrity counters

New issues observed during future runs go here, with log path/date, topic, group size, and the smallest description of the failure.

## 4. Resolved / dropped since the last revision

- **I18: Non-leading split probes; contested candidates never probed** — done 2026-07-03. `_split_probe_candidate` (extracted for testability) walks vote-getters by count and returns the first that (a) has no visible unresolved dealbreaker from anyone (`_candidate_blocked`) and (b) at least one dissenter can actually move to; if none qualifies the compromise pass is skipped and the run closes honestly as unresolved. The probe wording now presents the front-runner as "currently has the most support" and asks dissenters whether they could *genuinely* live with it or would rather stay with their own pick — "both answers are fine", no middle-ground framing. Verified: 3 no-LLM tests (`tests/test_moderator.py`, incl. the mascot blocked-leader case), runs `20260703_125737` (n=4 forced 4-way split + blocker: neutral probe, two switches with reasons, blocker holdout restates unpressured, honest majority close) and `20260703_125905` (n=3 default config: minority probe offers an out, honest majority close, zero truncated lines).
- **I17: Honest, status-aware closure wording** — done 2026-07-03. `moderator_closure_prompt` is now branched by outcome: majority closes name the winning option *as the majority choice* and acknowledge the holdouts by name (computed from `explicit_vote`, matching the consensus criterion, +8 words budget); successful closes wrap up plainly; unresolved closes state the group leaves it undecided and present nothing as chosen. The copyable "Great — X it is, then." example was removed from the prompt (it was being echoed verbatim every run — R9 class). Verified: 3 no-LLM tests (`tests/test_moderator.py`), run `20260703_125215` (n=4 forced blocker, majority 3/4: closure "…even though Diego was leaning toward a different spot"), successful runs `20260703_124904/124957/125050` close naturally without the old template. New I19 evidence: Diego repeated an identical fallback vote line twice in `20260703_125215`.
- **I16: Cross-option fact transfer caught by grounding** — done 2026-07-03. Two-part fix. (1) The judge prompt no longer permits paraphrasing "for ANY option" (the clause that waved misattribution through): a message is now unsupported when it ties a real card fact to the wrong option, misstates what an option is about, or compares values of different kinds as one unit. (2) The tripwire gained a cross-option branch on cached per-option distinctive tokens (card tokens on no other card, minus shared context and option names): it fires when a line names option X but uses option Y's distinctive fact, or mixes two cards' distinctive tokens inside an explicit comparison (`_COMPARATIVE` marker). Verified: 4 new no-LLM tests; the archived Amir "reliable backup power" vote line trips offline while the correct-option Hybrid line does not; runs `20260703_124303` (n=3 login, no misattribution, 24.1k tokens) and `20260703_124439` (n=5 cocktail: one live UNSUPPORTED_FACT caught and repaired, all comparisons like-for-like, 34.9k tokens). Cost note: judge calls up moderately (n=3 ≈ 19k→24k); the comparative gate keeps compares from always paying.
- **I15: Normalized unit caps + retry-before-clamp** — done 2026-07-03. `builders` now normalizes units within a family (hours→minutes, miles→km via `_UNIT_INFO` factors), reads a unit from the attribute key when the value is a bare number (`duration_minutes: 130`), and scopes activity-qualified caps ("within 15 minutes *walking* distance") to matching attributes only — a live n=4 brunch run exposed the walking-cap-clamps-wait-time false positive, now regression-tested. `enforce_shared_caps` gained report-only mode; `build` retries scenario generation on a violation and clamps (floored, in the attr's own unit) only on the final attempt, because rewriting a number can fabricate a false fact about a real-world named option (the archived Knives Out board is caught by both paths, verified offline). Verified: 8 new no-LLM tests, runs `20260703_123142` (n=3 documentary, cap respected at source), `20260703_123338` (n=4 brunch, retry+clamp fired), `20260703_123650` (n=4 day trip, drive cap clean, no false clamps).
- **I14: Truncated/incomplete utterances** — done 2026-07-03. Root cause was `utils.clean_generated`: any line over the word budget was hard-chopped at `max_words` and patched with fragment heuristics, producing the "maybe we just tweak Lora's." / "or if you?" tails. Fix: the budget is now a style target, not a correctness bound — a complete sentence within a soft cap (budget + max(8, 40%)) is kept whole; when cutting is required, the cut lands on the last real sentence boundary inside the soft window (decimal points excluded via lookahead); the lossy fragment-salvage only remains as last resort for a single runaway sentence, with an extended trailing-word blacklist (modals, pronouns, subordinators). Verified: 5 new no-LLM tests (`tests/test_clean_generated.py`), runs `20260703_122431` (n=3 science fair, 18.6k tokens) and `20260703_122528` (n=6 laundry schedule) — zero non-terminal line endings in both transcripts (checked mechanically), moderator holdout probes complete.

### Confirmed by the 2026-07-03 post-update logs

- **Old invalid printed-turn class is mostly gone.** In the ten reviewed runs, `invalid_printed_turn_count` is 0. Fallbacks still occur, but they no longer leave blocked contradictions in the transcript.
- **Visible vote/state consistency is much better.** Outcomes are generally based on visible vote maps, not hidden preference alone. Majority/unresolved statuses in metadata mostly match the visible votes.
- **Turn-taking distribution improved.** No actual same-speaker consecutive turns were observed in the reviewed transcripts; n=4 runs were often exactly balanced, and n=7 runs did not collapse into one dominant speaker.
- **Moderator dominance is reduced.** Moderator ratio is roughly 0.09-0.20 in the reviewed runs, which is acceptable for now. The remaining moderator problem is wording/candidate choice, not quantity.
- **Token usage improved.** n=3 input tokens are around 19k in the reviewed run, matching the rough target. Larger groups scale upward but are no longer in the earlier 50k+ range for normal n=6/n=7 runs.
- **Interaction is visibly better.** Several logs show real threads: mascot handler risk, hiking slippery falls, caterer portion/dietary tradeoffs, and standing-desk power/complexity concerns. The remaining issue is depth and correctness, not total absence of interaction.

- **I13: Docs aligned with implemented behavior** — done 2026-07-03. README's "current problem" paragraph replaced with the implemented-state summary; `info/00/04/06/07/08` gained "Implementation status (2026-07-03)" sections mapping intent to the actual mechanisms (thread-scored targeting, reactive acts, option-neutral vote calls, sanctioned-switch parsing, blocker lifecycle, integrity counters); CLAUDE.md mechanisms kept current per issue throughout.
- **I12: Length variation + anti-echo style control** — done 2026-07-03. `_word_bounds` gets ±10% per-turn jitter around the trait budget (verbosity ordering and switch headroom preserved, tested); decision turns pass `avoid_reasons` (justification snippets already used this round, extracted by `parsing.round_reason_snippets`) alongside the existing commitment-family avoid list, and the prompt demands a different reason in the voter's own words; BUILD move purposes rotate across three framings (grounded reason / practical consideration / personal priority). Verified: 4 no-LLM tests (`tests/test_style_variation.py`), runs `20260703_015624` (n=3 chant, 18.2k tokens: three distinct vote reasons, natural bridge switch) and `20260703_015733` (n=5 board game, 23.6k tokens: five distinct vote justifications, avg words per persona spread 18.0–23.4, switches concede the pressed point).
- **I11: Slim per-turn prompt + selective grounding** — done 2026-07-03. `sim_utterance` drops the raw OCEAN and simulator-parameter dumps (voice guidance + server-side length/tone notes already encode them), condenses the style block, and removes the "ask only on ask/invite" restriction; repair prompts scope cards to the intent's focus options. Grounding runs in `grounding_mode: tripwire` (default): the LLM judge is only called when a regex tripwire finds a suspicious concrete claim (number or policy/medical/weather-style term absent from the cards/context, cached per run); `grounding_acts` now includes vote/accept/reject (fixes the LastPass-2FA vote-turn leak class). Verified: 6 no-LLM tests (`tests/test_grounding_tripwire.py`); tokens: n=3 `20260703_015156` **19.4k** (was 25-35k, target ≤20k ✓, one invented fact tripped and repaired), n=6 `20260703_015258` **30.7k** (was 50-56k); repair rate stable; the n=6 dessert transcript shows full negotiation around a visible dealbreaker that stays respected.

- **I10: Targeted evidence-based moderator interventions** — done 2026-07-03. Vote calls are option-neutral: `_moderator_vote_nudge` passes no candidate name and no focus options, and the requested action forbids naming/suggesting options or asking about "leaning" (fixes the "which Space Station option" leak). The stall-nudge menu now prefers: unresolved visible blocker on the candidate → ask that person once what would make it workable (`mod:` probe key); visible split → ask the group to weigh the two live favorites head-to-head; then the existing holdout/generic branches. Nudge prompt instructs varied phrasing. Verified: 4 no-LLM tests (`tests/test_moderator.py`), runs `20260703_014630` (n=3 logo: neutral vote call, honest holdout, majority 2/3) and `20260703_014744` (n=5 cabin: clean "name your final pick" call, minority beat with one switch + one honest holdout, majority 4/5).
- **I9: Reactive act selection; agenda as weak fallback** — done 2026-07-03. `_reactive_intent` runs before the agenda with probability-gated adjacency-pair moves: a challenged option gets defended by an advocate (never the challenger), an answer gets a follow-up (agree/challenge/ask by traits), an unresolved blocker on the leading option gets probed exactly once (`state.blocker_probes`), and a visible split triggers a head-to-head compare or compromise test. Agenda firing probability cut to 0.25+0.25·initiative. `_reason_for_act` is stance-aware: a challenge never targets the speaker's own pick (restaurant-run flip-flop fix). Verified: 6 no-LLM tests (`tests/test_reactive_acts.py`), runs `20260703_014042` (n=3 password manager: defense beats, follow-ups, honest 3-way unresolved) and `20260703_014226` (n=6 feast, forced blocker: spit-roast thread asked→answered→acknowledged, blocker probed and honestly held, accurate split summary, honest unresolved). New I11 evidence noted: a vote turn claimed a card fact from another option (LastPass "two-factor" from Dashlane's card) — vote acts skip grounding.
- **I8: Thread-scored target selection** — done 2026-07-03. `_choose_target_turn` scores a pool of the last `routing.target_window` (6) participant turns: open questions +2.0, embedded questions +0.5, objections/blockers +1.0, leading-candidate turns +0.6, under-discussed-option turns +0.4, minority voices +0.6, mild recency decay, ×0.6 damping on re-targeting the same speaker; ANSWER acts deterministically target the pending question's turn. Verified: 4 no-LLM tests (`tests/test_targeting.py`), runs `20260703_013408` (n=5 escape room: multi-turn threads — Noir script-lead thread revisited across four non-adjacent turns) and `20260703_013615` (n=3 choir: two question threads asked, answered, and followed up; outcomes consistent).
- **I6: Setup hard-constraint validator + persona-goal coherence** — done 2026-07-03. `builders.shared_context_caps` extracts hard numeric caps (money with per-unit basis; distance/time with unit families; soft phrasings like "around $200" ignored) and `enforce_shared_caps` clamps violating option attrs in place (per-basis mismatch — "$500 total" vs "cost per person" — is deliberately skipped); repairs recorded in `Scenario.setup_notes` (in run.json) and synced into the persona-prompt JSON. Setup prompts gained two rules: options must satisfy stated caps; persona goals must be consistent with the assigned primary preference and never state a need the preferred option's card explicitly fails. Verified: 9 no-LLM tests (`tests/test_setup_constraints.py`), runs `20260703_012913` (n=3, $50 gift: all options at/below cap, unanimous) and `20260703_013003` (n=5, 10-mile venue: caps respected at source, majority 4/5, live fallback restated the holdout).

- **I5: Vote readiness/candidate from visible evidence** — done 2026-07-03. Early narrowing now requires a visible support cluster (≥2 sims' votes/acceptances; ≥1 for n=2) or visible support plus a visible compromise proposal, with no open question and no active blocker on the candidate; `concentration_to_vote` removed from config, `_latent_concentration` deleted. `_candidate_for_vote` scores visible votes (×2), acceptances, and visible proposals; latent lean only breaks ties or fills the no-evidence fallback (shapes whom the moderator asks, never the outcome). Stall-nudge candidate is visible-first. Verified: 8 no-LLM tests (`tests/test_vote_readiness.py`), runs `20260703_012304` (n=3 book circle: 3-way split → compromise → unanimous, all switches visible) and `20260703_012412` (n=4 surplus: 4-way split, one visible switch, honest unresolved; a "scheduling could be a dealbreaker" line correctly registered as an active blocker on the workshop option).
- **I4: Visible text is the only source of public stance movement** — done 2026-07-02. `moves_lean` removed entirely; latent lean moves only on parsed signals (compromise offer / proposal / conditional support) gated by `_can_shift_to`, which now also checks runtime `hard_rejections`. Votes/acceptances for an actively blocked option are skipped in `_apply_semantics` and flagged blocking in validation (`BLOCKED_OPTION_ACCEPTED`, waived when the same line resolves the blocker). Sanctioned switches may only land on offered/current/initial options (`OFF_TARGET_SWITCH`); the safe fallback is restate-first and blocker-aware. `_should_compromise_to_candidate` requires visible support or a visible proposal (no latent pressure); switch events (from→to, has_reason) recorded per sim. Verified: 12 no-LLM tests (`tests/test_visible_stance.py`), runs `20260702_230750` (n=3 hallway paint: round-1 votes all match argued positions, switch only in visible minority beat) and `20260702_230908` (n=5 dietary: split→compromise, one live fallback restated the holdout's own pick, outcome matches transcript).
- **I3: Parser/state vocabulary for blockers, conditions, switches** — done 2026-07-02. `parsing.py` gained `active_blocker_option` (option-tied vetoes with negation guard), `blocker_resolution_option` (explicit resolution heads; conditional residue blocks), `conditional_support_option`, `compromise_offer_option` (incl. question forms), `commitment_has_reason`; `_COMMIT`/`_DIRECT_VOTE` now cover "I'd switch to" and "I can live with". `DialogueAct` carries `resolves_blocker` / `conditional_support` / `offers_compromise`; parsed blockers land in the existing `ParticipantRuntime.hard_rejections` (reused instead of a new field), resolutions clear parser-derived entries only — never the persona-level setup rejection. Verified: 15 no-LLM tests (`tests/test_blockers.py`), regression runs `20260702_225756` (n=3 RPG, majority consistent) and `20260702_225926` (n=6 volunteering, split→compromise with two visible switches, majority 5/6 matches transcript).
- **I2: Sanctioned-switch bridge clauses parse as commitments** — done 2026-07-02. `visible_commitment(..., sanctioned_switch=True)` (wired from `intent.allow_vote_change`) accepts a commitment with a concessive rider ("as long as", "even though", "despite"); questions and genuine prerequisites (`only if`, `unless`, `would need`, `depends`) still block. Conservative rules unchanged everywhere else. Verified: 8 no-LLM tests (`tests/test_sanctioned_switch.py`, incl. the exact Gemma line), runs `20260702_224801` (n=3 synth, unanimous), `20260702_224909` (n=4 forced 4-way split → honest unresolved), `20260702_225046` (n=5 forced 3-2 → minority check, Faye's visible switch recorded, unanimous matches transcript).
- **I1: Blocked invalid decision turns never printed** — done 2026-07-02. `_generate_and_append` now replaces still-blocking text after repair with `_safe_fallback_text` (deterministic, parser-clean per intent: blockers commit to an allowed alternative, unclear votes become one clear commitment, coverage turns name the required option). Counters `fallback_turns` / `invalid_printed_turn_count` added to metrics. Verified: 6 no-LLM tests (`tests/test_fallback.py`), n=3 run `logs/archive/.../20260702_224207_607026` (dog name, majority consistent with transcript), n=5 forced-blocker run `20260702_224346_818603` (mural; blocker never accepts B, both counters 0). Also fixed: PowerShell BOM leaking into piped topics (`main.py`).
- **Duplicated `expected_act` field in `ResponseObligation`** — already fixed; `models.py` declares it once (verified 2026-07-02). Dropped.
- Old unordered P0s ("invalid blocking turns", "hidden preference movement", "blockers ignored in voting", "hard constraints", "vote readiness") → reorganized into I1-I6 with new evidence added (bridge-clause parser conflict, camp vote flips, third-option switch, persona-goal contradiction).
- Old P1/P2s (targeting, act selection, moderator, prompt cost, grounding, length, style, parser categories, metrics, docs) → I3, I7-I13.

---

## 5. Suggested discussion logic after refactor

### Turn-taking

1. Direct response obligation → the addressed sim speaks, unless impossible. Also counts when the Moderator points to a sim.
2. Otherwise score speakers by: not having spoken recently; relevance to an unresolved concern; minority/holdout status; ability to answer a question; engagement/initiative; quiet-speaker boost.
3. No same speaker twice in a row in normal dialogue.
4. Turn distribution must not collapse into one dominant speaker unless a corpus preset models it.

### Target selection

Target a thread, not the last line: active direct question; unresolved group question; blocker/dealbreaker; recent objection; leading-candidate concern; minority position; under-covered option; non-latest relevant turn from the last 4-6.

### Act selection

question → answer; answer → acknowledge/challenge/follow-up; repeated agreement → invite holdout or test consensus; visible split → compare or propose compromise; blocker → ask what resolves it; candidate emerging → ask minority if they can live with it; enough visible support → vote.

### Preference movement

Initial preferences guide early behavior. A sim softens or switches only with visible reason. Hard blockers never accept rejected options without a concrete resolving compromise. Hidden preference may guide likely next moves; visible support is parsed from text.

### Consensus and termination

Final support = explicit visible vote/acceptance. `successful` all, `majority` unique majority, `unresolved` otherwise. Never close on internal convergence. At hard max, force a visible vote, then finalize from visible votes.

---

## 6. Prompting strategy after refactor

The LLM renders a move; it does not control the discussion. A participant prompt contains only:

1. speaker name + compact voice capsule;
2. current private lean / blocker if relevant;
3. the move: act, purpose, addressee, target turn/concern;
4. focus option facts + compact option list;
5. relevant recent lines;
6. short style/length constraints;
7. strict no-invented-facts / no-metadata rule.

Repairs stay issue-specific (unclear commitment → one clear commitment; blocker contradiction → reject or alternative; unsupported fact → remove it; invalid reference → valid names; missing focus → mention it). When a known safe deterministic fallback exists, use it instead of a generic repair prompt (I1). Quoted example phrasings in guidance are acceptable for gpt-4.1-mini, but rotate/limit them so they don't become the only surface forms (I12).

---

## 7. What not to do

- No large theoretical prompt blocks; no mechanical paper implementations.
- The moderator never becomes the main speaker or the decision engine.
- Hidden preferences never count as consensus evidence.
- Blocked/invalid generated lines never remain visible (I1) — and visible valid commitments are never silently discarded (I2).
- No many tiny files before behavior is stable.
- Never optimize tokens by removing the visible evidence consensus needs.
- Successful execution ≠ successful dialogue quality.
