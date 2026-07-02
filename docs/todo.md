# TODO: Natural Option-Grounded Multi-User Discussion Simulator

Source of truth for open work. Restructured 2026-07-02 after a full repository + log review: issues were merged, newly observed failures added, and the list ordered by dependency (fix what later fixes must be verified against first). Log evidence referenced below lives in `logs/archive/` (archived 2026-07-02).

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

## 3. Open issues, dependency-ordered

Work top to bottom. Phase A restores transcript–state integrity (everything later is unverifiable without it). Phase B adds the metrics that measure Phases C/D. Phase C fixes interaction quality, Phase D cost and surface style, Phase E documentation.

### Phase A — transcript–state integrity

---

#### I3 (P0). Parser/state vocabulary for blockers, conditions, and switches

Promoted from old P2 because I4/I5 need this machinery. The parser already rejects hedged/conditional commitments well; what is missing are explicit categories the controller can act on.

**Fix:**

- Parser outputs / helper predicates: `active_blocker` (option-tied `dealbreaker`, `can't support`, `hard no`, `doesn't work for me`, `not okay with`, `cannot support`, `blocked`), `blocker_resolution` (`that addresses my concern`, `that fixes it`, `if we do X I can support Y` when X is affirmed, `I can live with Y because…` without hard-conditional residue), `conditional_support`, `visible_switch_reason` (commitment to a non-initial option + a because/so clause), `compromise_offer`.
- Runtime state: `ParticipantRuntime.active_blockers: dict[option_id, reason]`, populated from parsed active blockers (in addition to persona-level `rejection`); cleared only by a parsed resolution from the same sim.
- Keep conservative defaults: ambiguous lines never count as final votes.

**Verify (no-LLM tests):**

- `I can support A, but only if…` → conditional, not a vote.
- `Sunny Side is a dealbreaker for me` → active blocker on A.
- `That fixes my concern; I can live with A` → resolved blocker + acceptance.
- `I'd switch to B because it solves the budget issue` → visible switch with reason.
- Compare turn mentioning B → no support for B.

---

#### I4 (P0). Visible text is the only source of public stance movement

**Observed in:**

- `logs/archive/20260702_150459_577768`, brunch, n=3: Isla names Sunny Side's vegetarian gap "a dealbreaker if we care about food variety", Zeke agrees it's "a non-starter" — two turns later she votes Sunny Side with no visible resolution.
- `logs/archive/20260702_150822_781090`, summer camp, n=5: Amir and Callum argue for Sailing/Arts and against Robotics ("breaks the budget") for the whole discussion, then both vote Robotics "despite the higher cost" with no visible turnaround; asked whether he "can live with Robotics", Uri instead switches to a *third* option (Sailing) and the forced `allow_vote_change` accepts it.
- `logs/archive/20260702_150710_502883`, soundtrack, n=4: Pavel opens for World Music, votes Indie Pop, then switches back — movements driven by latent lean, not by anything said.

**Root cause:** `_apply_semantics()` updates `current_preference` from `intent.moves_lean`/`proposes_option`; `_vote_intent()` + `_should_compromise_to_candidate()` then roll dice toward the latent leader at vote time; `_can_shift_to()` checks only persona `rejection` and stubbornness, not parsed blockers/objections.

**Fix:**

- Never update `current_preference` from `moves_lean` alone; a lean shift requires the parsed act to show acceptance, proposal, or explicit softening toward that option.
- `_can_shift_to()` must also check `runtime.active_blockers` and unresolved parsed hard rejections (from I3).
- A sim cannot vote for an actively blocked option unless a `blocker_resolution` line from that sim exists; `_vote_intent` for a blocked candidate keeps the current blocked-alternative path.
- `_should_compromise_to_candidate()` may only fire when there is *visible* pressure: parsed votes/acceptances for the candidate from others, or a visible compromise proposal — not `_latent_leading_count`. The rendered switch reason should reference what visibly happened.
- Sanctioned switch turns (`allow_vote_change`) must constrain the accepted target: in `_minority_check`, only winner-or-restate counts; a commitment to a third option is rejected (repair → fallback per I1).
- Record `visible_switch_reason` (I3) whenever `explicit_vote`/`current_preference` moves away from the initial preference; log it in run.json.

**Verify:**

- No-LLM tests: dealbreaker then vote for same option is blocked without a resolution line; switch A→B requires a visible reason; minority-check turn committing to a third option is not applied.
- Run a 3-sim split topic and a dietary/accessibility-flavored topic: a sim switches only if the transcript states why; dealbreakers stay respected.

---

#### I5 (P0). Vote readiness and candidate choice from visible evidence, not latent concentration

**Observed in:** soundtrack run and several quick split-to-consensus endings — narrowing starts because internal preferences concentrate (`_latent_concentration >= concentration_to_vote`), and `_candidate_for_vote()` picks the latent leader, before the transcript shows agreement, resolved objections, or compromise.

**Fix:**

- `_ready_for_vote()` minimum conditions (all from visible/parsed state): participant turns ≥ `min_discussion_turns`; no active response obligation; no unanswered direct question about the candidate; no active blocker against the candidate; option coverage satisfied when configured; at least one visible support cluster (parsed acceptances/votes/proposals from ≥ 2 distinct sims) or a visible compromise/narrowing turn proposing the candidate.
- Keep the hard cap: at `hard_max_turns`, force a visible vote rather than endless discussion; `force_narrow_turns` stays as a soft trigger but the candidate must still be visibly grounded.
- `_candidate_for_vote()`: primary = visible votes/acceptances, secondary = visible proposals + coverage-weighted discussion; latent preference only as tie-breaker.
- Rename the vote reason away from `internal lean concentration held…`.

**Verify:**

- No-LLM tests: latent preferences converge but no visible support cluster → `_ready_for_vote()` false; visible cluster present → true after the turn gate.
- Rerun a soundtrack-like topic: the moderator calls the vote only after visible narrowing; the vote candidate is an option the transcript actually favors.

---

#### I6 (P0). Hard shared constraints enforced at setup; persona goals must not contradict their assigned preference

**Observed in:**

- Summer camp run: shared context fixes the budget at `$300 per child`, option D (Robotics) costs `$320` — and wins. No validator connects the two.
- Brunch run (new): `builders` assigns required primary preferences randomly *before* the persona LLM writes goals; Isla was assigned Sunny Side (card: "Limited vegetarian options") and the LLM gave her a vegan-lifestyle goal. Her dealbreaker-then-vote incoherence starts at setup.

**Fix (keep it lightweight, no constraint solver):**

- Post-generation scenario validator: regex-extract simple numeric caps from shared context (`budget … $300`, `fixed at`, `under`, `no more than`, `max`; capacity minimums; distance maximums) and check against `OptionCard.attrs` keys (`cost`, `price`, `budget`, `capacity`, `distance`, `duration`). On violation: regenerate once, else rewrite the offending attr to a valid nearby value, else mark the option not viable — a not-viable option never becomes a vote candidate.
- Persona coherence: `setup_personas` prompt must state that `background`/`private_goal` have to be *compatible with the participant's assigned primary preference* (the assignment is already in the prompt) — the goal should explain why they lean that way, and must not name a need the preferred option's card explicitly fails. Rely on the instruction plus the existing retry loop; keep persona reads a mandatory manual check in validation runs (a deterministic contradiction detector would be fuzzy — do not build one).

**Verify:**

- No-LLM unit tests for cap extraction and option validation.
- Prompt topics with budget/distance/availability constraints: violating options never win; setup logs show repair or not-viable marking.
- Read personas in validation runs: no goal contradicts its own assigned preference.

### Phase B — measurement

---

#### I7 (P1). Implement the planned metrics and the new integrity counters
Do not consider evaluation now - we'll do that in depth once the discussion is working, 

### Phase C — interaction quality

---

#### I8 (P1). Thread-scored target selection instead of latest-turn bias

**Observed in:** all recent logs — participants react almost only to the immediately previous line; questions get one answer and the thread dies. `_choose_target_turn()` returns `participant_turns[-1]` for most acts and with p=0.7 otherwise.

**Fix:**

- Replace with a scored target pool: open direct questions; group-directed questions; unresolved objections; active blockers; minority/holdout positions; recent claims about the leading candidate; under-discussed viable options; non-latest participant turns from the last 4-6.
- Hard obligations still win. Otherwise score, don't take recency; include social balance (don't always target the same high-engagement sim). Let non-latest turns be referenced by content or speaker name.

**Verify:** in n=4/n=5 runs, some turns respond to non-immediately-previous points; a direct question usually gets answer + acknowledgement/challenge/follow-up before the topic jumps; `direct_response_rate` (I7) improves against the archived baseline.

---

#### I9 (P1). Reactive act selection; agenda as weak fallback

**Observed in:** sequential preference/trade-off statements with little challenge, persuasion, or compromise; the agenda + weighted sampling in `_route_discussion_turn`/`_choose_discussion_act` isn't tied to adjacency-pair logic (agenda fires with p≈0.45–0.80 before context is even considered).

**Fix:**

- Add a small `conversation_need(state) -> need` helper: `answer_obligation`, `resolve_blocker`, `cover_option`, `compare_split`, `invite_holdout`, `narrow`, `continue_discussion`.
- Local dependencies drive acts: question → answer; answer → acknowledge/challenge/follow-up; challenge → defense/clarification/softening; two support turns → invite holdout or test consensus; visible split → compare or propose compromise; active blocker → ask what resolves it; stalled repeated reasons → practical constraint or reframe.
- Agenda items only fill `continue_discussion`; no run marches through the same act sequence; sims need not consume every item.

**Verify:** n=3 and n=5 transcripts each contain at least one question-answer pair, one challenge/concern, one visible narrowing/compromise attempt; act sequences differ across sims/runs; `question_answer_completion` (I7) improves.

---

#### I10 (P1). Moderator interventions must be targeted and evidence-based

Largely unblocked by I3–I5 (the moderator then has real visible state to point at). Remaining work:

- Interventions choose from: ask a blocker what would resolve their concern; ask a holdout whether a compromise works; ask the group to compare the two leading options; request missing practical evidence; call a vote only when I5's readiness holds.
- Never close right after private convergence; keep `moderator_max_interventions` low; phrasing stays plain (no "where everyone stands" boilerplate — vary it).

**Verify:** moderator turns reference a concrete visible issue; the moderator is never the only reason consensus forms; `moderator_ratio` stays low.

### Phase D — cost and surface quality

---

#### I11 (P1). Slim the per-turn prompt and make grounding checks selective

**Observed in:** n=3 runs at 29-32k input tokens, n=5/6 at ~50k (target for n=3: 10-20k). `sim_utterance()` resends full persona background, private goal, all OCEAN values, all seven simulator parameters, voice guidance, full style rules, option cards, shared context, recent chat, agenda, and move instructions every turn; the LLM grounding judge (`validation.grounding_check`) adds a call per eligible turn and still misses subtle claims (plant run `logs/archive/20260702_150339_911975`: unsupported biology/allergy-style claims).

**Fix:**

- Compact voice capsule (persona name + 1-2 line register + only behavior-critical parameters) instead of raw OCEAN + all params; build once per persona.
- Focus option cards + one-line board summary unless the act compares across the board; relevant recent lines (include the target turn even if older) instead of blindly last N.
- Drop the `Ask a question only when the move is ask or invite` rule (too restrictive; embedded questions are natural).
- Grounding: default the LLM judge off; run it only when a regex tripwire fires (numbers/units not present in cards, policy/service words, medical/allergy claims, weather/time claims). Repair or fallback on confirmed hits.
- Keep word budgets and recent-line counts in `config.yaml`. Do not slim so far that commitments become unparseable.

**Verify:** compare `total_tokens_in` on n=3 before/after (target ≤ 20k); repair rate does not spike; unsupported-fact flags do not rise on a plant/allergy-style topic; transcript reads less formulaic.

---

#### I12 (P1). Trait-consistent length variation and anti-echo style control

**Merged:** the two old length/style issues plus a new observation — verbatim reason echo: in the soundtrack run Nico and Emeka both say "for its unique and culturally rich atmosphere" word-for-word (vote + compromise turn); the anti-chorus mechanism covers commitment *phrase families* but not copied reason clauses.

**Fix:**

- Length: keep `verbosity`-derived budgets (`_word_bounds`), add small per-turn jitter around the sim's average; act type modulates (answers/compare longer, votes/acknowledgements shorter); low-verbosity sims may produce meaningful fragments; high-verbosity sims never essays. Config-tunable bounds stay.
- Style rotation at controller level, not more prompt prose: rotate move purposes (plain answer, short challenge, practical constraint, social preference, compromise test, explicit vote); suppress recently overused openings and commitment families (exists) **and** recently used reason clauses — pass the previous voters' reason phrases as an avoid-list on decision turns, mirroring `avoid_phrases`.
- Allow occasional casual acknowledgements (`Yeah`, `Fair`, `Honestly`) where persona-appropriate; don't force trade-off structure.

**Verify:** `avg_words_by_persona` spread matches verbosity ordering with per-turn variance; option-opening/name-prefix/I-opening rates stay under thresholds; no duplicated reason clause across two voters in a round; manual read remains mandatory.

### Phase E — documentation

---

#### I13 (P2). Align docs with implemented behavior

After Phases A-D: update `info/*.md`, the CLAUDE.md mechanisms section, and this file to match the code — never aspirational. `info/00_overview.md` pipeline, turn-taking, moderator, consensus, and evaluation docs must describe actual behavior; this todo keeps only open issues.

---

## 4. Resolved / dropped since the last revision

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
