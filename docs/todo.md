# Controller Cleanup TODO

This is the active queue for the **final consolidation and deletion pass**. The controller refactor already exists. Do not redesign it or add another layer. Preserve behavior while removing duplicate ownership, stale compatibility state, ineffective configuration, and dead code.

## Fixed behavior

Keep:

- outcomes: `successful`, `majority`, `unresolved`
- phases: `opening -> discussion -> narrowing -> voting -> [compromise_repair] -> closing`
- threads for local `question`, `concern`, `blocker`, and `comparison` handling
- routing order: required answer -> hot thread -> optional cooling thread -> coverage -> rare continuation -> normal turn
- normal acts: `support`, `concern`, `ask`, `answer`, `compare`, `comment`
- visible formal voting and bounded decision repair
- existing setup/simulator/persona generation
- existing trait meanings:
  - `engagement`: speaking frequency
  - `verbosity`: utterance length
  - `directness`: wording/challenge strength
  - `stubbornness`: stance defence during discussion
  - `switch_resistance`: final switching, compromise, holdout, and repair resistance

Ownership must end as:

```text
threads    -> local issue lifecycle and repetition state
policy     -> pure route/speaker/act/focus selection
observer   -> state updates from accepted public semantics
parsing    -> deterministic interpretation of visible text
validation -> output acceptance and mutation blocking
flow       -> phases, narrowing, voting, bounded repair
consensus  -> final visible outcome
```

Do not tune prompt wording/style. Structural prompt changes are allowed only when removing obsolete controller state. Do not update `README.md`, `CLAUDE.md`, or `info/*.md`; they are already current.

## Work rules

- Work through unchecked items in order and continue automatically.
- Mark completed items `[x]` with one short implementation note.
- Prefer deletion/consolidation over new abstractions, files, state, or config.
- Do not keep two active mechanisms for the same responsibility.
- Routing must not mutate persistent or hidden future-routing state before a turn is accepted.
- Do not reduce lines by changing intended behavior.

After every item:

```powershell
py -m compileall -q main.py src eval tests
py -m unittest discover -s tests
```

Fix failures before continuing.

---

## [x] 1. Record the cleanup baseline

- Run all deterministic tests and record the passing count.
- Record line counts for `src/` and the main controller files.
- Run `pyflakes src` and record current warnings.
- Search all production/test references before deleting symbols.

No behavior change in this item.

**Done when:** baseline numbers and warnings are noted below this item.

> **Baseline (2026-07-11, splitlines counts):** 158/158 tests pass; compile clean.
> `src/` total **9347** lines — controller/flow.py 1512, controller/policy.py 1306, builders.py 1051, prompts.py 683, observer.py 632, parsing.py 695, dialogue.py 557, validation.py 544, models.py 473, controller/threads.py 361, config_loader.py 379, utils.py 285, logger.py 219, style.py 196, llm_client.py 147, controller/state.py 107, consensus.py 78, aliases.py 60, simulator.py 55, controller/__init__.py 7.
> `pyflakes src` (6 warnings): dialogue.py:42 `controller.threads` unused; models.py:16 `BlockingStrength`/`ThreadStatus`/`ThreadType` unused (intentional re-export surface — see item 10); policy.py:14 `re` unused; policy.py:20 `Phase` unused.

---

## [x] 2. Make accepted text the only semantic authority

> **Done (2026-07-11):** `parse_dialogue_act` now derives the realized act from visible text
> (`_realized_act_type`: commitment > objection > question > comparison > benefit claim; only
> contextual acts — opening/answer/process/compromise/vote/closing — keep the routed label).
> Removed `DialogueAct.proposes_option` and all intent-act semantic fallbacks: concern threads
> open only from parsed `soft_rejects`/`hard_rejects`, comparison threads from
> `realized_comparison()` (two named options + comparative wording), opening leans from the
> option the line visibly names, SUPPORT requires a visible benefit claim (`_PRO_CLAIM`), and
> objection wording coverage was widened (`_SOFT_OBJECT`: worries/bothers/seems high/a bit steep).
> Remaining intent reads in the observer are contextual only (sanctioned switch, opening gate,
> reply targeting, routed answer identity). New `tests/test_intent_text_mismatch.py` (7 cases)
> proves text wins on mismatch. 165/165 tests pass.

Fix remaining cases where controller intent creates state even when the final text does not visibly realize it.

Targets:

- `parse_dialogue_act()` must not default ordinary realized acts to `intent.act` without textual evidence.
- `observer.py` must not open concern threads merely because the route intended `CONCERN`.
- opening preference changes must come from the accepted utterance, not only `intent.option_focus`.
- answer, support, coverage, thread, stance, and vote effects must require parsed/validated public evidence.

Keep `MoveIntent` as generation guidance. Use deterministic parsing plus validation; do not add an LLM classifier.

Add mismatch tests where intent and final text differ and verify that text wins.

**Delete:** intent-based semantic fallbacks once covered by tests.

**Done when:** selected and realized acts can genuinely differ and state follows the realized act.

---

## [x] 3. Separate private stance, public support, and formal votes

> **Done (2026-07-11):** Centralized public evidence in `consensus.py`: `public_support()`
> (transcript-based visible backing, optional phase filter and realized-SUPPORT acts) and
> `public_evidence()` (backing, formal votes/counts, compromise proposals, weighted candidate
> scores/leaders, public top pair). Policy's `_visible_support_count`/`_current_top_pair`/
> `_visibly_proposed` are now thin reads over it; `_visible_candidate` and the old weighted
> `_candidate_for_vote` merged into one `_public_candidate` (latent lean only breaks ties);
> the `acceptable_options()` private-rank leak into visible support is gone (the sole remaining
> use is the sim's own `_stance_consistent_vote_target`). Flow's repair camps (reservation
> supporters, holdout probe, split `votes_by_id`, switch pressure, holdout alternatives,
> two-person deadlock, peer closing) read formal `visible_votes_from_transcript` instead of
> runtime `explicit_vote`; `_discussion_support_options` lost its intent-act fallback and
> feeds narrowing only. New `PublicEvidenceTests` (6 cases) prove leans≠votes, discussion
> support≠consensus, private ranks don't count, and repair acceptance replaces a formal vote.
> 171/171 tests pass.

Use distinct evidence for:

```text
private rank/lean
public discussion support
formal voting commitment
repair-phase replacement commitment
```

Current overlap to remove:

- discussion acceptance and formal voting sharing `explicit_vote`
- candidate/support helpers counting `acceptable_options()` or unexpressed private ranks as visible support
- multiple current/visible candidate and top-pair calculations

Create one centralized public-evidence calculation used by policy and flow. It must provide discussion support, formal votes/counts, public candidate, and public top pair.

Private ranks remain for persona-consistent routing and switching, but never count as visible evidence.

Tests must prove:

- opening leans are not votes
- discussion support can affect narrowing but not consensus
- private acceptable ranks do not count publicly
- visible repair acceptance can replace an earlier formal vote

**Delete:** redundant candidate/top-pair/support helpers after migration.

**Done when:** narrowing, voting, repair, and consensus consume the correct single evidence source.

---

## [x] 4. Remove inactive and ineffective thread state

> **Done (2026-07-11):** Removed `ThreadType.REPAIR` (its age-skip and selection-priority
> branches, plus the repair step of the priority test) — `RepairState` alone owns voting
> repair. Removed never-written `DialogueState.primary_thread_id` (models/logger/trace);
> the routed thread is now carried on `MoveIntent.thread_id` (set by thread-hot/cooling/
> answer routes) and traced per turn as `routed_thread_id/type/status`. Removed unread
> `ThreadState.explicit_addressee_id` (redundant with `question_scope` +
> `required_respondent`). Added `ThreadState.contribution_count` (accepted turns folded
> into the thread; same-turn double-touches count once): the soft cap now blocks optional
> cooling continuations and the hard cap stops a thread from driving turns at all inside
> `select_primary_thread` — unique-participant counting as thread length is gone, and both
> configured limits are implemented. New `ContributionCapTests` (4 cases) + updated trace
> tests. 176/176 tests pass.

Keep persistent thread types only for:

```text
question
concern
blocker
comparison
```

`RepairState` owns voting repair. Therefore remove:

- `ThreadType.REPAIR`
- repair-thread aging/selection branches
- repair-thread-only tests

Clean thread state:

- remove persistent `DialogueState.primary_thread_id`; record the selected thread in the route/trace for that turn
- remove `explicit_addressee_id` if it is redundant with `question_scope` and `required_respondent`; otherwise prove its distinct use with tests
- add one accepted-contribution counter to `ThreadState`
- use that counter for soft/hard thread limits
- stop using unique participant count as thread length
- implement both existing limits or delete the unused soft-limit config

**Done when:** trace shows the actual routed thread and thread caps count accepted contributions.

---

## [x] 5. Make threads the only issue/blocker progression system

> **Done (2026-07-11):** Deleted `DialogueState.issue_ledger` + observer's `_update_issue_ledger`
> and `_LEDGER_ISSUES` lexicon; the settled-issue prompt suppression now derives from
> resolved/stale concern/blocker threads' normalized `issue_key`s (sig:/general keys excluded),
> and eval reports `settled_issue_keys` from threads instead of the ledger dump. Deleted
> `DialogueState.blocker_probes`: probes live on the blocker thread (`ThreadState.probe_count`,
> charged post-turn via `MoveIntent.thread_id` for the peer probe, and per selected thread for
> the moderator probe — shared cap of one natural probe per THREAD, so one sim's probe no
> longer suppresses another sim's blocker on the same option; the moderator probe also now
> requires a visible blocker thread instead of reading private rejected ranks). Existing tests
> already covered same-issue-two-options and resolved-repeat suppression; added
> two-blockers-two-probes and thread-derived prompt-suppression tests, and the routing-purity
> fingerprint now covers `contribution_count`/`probe_count`. 178/178 tests pass.

### Issue repetition

Replace the old fixed-category `issue_ledger` with thread history and normalized `issue_key` values. Prompt context for settled issues must derive from resolved/stale concern or blocker threads.

### Blocker probing

Replace global option-level `blocker_probes` with state on the specific blocker thread or deterministic accepted thread history. One participant's blocker probe must not suppress another participant's blocker against the same option.

Preserve one natural probe, relevant response/mitigation, and visible resolution/staleness behavior.

Add tests for:

- two blockers by different participants against one option
- the same issue category affecting different options
- repetition suppression after resolution/staleness

**Delete:** `DialogueState.issue_ledger`, `DialogueState.blocker_probes`, their update functions, prompt branches, comments, and config references.

**Done when:** threads are the single owner of local issue history and blocker progression.

---

## [x] 6. Make routing completely side-effect free

> **Done (2026-07-11):** Deleted the `_last_target_speaker` class attribute; the anti-pile-on
> damping in `_choose_target_turn` now derives the last targeted speaker from accepted
> `TurnRecord`s (`_last_targeted_speaker`: most recent accepted turn's `respond_to_turn`).
> Audited policy for writes — no assignments to DialogueState, runtimes, controller fields,
> or counters remain during selection (probe/coverage/procedural charges stay post-turn in
> `_post_turn_route_accounting`). Routing-purity tests now fingerprint controller instance
> state (including `vars(runner)` so any new routing memory fails the test) and a new
> reproducibility test proves repeated same-seed route selection over identical accepted
> history yields identical routes. 179/179 tests pass.

Audit policy and speaker selection for mutations before generation succeeds.

Current target: `_last_target_speaker` changes during selection and affects later routes even if no turn is accepted.

Derive recent targeting from accepted `TurnRecord`s or update routing memory only after observation.

Check for writes to:

- `DialogueState`
- participant runtimes
- controller instance fields affecting future routes
- coverage/thread/probe counters

Expand routing-purity tests to fingerprint both dialogue state and relevant controller state.

**Done when:** repeated route selection over identical accepted history does not consume or alter future behavior.

---

## [x] 7. Consolidate duplicated decision-repair code

> **Done (2026-07-11):** Merged `_split_reservation_exchange()` into one parameterized
> `_reservation_exchange(state, holdout, candidate, split=...)` (identical two-turn structure;
> only route source and instruction wording differ — wordings preserved verbatim). Deleted
> `_final_decision_intent()` entirely: it and the inline intent constructions in the
> majority-holdout and deadlock loops were dead weight — `_append_final_decision` recomputed
> target/outcome/reason and only read `route_source`/`length_hint` from the passed intent.
> `_append_final_decision(state, persona, candidate=, can_move=, route_source=, alternative=)`
> is now the single place a repair's final beat (target, expected outcome, grounded reason,
> generation intent) is calculated, used by all three repair variants. Extracted the shared
> supporter-choice op `_candidate_supporter()` (reservation responder + peer holdout probe).
> All repair reasons, bounds, and per-reason tests unchanged. 179/179 tests pass.

Preserve current repair reasons and limits:

```text
unclear_vote
majority_holdout
split_vote
two_person_deadlock
hard_blocker
```

Remove duplicated implementations:

- merge `_reservation_exchange()` and `_split_reservation_exchange()` into one parameterized exchange
- combine `_final_decision_intent()` and `_append_final_decision()` so the target, expected outcome, reason, and intent are calculated once
- reuse small shared operations for asking a reservation, choosing a respondent, applying one bounded response, and deciding revote versus closing

Do not replace explicit repair reasons with one giant conditional. Keep focused tests for each reason.

Preserve visible votes, outcome definitions, round limits, honest blockers/holdouts, and `switch_resistance`.

**Done when:** repair variants share primitives without parallel reservation/final-decision implementations.

---

## [x] 8. Use one validation mutation-blocking path

> **Done (2026-07-11):** Compared both paths: `_semantic_block()`'s two checks were exact
> duplicates of `UNCLEAR_VISIBLE_COMMITMENT` and `REQUIRED_VOTE_MISMATCH` in
> `_validate_turn_text` (both already blocking), so no checks needed moving. Deleted
> `_semantic_block()` and both call sites in `dialogue.py` — `report.block_state_mutation`
> is now the single block decision. New `tests/test_validation_blocks.py` (15 cases) covers
> every blocking reason (EMPTY, MALFORMED_UTTERANCE, INVALID_OPTION_REFERENCE,
> MISSING_REQUIRED_OPTION_FOCUS, UNCLEAR_VISIBLE_COMMITMENT, REQUIRED_VOTE_MISMATCH,
> HARD_BLOCKER_ACCEPTED_REJECTED_OPTION + BLOCKED_OPTION_ACCEPTED, HYBRID_COMPROMISE,
> CONTINUATION_REPEATS, CONTINUATION_TOPIC_JUMP, ANSWER_DOES_NOT_ADDRESS_QUESTION,
> OFF_TARGET_SWITCH, UNBRIDGED_SWITCH, UNSUPPORTED_FACT) plus a non-blocking telemetry
> check for thread-realization misses. Validation still selects nothing. 194/194 tests pass.

`validate_turn_text()` already reports semantic failures while `_semantic_block()` repeats part of the same protection.

- compare both paths
- move any missing checks into the validation report
- add focused tests for every block reason
- remove `_semantic_block()` and duplicate call sites

Validation must not choose routes, speakers, candidates, thread transitions, or repair objectives.

**Done when:** each invalid semantic mutation produces one validation code and one block decision.

---

## [x] 9. Simplify remaining complexity hotspots in place

> **Done (2026-07-11):** `_apply_semantics`: extracted the latent-lean elif chain into
> `_apply_lean_movement()` (single responsibility: visible lean movement) and deleted the
> `_softening_signal` wrapper (inlined `act.softens_toward`); removed duplicate enum entries
> in its two act-set literals. `_route_discussion_turn`: extracted the slot-6 tail into
> `_normal_intent()` so the router reads as the pure priority ladder, and dropped the dead
> `focus` pre-init. `_reason_for_act`/`_focus_options`: removed branches for acts the normal
> sampler can never produce (ANSWER/COMPROMISE/PROCESS/OPENING/CLOSING — contextual acts get
> reasons at their route/flow sites), which also made `_quietest_other` dead (deleted).
> `_vote_intent`/`_apply_style_flags`: removed `VOTE if switching else VOTE` and duplicate
> VOTE set entries. `parse_dialogue_act`, `_thread_intent`, `_choose_target_turn`, and
> `_ready_to_narrow` were already reshaped by items 2–6 and left as-is. Net −20 production
> lines in this item (9383 → 9363). 194/194 tests pass.

After deleting duplicate mechanisms, simplify only where it reduces branching or responsibility overlap.

Priority functions:

- `observer.py::_apply_semantics`
- `controller/policy.py::_route_discussion_turn`
- `controller/policy.py::_thread_intent`
- `controller/policy.py::_choose_target_turn`
- `controller/flow.py::_ready_to_narrow`
- `parsing.py::parse_dialogue_act`

Use small private helpers only for clear responsibilities. Do not add registries, rule engines, mixins, generic dispatch layers, new files, or new config.

Remove branches made obsolete by items 2–8. Total production lines must not increase in this item.

**Done when:** each function follows its module's ownership and duplicate conditionals are gone.

---

## [x] 10. Mechanical dead-code and stale-config pass

> **Done (2026-07-11):** `pyflakes src` is clean. Removed the unused `controller.threads`
> import in dialogue.py and unused `re`/`Phase` imports in policy.py. The three controller-enum
> re-exports in models.py are kept intentionally as the stable import surface, declared via
> `__all__` (nothing star-imports models) so static tools count them as used. Deleted ten
> unreferenced convenience helpers found by repo-wide search: `ParticipantRuntime.adjust_rank/
> options_at_rank/liked_options/is_acceptable/is_disliked/is_rejected`,
> `DialogueState.participant_ids`, `OptionResolver.option_text`, `parsing.has_commitment_hedge`,
> `utils.extract_numbers` (kept `challenges_received`/`commitment_min` — they are logged run
> telemetry). Duplicate enum values in set literals were already removed in item 9; no shadow-
> mode/repair-thread/old-trait comments remain in src (fixed the threads config comment and the
> soft/hard cap descriptions, plus a stale test docstring). All config keys verified to have
> live consumers. 194/194 tests pass.

After behavioral consolidation, remove confirmed dead artifacts.

Known items to review:

- unused `controller.threads` import in `src/dialogue.py`
- unused `re` and `Phase` imports in `src/controller/policy.py`
- unused helpers/convenience methods found by repository-wide search
- removed fields/config from earlier items
- duplicate enum values inside set literals
- stale source comments about shadow mode, repair threads, old trait counts, or removed systems

For controller-type imports re-exported through `models.py`, either keep them intentionally as the stable import surface or migrate callers and remove them. Do not delete symbols solely from a static-tool warning without checking production/tests/entry points.

Run:

```powershell
pyflakes src
```

**Done when:** no genuine unused imports, dead fields, ineffective config keys, duplicate literal entries, or stale implementation comments remain in scope.

---

## [x] 11. Minor tweak in run_eval_suite.py

run_eval_suite.py creates a config backup and stores it in root/ of project. Make it store this file in eval/
Also add/remove/adapt existing test cases to test new implementation correctly.

**Done when:** ren_eval_suite.py reflects current state and stores config backup in eval/

> **Done (2026-07-11):** The config safety copy now lands in `eval/config.yaml.eval_backup`
> instead of the project root (restore logic unchanged — it rewrites config.yaml from memory).
> Suite summary rows gained two cleanup-pass surfaces: `settled_issue_keys` (thread-owned
> issue history, item 5) and `route_source_distribution` (threads-vs-coverage/normal mix);
> no case referenced removed metrics, so the 12 cases stand as-is. 194/194 tests pass.

---


## [x] 12. Final verification and reduction review

> **Done (2026-07-11):** `compileall` clean, **196/196** deterministic tests pass, `pyflakes src`
> clean. Full eval suite (gpt-4.1-mini): **12/12 cases rc=0**, outcomes 6 successful /
> 5 majority / 1 unresolved (all three types; the unresolved is the honest n=2 stubborn
> deadlock), 0 unsupported printed turns, fallbacks 0–2 per run, repairs bounded (each reason
> ≤1×/run: 9× majority_holdout, 2× split_vote, 1× two_person_deadlock) with visible re-votes.
> Logs inspected: intended vs realized acts genuinely differ and state follows text; threads
> drive local moves (thread_hot 28 + cooling 21 + answer_required 12 vs coverage 9 across the
> suite); `routed_thread_id` traces populated; question/concern/blocker/comparison lifecycles
> visible; settled-issue prompt context now actually fires (6/12 runs — the old issue_ledger
> version fired 0/12 in the pre-cleanup suite, i.e. it was dead).
> **Observation:** `discussion_lean_shifts` dropped from 17 (old suite) to 0 — parsed
> conditional-support/softening signals in discussion fell 20→2, a generation-side
> distribution shift (stricter objection parsing changes dynamics; live settled-issue
> suppression; LLM drift). The mechanism itself is intact and now pinned by two new
> deterministic tests (softening and conditional support both move the lean in discussion).
> **No parallel mechanisms remain** for questions/concerns/blockers (threads), issue
> repetition (thread history), candidate calculation (`public_evidence`), vote evidence
> (`visible_votes_from_transcript`), repair exchange (one `_reservation_exchange` +
> `_append_final_decision`), or semantic blocking (`block_state_mutation`).
> **Line counts:** src 9347 → **9332** (−15 net), with flow −105, observer −42, policy −23,
> models −22, validation −10, utils −4, logger −1, offset by the two designated single-owner
> layers growing: consensus +103 (centralized public evidence) and parsing +67 (text-realized
> acts, realized-comparison predicate, widened objection lexicon), plus state/threads/dialogue/
> prompts +22 combined. Removed responsibilities: intent-based semantic fallbacks, private-rank
> support leak, ThreadType.REPAIR, primary_thread_id, explicit_addressee_id, issue_ledger,
> blocker_probes, `_last_target_speaker` routing memory, `_split_reservation_exchange`,
> `_final_decision_intent`, `_semantic_block`, ten unused helpers. Remaining intentional
> complexity: the act-specific wording in `_thread_intent`/repair intents (prompt content,
> not duplicated mechanism) and the models.py re-export surface.

Run:

```powershell
py -m compileall -q main.py src eval tests
py -m unittest discover -s tests -v
pyflakes src
```

Then run the full evaluation suite once and inspect representative logs for:

- intended versus realized acts
- question/concern/blocker/comparison lifecycle
- coverage after local threads permit it
- narrowing/voting readiness
- formal visible votes
- bounded repair and revoting
- all three outcomes
- fallback and validation-block counts

Compare final line counts with the baseline. A meaningful net reduction is expected, especially in controller/observer/validation code, but correctness and single ownership take priority over a fixed target.

Confirm there is no parallel mechanism left for questions, concerns, blockers, issue repetition, candidate calculation, vote evidence, repair exchange, or semantic mutation blocking.

**Done when:** all checks pass, the full suite completes, and final notes list code removed, responsibilities consolidated, remaining intentional complexity, and final line-count change.