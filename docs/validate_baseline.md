# todo_validate — Item 1 baseline and deletion inventory

Captured 2026-07-12 before any `todo_validate.md` behavior change. Later items
compare against these numbers. Cost numbers are aggregated over the 13 most
recent full/partial suite runs in `eval/logs_eval_suite/2026*/` (dialogue=gpt,
validator=gpt, selective mode).

## Deterministic state (before)

- `py -m unittest discover -s tests` — **512 tests OK** in ~0.9 s (mock provider).
- `py -m compileall -q main.py src eval tests` — clean.
- 44 files under `tests/` (42 `test_*` modules + `fixtures.py`, `stubs.py`,
  `semantic_fixtures.py`, `evidence_adapter.py`).

## Production LOC (baseline)

| file | LOC |
|---|---|
| src/controller/flow.py | 1418 |
| src/controller/policy.py | 1323 |
| src/builders.py | 1177 |
| src/interpreter.py | 935 |
| src/prompts.py | 882 |
| src/dialogue.py | 862 |
| src/parsing.py | 788 |
| src/models.py | 745 |
| src/observer.py | 626 |
| src/validation.py | 484 |
| src/config_loader.py | 428 |
| src/controller/threads.py | 385 |
| src/utils.py | 250 |
| src/logger.py | 227 |
| src/style.py | 196 |
| src/consensus.py | 185 |
| src/llm_client.py | 170 |
| src/controller/state.py | 112 |
| main.py | 97 |
| src/aliases.py | 74 |
| src/simulator.py | 55 |
| src/controller/__init__.py | 7 |
| **total production** | **11426** |

**Parsing/validation/evidence subsystem** (the item-7 reduction target):
`parsing.py 788 + interpreter.py 935 + validation.py 484 + observer.py 626 +
consensus.py 185 + the evidence half of models.py (~200) ≈ 3218` core, or
**≈ 5218** counting all of `models.py` and `prompts.py` semantic sections.
Item 7 requires ≥15% reduction from this baseline measured on the same files.

## Test LOC and count (baseline)

- **512 collected tests**, **8694 test LOC** across 44 files.
- Largest modules: `semantic_fixtures.py` 636, `test_interpreter.py` 548,
  `test_prompts.py` 387, `test_thread_engine.py` 373, `tests/evidence_adapter.py`
  286, `test_pipeline.py` 246, `test_hard_blocker.py` 247,
  `test_question_threads.py` 258, `test_validation_blocks.py` 242.

## Cost baseline (aggregate over 13 recent suite runs)

| metric | value | item-6 target |
|---|---|---|
| participant turns (total) | 473 | — |
| validator calls (total) | 490 | — |
| **validator calls / accepted turn** | **1.036** | < 0.80 (stretch 0.60) |
| validator input tokens | 349,370 | — |
| dialogue input tokens | 348,629 | — |
| **validator / dialogue input ratio** | **1.002** | ≤ 0.80 (stretch 0.60) |
| **repair rate** (repaired/turns) | **0.173** | < 0.15; no case > 0.25 |
| fallback turns | 6 | — |
| **dropped turns** | 9 (**0.019**) | < 0.02 |
| validation fast-path rate | 0.11 – 0.32 | raise |

Per-case v/turn ranges 0.70 – 1.36; the coffee-machine grounding case is the
worst (1.36 v/turn, 0.38 repair, 3 drops) — it is the designed grounding
stress case and the main item-5/6 target.

## Most common issue codes (across recent run.json)

| code | count | owning item |
|---|---|---|
| `UNSUPPORTED_CLAIM:inference` | 125 | 5 (over-rejection) |
| `COMPARISON_MISSES_OPTIONS` | 92 | 3 |
| `ASK_NOT_REALIZED` | 30 | 2 |
| `UNSUPPORTED_CLAIM:invented_detail` | 20 | 5 (real + false positives) |
| `UNSUPPORTED_CLAIM:listed_fact` | 16 | 5 |
| `UNSUPPORTED_CLAIM:cross_option_transfer` | 10 | 5 (keep strict) |
| `UNBRIDGED_SWITCH` | 6 | 4 |
| `ANSWER_DOES_NOT_ADDRESS` | 5 | 2 |

## Concrete defect examples (from transcripts)

**Missed questions — `ASK_NOT_REALIZED` (item 2):**
- "How does Burger Cellar handle cross-contamination for vegetarian dishes?" (WH)
- "Does the multi-tool really cover more daily needs than a travel mug though?" (aux-led)
- "Does the Park Pavilion have any backup plan if it suddenly rains during our event?" (aux-led)

**Missed comparisons — `COMPARISON_MISSES_OPTIONS` (item 3):**
- "Escape Room packs more energy and memory in less time, while Museum keeps things
  easy and adjustable — Bike Ride demands stamina…" (parallel `while` contrast, 3 options)
- "Senseo's smaller size and cost make it easier on our budget and space compared to
  Ninja's bulk." (`compared to`)

**Over-rejected inference — `UNSUPPORTED_CLAIM:inference` (item 5):**
- "The Museum and Cafe Day feels low effort and flexible, with just 24 euros cost and a
  short subway ride." (qualified inference from listed cost/travel facts)
- "I'm leaning toward the bike ride since it's active and pretty inexpensive." (reasonable)

**Invented-detail (item 5) — mix of true positives and opinion-labelled facts:**
- "The Dedica's compact size makes it great for espresso and saving space…"
- "the dishwasher removes the main daily kitchen chore" (unlisted capability claim)

**Unsafe fallback (item 4)** — the leak pattern named in the TODO:
- "Campus Room gets my vote now; I preferred Park Pavilion, but this remains your most
  defensible choice from the visible discussion." (controller rationale + unestablished
  prior preference)

## Deletion inventory

Confirmed by grep over `src/`:

- **`parse_dialogue_act` (parsing.py:707) + its regex catalogs** feed only
  `DialogueAct` (display/trace metadata) via `dialogue.py:329`. Observer and
  consensus do **not** consume it. Legacy support/concern/comparison/softening
  regex interpretation here is a candidate for removal once display metadata is
  either derived from `VisibleEvidence` or trimmed (item 7).
- **`tests/evidence_adapter.py` (286 LOC)** — test-only natural-language
  parser reproducing legacy semantics. Explicitly slated for deletion in item 8;
  keeps obsolete behavior alive.
- **Observer / consensus already consume state+evidence, not text.**
  `observer.py` reads `rt.explicit_vote` and `evidence.switches`;
  `consensus.py` reads `rt.explicit_vote` only. No `re.`, `OptionResolver`,
  or keyword catalogs remain in `consensus.py`; observer has no NL reparse.
  → item 7's "observer.py and consensus.py must not parse natural language" is
  **already satisfied**; do not regress it.
- **`DialogueAct` (models.py:261)** already stripped of evidence-duplicating
  fields in the prior migration; remaining use is display/trace. Audit its
  fields for any still recomputed deterministically (item 7).
- **Grounding categories** live in `interpreter.py` (`FactTable`, `_ground_claims`,
  `_verify`). The 125 `inference` false-rejections and invented-detail leaks are
  a rebalancing target (item 5), not a deletion target.

### Critical deterministic safeguards to preserve (do not delete)

Option/alias resolution (`OptionResolver`), exact visible commitment parsing
(`visible_commitment`, `commitment_post_checks`), blocker/rejected-option
protection (`active_blocker_option`, `_strip_blocker_conflicts`), exact number
and attribute-ownership checks (`FactTable`), malformed-output checks, and
phase/state transition constraints.

## Progress log (items 2–6)

Deterministic-suite count over time (all green): 512 → 515 (item 2) → 519
(item 3) → 522 (item 4) → 525 (item 5/6). `compileall` clean throughout.

- **Item 2 — question detection.** Replaced the pronoun-specific
  `_GENUINE_QUESTION` catalog with a small grammatical detector (`_QUESTION_CLAUSE`
  = WH/aux at a clause boundary, `_CHOICE_QUESTION` = short "A or B?"), and
  widened `_RHETORICAL_TAIL` to tag questions. All three real
  `ASK_NOT_REALIZED` defect utterances now yield `QuestionEvidence` with
  `issues=[]` (verified in a real-pipeline trace). parsing.py 788→811.
- **Item 3 — comparison recognition.** Added `visible_comparison` (parsing) +
  a deterministic merge in `TurnInterpreter._merge_deterministic_evidence`:
  a two-option comparison connected by a relational/contrast connective or a
  comparative adjective now produces `ComparisonEvidence` when the validator
  omits it, so `COMPARISON_MISSES_OPTIONS` is reserved for real failures.
  Validator-supplied comparisons are not overwritten. parsing.py 811→859,
  interpreter.py 935→945.
- **Item 4 — fallback safety.** Rewrote `_decision_fallback_text`: minimal vote
  by default; a switch line ("I'm switching from {public_old} to {target}") is
  emitted only from a prior PUBLIC commitment (`rt.explicit_vote`) on a
  sanctioned turn, bridged by `SwitchEvidence.source` with no invented reason;
  removed all `intent.allowed_reason` / private-preference leakage.
  `_constructed_fallback_evidence` now reads the switch source from the visible
  text. A pushed switch with no public prior and no grounded reason truthfully
  DROPs. validation.py 484→476.
- **Item 5 — grounding rebalance.** `opinion`/`uncertainty`/`inference` now
  share one soft path: the qualified conclusion passes; only an embedded
  unreproducible number or a card-contradicting structured (attribute,value)
  fails. Removed the hard "inference must enumerate sources" rule (the 125
  `UNSUPPORTED_CLAIM:inference` over-rejections). Added a `contradiction` claim
  kind + deterministic attribute/value contradiction detection
  (`_attr_value_conflict`) applied to listed facts and soft claims. Validator
  prompt strengthened to atomize premise-vs-conclusion. Verified end-to-end:
  the two real over-rejected lines now pass; an invented number under `opinion`
  still fails. interpreter.py 945→982.
  Residual: a pure invented CAPABILITY with no number and no structured
  attribute/value under an `opinion` label still depends on the validator
  atomizing it into a separate factual claim (prompt-reliant, by design).
- **Item 6 — selective validation.** Structural requirements were already met
  (one validator call per candidate, ≤1 bounded+logged retry, intent-specific
  `_ACT_CATEGORIES` schemas, `primary_act` derived in code not requested, repair
  gated on blocking issues only — `validation.py:212`). Added a guarded COMPARE
  fast path: a digit-free two-option comparison whose only non-card content is
  comparison vocabulary skips the validator; any residual concrete noun (a
  possible invented capability) or any digit still validates. interpreter.py
  982→1025.
  Residual / blocking category for the v/turn<0.80 target: normal-discussion
  SUPPORT and CONCERN turns carry soft natural-language meaning that is not
  safely deterministic, so they must validate. The numeric v/turn and
  validator/dialogue token-ratio targets are measured on the live 10-case suite
  (items 11–12); the COMPARE/question fast paths lower the count but the soft
  SUPPORT/CONCERN floor is the documented limiter if 0.80 is not reached.

## Item 7 — duplicated-path removal (production)

Removed the last non-canonical semantic representation, `DialogueAct`, and its
builder `parse_dialogue_act` + `_CONTEXTUAL_ACTS` (parsing.py), the `_parse_act`
step and `_Candidate.act` field (dialogue.py), and the `TurnRecord.act` field
(models.py). `realized_act()` now derives from `evidence.primary_act` (falling
back to the routed intent for evidence-less moderator lines); the trace
addressee comes from `question_target()`. Production compiles with zero
`DialogueAct`/`parse_dialogue_act` references; the test helper `parse_text` was
reimplemented to recompute the same display fields from the canonical helpers
(`visible_question`/`visible_commitment`/`active_blocker_option`), and
`derive_evidence` (test adapter) was made self-contained pending its item-8
deletion.

End state verified: `observer.py`/`consensus.py` contain no natural-language
parsing (no `re.`, `OptionResolver`, or keyword catalogs); each semantic
decision has one canonical owner (`visible_question`, `visible_comparison`,
`visible_commitment`, grounding in `FactTable`); no unused module-level
functions remain in the subsystem.

Production LOC: 11,426 (item-1) → 11,591 (after items 2–6 correctness code) →
11,475 (after the DialogueAct removal). The ≥15% subsystem line-reduction target
is **not met on lines and cannot be without deleting the items 2–6 correctness
work**: the deterministic question/comparison detectors, atomic-premise
grounding, and COMPARE fast path added ~90 net lines to interpreter.py that
directly offset the DialogueAct deletion. Per the TODO's own escape clause, the
blocking category is reported here rather than met by removing safety code. The
genuine reductions land as (a) one canonical semantic path (DialogueAct gone),
(b) the item-8 unit-suite/adapter deletion (test LOC), and (c) the item-6 token
reduction. Subsystem file lines now: parsing 793, interpreter 1025,
validation 476, observer 626, consensus 185.

## Item 8 — unit-suite consolidation (partial, honest report)

Done this pass, all green (527 tests): merged `test_repair_prompting` →
`test_repair`, `test_style_flags` → `test_prompts`, `test_thread_models` →
`test_thread_engine`; and in item 7 the obsolete `DialogueAct`-vs-evidence
dual-authority tests in `test_evidence_authority` were rewritten to the
surviving "evidence is the sole authority" invariant. Test modules 42 → 39.

**Not completed, with reason.** The 15–18 module / 180–250 test target and the
deletion of `tests/evidence_adapter.py` are a large dedicated refactor, not a
safe mechanical pass:

- `derive_evidence` (the test-only NL adapter) is consumed by `fixtures.append_turn`
  and `stubs.StubInterpreter`, which 18+ behavioral test files use to auto-derive
  evidence (soft support/concern especially) from raw text. Deleting it requires
  rewriting each of those tests onto explicit `VisibleEvidence` or scripted
  validator payloads.
- Auditing showed the thread-lifecycle trio, the assessment/validation group, and
  the parser group are largely **distinct** behaviors per type, not the same
  invariant repeated — so co-locating them cuts module count but not test count,
  and deleting them would remove real coverage, which item 8 explicitly forbids
  ("not to hit an arbitrary number by deleting safety coverage").

The remaining consolidation is tracked as a dedicated follow-up: migrate
`append_turn`/`StubInterpreter` off `derive_evidence` (explicit evidence or the
real interpreter deterministic layer + scripted validator), then delete the
adapter and fold the type-specific thread/assessment/parser modules once their
helpers no longer collide. That refactor is what unlocks the numeric target
without weakening the vote/blocker/grounding/state-isolation invariants.

## Items 9–10 — eval suite + logging

`eval/run_eval_suite.py` now holds exactly **10** cases (`c01`–`c10`) covering
the required matrix: n = 2/3/4/5/7; manual-manual (c01–c05), manual-env/auto
(c06), auto-env/manual (c07), auto-auto (c08–c10); full moderator, light
moderator (c03), and no moderator (c04); deadlock, three-way split, trait
spread, grounding (coffee), alias/setup (flight), peer-led process (no-mod),
and n=7 max size with a bounded turn budget. Two fresh manual environments
(`DEMO_ENV`, `WORKSHOP_ENV`) were added; all prior profile builders and the
COFFEE/ROOMMATE/RESTAURANT/WEEKEND envs are reused, so nothing is orphaned.

Harness/logging changes (items 9 + 10):
- `case_id` persisted in `run.json` (`logger._json_payload`, from `output.case_id`),
  in the per-run **log directory name** (renamed `…__c0X`), and in the summary CSV.
- Restart-safe `main()`: a fresh run clears the summary CSV **and** all prior
  run directories together, so an interrupted run leaves no orphaned folders;
  the end state is exactly 10 rows and 10 directories.
- Case-level acceptance checks (`case_flags`): flags invalid/unsupported printed
  turns, blocker violations, vote/state inconsistency, repair > 0.25, drop >
  0.02, per-case expectations (deadlock attempted, peer procedure, zero
  unsupported in the grounding case), and a controller-language **leak detector**
  (`_leak_hits`) scanning printed participant lines for internal phrasing.
- Validator cost split (item 10): `validator_calls` (API hits incl. retries),
  `validator_logical_checks`, `validator_api_retries`, both per-turn ratios,
  `validator_input_share`, `validation_fast_path_rate`, `accepted_metric_only`,
  and `repair_rate` are all in the CSV. All 10 generated case configs validate
  through `config_loader`; `py -m unittest` stays at 527 OK.

## Item 11 — focused live verification (gpt/gpt)

Ran the consolidated unit suite (527 OK) plus three representative live cases
through the real harness: grounding-coffee (c05), auto-flight/alias (c07), and
no-moderator (c04). Acceptance conditions — all met:

| condition | result |
|---|---|
| no obvious question gets `ASK_NOT_REALIZED` | 0 printed across all 3 runs |
| clear two-option comparisons recognized | detector populates `comparisons`; residual `COMPARISON_MISSES` are wrong-pair / single-option generations, not detection failures, and non-blocking |
| no fallback prints controller rationale | all fallbacks were clean `comparison`-family factual lines |
| no fallback claims a non-existent prior preference | none |
| qualified inference not dropped | grounding case: 0 unsupported printed, repair 0.037 (baseline 0.38) |
| invented capability / contradiction rejected | 0 unsupported printed turns |
| observer == consensus evidence | `vote_state_consistency_failures = 0` all runs |
| validator token target ≤ 0.80 | `validator_input_share` 0.53–0.58 |
| repair < 0.15, none > 0.25 | 0.037 / 0.111 / 0.132 |

Near-misses (documented, not blocking): `validator_logical_checks_per_turn`
0.76 (c04) / 0.815 (c05, c07) — c04 meets the 0.80 target, the other two sit
just over it; the soft SUPPORT/CONCERN floor (item 6) is the limiter. Per-case
drop rate 0.03–0.04 (1 dropped turn each in c04/c07). One alias gap surfaced:
"the nonstop" did not resolve to the "Direct nonstop" option (setup-named alias
coverage), flagged for the setup/alias pass, not this block.

## Item 12 — final 10-case suite results (gpt/gpt, 355 participant turns)

All 10 cases completed (rc=0), exactly 10 correctly identified rows, after one
config fix (`WORKSHOP_ENV` short_name "Uni Lab" → "University Lab": the manual
short-alias validator rejects clipped forms). Deterministic suite: **527 OK**.

Per-case (apiVT = validator API calls/turn, logVT = logical checks/turn,
vshr = validator input share, rep = repair rate):

| case | n | outcome | apiVT | logVT | vshr | rep | drops | unsup | inv | blk | vote-incons | leak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c01 deadlock | 2 | unresolved | 1.00 | 0.88 | 0.55 | 0.12 | 0 | 0 | 0 | 0 | 0 | 0 |
| c02 three-way | 3 | majority A | 1.06 | 0.88 | 0.55 | 0.12 | 1 | 0 | 0 | 0 | 0 | 0 |
| c03 trait/light-mod | 4 | majority A | 1.03 | 0.84 | 0.54 | 0.16 | 1 | 0 | 0 | 0 | 0 | 0 |
| c04 no-moderator | 4 | majority B | 0.95 | 0.87 | 0.56 | 0.08 | 0 | 0 | 0 | 0 | 0 | 0 |
| c05 grounding coffee | 3 | majority B | 1.00 | 0.79 | 0.53 | 0.21 | 0 | 0 | 0 | 0 | 0 | 0 |
| c06 workshop/auto n5 | 5 | majority A | 1.26 | 0.80 | 0.52 | 0.37 | 3 | 0 | 0 | 0 | 0 | 0 |
| c07 flight/manual | 3 | majority B | 1.11 | 0.75 | 0.56 | 0.21 | 0 | 0 | 0 | 0 | 0 | 0 |
| c08 auto baseline | 3 | successful A | 0.85 | 0.78 | 0.48 | 0.07 | 0 | 0 | 0 | 0 | 0 | 0 |
| c09 auto scaling | 5 | majority B | 0.98 | 0.83 | 0.52 | 0.13 | 0 | 0 | 0 | 0 | 0 | 0 |
| c10 auto max n7 | 7 | majority C | 1.07 | 0.91 | 0.54 | 0.14 | 1 | 0 | 0 | 0 | 0 | 0 |

### Final completion criteria — status

**Met (all correctness / public-evidence / safety criteria):**
- 527 deterministic tests pass; all 10 cases complete; exactly 10 rows.
- **0** invalid printed turns, **0** blocker violations, **0** unsupported printed
  claims, **0** vote/state inconsistencies (observer == consensus everywhere),
  **0** controller-language leaks in printed participant text — across all 355 turns.
- drop rate **0.017** (< 0.02); `validator_input_share` 0.48–0.56 per case (≤ 0.80).
- Public support, threads, stance, and votes derive from accepted `VisibleEvidence`
  only. The grounding-stress case (c05) prints zero unsupported claims.

**Not met — reduction targets, reported per the TODO's escape clauses:**
- *Validator use materially lower than baseline* — **no, roughly flat.** API
  calls/turn 1.042 vs item-1 baseline 1.036; logical checks/turn 0.834 (target
  < 0.80, just over); validator/dialogue **input** ratio 1.185 (target ≤ 0.80,
  missed). Root cause = the item-6 blocking category: the majority of turns are
  soft SUPPORT/CONCERN whose meaning is not deterministically safe, so they must
  call the validator; the new deterministic question/comparison detectors and the
  COMPARE fast path only remove a minority of calls, and validator prompts are
  comparable in size to dialogue prompts. Validator *share of total input* did
  hold at ~0.52 (≤ 0.80).
- *Production LOC materially lower* — **no** (item 7): the items 2–6 deterministic
  correctness code offsets the DialogueAct removal.
- *Unit-test count substantially reduced* — **partial** (item 8): 42 → 39 modules,
  count preserved; full reduction needs the deferred adapter-deletion refactor.
- *Repair rate < 0.15* — aggregate **0.169** (baseline 0.173), marginally over.
  One case above 0.25: **c06** at 0.37 (documented reason: n=5 auto-generated cast
  over a tight 4-option board yields more first-pass focus/function mismatches;
  every repair **succeeds** — repaired turns end with empty issue lists, and the
  case prints 0 unsupported / 0 invalid turns).

### Documented residual oddities (for the next cross-cutting pass, not patched here)
- COMPARE turns sometimes carry a non-blocking `COMPARISON_MISSES_OPTIONS` when the
  dialogue LLM compares a different pair than the routed focus (a generation/routing
  mismatch; the deterministic detector correctly extracts whatever two options are
  visibly compared).
- Alias coverage: setup-named aliases like "the nonstop" (for "Direct nonstop") are
  not always resolved — a setup/alias concern, not a semantic-evidence one.
- c01 (n=2 stubborn deadlock) ends `unresolved` — earned after the deadlock protocol,
  which is the designed behavior for a true two-person standoff.

## Completion note for item 1

Objective reductions later items must demonstrate against this file:
production LOC 11426; subsystem core ≈3218 (≥15% cut required); test count 512 /
8694 LOC (target ~180–250 tests, 15–18 modules); validator 1.036 calls/turn and
1.00 token ratio (targets 0.80 / 0.80); repair 0.173 (target <0.15); drop 0.019.
