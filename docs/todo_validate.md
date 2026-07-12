# TODO — Final stabilization, simplification, and test-suite reduction

## Purpose

The parsing, validation, repair, fallback, and visible-evidence migration is substantially improved, but the current implementation is still larger and more expensive than intended. The remaining work must be corrective and subtractive rather than another redesign.

Current baseline from the reviewed repository:

- approximately 11.3k production Python LOC;
- approximately 5.7k LOC in the parsing/validation/evidence subsystem;
- 42 unit-test modules and 512 collected tests;
- 12 dialogue evaluation cases in `eval/run_eval_suite.py`;
- validator calls still occur approximately once per participant turn in the available full-suite logs;
- validator input tokens remain roughly comparable to dialogue-generation input tokens;
- all current deterministic tests pass, but several runtime transcript defects remain.

The goals of this TODO are to:

- fix the remaining correctness defects in question, comparison, grounding, repair, and fallback handling;
- reduce validator calls, tokens, retries, and latency without weakening public-evidence safety;
- remove obsolete, duplicated, compatibility-only, and test-only semantic paths;
- materially reduce production code;
- replace the oversized and overlapping unit suite with a smaller behavior-focused suite;
- replace the current repetitive evaluation suite with exactly 10 representative cases and fresh topics;
- inspect generated transcripts and traces, not merely subprocess return codes or unit-test success.

Do not reopen scenario/persona generation, controller/routing, prompt design, or consensus architecture unless a concrete dependency in this TODO requires a small correction.

---

## Non-negotiable behavioral rules

1. Controller intent states what the turn should express.
2. The accepted visible utterance states what was publicly said.
3. Validated `VisibleEvidence` is the sole semantic authority for public support, concerns, questions, answers, comparisons, proposals, commitments, blockers, switches, thread updates, and participant state changes.
4. Hidden intent must never count as visible evidence.
5. Only a participant's own accepted visible evidence may change that participant's private ranks, preferred option, vote, or blocker state.
6. Direct votes and switches must remain conservative and unambiguous.
7. Unsupported concrete facts and cross-option fact transfers must not be accepted.
8. Qualified opinions and reasonable inferences from listed facts must not be rejected merely because their exact conclusion is not stated verbatim in the scenario.
9. Repair and fallback text must pass the same evidence, grounding, and validation pipeline as ordinary generated text.
10. Controller-facing rationale, trace terminology, or hidden state descriptions must never appear in participant-visible fallback text.
11. There remain exactly two configurable LLM roles: `dialogue` and `validator`. Do not add a third runtime checker.
12. Do not reintroduce generator self-report metadata.

---

## Required procedure for every implementation item

For each item:

1. Inspect the relevant current implementation and tests.
2. Identify the smallest coherent change.
3. Implement only that item.
4. Add, update, consolidate, or delete focused tests as appropriate.
5. Run the targeted tests.
6. Generate a few representative dialogue samples when runtime behavior is affected.
7. Inspect transcript and trace output, not only test success.
8. Resolve regressions before moving to the next item.
9. Summarize:
   - what changed;
   - what code and tests were removed;
   - production and test LOC before/after;
   - tests and sample commands run;
   - transcript and trace observations;
   - dialogue and validator calls/tokens for affected samples.

Do not run the complete dialogue evaluation suite after every item. Use focused unit tests and small representative generations first.

---

# Remaining implementation items

## [x] 1. Record the current baseline and make deletions measurable

### Relevant files

- `src/dialogue.py`
- `src/interpreter.py`
- `src/models.py`
- `src/observer.py`
- `src/parsing.py`
- `src/prompts.py`
- `src/validation.py`
- `src/consensus.py`
- `tests/`
- `eval/run_eval_suite.py`
- current full-suite logs

### Required work

Before modifying behavior, record:

- production LOC by affected file;
- test LOC and test count by module;
- validator calls per participant turn;
- validator/dialogue input-token ratio;
- repair, fallback, and drop rates;
- the most common repair and validation issue codes;
- examples of missed questions, missed comparisons, unsupported accepted claims, over-rejected inferences, and unsafe fallback text.

Create a short deletion inventory identifying:

- legacy semantic helpers still duplicated by `VisibleEvidence`;
- observer or consensus code that reparses visible text;
- test-only compatibility parsing;
- unused models, fields, helpers, prompt outputs, comments, and documentation;
- tests that assert helper implementation rather than a public behavioral contract;
- multiple tests that exercise the same invariant at several layers without adding unique coverage.

### Completion criteria

- Later items can report objective code, test, and token reductions.
- No change is justified only by making an individual file look shorter.
- The inventory distinguishes critical deterministic safeguards from removable soft-semantic duplication.

---

## [x] 2. Correct genuine-question detection and thread evidence

### Current problem

Ordinary questions can be accepted with `ASK_NOT_REALIZED`, including forms such as:

```text
Does the Park Pavilion have a backup plan if it rains?
Does the Moccamaster allow half-pot brewing?
How reliable is the weather forecast?
How does the Ninja's footprint compare?
```

This can prevent question threads from opening and can make later answers appear unrelated.

### Required change

- Replace narrow pronoun-specific question patterns with a small grammatical detector for:
  - auxiliary-led questions such as `does/is/can/would/has + noun phrase`;
  - ordinary WH-questions;
  - short option-choice questions;
  - direct questions with an explicit participant addressee.
- Preserve semantic-validator interpretation for ambiguous question scope or answer relevance.
- Do not add option-specific or endpoint-specific phrases.
- Ensure a visibly genuine question produces `QuestionEvidence` even if its primary act also includes comparison, concern, or support.
- Ensure question detection alone does not infer an answer target that is not visible or supplied by the active thread.

### Tests and samples

Retain only representative boundaries:

- clear auxiliary question;
- clear WH-question;
- comparative question;
- statement ending in a rhetorical check-in;
- conditional statement that is not a question;
- direct addressee;
- ambiguous focus.

Run a question-heavy dialogue and verify:

- question threads open correctly;
- the required respondent is routed;
- a relevant answer cools/resolves the thread;
- unrelated replies do not resolve it;
- no obvious question receives `ASK_NOT_REALIZED`.

---

## [x] 3. Correct basic comparison recognition without expanding regex vocabularies

### Current problem

Visible comparisons are sometimes repaired or downgraded because the current logic misses natural structures such as:

```text
Moccamaster's upfront cost is almost double Ninja's—worth it?
A is cheaper, while B has more capacity.
A costs more but B takes longer.
```

### Required change

- Use canonical option spans and clause structure first.
- When two distinct options are visibly present and connected by a clear comparative or contrast construction, create basic `ComparisonEvidence` deterministically.
- Support:
  - comparative adjectives/adverbs;
  - `while`, `but`, `whereas`, `compared with/to`, `than`, `versus`, and parallel contrast clauses;
  - comparative questions.
- Let the semantic validator determine subtle comparison direction or dimension only when deterministic extraction is insufficient.
- Do not maintain a growing endpoint-tuned phrase catalogue.
- A line may simultaneously be a comparison and a question/support/concern.

### Tests and samples

Use a compact parameterized set covering:

- explicit `A versus B`;
- parallel `A ..., while B ...`;
- comparative adjective with possessives;
- two mentions without comparison;
- one-option comparative wording;
- comparison plus question;
- comparison plus concern.

Run one comparison-heavy sample and confirm that `COMPARISON_MISSES_OPTIONS` is reserved for actual failures.

---

## [x] 4. Make vote and blocker fallback minimal, public, and truthful

### Current problem

Fallback text can leak controller rationale or state a previous preference that was never publicly established, for example:

```text
Campus Room gets my vote now; I preferred Park Pavilion, but this remains your most defensible choice from the visible discussion.
```

### Required change

- Never insert `intent.allowed_reason`, route rationale, trace text, policy explanations, or controller-facing wording directly into public fallback text.
- The default vote fallback must be minimal:

```text
Campus Room gets my vote.
```

- Mention a switch only when a prior public commitment by the same participant exists and is present in accepted evidence/state:

```text
I'm switching from A to B.
```

- Mention a reason only when it is already available as grounded participant-facing evidence and can be rendered without inventing meaning.
- Do not say `I preferred A` based only on private ranks or controller state.
- Keep blocker fallback tied to an already stored grounded blocker reason.
- Drop the turn when no safe act-specific fallback exists.
- Revalidate fallback through the full normal pipeline.

### Tests and samples

Cover only the behavioral boundaries:

- minimal vote fallback;
- validated public switch;
- no public old commitment;
- rejected target;
- grounded blocker restatement;
- internal rationale never appears;
- fallback is reinterpreted and grounded;
- unsafe fallback drops.

Force vote repair/fallback in a small generation and inspect the exact printed text and trace.

---

## [x] 5. Rebalance grounding around atomic factual premises

### Current problems

The current grounding path has both failure types:

- reasonable qualified inferences are frequently rejected as `UNSUPPORTED_CLAIM:inference`;
- invented capabilities or concrete details can pass when the full sentence is labeled as opinion or inference.

Examples requiring different handling:

```text
The cheaper option may be easier on the budget.
```

This is a reasonable qualified inference from listed cost facts.

```text
Automatic reminders make Teams easier to use.
```

`easier to use` is an opinion, but `automatic reminders` is a concrete capability claim that must be separately grounded.

### Required change

- Extract and validate factual premises separately from their subjective conclusion.
- Grounding categories must remain explicit:
  - listed fact;
  - reproducible arithmetic;
  - reasonable opinion;
  - qualified inference from listed facts;
  - uncertainty/question;
  - unsupported concrete fact;
  - cross-option transfer;
  - contradiction of listed facts.
- A qualified inference passes when its premises are grounded and the conclusion is not presented as a new concrete fact.
- An opinion label must not hide an unlisted number, capability, facility, event, guarantee, or option attribute.
- Validate exact option-attribute-value ownership deterministically.
- Add direct contradiction detection where the utterance conflicts with an option card, not merely where a value is absent.
- The validator must return atomic claim spans; code must confirm that each span exists in the visible utterance.
- Do not add another grounding model call. Keep grounding within the existing validator interpretation plus deterministic verification.
- Remove blanket rules such as “all opinions pass” when the sentence contains factual premises.

### Focused regression corpus

Keep a small curated corpus covering:

- exact listed fact;
- shared-context fact;
- exact cross-option transfer;
- unlisted number;
- unlisted product/venue capability;
- opinion with no factual premise;
- opinion containing an unsupported factual premise;
- qualified inference with valid sources;
- qualified inference with invented premise;
- reproducible arithmetic;
- direct contradiction of a listed concern/attribute;
- uncertainty phrased as a question rather than an assertion.

Run one grounding-heavy sample, preferably the coffee-machine case, and manually audit every concrete product capability in the transcript.

---

## [x] 6. Make semantic validation genuinely selective and reduce validator cost

### Current problem

Despite selective fast paths, the available suite still uses approximately one validator call per participant turn, and validator input tokens remain roughly equal to dialogue-generation input tokens. Repairs add further calls and latency.

### Required change

Classify turns before calling the semantic validator.

Always use cheap deterministic checks for:

- envelope/malformed output;
- explicit option aliases and spans;
- direct unambiguous vote/acceptance;
- required vote target;
- visible switch source/target when explicit;
- hard blocker phrases and rejected-option protection;
- exact numerical and option-attribute-value claims;
- obvious questions and basic comparisons after items 2 and 3.

Use the validator only when needed for state-changing or genuinely ambiguous semantics, including:

- natural support or concern not deterministically clear;
- answer relevance;
- softening/concession;
- compromise/proposal;
- ambiguous implicit reference;
- ambiguous factual premise versus opinion/inference;
- subtle comparison direction/dimension.

Do not call the validator for:

- process comments with no state-changing semantics;
- closings;
- deterministic direct votes;
- deterministic blocker restatements;
- ordinary comments with no option/claim/thread relevance;
- turns whose complete required evidence is already safely extracted.

Additional requirements:

- Keep one validator call per candidate; do not split interpretation and grounding into separate LLM calls.
- Do not ask the validator to return fields that are derived in code, including redundant primary-act or intent-alignment explanations.
- Use compact intent-specific schemas rather than the full universal evidence schema when practical.
- Bound provider-level retries and log each retry separately.
- Repair only blocking failures. Metric-only disagreement must not cause another generation and validation cycle.

### Acceptance targets

Across focused samples and the final 10-case suite:

- validator calls per participant turn: target below `0.80`, stretch target `0.60`;
- validator input tokens: no more than `80%` of dialogue input tokens, stretch target `60%`;
- overall repair rate below `15%`;
- no individual case above `25%` repair rate without an explicit documented reason;
- dropped participant turns below `2%` overall;
- no vote or final commitment is accepted without clear public evidence.

If these targets cannot be met without weakening correctness, report the exact blocking categories rather than adding more validation layers.

---

## [x] 7. Remove duplicated semantic paths and materially reduce production code

### Required architectural end state

The final participant-turn path must be:

```text
controller intent
    -> generate visible utterance
    -> conservative cleanup
    -> canonical option/addressee resolution
    -> deterministic critical evidence extraction
    -> selective semantic interpretation when required
    -> deterministic evidence/grounding verification
    -> ACCEPT / ACCEPT_WITH_METRIC / REPAIR / FALLBACK / DROP
    -> observer and consensus consume the same accepted VisibleEvidence
```

### Required deletions and consolidation

Inspect and remove or merge the following where they remain:

- broad legacy support/concern/comparison/softening/compromise regex interpretation in `parsing.py` after equivalent consumers use `VisibleEvidence`;
- observer-side reparsing of commitment reasons, switch bridges, issue relevance, or semantic functions already present in accepted evidence;
- consensus/public-support dependence on legacy act labels;
- compatibility adapters that reconstruct evidence from a legacy `DialogueAct`;
- redundant LLM-returned `primary_act`, intent-realization explanation, or other fields recomputed deterministically;
- unused evidence fields or model wrappers;
- obsolete TODO-number comments and migration comments;
- stale configuration paths and documentation;
- dead helper functions and unreachable checks;
- duplicate prompt instructions shared by generation, validation, and repair when one canonical helper is sufficient.

Preserve deterministic code that still adds precision:

- option and alias resolution;
- exact visible commitment parsing;
- blocker/rejected-option protection;
- exact number and attribute ownership checks;
- malformed output checks;
- state-transition constraints.

### Code-reduction targets

Do not optimize to a cosmetic per-file limit. Reduce total responsibility overlap.

Targets after all correctness fixes:

- reduce the affected parsing/validation/evidence subsystem by at least `15%` from the recorded item-1 baseline;
- reduce total production Python LOC rather than moving code to new files;
- no new production module unless it replaces and deletes more code than it adds;
- `observer.py` and `consensus.py` must not parse natural language;
- one canonical implementation for each semantic decision.

Every retained helper should have a current caller and a clear responsibility.

---

## [x] 8. Replace the 512-test accumulation with a smaller behavior-focused unit suite

### Current problem

The repository contains 42 test modules and 512 tests. Many are useful regressions, but the suite also repeats the same contract across parser, adapter, validator, observer, pipeline, thread, prompt, and trace layers. A large test-only evidence adapter reproduces legacy semantics and can keep obsolete behavior alive.

Do **not** reduce the entire unit suite to 10 tests; that would be insufficient for this stateful system. The exact target of 10 applies to the dialogue evaluation suite in item 9. The unit suite should nevertheless be reduced substantially.

### Test design rules

- Test externally meaningful behavior and safety invariants, not every private helper branch.
- Keep one direct unit test for a low-level critical helper and one integration test for its cross-layer contract; avoid repeating the same assertion in four layers.
- Use parameterization for equivalent language forms and boundary tables.
- Keep regressions for real observed bugs.
- Delete tests whose only purpose is to preserve removed compatibility code.
- Tests should construct explicit `VisibleEvidence` when testing observer/consensus behavior; they must not run a separate test-only natural-language parser.
- Natural-language interpretation belongs in a compact interpreter corpus, not duplicated throughout unrelated test modules.
- Live network/LLM calls must not be part of deterministic unit tests.
- Keep only a few bounded end-to-end tests.

### Required consolidation

Audit and consolidate these overlapping groups:

1. `test_acts.py`, `test_parsing.py`, `test_critical_parser.py`, `test_evidence_model.py`, `test_interpreter.py`, `test_semantic_fixtures.py`.
2. `test_assessment.py`, `test_validation_blocks.py`, `test_intent_text_mismatch.py`, `test_responsibilities.py`.
3. `test_repair.py`, `test_repair_prompting.py`, `test_fallback.py`.
4. `test_question_threads.py`, `test_concern_threads.py`, `test_comparison_threads.py`, `test_thread_engine.py`, `test_thread_models.py`.
5. `test_models.py`, `test_phases.py`, `test_progress.py`, `test_pipeline.py`, `test_trace.py` where the same state-transition behavior is repeated.
6. `test_prompts.py` and `test_style_flags.py`, retaining only prompt contracts that affect correctness rather than exact prose or section implementation.
7. `test_evidence_authority.py`, `test_observer_evidence.py`, and relevant `test_consensus.py` cases, retaining the key same-evidence-authority integration tests.

Delete `tests/evidence_adapter.py` after all tests use explicit accepted evidence or the real interpreter path.

Retain focused coverage for:

- CLI and alias setup regressions;
- option resolution and strict commitment safety;
- visible-evidence authority;
- grounding categories and contradictions;
- validation action severity;
- repair/fallback safety;
- speaker-local stance mutation;
- question/concern/comparison thread lifecycle;
- consensus vote/public-support rules;
- controller routing priorities;
- logging/evaluation metrics;
- a small number of bounded end-to-end outcomes.

### Size target

Aim for:

- no more than approximately `15–18` unit-test modules;
- approximately `180–250` collected tests, unless the audit demonstrates that a specific unique invariant requires more;
- materially lower test LOC;
- no reduction in coverage of critical public-evidence, vote, blocker, grounding, and state-isolation invariants.

The objective is not to hit an arbitrary number by deleting safety coverage. Every retained test must defend a distinct behavior or observed regression.

---

## [x] 9. Replace the dialogue evaluation suite with exactly 10 representative cases

### Current problems

The current suite contains 12 cases, repeats several environments/topics, and treats subprocess completion as success even when a run has excessive repairs, drops, token cost, or transcript defects. Summary output can become incomplete across interrupted/restarted runs.

### Required suite structure

`eval/run_eval_suite.py` must contain exactly **10** dialogue cases. Keep one sequential full-suite command; do not add a large flag surface.

Use a balanced matrix with fresh topics and minimal duplication:

1. **Manual/manual, n=2, stubborn deadlock**  
   Keep a two-person opposing-preference regression, using the shared-home upgrade topic or a refreshed equivalent.

2. **Manual/manual, n=3, three-way split and narrowing**  
   Fresh topic suggestion: choose a Saturday plan under uncertain weather and limited time.

3. **Manual/manual, n=4, strong trait spread**  
   Fresh topic suggestion: choose how a student software project should present its final demo.

4. **Manual/manual, n=4, no moderator**  
   Fresh topic suggestion: choose a group dinner plan with dietary and travel constraints.

5. **Manual/manual, n=3, grounding stress case**  
   Keep the coffee-machine case because it exposed real capability, inference, repair, and token defects; update its facts if needed but preserve its regression value.

6. **Manual environment/automatic participants, n=5**  
   Fresh topic suggestion: choose a format and venue for a community workshop.

7. **Automatic environment/manual participants, n=3**  
   Use `Book a flight from Miami to Stockholm` to retain the automatic setup and alias-repair regression with controlled personas.

8. **Automatic/automatic, n=3 baseline**  
   Fresh practical topic suggestion: choose a shared scheduling method for a volunteer group.

9. **Automatic/automatic, n=5 scaling case**  
   Fresh topic suggestion: choose a one-day team retreat format with mixed budgets and energy levels.

10. **Automatic/automatic, n=7 maximum-size case**  
    Fresh topic suggestion: choose a format for a student hackathon or project showcase. Use a bounded turn budget so this case tests scale without dominating total cost.

Requirements:

- Avoid reusing the same manual environment in several cases unless it targets a distinct known regression.
- Cover participant counts 2, 3, 4, 5, and 7.
- Cover manual/manual, manual/auto, auto/manual, and auto/auto combinations.
- Cover full moderator, light moderator, and no-moderator behavior.
- Preserve cases for deadlock, three-way split, trait visibility, grounding, alias/setup, peer-led process, and maximum group size.
- Store the suite case ID in each run's metadata and log directory.
- Make summary generation restart-safe: interrupted or resumed runs must not silently erase previously completed rows while leaving orphaned run directories.
- Remove old environments, persona fixtures, and cases no longer used by the final 10.

### Case-level acceptance checks

A case must be more than `returncode == 0`.

At minimum, report and flag:

- invalid printed turns;
- unsupported printed claims detected by the pipeline;
- repair/fallback/drop counts and rates;
- validator calls and tokens;
- intended-function realization;
- final visible votes and outcome consistency;
- controller/internal-language leakage;
- question/answer and concern-response behavior;
- case-specific expectation, such as deadlock attempted or peer vote call used.

Do not hard-code that a particular option must win unless the case is specifically designed for that invariant. Flag suspicious behavior for manual transcript review rather than forcing a predetermined outcome.

---

## [x] 10. Simplify evaluation logging needed for this block

This is not the full later evaluation/logging review. Only make the changes required to assess the current fixes reliably.

### Required work

- Persist `case_id` in `run.json`, trace metadata, summary CSV, and the case log directory name.
- Log validator endpoint calls separately from logical validation attempts and retries.
- Report:
  - participant turns;
  - validator logical checks;
  - validator API calls/retries;
  - validator calls per participant turn;
  - dialogue and validator input/output tokens;
  - validator/dialogue token ratio;
  - repair, fallback, and drop rates;
  - issue-code counts before and after repair;
  - fast-path family counts;
  - accepted turns carrying metric-only issues.
- Add an explicit leak detector for known controller-facing phrases in printed participant text.
- Keep logs concise enough for manual inspection. Do not duplicate the same evidence object in several files unless each representation has a distinct use.
- Update stale baseline documents or remove them if they no longer provide value.

---

## [x] 11. Focused verification before the final suite

After items 2–10, run only:

1. the consolidated deterministic unit suite;
2. a question-heavy sample;
3. a comparison-heavy sample;
4. a grounding-heavy coffee-machine sample;
5. a forced vote-repair/fallback sample;
6. one automatic scenario using `Book a flight from Miami to Stockholm`;
7. one no-moderator sample.

Inspect transcript and trace output manually.

### Required acceptance conditions

- no obvious question receives `ASK_NOT_REALIZED`;
- clear two-option comparisons are recognized;
- no fallback prints controller-facing rationale;
- no fallback claims a previous public preference that did not exist;
- ordinary qualified inference from listed facts is not dropped;
- invented concrete capability and direct contradiction regressions are rejected or repaired;
- observer and consensus consume the same accepted evidence;
- validator-call and token targets from item 6 are met or any misses are explicitly explained;
- production and test reductions from items 7 and 8 are demonstrated.

Do not run the 10-case full suite until these conditions hold.

---

## [x] 12. Run and inspect the final 10-case suite

Once focused verification passes:

1. run all 10 cases sequentially;
2. run the complete consolidated deterministic test suite;
3. inspect every transcript, `run.json`, and trace—not only the CSV;
4. compare code, test, token, repair, fallback, and drop metrics with the item-1 baseline;
5. resolve regressions before marking this block complete.

### Final completion criteria

- all deterministic tests pass;
- all 10 cases complete;
- suite summary contains exactly 10 correctly identified rows;
- no invalid public vote or blocker transition is printed;
- no detected unsupported concrete claim remains printed;
- no controller-facing fallback language is present;
- public support, threads, stance changes, and votes derive from accepted `VisibleEvidence` only;
- validator use is materially lower than the reviewed baseline;
- production LOC is materially lower, not merely redistributed;
- unit-test count and LOC are substantially reduced while preserving unique safety invariants;
- remaining transcript oddities are documented for the next consensus/public-evidence or final cross-cutting review rather than patched with another broad validation layer.