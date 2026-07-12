# TODO — Finish and simplify validation migration

## Purpose

The previous parsing/validation migration added useful foundations, but it is not yet a finished simplification. The current repository contains both the new `VisibleEvidence` path and substantial legacy `DialogueAct`/regex interpretation. This creates duplicated semantic authority, unnecessary source growth, excessive validator token usage, and cases where observer state, consensus/public support, and validation disagree.

This TODO contains only the remaining work. Completed items from the previous `docs/todo_validation.md` are intentionally omitted.

The goal is to finish the migration **subtractively**:

- make validated visible evidence the sole semantic authority;
- remove obsolete parsing, validation, observer, prompt, model, and compatibility code;
- validate only when semantic interpretation is actually needed;
- substantially reduce validator calls, prompt size, latency, and tokens;
- preserve strict public-vote, blocker, grounding, and speaker-local state guarantees;
- restore reliable command-line runs;
- fix automatic scenario failure caused only by an invalid generated alias without splitting the full setup call prematurely.

Do not redesign the already reviewed scenario, persona, controller, routing, discussion, narrowing, voting, or utterance-generation architecture unless a concrete dependency in this TODO requires a small correction.

---

## Non-negotiable semantic rules

1. Controller intent defines what the turn was requested to express.
2. The visible accepted utterance defines what was publicly said.
3. Validated evidence extracted from that visible utterance is the only semantic input allowed to update public dialogue state.
4. Hidden intent must never count as support, concern, compromise, vote, blocker, switch, or stance movement when the visible utterance does not express it.
5. The participant whose accepted utterance contains the evidence is the only participant whose private ranks, preference, vote, or blocker state may change from that evidence.
6. Another participant’s statement may create public pressure, a challenge, or a thread response, but must not silently alter someone else’s private ranks.
7. Direct votes and vote switches remain conservative: exactly one visible commitment target must be identifiable.
8. Unsupported concrete claims and cross-option fact transfers must not be accepted merely because the controller intended a grounded move.
9. No generator self-report metadata is required or trusted.
10. There are exactly two configurable LLM roles:

```yaml
llm:
  dialogue: "gpt"
  validator: "uni"
```

The same provider may be used for both. Do not add a third runtime checker.

---

## Required procedure for every item

For each item:

1. Inspect the relevant current implementation and tests.
2. Identify the smallest coherent change.
3. Implement only that item.
4. Add or update focused tests.
5. Run the targeted tests.
6. Generate a few representative dialogue samples where the behavior is visible when the item affects runtime dialogue.
7. Inspect the transcript and structured trace output, not only test success.
8. Resolve regressions before moving to the next item.

Do not run the expensive full evaluation suite after every item.

After the implementation items are complete:

- run the complete deterministic test suite;
- run a bounded representative dialogue sample set;
- compare validator cost against the current baseline;
- do **not** run the full expensive evaluation suite

---

# Implementation items

## [ ] 1. Capture the current failure and cost baseline

### Relevant files

- `main.py`
- `config.yaml`
- `src/logger.py`
- `eval/eval.py`
- current logs under `logs/` and `eval/logs_eval_suite/`
- focused tests for CLI, setup, validation metrics, and trace output

### Required work

Before changing behavior, record a compact baseline that later items can compare against.

Capture:

- complete deterministic test result;
- production LOC for the affected subsystem listed above;
- validator calls per accepted participant turn;
- validator input/output tokens;
- dialogue input/output tokens;
- validator share of total input tokens;
- repair, fallback, and drop rates;
- intended-move realization;
- option-focus agreement;
- accepted turns that still carry realization issues;
- examples where legacy `DialogueAct` and `VisibleEvidence` disagree.

Reproduce and add focused regression coverage for:

- explicit CLI topic execution;
- topic-file execution;
- piped topics;
- configured manual environment execution;
- automatic scenario generation failing only because `short_name` is invalid;
- public support being present in `VisibleEvidence` but missing from `consensus.public_support()` because the legacy act says `COMMENT`.

Do not run a new full evaluation suite for this baseline. Use the existing logs and a few bounded runs.

### Completion criteria

- The current semantic inconsistency and token cost are measurable.
- The alias-only setup failure is reproducible in a focused test or stubbed setup path.
- Later changes can show objective deletions and cost reduction.

---

## [ ] 2. Restore predictable command-line run behavior

### Relevant files

- `main.py`
- `src/config_loader.py` only if required
- `src/builders.py` only if an explicit runtime environment override is required
- `README.md`
- `info/08_configuration_and_running.md`
- new focused CLI tests

### Required behavior

Restore these commands without silently ignoring user input:

```powershell
py .\main.py
py .\main.py "Book a flight from Miami to Stockholm"
py .\main.py topics.txt
"Book a flight from Miami to Stockholm" | py .\main.py
```

Use this precedence:

1. An explicit CLI topic, topic file, or piped topic requests automatic scenario generation for that topic, even when `environment.mode` in `config.yaml` is `manual`.
2. With no explicit or piped topic:
   - `environment.mode: manual` runs the configured manual environment once;
   - `environment.mode: auto` prompts interactively for a topic.
3. `participants.mode` remains independent. A CLI topic must work with either automatic or manually configured participants.
4. Never print “CLI topic ignored” for a valid explicit topic.
5. Batch topic files continue to ignore blank lines and `#` comments and run every topic in order.
6. A failure for one topic must identify that topic clearly. Preserve the current exit behavior unless tests show that batch mode should continue after individual failures; do not redesign batch error policy in this item.

Do not add unnecessary CLI frameworks or many flags. Keep the entry point small.

### Focused tests

Cover:

- no args + automatic environment;
- no args + manual environment;
- explicit topic + automatic environment;
- explicit topic + manual-configured environment, where the explicit topic wins;
- topic file;
- piped single and multiple topics;
- manual participants combined with a CLI-generated automatic scenario.

### Completion criteria

- All four documented commands work.
- Explicit command input is never silently discarded because of configuration mode.
- README and running documentation match actual precedence.

---

## [ ] 3. Repair invalid generated aliases without regenerating the full scenario

### Relevant files

- `src/builders.py`
- `src/aliases.py`
- `src/prompts.py`
- `src/llm_client.py` only if a small dialogue-role helper is needed
- setup token logging
- alias and scenario-generation tests

### Current failure

A substantively valid automatic scenario is discarded when one generated `short_name` is unusable, for example:

```text
name: Lufthansa flight via London
short_name: London Stop
```

The alias validator correctly rejects `Stop` because it is not part of the option name, but the whole scenario is then regenerated. Repeated variants such as `London Stop`, `Lufthansa Stop`, and `London stop` can exhaust all scenario attempts.

### Required change

Do not split the full scenario-generation call at this stage. The observed failure is caused by alias handling, not evidence that the entire setup prompt is too complex.

Use this sequence:

```text
generate full scenario
    -> validate substantive scenario fields
    -> collect invalid or duplicate short aliases
    -> run one small alias-only repair for the affected options
    -> deterministically validate repaired aliases
    -> keep the original scenario and options
```

Alias repair must:

- use the `dialogue` LLM role;
- receive only the option IDs, full names, rejected aliases, and aliases already used by other options;
- return concise unique aliases;
- preserve the established rule that aliases use words from the option name and are not arbitrary invented labels;
- not silently derive aliases by clipping the first words of the name;
- not regenerate shared context, option facts, upside, concern, or personas;
- have a small explicit retry limit;
- produce a precise final error if alias-only repair still fails.

Add setup diagnostics that distinguish:

- substantive scenario failure;
- option-field failure;
- invalid alias;
- duplicate alias;
- alias-repair failure.

After this fix, generate several automatic scenarios across varied topics. Only reconsider splitting the full setup call if the logs still show frequent substantive scenario failures unrelated to aliases. Do not pre-emptively split it.

### Completion criteria

- The reported Miami-to-Stockholm type of failure succeeds when only the alias is invalid.
- A bad alias no longer discards an otherwise valid scenario.
- Alias repair uses far fewer tokens than full scenario regeneration.
- Truly invalid substantive scenarios still fail or retry normally.

---

## [ ] 4. Make `VisibleEvidence` the sole public semantic authority

### Relevant files

- `src/consensus.py`
- `src/observer.py`
- `src/dialogue.py`
- `src/models.py`
- public-support, narrowing, proposal, vote, and outcome tests

### Current problem

Some runtime consumers still use legacy `turn.act` fields while others use `turn.evidence`. This allows the same accepted utterance to be interpreted differently by observer state and public-support/consensus logic.

Known examples:

- `consensus.public_support()` reads `act.accepts`, `act.explicit_vote`, and `act.act_type is SUPPORT` rather than evidence support/commitments;
- compromise proposal counts read `turn.act.offers_compromise`;
- compatibility properties in `TurnRecord` may still fall back to legacy act fields.

### Required change

Migrate every public semantic consumer to accepted `VisibleEvidence`:

- support and acceptance;
- concern and hard rejection;
- comparisons;
- proposals and compromises;
- votes and vote switches;
- blocker introduction and resolution;
- question/answer evidence;
- stance movement;
- thread-response relevance.

`DialogueAct` may temporarily remain as a derived display/trace label, but it must not decide state, public support, narrowing, consensus, proposal counts, or outcome.

Add integration tests where:

```text
legacy act = COMMENT
validated evidence = support A
```

and verify that:

- public support sees A;
- observer records support for A;
- narrowing and consensus consumers see the same evidence;
- no consumer records support for the legacy act’s interpretation.

Also test the inverse: legacy parser claims support, validated evidence does not. No public support may be created.

### Completion criteria

- All semantic state/public-evidence consumers use one accepted evidence object.
- There is no runtime path where `DialogueAct` overrides validated evidence.
- Consensus, observer, thread state, and traces agree on the same turn.

---

## [ ] 5. Remove semantic reparsing from `observer.py`

### Relevant files

- `src/observer.py`
- `src/interpreter.py`
- `src/models.py`
- `src/parsing.py`
- thread, switch, stance, and observer tests

### Current duplication

`observer.py` still imports or uses legacy parsing helpers for behavior such as:

- `evidence_from_dialogue_act()` fallback;
- `parse_dialogue_act()`;
- `commitment_has_reason()`;
- `switch_bridge_ok()`;
- raw-text vote-change regexes;
- separate raw-text issue/thread relevance checks.

### Required change

The observer must consume the final accepted evidence and controller/thread metadata only. It must not reinterpret natural language.

Move any genuinely required fields into the accepted evidence contract before deleting observer parsing, for example:

- switch source and target;
- whether a visible bridge/reason exists;
- response target/thread ID;
- issue/option relevance;
- blocker resolution target.

Do not add broad new fields merely to mirror old helpers. Keep only evidence that changes state or is required for traceability.

Delete observer fallback conversion from `DialogueAct`. Tests must construct accepted evidence explicitly.

### Completion criteria

- `observer.py` performs state transitions, not text classification.
- No natural-language regex or parser helper remains in observer state-update paths.
- Switch, thread, and stance tests still pass using explicit evidence fixtures.

---

## [ ] 6. Reduce `parsing.py` to deterministic critical responsibilities

### Relevant files

- `src/parsing.py`
- `src/interpreter.py`
- `src/validation.py`
- `src/prompts.py`
- parsing, resolver, commitment, blocker, and semantic fixture tests

### Keep

Retain deterministic code where high precision and stable public rules are valuable:

- option/full-name/short-name/letter resolution;
- alias spans and textual order;
- participant/addressee resolution where deterministic;
- unambiguous public-context reference resolution;
- strict visible commitment/vote detection;
- strict hard blocker detection when used as a critical safeguard;
- malformed output and exact structural checks;
- exact option/attribute/value helpers used by grounding.

### Remove or demote from authority

After items 4 and 5 migrate all consumers, remove broad phrase-list/regex authority for:

- ordinary support;
- ordinary concern;
- comparison realization;
- answer relevance;
- concession and softening;
- compromise/proposal realization;
- general commitment reasons;
- dominant dialogue-act classification.

Delete helpers that have no remaining runtime consumer, including legacy compatibility adapters and tests that only preserve obsolete parser vocabulary behavior.

If a primary act is still useful for logging, derive it from accepted evidence in one small deterministic function. Do not reparse the utterance to obtain it.

### Completion criteria

- `parsing.py` is a critical deterministic parser/resolver, not a second semantic interpreter.
- Natural soft semantics have one authority.
- Removed parser behavior is replaced by evidence-based tests, not retained through compatibility shims.

---

## [ ] 7. Minimize the evidence contract and validator response schema

### Relevant files

- `src/models.py`
- `src/interpreter.py`
- `src/prompts.py`
- `src/validation.py`
- interpreter, evidence model, prompt, and assessment tests

### Current problem

The validator is asked to return a large universal schema for every candidate, including categories irrelevant to the requested move. It also returns fields that code recomputes, such as `primary_act` and `intended_move.realized`.

### Required change

Remove redundant validator outputs:

- do not ask the LLM for `primary_act` when it can be derived from verified evidence;
- do not ask the LLM for a separate intended-move realization verdict when validation can compare verified evidence with controller intent;
- remove explanation fields that are not used for repair, trace, or state;
- remove evidence fields that have no state, validation, grounding, or diagnostic consumer.

Use a compact common result plus an intent-specific semantic payload instead of requesting every possible category on every turn.

The common portion should contain only what is needed across calls, such as:

- exact evidence spans;
- option binding;
- ambiguity;
- factual claims that require grounding;
- target/thread relevance when a target exists.

The intent-specific portion should request only the semantic evidence relevant to the intended state-changing move, while still allowing critical visible commitments/blockers to be caught deterministically.

Examples:

- support turn: support target/strength/reason span plus grounding claims;
- concern turn: concern target/issue/severity span plus grounding claims;
- compare turn: compared options/dimensions/direction plus grounding claims;
- answer turn: target relevance and answer span plus grounding claims;
- compromise turn: proposed target/condition plus grounding claims.

Do not create a separate validator call for each category.

### Completion criteria

- Validator prompt and response are substantially smaller.
- Removed fields are not duplicated elsewhere.
- Multi-function turns remain representable where they affect state or thread continuity.
- State-critical evidence remains explicit and span-backed.

---

## [ ] 8. Introduce selective semantic validation with deterministic fast paths

### Relevant files

- `src/dialogue.py`
- `src/interpreter.py`
- `src/validation.py`
- `src/config_loader.py`
- `config.yaml`
- `src/logger.py`
- validation mode, pipeline, and token-accounting tests

### Target configuration

Support an explicit validation mode, with selective behavior as the normal runtime mode:

```yaml
validation:
  mode: selective   # selective | full
```

`full` is for debugging/evaluation. It must not be required for normal generation.

### Deterministic fast paths

Skip the validator LLM when deterministic code can fully establish correctness and no soft semantic state update is required, for example:

- direct unambiguous vote to the required option;
- direct unambiguous vote switch with already approved grounded reason data;
- simple explicit hard blocker using known grounded option facts;
- process or closing text that cannot change option state;
- non-state-changing light comments;
- exact factual statements fully matched to normalized scenario data;
- deterministic fallback forms whose complete evidence is known from construction and then checked.

Do not skip semantic validation when the turn may change state through open-ended natural language, including:

- support;
- concern;
- comparison;
- answer relevance;
- softening/concession;
- compromise/proposal;
- ambiguous reference;
- ordinary discussion-phase stance movement;
- factual claims not fully verifiable deterministically.

A skipped validator call must be explicit in trace output, including the fast-path reason. It must not silently fabricate semantic evidence from hidden intent. A fast path may accept only evidence deterministically visible in the utterance.

Keep a maximum of one semantic validator call per candidate. A repaired candidate must be rechecked when its semantics are not deterministically verifiable, but do not perform several validator calls for separate concerns.

### Cost goals

On representative runs:

- average validator calls per accepted participant turn must be below 1.0;
- validator input tokens must no longer dominate total input tokens;
- target validator input tokens at or below dialogue input tokens before approving a full evaluation run;
- retain separate dialogue and validator token reporting.

If these goals are not met, continue simplifying prompt/context size before running the expensive suite.

### Completion criteria

- Normal runs do not validate every turn through the full LLM schema.
- Full mode remains available for diagnostics.
- Selective mode preserves state/public-evidence correctness in focused tests and transcripts.

---

## [ ] 9. Reduce validator prompt context and repeated tokens

### Relevant files

- `src/prompts.py`
- `src/interpreter.py`
- `src/dialogue.py`
- `src/logger.py`
- prompt snapshot and token tests

### Required change

The current validator prompt repeatedly sends a large annotation manual and broad scenario context. Reduce it to the minimum context needed for the specific candidate.

Apply all relevant reductions:

- send compact normalized option facts rather than verbose option cards when possible;
- include only focused options plus any explicitly mentioned options, unless the move genuinely requires the full board;
- include only the target turn/thread excerpt needed for response relevance;
- do not send unrelated recent dialogue;
- shorten schema instructions after deterministic verification guarantees are in place;
- use concise controlled vocabularies;
- cap validator output tokens tightly;
- avoid repeating controller data that validation does not use;
- do not include private persona information unless a concrete validation rule requires it;
- do not ask the validator to repeat the utterance or explain every accepted field.

Do not rely on provider-specific prompt caching as the only optimization. The compact prompt must be efficient across `gpt`, `uni`, `gemini`, and other configured providers.

### Completion criteria

- Average validator input tokens per validator call are materially lower than the current roughly 1.4k–1.5k range.
- Prompt tests verify that irrelevant options/context are omitted.
- Validator accuracy on the focused semantic fixtures does not regress.

---

## [ ] 10. Simplify and tighten grounding without adding another LLM call

### Relevant files

- `src/interpreter.py`
- `src/validation.py`
- `src/models.py`
- `src/prompts.py`
- grounding tests and trace metrics

### Required design

Grounding remains hybrid but must use the same selective validator call when semantic judgment is needed. Do not add a second grounding endpoint or a third checker.

Use deterministic verification for:

- exact option/attribute/value relations;
- numbers, prices, capacities, dates, durations, distances, and units;
- cross-option value transfer;
- reproducible arithmetic;
- claims that directly match normalized shared context or option facts.

Use the validator only for ambiguity such as:

- fact versus opinion;
- qualified inference;
- uncertainty;
- whether a sentence introduces an unstated concrete capability, event, or logistical detail.

A broad `opinion` label must not bypass an embedded concrete premise. For example:

```text
Automatic reminders make Teams easier to use.
```

contains a factual capability claim (`automatic reminders`) and a subjective conclusion (`easier to use`). They must be assessed separately.

Likewise, an inference must identify its source fact. Code must verify that the cited source relation exists.

Required behavior:

- final generated, repaired, and fallback candidates use the same grounding guarantees;
- validator operational failure must not fail open for unverified concrete claims;
- exact unsupported spans and missing/correct source relations must be available to targeted repair and traces;
- `unsupported_printed_turns == 0` must reflect final checked claims, not merely absence of a retained issue code;
- do not force a costly claim audit on turns that contain no concrete factual assertion and cannot affect grounding.

### Completion criteria

- Known invented capabilities and cross-option transfers are blocked.
- Reasonable opinions, uncertainty, and source-backed inference remain natural.
- Grounding uses no additional LLM call beyond the semantic validator call already required for that candidate.

---

## [ ] 11. Repair only genuine blocking semantic failures

### Relevant files

- `src/validation.py`
- `src/dialogue.py`
- `src/prompts.py`
- `src/models.py`
- repair and assessment tests

### Required change

Keep the action model:

- `ACCEPT`
- `ACCEPT_WITH_METRIC`
- `REPAIR`
- `FALLBACK`
- `DROP`

But simplify its use.

Repair only when the visible candidate cannot safely realize the required move, for example:

- required option focus is contradicted or missing;
- required vote is absent, conditional, ambiguous, or targets the wrong option;
- answer does not address the required target;
- switch lacks required visible movement/reason;
- unsupported concrete claim;
- invalid blocker/acceptance contradiction;
- unresolved ambiguous reference that would change state;
- malformed or empty utterance.

Do not repair merely because:

- a derived primary label differs from the intended label while the requested function is visibly realized;
- the turn contains an additional harmless function;
- telemetry is incomplete but state/public evidence is safe;
- wording differs from prompt examples.

Repair receives exact failures and exact offending spans. Keep at most one targeted repair attempt by default.

Candidate selection must be severity-based, not issue-count-based. Never replace a safe candidate with one containing an equally numerous but more severe issue.

If repair still has a non-blocking metric issue, it may be accepted. If it still has a blocking issue, use only a safe act-specific fallback or drop it.

### Completion criteria

- Repair rate falls materially in representative runs.
- Repaired turns do not remain printed with unresolved blocking issues such as `ANSWER_DOES_NOT_ADDRESS_QUESTION` or `SUPPORT_NOT_REALIZED`.
- Repair traces clearly state why repair was necessary and what changed.

---

## [ ] 12. Restrict fallback to truthful, state-safe forms

### Relevant files

- `src/validation.py`
- `src/dialogue.py`
- `src/models.py`
- fallback tests and trace output

### Required change

Retain deterministic fallback only where the system can construct truthful public evidence from known grounded data:

- explicit vote;
- explicit vote switch using an already grounded approved reason;
- explicit blocker restatement using a known blocker reason;
- coverage request;
- exact factual answer or comparison when the required facts are present;
- explicit “the listed information does not say” answer when appropriate.

Remove generic fallback that pretends to realize arbitrary support, concern, answer, compromise, comparison, or softening acts without sufficient grounded content.

When no truthful act-specific fallback exists, drop the turn and let the controller continue rather than printing false state evidence.

Every fallback must pass the same final deterministic/semantic/grounding checks applicable to an equivalent generated turn. A fallback must never receive an intended act label only because the controller requested that act.

### Completion criteria

- Fallback text cannot create support, concern, compromise, thread resolution, vote, blocker, or rank movement not visibly present.
- Fallback use is narrow and explainable.
- Dropping an unsafe turn does not corrupt controller/thread state.

---

## [ ] 13. Delete compatibility paths, obsolete model fields, issue codes, prompts, and tests

### Relevant files

- `src/interpreter.py`
- `src/models.py`
- `src/parsing.py`
- `src/validation.py`
- `src/prompts.py`
- `src/dialogue.py`
- `src/observer.py`
- `src/consensus.py`
- related tests and fixtures

### Required cleanup

After the previous migrations are complete, delete stale code rather than leaving it dormant.

Inspect and remove where no longer used:

- `evidence_from_dialogue_act()` runtime/test compatibility paths;
- `DialogueAct` fields that duplicate accepted evidence;
- fallback properties that silently read legacy act fields;
- broad support/concern/comparison/compromise regex helpers;
- raw-text switch-reason/bridge helpers superseded by evidence;
- observer text parsing;
- consensus legacy act consumption;
- validator `primary_act` and self-reported intent-alignment fields;
- unused issue codes and unreachable branches;
- prompt instructions for outputs no consumer uses;
- tests that only preserve deleted implementation details;
- duplicated semantic fixture adapters;
- dead comments referring to previous TODO item numbers or obsolete architecture.

Consolidate tests around behavior and accepted evidence rather than parser phrase catalogs. Do not delete coverage for critical commitments, blockers, grounding, thread continuity, or state isolation.

Run an unused-import/dead-reference search and compile check after deletion.

### Completion criteria

- There is one semantic path from accepted text to state.
- No production runtime consumer requires a compatibility adapter.
- Combined affected production LOC is materially lower than at baseline.
- Tests assert behavior, not coexistence of old and new systems.

---

## [ ] 14. Make validation behavior and cost visible in logs

### Relevant files

- `src/logger.py`
- `eval/eval.py`
- trace/metrics tests
- transcript metadata only where concise and useful

### Per-turn trace requirements

For every participant candidate, log:

- whether validation used a deterministic fast path, selective LLM interpretation, or full mode;
- why an LLM validator call was required or skipped;
- validator provider;
- validator tokens in/out;
- verified visible evidence used for state;
- intended function realization result;
- grounding claims and sources only when relevant;
- assessment action;
- repair/fallback/drop reason;
- whether observer state changed and exactly which speaker-local/public fields changed.

### Per-run summary requirements

Report:

- participant turns;
- validator calls;
- validator calls per accepted participant turn;
- validation skip/fast-path rate;
- validator and dialogue tokens separately;
- validator share of total input tokens;
- repair/fallback/drop rates;
- intended-function realization;
- option-focus agreement;
- unsupported final claims;
- public-evidence/observer consistency failures;
- discussion lean shifts and their source turns.

Do not keep a misleading metric that treats every primary-act label difference as semantic failure. If `act_mismatch` remains, label it as diagnostic only and prioritize intended-function realization and evidence alignment.

### Completion criteria

- The user can determine from `run.json` why each validator call occurred and what it changed.
- Token regressions are visible immediately.
- Zero unsupported printed turns has a clear, defensible meaning.

---

## [ ] 15. Focused end-to-end verification before any full evaluation suite

### Required deterministic commands

Run:

```powershell
py -m unittest discover -s tests
py -m compileall -q main.py src eval tests
```

### Required command-line samples

Verify:

```powershell
py .\main.py
py .\main.py "Book a flight from Miami to Stockholm"
py .\main.py topics.txt
"Book a flight from Miami to Stockholm" | py .\main.py
```

Use both:

- automatic and manual environments;
- automatic and manual participants;
- same-provider dialogue/validator configuration;
- cross-provider dialogue/validator configuration when the endpoint is available.

### Bounded representative dialogue samples

Run a small set, not the full evaluation suite, covering:

- natural support and concern;
- comparison-heavy discussion;
- direct question and answer;
- menu-less vote;
- vote switch;
- blocker and blocker resolution;
- compromise/proposal;
- concrete grounding and deliberate unsupported detail;
- automatic scenario with an initially invalid alias repaired separately;
- validator outage/failure through a stubbed or controlled test path.

For each sample inspect:

- transcript naturalness;
- validated evidence;
- public support/consensus agreement;
- speaker-local rank and preference changes;
- thread opening/resolution;
- grounding decisions;
- repair/fallback/drop behavior;
- validator call and token cost.

### Cost gate before full evaluation

Do not run `eval/run_eval_suite.py` automatically.

First require focused samples to show:

- validator calls per accepted participant turn below 1.0 on average;
- validator input tokens no greater than dialogue input tokens on the representative aggregate;
- no known dual-authority discrepancy;
- no blocking issue printed after repair;
- no false state evidence from fallback;
- automatic CLI generation works reliably;
- alias-only problems do not regenerate the full scenario.

If the cost gate fails, return to items 7–10 and simplify further.

Once the gate passes, summarize the estimated cost of the full suite and wait for explicit user approval before running it.

---

## [ ] 16. Update existing documentation and remove the obsolete TODO

### Relevant files

- `README.md`
- `CLAUDE.md`
- `info/01_scenario_generation.md`
- `info/05_discussion_and_decision.md`
- `info/06_consensus_and_outcomes.md`
- `info/07_evaluation_and_logging.md`
- `info/08_configuration_and_running.md`
- `config.yaml` comments
- `docs/todo_validation.md`

### Required documentation

Document the final actual behavior:

- CLI/config precedence;
- automatic alias-only repair;
- two configurable LLM roles;
- selective versus full validation;
- deterministic fast paths;
- visible validated evidence as the sole state authority;
- no generator metadata;
- grounding policy;
- repair and fallback policy;
- validation token/call metrics;
- full evaluation suite is an explicitly approved costly operation.

Once this TODO is complete and the relevant information has been integrated, remove or archive `docs/todo_validation.md` so the repository does not contain a mostly completed obsolete implementation plan alongside the current TODO.

Do not add redundant documentation files.

### Final completion criteria

- Documentation matches code and tested command behavior.
- No completed old TODO remains as the apparent active plan.
- The repository presents one coherent parsing/validation architecture.

---

# Final target architecture

The simplified runtime path should be:

```text
controller intent
    -> dialogue model generates visible utterance
    -> conservative envelope extraction/cleanup
    -> deterministic option/reference and critical commitment checks
    -> selective semantic validator only when soft meaning or ambiguous grounding matters
    -> deterministic verification of validator spans, option bindings, facts, and transitions
    -> ACCEPT / ACCEPT_WITH_METRIC / REPAIR / FALLBACK / DROP
    -> at most one targeted dialogue-model repair when genuinely required
    -> same relevant checks for repaired/fallback text
    -> append accepted visible turn
    -> observer consumes accepted VisibleEvidence without reparsing
    -> consensus/public support consume the same VisibleEvidence
```

The intended end state is not “more validation.” It is **one coherent evidence path with less code and fewer calls**.