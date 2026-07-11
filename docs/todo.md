# TODO — Closeout Fixes and Remaining Code Blocks

## Purpose

Finish the currently known correctness issues, then review the remaining untouched or only lightly reviewed code blocks one by one.

The established simulator generation, option generation, dialogue phases, routing model, thread model, traits, and three outcomes must remain intact.

This is **not** another architecture rewrite. Do not add new folders or split files merely to make them smaller. Prefer deletion, consolidation, and clearer ownership inside the current structure.

Work through the items in order. Complete and verify one item before moving to the next. Mark completed items with `[x]` and keep brief implementation notes below them.

---

# Part A — Immediate correctness fixes

## [x] 1. Prevent repair questions from leaving false active threads

> **Done (2026-07-11):** Question-thread creation in the observer is now phase-gated to
> `OPENING/DISCUSSION/NARROWING`; voting, compromise-repair, and closing questions belong to
> the bounded decision flow that asked them and open no ordinary question threads. Regression
> tests: a repair probe + its answer in VOTING and COMPROMISE_REPAIR leave zero active threads;
> ordinary discussion questions still open threads. 198/198 tests pass.

### Problem

Questions asked during `VOTING` or `COMPROMISE_REPAIR` can create ordinary question threads. The repair flow answers them independently, so those threads may remain falsely `hot` or `cooling` at closing.

### Required change

- Do not create normal discussion question threads from voting or repair turns.
- Let the bounded repair flow own those exchanges.
- Ensure completed repair exchanges leave no unrelated active question thread behind.

### Verify

- Add a deterministic regression test covering a repair question followed by its answer.
- Confirm no false hot/cooling question remains at closing.
- Preserve ordinary discussion question routing.

---

## [x] 2. Require semantic relevance before an answer closes a question

> **Done (2026-07-11):** `_observe_question_answers` no longer trusts the routed act: a
> question cools only when the required respondent's accepted text overlaps the thread's
> focused options or matches its normalized issue key (shared `_issue_relevant` helper, same
> relevance logic concern threads use); fallback lines never count. New tests: a ROUTED answer
> with unrelated text leaves the question hot, and an issue-key-related answer cools a
> focus-less group question. Existing direct/group answer-resolution tests unchanged and
> passing. 200/200 tests pass.

### Problem

A routed `answer` still receives too much trust from controller intent. An accepted but irrelevant response can potentially move a question thread out of `hot` state.

### Required change

A question may only be treated as answered when the accepted text visibly relates to the question, for example through:

- the question's focused option,
- its normalized issue key,
- or another explicit parsed/validated answer relation.

The routed act alone must not be sufficient.

### Verify

- Add a test where the required respondent gives an unrelated but otherwise valid statement.
- The question must remain hot.
- Existing valid direct and group answers must still resolve correctly.

---

## [x] 3. Fix opening contrasts that promote the rejected routed option

> **Done (2026-07-11):** The opening lean now only ever promotes a POSITIVELY named option:
> options the same accepted turn soft/hard-rejects are excluded, the routed favorite wins only
> when named positively, a unique positive alternative wins otherwise, and ambiguous or
> all-negative naming moves nothing. Root-causing the test also exposed a parser attachment
> bug: a soft objection bound to `option_refs[0]` (alias-match order), so "A seems too
> expensive, while B fits better" objected to B — objections now bind to the option nearest
> the objection phrase (`_nearest_option`). Four tests cover rejected-routed/supported-
> alternative, normal supported opening, ambiguous multi-option, and no-positive-option.
> 203/203 tests pass.

### Problem

An opening may mention the routed option negatively and support another option, but the routed option can still become the participant's lean merely because it appears in the text.

Example:

```text
A seems too expensive, while B fits us much better.
```

### Required change

- Never promote an option that the same accepted turn visibly soft-rejects or hard-rejects.
- Prefer the uniquely positively supported option when one exists.
- Keep ambiguous multi-option openings conservative rather than guessing.

### Verify

Add tests for:

- routed option rejected, alternative supported,
- routed option supported normally,
- ambiguous multi-option opening,
- no visible positive option.

---

## [x] 4. Improve issue-key normalization for equivalent concerns

> **Done (2026-07-11):** `normalize_issue_key` now derives keys in the mandated order:
> option attribute key > focused option's normalized card concern > its card upside >
> deterministic category > sig fallback. Concern/upside keys are built from the CARD's
> distinctive tokens (stopwords/generic/evaluative/name/alias tokens dropped), so every
> paraphrase of a card-listed issue converges on one key (`concern:...`/`upside:...`);
> matching is restricted to the focus options so another card's concern cannot capture the
> focused issue. Fully deterministic, no LLM. New tests: paraphrase convergence, distinct
> issues on one option stay separate, upside relevance, cross-card capture guard; existing
> attr/category/sig, reactivation, and repetition-suppression tests unchanged. 207/207 pass.

### Problem

Equivalent concerns can receive different fallback signatures and therefore create several separate threads.

### Required change

Derive issue keys in this order:

1. matching option attribute key,
2. matching normalized option concern,
3. matching normalized option upside when relevant,
4. existing general concern category,
5. free-form fallback signature only as a last resort.

Keep the process deterministic. Do not add an LLM summarizer.

### Verify

- Equivalent paraphrases about the same option issue produce the same key.
- Different concerns about the same option remain separate.
- The same issue on different options remains option-specific.
- Thread reactivation and repetition suppression still work.

---

## [x] 5. Make coverage use independent semantic evidence

> **Done (2026-07-11):** Coverage now updates per option from parsed evidence: every named
> option counts a mention; an option the line soft/hard-rejects counts an objection (the old
> blanket "CONCERN act → objection for every ref" is gone); a non-challenged option counts a
> reason when the line commits to it (vote/accept) or carries support evidence — a realized
> comparison (`realized_comparison`, so comparative QUESTIONS count), a supportive act label,
> or visible benefit-claim wording (new `parsing.has_support_claim`). `act_type` remains for
> routing/reporting only. Five new tests: comparative question, answer-with-objection,
> mixed support+concern opening (split evidence), neutral mention, ordinary support/compare.
> 212/212 tests pass.

### Problem

Coverage still relies too heavily on one dominant `act_type`. A multi-function utterance can contain useful support, concern, comparison, or mention evidence that is ignored because another act won precedence.

### Required change

Update coverage from parsed semantic evidence rather than only the dominant act label:

- option mentions,
- positive support evidence,
- objection evidence,
- comparison evidence.

Keep `act_type` for routing and reporting, but do not make it the only coverage signal.

### Verify

Add tests for:

- comparative questions,
- answers containing objections,
- openings containing both support and concern,
- neutral mentions,
- ordinary support and comparison turns.

---

## [x] 6. Fix the question-answer evaluation off-by-one error

> **Done (2026-07-11):** `_answered_by_target` sliced `state.turns` by the 1-based turn index
> (list positions are 0-based), skipping the immediately following turn. It now iterates all
> turns and counts any with `turn.index > question_turn.index`. Three regression tests:
> immediate answers count, pre-question turns and the question turn itself do not, and a
> late answer counts as answered but not as prompt (window logic intact). 215/215 tests pass.

### Problem

The evaluation search starts one position too late and can skip the immediate answer following a question.

### Required change

Correct `_answered_by_target` so every turn with:

```text
turn.index > question_turn.index
```

is eligible, including the immediately following turn.

### Verify

- Add or update the metric regression test.
- Confirm immediate answers count correctly.
- Confirm earlier turns and the question turn itself do not count.


### Final verification

Run:

```powershell
py -m compileall -q main.py src eval tests
py -m pytest -q
```

Then run the full LLM evaluation suite and record:

- pass/failure count,
- outcome distribution,
- unsupported printed turns,
- fallback count,
- repair count,
- intended/realized act distribution,
- visible stance shifts.

---
