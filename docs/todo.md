# TODO: current open issues

This file is the active implementation guide. It must contain only current open work, not historical fixes. Completed issues should be removed or moved to documentation only if they are important for understanding the current workflow.

## 0. Project framing

The project is an **option-grounded multi-user decision simulator**.

A run should model this pipeline:

```text
one-line topic or manual environment
  -> option-grounded decision environment
  -> 2-7 configurable simulated users
  -> controller selects speaker / addressee / dialogue act / option focus
  -> LLM writes one visible utterance
  -> observer updates public state from visible text
  -> discussion narrows through reactions, concerns, stance movement, reservations, and votes
  -> outcome = successful / majority / unresolved from visible votes only
```

The option board is intentional. It provides concrete facts and makes outcomes observable. Do not broaden the system into open-ended group chat.

## 1. Implementation protocol

1. Work top to bottom through the issues below.
2. Implement one behavioral issue at a time.
3. Prefer deterministic controller logic over more LLM calls.
4. Keep prompts smaller and more act-specific where possible.
5. Validate with generated transcripts and `run.json`, not only execution success.
6. Test all four mode combinations across the round:

```text
auto environment + auto participants
manual environment + auto participants
auto environment + manual participants
manual environment + manual participants
```

7. Include group sizes `n=2`, `n=3`, `n=4`, and at least one `n=5` run before claiming completion.
8. Update relevant `info/*.md`, `README.md`, and `CLAUDE.md` when behavior changes.
9. Do not remove an issue until fresh logs or deterministic code prove it is fixed.
10. For live validation, prefer `py run_eval_suite.py --full` and compare against the latest full-suite logs.

## 2. Open issue 1 — Improve split-vote candidate selection

### Problem

The current system detects no-majority votes and starts a narrowing pass, but candidate selection can be socially implausible. In the latest full logs, some cases tested a one-vote option even when another option already had two votes.

### Evidence

From the latest full evaluation, split-vote handling improved overall, but cases such as no-/light-moderator restaurant runs still showed candidate choice that did not match the visible support structure.

### Correct behavior

Candidate selection should follow clear rules:

```text
if one option has a strict plurality:
    test the leading option first
elif multiple options tie for lead:
    choose the candidate with lower visible resistance / better compromise fit
elif all options are tied:
    choose by compromise potential, not arbitrary order
```

If the first candidate fails and the turn budget allows, test at most one alternative. Do not cycle indefinitely.

### Validation

Inspect transcripts around the first final vote. The tested compromise candidate should be explainable from visible votes and concerns.

## 3. Open issue 2 — Add explicit post-reservation decision steps

### Problem

After a holdout gives a reservation and a supporter responds, the holdout often just repeats the old vote or the system closes without a clear visible stance update.

### Correct behavior

After a reservation response, the holdout should produce exactly one of these visible outcomes:

```text
switch to the tested candidate
stay with the original option
propose one concrete alternative candidate
state that no acceptable compromise exists
```

This should update the internal state and be visible in transcript text. Avoid vague re-votes.

### Validation

In `q01`, `q03`, `q05`, `q06`, and no-/light-moderator split cases, post-reservation turns should clearly indicate switch/stay/alternative before closure.

## 4. Open issue 3 — Validate and improve n=2 deadlock handling

### Problem

The code now contains a two-person deadlock path, but the latest full suite did not trigger it because the n=2 run converged before deadlock. Therefore, the protocol is not validated.

### Correct behavior

Add or adjust an evaluation case with two stubborn manual participants and opposing fixed preferences. A 1-1 tie should trigger:

```text
1. each person states their strongest blocker;
2. each person proposes one condition or concession;
3. if neither moves, unresolved is valid.
```

### Validation

A forced n=2 stubborn-deadlock case should show `two_person_deadlock_attempted = true` and the transcript should include symmetric negotiation before unresolved closure or a justified switch.

## 5. Open issue 4 — Fix option-target confusion in compromise prompts

### Problem

Some compromise reservations attach to the wrong option. A concern that belongs to one option can appear while testing another candidate. This makes negotiation feel incoherent and weakens grounding.

### Evidence

Latest full logs included cases where a concern like “less flexible once booked” appeared while testing a different option. This suggests the prompt or controller does not bind the tested candidate strongly enough.

### Correct behavior

When testing candidate `X`:

- reservations must refer to `X` and its actual facts/tradeoffs;
- comparisons to the speaker’s original favorite are allowed;
- attributes from unrelated options must not be transferred to `X`;
- the prompt should name the candidate and the speaker’s previous option clearly.

## 6. Open issue 5 — Reduce token cost

### Problem

The latest full suite still used roughly 460k input tokens across 12 runs. Cost did not improve compared to the previous suite.

Approximate distribution from token diagnostics:

```text
utterance calls: ~65% of input tokens
grounding calls: ~27%
repair calls:    ~5%
setup/moderator: small share
```

### Correct behavior

Reduce cost without removing behavioral controls.

Prioritize:

```text
smaller participant prompts
fewer grounding calls
compact option facts
deterministic pre-checks before LLM grounding
skip LLM grounding for short non-factual vote lines unless suspicious
reduce repair rate by simplifying act-specific prompts
avoid extra LLM calls for split summaries when deterministic text is enough
```

Do not add prompt complexity to solve dialogue quality.

### Validation

`run.json` and `metrics.csv` should show lower input-token totals by call type. Grounding calls should not account for roughly a quarter of all input tokens unless a run is unusually fact-heavy.

## 7. Open issue 6 — Tighten grounding for unsupported logistical workarounds

### Problem

Grounding is better, but sims still sometimes state unsupported practical details or mitigations, such as shelters, quiet corners, weather reliability, parking, or other logistical fixes not present in the option board.

### Correct behavior

Allowed:

```text
Maybe we could pick a quieter corner, but we do not know if that is possible.
```

Not allowed:

```text
We can pick a quieter corner.
```

Sims may propose hypothetical mitigations only when uncertainty is explicit. Concrete unsupported facts should be repaired or blocked.

## 8. Open issue 7 — Monitor trait-weighted participation without overfitting

### Current state

Trait-weighted participation is much better than earlier versions, especially in manual trait-spread cases. Some auto/auto runs still show weak or negative engagement correlation, but this is not currently the highest-priority issue.

### Correct behavior

Low-engagement sims should be quieter but visible. High-engagement sims should be more active but not accidentally dominant. Opening and voting phases will naturally compress turn-share differences, so judge the discussion phase separately where possible.

### Action

Do not make large changes unless a clear bug appears. Keep the metrics visible and compare across manual/auto runs.

## 9. Non-goals for the next round

Do not prioritize:

- new research-paper integrations,
- new personality traits,
- broad open-domain chat,
- large code rewrites unrelated to the issues above,
- cosmetic transcript polish without behavior changes.

The critical next step is to make disagreement handling more socially plausible while reducing prompt/grounding cost.
