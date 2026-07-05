# TODO: active implementation plan

This file is the authoritative work queue for the option-grounded multi-user decision simulator. It is not a changelog. Keep only open work here. Remove an item only after code inspection plus fresh evaluation logs show the issue is solved.

## 0. Project framing

The project is an **option-grounded multi-user decision simulator**, not a generic chatbot and not an open-ended society simulator.

A run should follow this pipeline:

```text
one-line topic or manual environment
  -> fixed option-grounded decision environment
  -> 2-7 configurable simulated users
  -> controller selects speaker / addressee / dialogue act / option focus
  -> LLM renders one visible utterance unless the controller owns a procedural/decision line
  -> observer updates public state from visible text
  -> discussion narrows through reactions, concerns, stance movement, reservations, and votes
  -> outcome = successful / majority / unresolved from visible transcript commitments only
```

Final states remain exactly:

- `successful`: all sims visibly agree on the same winning option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement is reached after bounded discussion and narrowing.

Do **not** add a fourth outcome such as `successful_but_faulty`. If an option violates a participant's blocker or still lacks enough acceptance, the transcript should show that participant refusing or staying elsewhere, which naturally yields `majority` or `unresolved`.

Participant parameters must remain behaviorally visible: engagement, initiative, responsiveness, verbosity, stubbornness, directness, and compromise tendency should affect turn-taking, response timing, stance movement, and willingness to compromise.

Speaking should **not** be mechanically balanced. Dominant/high-engagement/high-initiative sims may speak more. Quiet sims should not disappear. Opening and vote rounds are intentionally more uniform and should be excluded from most trait-realization judgments.

Use `gpt` as the LLM provider for dialogue generation unless explicitly testing provider differences.

## 1. Implementation protocol

1. Work one issue at a time, in the priority order below.
2. Before editing an issue, inspect the relevant code.
3. Prefer deterministic controller/state logic over additional LLM calls.
4. Do not solve quality problems by broadly adding prompt text. Prompt changes should be local, act-specific, and shorter where possible.
5. Validate with transcript behavior and `run.json`, not execution success alone.
6. After each fix, run static checks:

```powershell
py -m py_compile main.py run_eval_suite.py src\*.py
```

7. Run targeted evaluation for the touched behavior. Before removing any issue from this file, run and inspect:

```powershell
py run_eval_suite.py --full
```

8. Inspect at least these artifacts manually: `logs_eval_suite/eval_suite_runs.csv`, `run.json`, `transcript.md`.
9. After a verified fix, update `README.md`, `CLAUDE.md`, and the relevant `info/*.md` files. Then remove or narrow the completed item here.

## 2. Open work

The 2026-07-06 behavioral round closed the previous P1-P9 and P11 items (shorter trait-shaped utterances, question-chain reduction, group-size-aware naming, free-discussion dominance, agreeable manual blockers, earned switches, the issue ledger, bounded compromise wording, cheaper grounding, and the new diagnostics). Validation: full 12-case suite on 2026-07-06 (`logs_eval_suite/eval_suite_runs.csv`) — avg words/turn 12.4-16.4, repeated_unknown_mentions 0 everywhere, final_blocker_violations 0 everywhere, n=2 name-prefix rate 0.0, top free-discussion share 0.26-0.53, unsupported printed turns 0 in 10/12 cases.

### O1 — Simplify code paths where fixes accumulated (was P10)

The controller has many local patches: split-vote handling, post-reservation decisions, continuation guards, grounding repair, option alias parsing, surface style suppression, deadlock protocol, trait-share routing, and now the issue ledger. Behavior is good but harder to reason about.

Direction: do not rewrite the architecture. Isolate or simplify persona constraints/preferences, thread state, turn routing priority, stance eligibility, surface flags, and validation fallback — one area at a time, each protected by a fresh `--full` suite run before and after. Static compile is not enough.

### O2 — Grounding judge false positives on short grounded lines (narrowed from P9)

Two lines in the f05 suite case were flagged UNSUPPORTED_FACT although fully card-grounded ("Good filter speed, no espresso option."). Non-blocking flags print anyway, so this only pollutes `unsupported_fact_flags` / `unsupported_printed_turns`. Direction: check the grounding-judge prompt on terse fragment lines before touching thresholds; do not weaken the deterministic asserted-workaround path.

### O3 — Bounded option combinations: keep monitoring (narrowed from P8)

`propose_compromise` turns now carry an explicit one-option instruction, and split/deadlock cases in the suite ended on single-option winners everywhere. Blend proposals ("A for food, B for parking") appeared before the instruction landed and have not been re-observed since. Keep an eye on split runs; if blends reappear, add a deterministic observer check instead of more prompt text.

### O4 — switch_explanation_rate misses em-dash reasons (metric artifact)

The `has_reason` detector keys on because/since/for-style markers; deterministic decision lines like "Okay, X works for me; Y clearly isn't getting the group there." carry a visible reason but are not counted. `switch_bridge_rate` (the P6 signal) is unaffected and stayed 1.0 across the suite. Fix the detector, not the phrasing menus.

## 3. Literature usage guidance

Use the literature selectively. Do not implement papers 1:1.

- ConvLab3: component idea only (environment, policy/controller, simulator, evaluation) — already followed.
- MUCA: multi-user routing (what to say, when, to whom) — improve controller state, never add a parallel system.
- Generative Agents: lightweight persistent state only. No full reflection/memory loops.
- SOTOPIA: judge social plausibility, not just fluent text.

## 4. Non-goals

Do not prioritize: a full agenda simulator; Generative Agents-style memory/reflection; open-ended society simulation; new outcome labels; broad prompt expansion; more personality traits; provider comparisons before the `gpt` baseline is revalidated; large rewrites unrelated to the open items above.
